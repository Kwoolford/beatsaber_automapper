#!/usr/bin/env python
"""THE WORKSPACE — build a map incrementally, in the coordinates the brief prints.

The action half of `agent_mapper/`. `brief.py` shows a bar as sixteen cells; this
accepts notes in **exactly the same cells**, so reading and writing share one
coordinate system:

    brief says   59  1:14.02  |x...x.x.x...x.x.| D    I've been dying for my
    you write    59.0 L 1 0 D        # bar 59, slot 0, left hand, col 1, row 0, down

★**Why a session on disk rather than one big call.** 1 300 notes will not survive a
single generation, and an agent that has to hold the whole map in context cannot spend
that context on the music. A session is a plain TSV: append a phrase, check it, move
on. It is greppable, diffable, and a bad section can be cleared and redone without
touching the rest.

★**Everything is stored in SECONDS as well as beats.** 30 % of our maps are at the
wrong tempo, so beats are not portable between analyses of the same song; seconds are.

⚠️`check` runs the same swing simulator the evaluation suite uses. A map that fails
parity is not a rough draft, it is unplayable — run it before export, not after.

Commands:
    init <audio> --name N            start a session (analyses the song, stores the grid)
    add  N --from notes.txt          append notes  (or --bar/--hand/--slots for a run)
    clear N --bars 33-40             remove a bar range and redo it
    status N                         coverage per 8-bar phrase, and what is still empty
    view N --bars 59-62              read back what is placed, next to the stems
    check N                          parity / doubles / reachability
    export N --out map.zip           write a playable .zip

Note format (one per line, '#' comments ignored):
    <bar>.<slot> <L|R> <col 0-3> <row 0-2> <dir>
    dir: U D L R  UL UR DL DR  or  X (dot / any direction)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))

SESSIONS = HERE / "sessions"
BEATS_PER_BAR = 4
SUBDIV = 4

# Beat Saber cut directions.
DIRS = {"U": 0, "D": 1, "L": 2, "R": 3, "UL": 4, "UR": 5, "DL": 6, "DR": 7, "X": 8}
DIR_NAME = {v: k for k, v in DIRS.items()}
ARROW = {0: "^", 1: "v", 2: "<", 3: ">", 4: "\\", 5: "/", 6: "/", 7: "\\", 8: "o"}


def sess_dir(name: str) -> pathlib.Path:
    return SESSIONS / name


def load_session(name: str) -> dict:
    d = sess_dir(name)
    f = d / "session.json"
    if not f.exists():
        print(f"no session '{name}' — run: mapctl.py init <audio> --name {name}",
              file=sys.stderr)
        raise SystemExit(2)
    return json.loads(f.read_text())


def notes_path(name: str) -> pathlib.Path:
    return sess_dir(name) / "notes.tsv"


def read_notes(name: str) -> list[dict]:
    f = notes_path(name)
    if not f.exists():
        return []
    out = []
    for ln in f.read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        bar, slot, hand, col, row, d, beat, t = ln.split("\t")
        out.append({"bar": int(bar), "slot": int(slot), "hand": hand,
                    "col": int(col), "row": int(row), "dir": int(d),
                    "beat": float(beat), "t": float(t)})
    return out


def write_notes(name: str, notes: list[dict]) -> None:
    notes = sorted(notes, key=lambda n: (n["beat"], n["hand"]))
    lines = ["# bar\tslot\thand\tcol\trow\tdir\tbeat\tseconds"]
    for n in notes:
        lines.append(f"{n['bar']}\t{n['slot']}\t{n['hand']}\t{n['col']}\t{n['row']}\t"
                     f"{n['dir']}\t{n['beat']:.4f}\t{n['t']:.4f}")
    notes_path(name).write_text("\n".join(lines) + "\n")


def to_beat(s: dict, bar: int, slot: int) -> float:
    """Bar+slot -> beat. Bars are 1-indexed; the grid is anchored on the fitted phase."""
    return (bar - 1) * BEATS_PER_BAR + slot / SUBDIV


def to_time(s: dict, beat: float) -> float:
    return s["phase"] + beat * (60.0 / s["bpm"])


def cmd_init(a) -> int:
    import brief as B
    audio = a.audio.resolve()
    if not audio.exists():
        print(f"no such audio: {audio}", file=sys.stderr)
        return 2
    an = B.analyse(audio)
    g = B.grid(an)
    d = sess_dir(a.name)
    d.mkdir(parents=True, exist_ok=True)
    s = {"name": a.name, "audio": str(audio), "song": audio.stem,
         "bpm": g["bpm"], "phase": g["phase"], "bar_s": g["bar_s"],
         "spb": g["spb"], "slot_s": g["slot"], "n_bars": g["n_bars"],
         "duration": an["dur"], "fit_r": an["r"]}
    (d / "session.json").write_text(json.dumps(s, indent=1))
    if not notes_path(a.name).exists():
        write_notes(a.name, [])
    print(f"session '{a.name}'  {audio.name}")
    print(f"  {g['bpm']:.2f} bpm, {g['n_bars']} bars, bar = {g['bar_s']:.3f}s, "
          f"1/16 = {g['slot']*1000:.1f}ms, downbeat {g['phase']*1000:+.0f}ms")
    if an["r"] < 0.35:
        print(f"  ⚠️tempo fit is WEAK (r={an['r']:.3f}) — check the grid against the "
              "audio before mapping a whole song onto it")
    print(f"  -> {d.relative_to(REPO)}")
    return 0


def parse_note_line(s: dict, ln: str) -> dict | None:
    ln = ln.split("#")[0].strip()
    if not ln:
        return None
    p = ln.split()
    if len(p) != 5:
        raise ValueError(f"expected '<bar>.<slot> <L|R> <col> <row> <dir>', got: {ln}")
    pos, hand, col, row, dname = p
    bar, _, slot = pos.partition(".")
    bar, slot = int(bar), int(slot or 0)
    hand = hand.upper()
    if hand not in ("L", "R"):
        raise ValueError(f"hand must be L or R, got {hand!r}")
    dname = dname.upper()
    if dname not in DIRS:
        raise ValueError(f"dir must be one of {sorted(DIRS)}, got {dname!r}")
    col, row = int(col), int(row)
    if not (0 <= col <= 3 and 0 <= row <= 2):
        raise ValueError(f"col must be 0-3 and row 0-2, got {col},{row}")
    if not (1 <= bar <= s["n_bars"]):
        raise ValueError(f"bar {bar} is outside 1-{s['n_bars']}")
    if not (0 <= slot < BEATS_PER_BAR * SUBDIV):
        raise ValueError(f"slot must be 0-{BEATS_PER_BAR*SUBDIV-1}, got {slot}")
    beat = to_beat(s, bar, slot)
    return {"bar": bar, "slot": slot, "hand": hand, "col": col, "row": row,
            "dir": DIRS[dname], "beat": beat, "t": to_time(s, beat)}


def cmd_add(a) -> int:
    s = load_session(a.name)
    cur = read_notes(a.name)
    new: list[dict] = []
    if a.from_file:
        text = pathlib.Path(a.from_file).read_text()
        for i, ln in enumerate(text.splitlines(), 1):
            try:
                n = parse_note_line(s, ln)
            except ValueError as exc:
                # Refuse the whole batch: a half-applied phrase is worse than none,
                # because you cannot tell by looking which half landed.
                print(f"line {i}: {exc}", file=sys.stderr)
                print("nothing was added.", file=sys.stderr)
                return 2
            if n:
                new.append(n)
    elif a.bar is not None:
        slots = [int(x) for x in a.slots.split(",")] if a.slots else [0]
        for sl in slots:
            new.append(parse_note_line(
                s, f"{a.bar}.{sl} {a.hand} {a.col} {a.row} {a.dir}"))
    else:
        print("give --from FILE or --bar/--hand/--col/--row/--dir", file=sys.stderr)
        return 2

    have = {(n["beat"], n["hand"]) for n in cur}
    dup = [n for n in new if (n["beat"], n["hand"]) in have]
    if dup and not a.replace:
        print(f"⚠️{len(dup)} note(s) collide with an existing note for the same hand "
              f"at the same beat (first: bar {dup[0]['bar']}.{dup[0]['slot']} "
              f"{dup[0]['hand']}). Use --replace to overwrite.", file=sys.stderr)
        return 2
    if a.replace:
        drop = {(n["beat"], n["hand"]) for n in new}
        cur = [n for n in cur if (n["beat"], n["hand"]) not in drop]
    write_notes(a.name, cur + new)
    print(f"added {len(new)} notes  (total {len(cur) + len(new)})")
    return 0


def cmd_clear(a) -> int:
    s = load_session(a.name)
    b0, b1 = (int(x) for x in a.bars.split("-"))
    cur = read_notes(a.name)
    keep = [n for n in cur if not (b0 <= n["bar"] <= b1)]
    write_notes(a.name, keep)
    print(f"cleared bars {b0}-{b1}: removed {len(cur) - len(keep)}, "
          f"{len(keep)} remain")
    return 0


def cmd_status(a) -> int:
    s = load_session(a.name)
    notes = read_notes(a.name)
    if not notes:
        print(f"session '{a.name}': EMPTY ({s['n_bars']} bars to map)")
        return 0
    import brief as B
    an = B.analyse(pathlib.Path(s["audio"]))
    dur = s["duration"]
    nps = len(notes) / dur
    print(f"session '{a.name}'  {len(notes)} notes  {nps:.2f} nps  "
          f"(human Expert median 3.91)")
    per = {}
    for n in notes:
        per[(n["bar"] - 1) // 8] = per.get((n["bar"] - 1) // 8, 0) + 1
    print(f"\n{'bars':>9} {'notes':>6} {'onsets':>7}  coverage")
    for p in range((s["n_bars"] + 7) // 8):
        b0 = p * 8 + 1
        t0 = s["phase"] + (b0 - 1) * s["bar_s"]
        t1 = t0 + s["bar_s"] * 8
        if t0 >= dur:
            break
        on = sum(1 for t in an["onsets"]["union"] if t0 <= t < t1) \
            if "union" in an["onsets"] else \
            sum(1 for k in an["onsets"] for t in an["onsets"][k] if t0 <= t < t1)
        c = per.get(p, 0)
        bar_g = "#" * min(20, int(c / 4)) if c else "— empty"
        print(f"{b0:>4}-{b0+7:<4} {c:>6} {on:>7}  {bar_g}")
    return 0


def _bm_from_notes(s: dict, notes: list[dict]):
    from beatsaber_automapper.data.beatmap import ColorNote, DifficultyBeatmap
    cn = [ColorNote(beat=n["beat"], x=n["col"], y=n["row"],
                    color=0 if n["hand"] == "L" else 1, direction=n["dir"])
          for n in sorted(notes, key=lambda n: n["beat"])]
    # v3 is what the rest of the pipeline writes and what map_view.py reads.
    return DifficultyBeatmap(version="3.0.0", color_notes=cn)


def cmd_view(a) -> int:
    s = load_session(a.name)
    notes = read_notes(a.name)
    import brief as B
    an = B.analyse(pathlib.Path(s["audio"]))
    b0, b1 = (int(x) for x in a.bars.split("-"))
    words = B.lyric_words(s["song"])
    print(f"BARS {b0}-{b1}   L/R = your notes (arrow = cut dir), D/B/O/V = the song")
    for bar in range(b0, b1 + 1):
        t0 = s["phase"] + (bar - 1) * s["bar_s"]
        t1 = t0 + s["bar_s"]
        said = " ".join(w["word"] for w in words if t0 <= w["t"] < t1)
        for hand in ("L", "R"):
            cells = ["."] * (BEATS_PER_BAR * SUBDIV)
            for n in notes:
                if n["bar"] == bar and n["hand"] == hand:
                    cells[n["slot"]] = ARROW[n["dir"]]
            head = f"{bar:>4} {B._mmss(t0):>8}" if hand == "L" else " " * 13
            print(f"{head}  |{''.join(cells)}| {hand}"
                  + (f"   {said}" if (hand == 'L' and said) else ""))
        for k, st in zip("DBOV", B.STEMS):
            row = B.stem_row(np.asarray(an["onsets"][st]), t0, t1, s["slot_s"])
            if row.strip("."):
                print(f"{' ' * 13}  |{row}| {k.lower()}")
    return 0


def cmd_check(a) -> int:
    s = load_session(a.name)
    notes = read_notes(a.name)
    if not notes:
        print("no notes to check")
        return 0
    from beatsaber_automapper.evaluation import swing_sim as ss
    bm = _bm_from_notes(s, notes)
    card = ss.simulate(bm, bpm=s["bpm"])
    times = sorted(n["t"] for n in notes)
    doubles = sum(1 for i in range(1, len(times)) if abs(times[i] - times[i - 1]) < 1e-4)
    print(f"notes {len(notes)}   swings {card.n_swings}   "
          f"resets {card.resets}   violations {card.violations}")
    print(f"double share {doubles / max(len(times), 1):.3f}  (human median 0.137)")
    if card.violations:
        print(f"\n🔴{card.violations} PARITY VIOLATION(S) — the map is unplayable as is.")
        for b in card.violation_beats[:10]:
            bar = int(b // BEATS_PER_BAR) + 1
            print(f"    beat {b:.2f}  (bar {bar})")
    else:
        print("✅no parity violations")
    return 0 if card.violations == 0 else 1


def cmd_export(a) -> int:
    s = load_session(a.name)
    notes = read_notes(a.name)
    if not notes:
        print("nothing to export", file=sys.stderr)
        return 2
    from beatsaber_automapper.generation.export import package_level
    bm = _bm_from_notes(s, notes)
    out = pathlib.Path(a.out).resolve()
    # ⚠️The map is written on the FITTED grid, whose downbeat is `phase` seconds after
    # t=0. Beat Saber applies `song_time_offset` to the audio, so the offset must be
    # the phase or every note lands `phase` early.
    package_level({a.difficulty: bm}, pathlib.Path(s["audio"]), out,
                  song_name=a.song_name or f"AGENT {s['song']}",
                  song_author="agent_mapper", bpm=s["bpm"],
                  song_time_offset=s["phase"])
    print(f"wrote {out}  ({len(notes)} notes, {s['bpm']:.2f} bpm, "
          f"offset {s['phase']*1000:+.0f}ms)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init"); p.add_argument("audio", type=pathlib.Path)
    p.add_argument("--name", required=True); p.set_defaults(fn=cmd_init)

    p = sub.add_parser("add"); p.add_argument("name")
    p.add_argument("--from", dest="from_file")
    p.add_argument("--bar", type=int); p.add_argument("--slots")
    p.add_argument("--hand", default="L"); p.add_argument("--col", type=int, default=1)
    p.add_argument("--row", type=int, default=0); p.add_argument("--dir", default="D")
    p.add_argument("--replace", action="store_true"); p.set_defaults(fn=cmd_add)

    p = sub.add_parser("clear"); p.add_argument("name")
    p.add_argument("--bars", required=True); p.set_defaults(fn=cmd_clear)

    p = sub.add_parser("status"); p.add_argument("name"); p.set_defaults(fn=cmd_status)

    p = sub.add_parser("view"); p.add_argument("name")
    p.add_argument("--bars", required=True); p.set_defaults(fn=cmd_view)

    p = sub.add_parser("check"); p.add_argument("name"); p.set_defaults(fn=cmd_check)

    p = sub.add_parser("export"); p.add_argument("name")
    p.add_argument("--out", required=True); p.add_argument("--difficulty", default="Expert")
    p.add_argument("--song-name", default=None); p.set_defaults(fn=cmd_export)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
