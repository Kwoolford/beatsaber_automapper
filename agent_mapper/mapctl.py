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
import bisect
import json
import pathlib
import random
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
    # 🔴🔴**A REUSED SESSION NAME SILENTLY BUILDS ON TOP OF THE LAST RUN.** `init` used
    # to leave existing notes alone, which is right for a human resuming a hand-built
    # map and WRONG for any sweep: re-running an arm with the same name appended a
    # second full map onto the first, and the result did not error -- it just scored
    # worse. That is exactly how a 2026-08-20 sweep invented a PASS-rate "regression"
    # from 23/23 to 20/23 that did not exist. `--fresh` is what every automated
    # builder should pass; the default still protects a hand-authored session.
    if getattr(a, "fresh", False):
        write_notes(a.name, [])
    elif not notes_path(a.name).exists():
        write_notes(a.name, [])
    elif read_notes(a.name):
        print(f"  ⚠️session '{a.name}' already has {len(read_notes(a.name))} notes; "
              f"they are KEPT. Pass --fresh to start empty.")
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


def cmd_plan(a) -> int:
    """Print the song's own section plan — the longitudinal view, as a work list.

    This is `structure.py`'s CONFIRMED result (repeated lyric lines land under the same
    letter 0.485 vs a shuffled null 0.317, p = 0.019 on held-out songs) turned into
    something you map against: which bars, which letter, and which instances are the
    same music as which.
    """
    import structure as ST
    s = load_session(a.name)
    secs = ST.analyse(pathlib.Path(s["audio"]))["sections"]
    notes = read_notes(a.name)
    print(f"session '{a.name}'   {len(secs)} sections")
    print(f"{'sec':>4} {'bars':>10} {'len':>5} {'notes':>6}   reuse")
    byl: dict[str, list[dict]] = {}
    for sec in secs:
        byl.setdefault(sec["label"], []).append(sec)
    for sec in secs:
        b0, b1 = sec["bar0"], sec["bar0"] + sec["bars"] - 1
        n = sum(1 for x in notes if b0 <= x["bar"] <= b1)
        peers = [x["bar0"] for x in byl[sec["label"]] if x["bar0"] != sec["bar0"]]
        print(f"{sec['label']:>4} {f'{b0}-{b1}':>10} {sec['bars']:>4}b {n:>6}   "
              + (f"same as bar {peers}" if peers else "—"))
    return 0


def cmd_reuse(a) -> int:
    """★Map a section once, then reuse it at every repeat — deliberately.

    `BEAT_STRUCTURE_REUSE` exists to infer this from an audio self-similarity matrix and
    apply it to the generator. An agent does not have to infer it: `structure.py` reads
    it off the song, and this copies the pattern across.

    ⚠️**Vary it on purpose.** The open question on review set A is whether the repetition
    reads INTENTIONAL or LAZY, and a byte-identical repeat is the definition of lazy.
    `--vary` drops a fraction of the copied notes, deterministically per instance, so
    the second chorus is the same idea played a little differently rather than the same
    file pasted twice. `--vary 0` gives an exact copy when that is what you want.

    ⚠️Sections that repeat are rarely the same LENGTH. The copy is truncated to the
    target's bars rather than overrunning into the next section, which is the failure
    mode that would otherwise quietly corrupt everything downstream of it.
    """
    import structure as ST
    s = load_session(a.name)
    secs = ST.analyse(pathlib.Path(s["audio"]))["sections"]
    notes = read_notes(a.name)

    byl: dict[str, list[dict]] = {}
    for sec in secs:
        byl.setdefault(sec["label"], []).append(sec)
    labels = [a.label] if a.label else sorted(byl)

    def count(sec) -> int:
        b0, b1 = sec["bar0"], sec["bar0"] + sec["bars"] - 1
        return sum(1 for x in notes if b0 <= x["bar"] <= b1)

    total_added = 0
    for lb in labels:
        insts = byl.get(lb, [])
        if len(insts) < 2:
            continue
        src = max(insts, key=count)
        if count(src) == 0:
            print(f"{lb}: no instance is mapped yet — map bars "
                  f"{src['bar0']}-{src['bar0']+src['bars']-1} first, then reuse")
            continue
        s0, s1 = src["bar0"], src["bar0"] + src["bars"] - 1
        pattern = [x for x in notes if s0 <= x["bar"] <= s1]
        for i, tgt in enumerate(insts):
            if tgt["bar0"] == src["bar0"]:
                continue
            t0, t1 = tgt["bar0"], tgt["bar0"] + tgt["bars"] - 1
            notes = [x for x in notes if not (t0 <= x["bar"] <= t1)]
            shift = tgt["bar0"] - src["bar0"]
            kept = 0
            for j, x in enumerate(pattern):
                nb = x["bar"] + shift
                if nb > t1:
                    continue                      # never overrun into the next section
                # Deterministic thinning, so the same call always gives the same map.
                if a.vary > 0 and ((j * 7 + i * 3) % 100) < a.vary * 100:
                    continue
                beat = to_beat(s, nb, x["slot"])
                notes.append({**x, "bar": nb, "beat": beat, "t": to_time(s, beat)})
                kept += 1
            total_added += kept
            print(f"{lb}: bars {s0}-{s1} -> {t0}-{t1}   {kept} notes"
                  + (f" ({a.vary:.0%} varied away)" if a.vary > 0 else " (exact copy)"))
    write_notes(a.name, notes)
    print(f"\n{total_added} notes placed by reuse; {len(notes)} in the map. "
          "Run `check` before export — reuse can create parity work.")
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


def _agree(an: dict, t: float, slot_s: float) -> int:
    """How many stems have an onset within one grid slot of `t`.

    A downbeat that drums, bass and the melody all hit is an accent worth marking with
    both hands; one that only the bass touches is not. This is the coincidence signal
    the 2026-08-03 work measured — humans map a 4-stem collision 84.5 % of the time.
    """
    n = 0
    for ts in an["onsets"].values():
        arr = np.asarray(ts, dtype=float)
        if arr.size and np.min(np.abs(arr - t)) < slot_s:
            n += 1
    return n


def _pitch_levels(audio: pathlib.Path, stem: str) -> list[tuple[float, int]]:
    """(time, level 0-9) for every pitched onset of a melodic stem, or [] if none.

    ★**This is where pitch becomes PLACEMENT.** `travel` is ours 4.60 against a human
    12.53 — our hands barely move — and the reason is visible in the two lines below
    this function: two columns and two rows per hand, chosen by parity alone. Nothing
    in the placer ever knew whether the melody went up or down, so there was nothing
    for it to follow. A human walks the grid with the line.
    """
    import melody as M
    try:
        res = M.analyse(audio)
    except Exception:                                             # noqa: BLE001
        return []
    ev = res["stems"].get(stem) or []
    meta = res.get("meta", {}).get(stem, {})
    # Refuse to place off a line the melody tool itself does not trust: a screamed
    # vocal has no f0 and a level derived from one is a number, not a pitch.
    if len(ev) < 20 or meta.get("coverage", 0) < 0.45:
        return []
    M.levels(ev)
    return [(e["t"], e.get("level", 4)) for e in ev]


def _level_at(levels: list[tuple[float, int]], t: float, tol: float = 0.12) -> int | None:
    """The pitch level of the note nearest `t`, if one is close enough to be it."""
    if not levels:
        return None
    import bisect as _b
    ts = [x[0] for x in levels]
    i = _b.bisect_left(ts, t)
    best, bd = None, tol
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(levels) and abs(levels[j][0] - t) < bd:
            best, bd = levels[j][1], abs(levels[j][0] - t)
    return best


def _follow_times(audio: pathlib.Path, an: dict, follow: str,
                  min_accent: float | None = None,
                  accent_pct: float | None = None):
    """Onset times for `--follow`, from a stem OR from a typed event class.

    Three forms, in increasing specificity:

        --follow drums              one of the four brief stems  (UNCHANGED PATH)
        --follow piano              any of the six `events.py` stems
        --follow other/hi-stab      one class within a stem

    ★**The four-stem path is deliberately left byte-identical.** It reads
    `brief.analyse`'s cached onsets exactly as before, because every number recorded
    for `auto` in this repo was measured on those onsets and rerouting them through a
    second detector would silently move all of them. The new forms are additive.

    ★**Why class-level following is the point.** `events.py` finds 14-20 distinct note
    types in a song where the old view had four. "Follow the lead" and "accent the
    crashes" are the decisions a human mapper actually makes, and they were not
    expressible until the classes existed. `--min-accent` is the other half: play only
    the hits above a loudness, in dB relative to that stem's own median.

    ⚠️A class inside a stem the control does NOT trust is refused, not silently used.
    `events.py` reports that verdict per stem; drawing a distinction the shuffled-label
    null does not support would be inventing structure.
    """
    import brief as B

    if ("/" not in follow and follow in B.STEMS
            and min_accent is None and accent_pct is None):
        return list(an["onsets"][follow])

    import events as E
    d = E.analyse(audio)
    stem, _, cls = follow.partition("/")
    if stem not in d["stems"]:
        raise ValueError(f"--follow: no stem `{stem}`; have {sorted(d['stems'])} "
                         f"or one of {B.STEMS}")
    if cls:
        trust = (d.get("trust") or {}).get(stem)
        if trust is False:
            have = sorted(d["stems"][stem].get("classes", {}))
            raise ValueError(
                f"--follow {follow}: the class labelling of `{stem}` FAILED its "
                f"control on this song (labels repeat no better than shuffled), so "
                f"its classes are not real here. Follow `{stem}` as one lane instead. "
                f"(classes it would have offered: {have})")
        avail = set(d["stems"][stem].get("classes", {}))
        if cls not in avail:
            raise ValueError(f"--follow {follow}: `{stem}` has {sorted(avail)}")

    sel = [e for e in d["events"] if e["stem"] == stem
           and (not cls or e.get("cls") == cls)]

    # ★**Accent as a PERCENTILE of this stem's own events, not as absolute dB.**
    # Measured on 1f767: the drums span p90 = +1.2 dB and max +2.8 -- a compressed
    # electronic drum bus -- while `other` spans +20.3. So `--min-accent 2` keeps 178
    # of 422 `other` events and **8 of 637** drum events, i.e. the same number means
    # something different on every stem and every song. This is the same lesson the
    # perception scorecard records for spectral thresholds: **absolute levels do not
    # transfer between mixes.** `--accent-pct 0.25` ("the loudest quarter of this
    # stem") does transfer, and is what a mapper actually means by "the accents".
    if accent_pct is not None and sel:
        import numpy as _np
        loud = _np.array([e.get("loud", 0.0) for e in sel])
        thr = float(_np.quantile(loud, 1.0 - max(min(accent_pct, 1.0), 0.0)))
        sel = [e for e in sel if e.get("loud", 0.0) >= thr]
    if min_accent is not None:
        sel = [e for e in sel if e.get("loud", 0.0) >= min_accent]
    return sorted(e["t"] for e in sel)


def cmd_auto(a) -> int:
    """Follow a stem over a bar range: the bulk-placement primitive.

    ★**This is what makes a full map reachable.** Hand-writing 1 300 notes one line at
    a time is not a workflow; deciding *which instrument carries each section and how
    densely* is. The agent supplies the musical judgement — follow the drums here, the
    vocal line there, thin the breakdown, swap which hand leads — and this handles the
    bookkeeping that has exactly one right answer: parity, hand alternation, and
    keeping each hand on its own side.

    ★**`--follow vocals` is the point of the whole folder.** It is precisely what the
    generator cannot do: `follow_vocals` is ours 0.020 against a human 0.149.

    ★★**HANDS ARE ASSIGNED OVER THE MERGED TIMELINE, not over this pass in isolation.**
    Layering a second `auto` on the same bars (drums, then the vocal line) doubled
    `ebpm_burst` — 376 to 752 against a human 376 — because each pass tracked its own
    hand and parity state, so the new notes landed *between* the old ones and handed
    one hand two fast consecutive swings. Found by measuring the first full map.

    ⚠️**A hypothesis the same measurement refuted.** I expected hand RUNS to look more
    human than strict alternation, since `role_asymmetry` is human 0.115 against our
    0.026. For burst speed it is the opposite: `runs=1` gives exactly the human's 376
    and `runs>=2` gives 752, because `ebpm_burst` is a PER-HAND rate and alternating is
    what keeps each hand slow. `--runs` stays because it is a real stylistic knob, but
    its default is 1 for a measured reason.
    """
    import brief as B
    s = load_session(a.name)
    an = B.analyse(pathlib.Path(s["audio"]))
    try:
        # ★A comma-separated `--follow` unions the streams into ONE pass. Layering two
        # passes is measurably what costs the pulse: drums alone scores
        # `pulse_stability` 0.387 and the union of drums+carrier 0.329 (n=23), because
        # each pass picks its own times and the interleaving is nobody's rhythm.
        follow_times = []
        for spec in str(a.follow).split(","):
            follow_times.extend(_follow_times(pathlib.Path(s["audio"]), an, spec.strip(),
                                              getattr(a, "min_accent", None),
                                              getattr(a, "accent_pct", None)))
        follow_times = sorted(set(follow_times))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    b0, b1 = (int(x) for x in a.bars.split("-"))
    a.double_slots = {int(x) for x in str(a.accent_slots).split(",")}
    plevels: list[tuple[float, int]] = []
    if getattr(a, "pitch", False):
        pstem = a.follow if a.follow in ("vocals", "other") else "vocals"
        plevels = _pitch_levels(pathlib.Path(s["audio"]), pstem)
        if not plevels:
            print(f"⚠️--pitch asked for, but `{pstem}` has no trustworthy melodic line "
                  "(screamed vocals, or coverage below 0.45). Falling back to the "
                  "parity-only layout — this is the honest answer, not a failure.")
    cur = read_notes(a.name)
    occupied = {(n["bar"], n["slot"]) for n in cur}
    n_cells = BEATS_PER_BAR * SUBDIV

    picks: list[tuple[int, int]] = []
    for bar in range(b0, b1 + 1):
        t0 = s["phase"] + (bar - 1) * s["bar_s"]
        t1 = t0 + s["bar_s"]
        seen = set()
        for t in follow_times:
            if not (t0 <= t < t1):
                continue
            i = int(round((t - t0) / s["slot_s"]))
            if 0 <= i < n_cells and i not in seen:
                seen.add(i)
                picks.append((bar, i))
    picks.sort()
    if a.every > 1:
        picks = picks[::a.every]
    if a.max_per_bar:
        by_bar: dict[int, list[int]] = {}
        keep = []
        for bar, sl in picks:
            got = by_bar.setdefault(bar, [])
            if len(got) < a.max_per_bar:
                got.append(sl)
                keep.append((bar, sl))
        picks = keep
    if getattr(a, "pulse", False):
        import pulse as PU
        before = len(picks)
        picks = PU.quantise(picks, n_cells, b0,
                            phrase_bars=getattr(a, "phrase_bars", 4),
                            max_empty=getattr(a, "pulse_fill", PU.MAX_EMPTY_RUN),
                            sync=getattr(a, "pulse_sync", PU.SYNC_FRAC))
        print(f"pulse: {before} -> {len(picks)} cells, one interval held per "
              f"{getattr(a, 'phrase_bars', 4)}-bar phrase")
    picks = [p for p in picks if p not in occupied]
    if not picks:
        print("nothing to place (no onsets in range, or all slots already taken)")
        return 0

    run = max(1, a.runs)
    merged = sorted([(n["t"], 0, n) for n in cur]
                    + [(to_time(s, to_beat(s, b, sl)), 1, (b, sl)) for b, sl in picks],
                    key=lambda r: (r[0], r[1]))
    last_hand = None
    # Every time each hand plays, INCLUDING notes already in the session, kept sorted
    # so a new note can be checked against its neighbours on both sides.
    hand_times = {"L": sorted(n["t"] for n in cur if n["hand"] == "L"),
                  "R": sorted(n["t"] for n in cur if n["hand"] == "R")}
    # (time -> was it a DOWN cut) per hand, so an inserted note can be checked against
    # the note that FOLLOWS it, not only the one before.
    hand_dir = {"L": {n["t"]: n["dir"] not in (1, 6, 7) for n in cur if n["hand"] == "L"},
                "R": {n["t"]: n["dir"] not in (1, 6, 7) for n in cur if n["hand"] == "R"}}
    last_t: dict[str, float] = {}
    skipped = 0
    n_doubles = 0
    last_down = {"L": True, "R": True}
    since_swap = 0
    # Seeded so a build is reproducible; the lead bias is the only stochastic part of
    # placement and an unseeded one would make every arm unrepeatable.
    _lead_rng = random.Random(getattr(a, "seed", 0))
    cols = {"L": [1, 0], "R": [2, 3]}
    new = []
    k = 0
    # ⚠️Per-HAND counter. `--wide` used the global note counter `k`, but hands strictly
    # alternate, so `k % 2` was perfectly correlated with which hand was playing: the
    # left hand only ever saw even k and the right only odd. `--wide` therefore pinned
    # L to column 1 and R to column 3 and never widened anything — measured as exactly
    # two distinct columns across a 449-note map.
    wide_k = {"L": 0, "R": 0}
    last_lvl: dict[str, int] = {}
    for _t, kind, payload in merged:
        if kind == 0:
            last_t[payload["hand"]] = _t
            last_hand = payload["hand"]
            last_down[payload["hand"]] = payload["dir"] not in (1, 6, 7)
            since_swap = 1
            continue
        bar, sl = payload
        # ★★A PASSAGE HAS A LEAD HAND. Strict alternation splits every window evenly,
        # which is why `role_asymmetry` sits at the 1.1st human percentile on 21 of 23
        # maps: humans are globally balanced but LOCALLY lopsided, and balance at
        # every scale is the unnatural thing (evaluation/handrole.py). The lead hand
        # is held for a phrase and then handed over, the same shape as the pulse fix
        # -- hold, then break at a boundary. `--lead-bias 0` is the old behaviour.
        lead_h = None
        if a.lead_bias > 0:
            phrase = (bar - b0) // max(1, a.lead_phrase_bars)
            lead_h = ("L", "R")[(phrase + (0 if a.lead.upper() == "L" else 1)) % 2]
        if a.hands != "alternate":
            h = a.hands.upper()
        elif last_hand is None:
            h = a.lead.upper()
        elif since_swap >= run:
            other = "R" if last_hand == "L" else "L"
            # Only the LEAD hand gets to repeat, and only sometimes: a lead that
            # always repeats gives a 2:1 split (`role_asymmetry` 0.33) where the human
            # sits at 0.11, i.e. roughly 55/45. Two-sided -- overshooting asymmetry is
            # as wrong as our current evenness.
            if (lead_h is not None and last_hand == lead_h
                    and _lead_rng.random() < a.lead_bias):
                h = last_hand
            else:
                h = other
        else:
            h = last_hand
        # ★THE PER-HAND FLOOR, measured from 31 723 human gaps over 40 songs: a human
        # hand almost never swings twice inside ~150 ms (cohort p5 = 148 ms; Hunger's
        # human map has a hard 160 ms floor). Without this the first agent map allowed
        # 80 ms — one sixteenth at 188 bpm — and scored ebpm_burst 752 against a human
        # 376 while its AVERAGE per-hand rate matched the human's exactly (3.96 vs
        # 3.99). The defect was never the average, it was the fast tail.
        # ⚠️Check BOTH neighbours, not just the previous note. A later `auto` pass
        # inserts notes BETWEEN ones already fixed, so a hand can satisfy its backward
        # gap and still land 40 ms before an existing note of the same hand. Checking
        # only backwards left 70 violations of the floor in place and the burst rate
        # unchanged at 752 — the fix looked applied and did nothing.
        gap = a.min_gap_ms / 1000.0 if a.min_gap_ms > 0 else 0.0

        def _free(hand: str) -> bool:
            if gap <= 0:
                return True
            ts = hand_times[hand]
            i = bisect.bisect_left(ts, _t)
            if i > 0 and _t - ts[i - 1] < gap:
                return False
            if i < len(ts) and ts[i] - _t < gap:
                return False
            return True

        if a.min_gap_ms > 0:

            def _unused(hand: str) -> bool:
                return True

            if not _free(h):
                o = "R" if h == "L" else "L"
                if _free(o):
                    h = o          # the other hand is free — give it the note
                else:
                    skipped += 1   # neither hand can play it; a human would not either
                    continue
        # Parity is left to `postprocess.fix_parity` on check/export: it has
        # flow-aware look-ahead and is the model `swing_sim` actually scores. Two
        # rounds of hand-rolled repair here cost 380 notes in skips and still left 5
        # violations. The simple alternation below is a good STARTING guess, nothing more.
        want_down = last_down[h]
        d = "D" if want_down else "U"
        row = 0 if want_down else 2
        col = cols[h][wide_k[h] % 2] if a.wide else cols[h][0]
        wide_k[h] += 1
        lvl = _level_at(plevels, _t) if plevels else None
        if lvl is not None and a.pitch_span == "full":
            # ★INTERVAL, not absolute level. Mapping level->position directly made
            # `travel` WORSE (4.77 -> 3.56 against a human 12.53), and the reason is
            # musical: a melody moves in small steps, so following its contour
            # literally parks consecutive notes in the same cell. What a human
            # actually mirrors is the LEAP — a big interval becomes a big move.
            prev = last_lvl.get(h)
            span = cols[h] + cols["R" if h == "L" else "L"]      # all four columns
            if prev is None:
                col = cols[h][0]
            else:
                jump = min(abs(lvl - prev), 5)
                # a small interval stays on this hand's side, a big one crosses over
                col = span[min(jump * 3 // 2, 3)]
            row = 1 if (prev is not None and abs(lvl - prev) >= 4) else row
            last_lvl[h] = lvl
        elif lvl is not None:
            # Column follows the line outward: a high note sits on the OUTER column of
            # the hand playing it, a low note on the inner one. Deliberately kept
            # inside the hand's own two columns — crossing hands over is a separate,
            # bigger change and mixing the two would make neither measurable.
            col = cols[h][1] if lvl >= 5 else cols[h][0]
            # Row nudges toward the pitch, but only by one, and only away from the
            # parity-natural extreme. ⚠️Parity decides the cut DIRECTION and is what
            # makes the map playable; this moves the note within what that allows and
            # never overrides it. The middle row was previously never used at all.
            if want_down and lvl >= 7:
                row = 1
            elif (not want_down) and lvl <= 2:
                row = 1
        new.append(parse_note_line(s, f"{bar}.{sl} {h} {col} {row} {d}"))
        hand_dir[h][_t] = want_down
        # ★DOUBLES MARK THE DOWNBEAT. Humans put both hands on an accent —
        # `hands_x_downbeat` is human 0.182 against our 0.036, and the note is that
        # "we spend doubles on 2/3 of all events so they mark nothing". A double also
        # buys density WITHOUT speeding either hand up, which is how the human map
        # reaches 8.35 nps at the same ebpm_burst as ours at 4.00.
        # Only on a bar downbeat, and only where several stems agree it is an accent.
        # ⚠️Bar downbeats ALONE are too few: 24 bars give at most 24 chances, and after
        # stem agreement and the per-hand floor only ~3 survive — a double share of
        # 0.016 against a human 0.146. Humans accent every STRONG beat, so slots 0 and
        # 8 (beats 1 and 3) both qualify.
        if a.doubles and sl in a.double_slots and \
                _agree(an, _t, s["slot_s"]) >= a.doubles_stems:
            o = "R" if h == "L" else "L"
            if a.min_gap_ms <= 0 or _free(o):
                od = "D" if last_down[o] else "U"
                orow = 0 if last_down[o] else 2
                new.append(parse_note_line(
                    s, f"{bar}.{sl} {o} {cols[o][0]} {orow} {od}"))
                hand_dir[o][_t] = last_down[o]
                last_down[o] = not last_down[o]
                bisect.insort(hand_times[o], _t)
                last_t[o] = _t
                n_doubles += 1
        last_down[h] = not last_down[h]
        bisect.insort(hand_times[h], _t)
        last_t[h] = _t
        since_swap = since_swap + 1 if h == last_hand else 1
        last_hand = h
        k += 1

    write_notes(a.name, cur + new)
    dens = len(new) / max((b1 - b0 + 1) * s["bar_s"], 1e-9)
    if n_doubles:
        print(f"  {n_doubles} downbeat double(s) — both hands on an accent")
    print(f"placed {len(new)} notes over bars {b0}-{b1} following {a.follow} "
          f"({dens:.2f} nps)")
    if skipped:
        print(f"  skipped {skipped} onset(s) that no hand could reach inside "
              f"{a.min_gap_ms:.0f}ms — the human floor. Follow a sparser stem, or "
              f"--every 2, if you wanted them.")
    if occupied:
        print("  (slots already occupied were skipped; `clear` a range to redo it)")
    return 0


def _fix_parity(bm):
    """Run the pipeline's own parity fixer over an authored map.

    ★**Do not reimplement this.** `auto` keeps a simple down/up alternation per hand,
    which is right for placement but is NOT the model `swing_sim` scores against — that
    one accounts for reset timing and swing angles. Two rounds of hand-rolled parity
    repair here got 13 violations down to 5 and cost 380 notes in skips;
    `postprocess.fix_parity` has flow-aware look-ahead, is already validated, and is
    what the ML pipeline uses. Reuse beats rebuild.
    """
    from beatsaber_automapper.generation.postprocess import fix_parity
    return fix_parity(bm)


def cmd_check(a) -> int:
    s = load_session(a.name)
    notes = read_notes(a.name)
    if not notes:
        print("no notes to check")
        return 0
    from beatsaber_automapper.evaluation import swing_sim as ss
    bm = _bm_from_notes(s, notes)
    raw = ss.simulate(bm, bpm=s["bpm"]).violations
    bm = _fix_parity(bm)
    card = ss.simulate(bm, bpm=s["bpm"])
    if raw and not card.violations:
        print(f"({raw} raw violation(s) repaired by postprocess.fix_parity — the "
              f"export applies the same fix)")
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


def cmd_judge(a) -> int:
    """Score the working map against the human corpus AT n=1, mid-authoring.

    ★**This is the step that lets the agent stop asking Kyle.** `check` answers "is
    this legal?" -- parity, reachability -- which a map can pass while still being
    nothing a human would write. `judge` answers "is this INSIDE the human
    distribution?", per metric, as a percentile, with a conformal p-value calibrated
    on 1 100 human maps and validated by rejecting all eight degenerate controls at
    n=1 (`scripts/audit_mapjudge.py`).

    ⚠️**Read it as a DEFECT DETECTOR, never as a score to maximise.** It gates against
    the human corpus MEDIAN, and Kyle's standing instruction is *"my target is the
    best mappers"* -- so a corpus median is a **floor, not a target**. `rank_score` is
    a distance-from-typical: driving it to zero produces the *average* map, which is
    precisely the `h_dist` failure in a new costume. Use the per-metric percentiles as
    a to-do list, and stop when nothing is flagged.

    ⚠️**No audio axis yet.** The judge scores note attributes and their sequencing; it
    is structurally blind to whether the notes sit on the MUSIC. `alignment` is the
    missing axis and until it is wired in, `judge` passing does not mean the map is on
    the beat.
    """
    s = load_session(a.name)
    notes = read_notes(a.name)
    if not notes:
        print("no notes to judge")
        return 0
    sys.path.insert(0, str(REPO / "src"))
    from beatsaber_automapper.evaluation import mapjudge as mj

    bm = _fix_parity(_bm_from_notes(s, notes))
    rec = mj.map_record(list(bm.color_notes), s["bpm"])
    res = mj.judge(rec, mj.load_reference(), label=a.name)
    print(mj.report(res, alpha=a.alpha, top=a.top))
    if res.verdict(a.alpha) == "PASS":
        print("\n  Inside the human distribution. ⚠️That means NOT DEFECTIVE, not good "
              "-- the bar is the corpus median and his target is the best mappers.")
    else:
        print("\n  Outside it. The flagged rows above are the to-do list; fix the "
              "highest percentile first and re-run.")
    return 0


def cmd_export(a) -> int:
    s = load_session(a.name)
    notes = read_notes(a.name)
    if not notes:
        print("nothing to export", file=sys.stderr)
        return 2
    from beatsaber_automapper.generation.export import package_level
    bm = _fix_parity(_bm_from_notes(s, notes))
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

    p = sub.add_parser("init")
    p.add_argument("--fresh", action="store_true",
                   help="discard any existing notes for this session name"); p.add_argument("audio", type=pathlib.Path)
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

    p = sub.add_parser("plan", help="the song's section plan, and which repeat which")
    p.add_argument("name"); p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("reuse", help="copy a mapped section to its repeats")
    p.add_argument("name")
    p.add_argument("--label", default=None, help="one section letter; default = all")
    p.add_argument("--vary", type=float, default=0.15,
                   help="fraction of copied notes to drop, so a repeat reads as "
                        "INTENTIONAL rather than pasted (0 = exact copy)")
    p.set_defaults(fn=cmd_reuse)

    p = sub.add_parser("view"); p.add_argument("name")
    p.add_argument("--bars", required=True); p.set_defaults(fn=cmd_view)

    p = sub.add_parser("auto"); p.add_argument("name")
    p.add_argument("--bars", required=True)
    p.add_argument("--follow", default="drums",
                   help="a stem (drums|bass|other|vocals|guitar|piano) or a typed "
                        "class within one (e.g. other/hi-stab). "
                        "`events.py <audio>` lists a song's classes.")
    p.add_argument("--lead", default="L", help="which hand starts")
    p.add_argument("--hands", default="alternate", help="alternate|L|R")
    p.add_argument("--every", type=int, default=1, help="thin: keep every Nth onset")
    p.add_argument("--max-per-bar", type=int, default=0, help="cap notes per bar")
    p.add_argument("--accent-pct", type=float, default=None,
                   help="follow only the loudest FRACTION of that stem's events "
                        "(0.25 = the loudest quarter). Self-relative, so it means "
                        "the same thing on every stem and song -- prefer this to "
                        "--min-accent.")
    p.add_argument("--min-accent", type=float, default=None,
                   help="only follow events at least this loud, in dB relative to "
                        "that stem's own median hit (a style knob: play the accents)")
    p.add_argument("--wide", action="store_true", help="use both columns per hand")
    p.add_argument("--lead-bias", type=float, default=0.0,
                   help="probability the phrase's LEAD hand repeats instead of "
                        "alternating; 0 = strict alternation (P0.6: role_asymmetry "
                        "ours 0.03 vs human 0.11)")
    p.add_argument("--lead-phrase-bars", type=int, default=4,
                   help="bars a hand keeps the lead before handing it over")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pulse", action="store_true",
                   help="hold ONE interval per phrase instead of playing every onset "
                        "(P0.5: our pulse_stability 0.329 vs human 0.514)")
    p.add_argument("--pulse-sync", type=float, default=0.3,
                   help="how far off the lattice (as a fraction of the period) an "
                        "event must sit to be restored as a syncopation; LOWER "
                        "restores more real onsets, breaking the pulse AND raising "
                        "onset_precision together")
    p.add_argument("--pulse-fill", type=int, default=1,
                   help="lattice points to hold across a quiet gap; 0 = never invent "
                        "a note (costs pulse, buys onset_precision)")
    p.add_argument("--phrase-bars", type=int, default=4,
                   help="bars per phrase for --pulse; the interval may change at each "
                        "boundary and not within (default 4)")
    p.add_argument("--pitch-span", default="hand", choices=("hand", "full"),
                   help="hand = pitch picks the column inside the hand's own two; "
                        "full = the pitch INTERVAL picks how far to jump, across all "
                        "four columns (crossovers allowed)")
    p.add_argument("--pitch", action="store_true",
                   help="★place by the MELODY: column and row follow the pitch "
                        "contour of the followed stem (needs agent_mapper/melody.py; "
                        "falls back with a warning if the line is not trustworthy)")
    p.add_argument("--doubles", action="store_true",
                   help="both hands on bar downbeats where stems agree; how a human "
                        "adds density without speeding either hand up")
    p.add_argument("--accent-slots", default="0,8",
                   help="slots that count as a strong beat for doubles (0,8 = beats "
                        "1 and 3). Downbeats alone are too few to reach a human rate.")
    p.add_argument("--doubles-stems", type=int, default=2,
                   help="how many stems must agree for a downbeat to count as an accent")
    p.add_argument("--min-gap-ms", type=float, default=150.0,
                   help="floor on the gap between two swings of the SAME hand; "
                        "150 = the human cohort p5 (n=31723 gaps). 0 disables.")
    p.add_argument("--runs", type=int, default=1,
                   help="notes one hand plays before the other takes over; 1 (strict "
                        "alternation) measured as the human burst rate")
    p.set_defaults(fn=cmd_auto)

    p = sub.add_parser("check"); p.add_argument("name"); p.set_defaults(fn=cmd_check)

    p = sub.add_parser("judge", help="score the map against the human corpus at n=1")
    p.add_argument("name")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--top", type=int, default=8)
    p.set_defaults(fn=cmd_judge)
    p = sub.add_parser("export"); p.add_argument("name")
    p.add_argument("--out", required=True); p.add_argument("--difficulty", default="Expert")
    p.add_argument("--song-name", default=None); p.set_defaults(fn=cmd_export)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
