#!/usr/bin/env python
"""Read a Beat Saber map the way a mapper reads one — as a score, in text.

Why this exists: every other channel we have onto a map is either an AGGREGATE
(the eval suite) or an IMAGE (render_map.py). Aggregates provably lie — `h_dist`
ranked our maps as *more human than human* for weeks before the control battery
caught it. Images can be looked at but not queried, diffed, or edited. This gives
a third channel: the actual notes, in musical context, as text I can read, slice,
compare, and eventually write back.

The layout is a tracker/score: one row per grid slot, time running down, the two
hands side by side, audio stems in their own lanes. Reading down a column shows
one hand's flow; reading across a row shows what both hands and the music are
doing at that instant; the rows above and below are the surrounding context.

    bar  beat  │ L        │ R        │ K S H │ bass lead
     33 132.00 │ 0,0 ↓ F  │          │ █ · · │ E2   —
     33 132.50 │          │ 3,1 ↑ B  │ · · ▃ │ E2   G4

Cells are `col,row`, then the cut-direction arrow, then parity (F/B) from the
swing simulator. Audio lanes come from the same per-stem transcription the model
trains on (data/instrument_features.py), so what I see here is what the model
saw — which is the point when auditing whether a note should be there at all.

Usage:
  python scripts/map_view.py <map.zip> --bars 33-40
  python scripts/map_view.py <map.zip> --bars 33-40 --audio data/eval_songset/1f333.ogg
  python scripts/map_view.py <map.zip> --sections
  python scripts/map_view.py <map.zip> --compare 33-40 65-72
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import swing_sim as ss  # noqa: E402

sys.path.insert(0, str(REPO / "agent_mapper"))
import elements as _EL  # noqa: E402

ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "↖", 5: "↗", 6: "↙", 7: "↘", 8: "•"}
BEATS_PER_BAR = 4.0
SUB = 4          # rows per beat (1/16-note resolution)
BLOCKS = " ▁▂▃▄▅▆▇█"


class _BM:
    def __init__(self, notes):
        self.color_notes = sorted(notes, key=lambda n: n.beat)
        self.bomb_notes = []


def load(map_path: pathlib.Path):
    from audit_eval_suite import _load_generated, _load_human
    r = _load_generated(map_path)
    if r is None:
        r = _load_human(map_path)
    if r is None:
        raise SystemExit(f"could not read {map_path}")
    return r


def _parity_map(notes, bpm: float) -> dict:
    """(beat, color) -> 'F'/'B' plus a marker for simulator violations."""
    card = ss.simulate(_BM(notes), bpm=bpm)
    out = {}
    for color, hand in card.per_hand.items():
        for sw in hand.swings:
            tag = "F" if sw.parity is ss.Parity.FOREHAND else "B"
            if sw.reset_kind == "violation":
                tag += "!"
            out[(round(sw.beat, 3), color)] = tag
    return out, card


def _idiom_map(notes) -> dict:
    """(beat, color) -> (rank, per-mille corpus frequency) of the idiom ENDING here.

    Naming each transition against the mined human vocabulary is what turns the
    score from "some notes" into something diagnosable: a transition is either a
    pattern human mappers actually use (and how often), or it is out of
    vocabulary entirely, which is usually where a map reads wrong.
    """
    try:
        from beatsaber_automapper.evaluation import idiom as idm
        counts, ranked, probs = idm.load_vocab()
    except Exception:  # noqa: BLE001
        return {}
    if not ranked:
        return {}
    rank = {k: i for i, k in enumerate(ranked)}
    out = {}
    for color in (0, 1):
        ns = sorted((n for n in notes if n.color == color), key=lambda n: n.beat)
        for a, b in zip(ns, ns[1:]):
            dt = round(b.beat - a.beat, 3)
            if dt <= 0 or dt > idm.MAX_DT:
                continue
            key = (b.x - a.x, b.y - a.y, a.direction, b.direction, idm.dt_class(dt))
            out[(round(b.beat, 3), color)] = (rank.get(key), probs.get(key, 0.0) * 1000.0)
    return out


def _stem_lanes(audio_path: pathlib.Path, bpm: float, max_beat: float):
    """Per-stem energy per slot, from the same features the model trains on."""
    try:
        from beatsaber_automapper.data.instrument_features import (
            INSTR_FEATURE_NAMES, compute_instrument_features,
        )
        import librosa
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        n_slots = int(max_beat * SUB) + 1
        feats = compute_instrument_features(y, sr, bpm=bpm, n_slots=n_slots,
                                            subdiv=SUB)
        return np.asarray(feats), list(INSTR_FEATURE_NAMES)
    except Exception as e:  # noqa: BLE001
        print(f"(audio lanes unavailable: {e})")
        return None, []


def _bar_range(spec: str) -> tuple[float, float]:
    a, _, b = spec.partition("-")
    lo = (float(a) - 1) * BEATS_PER_BAR
    hi = (float(b) if b else float(a)) * BEATS_PER_BAR
    return lo, hi


def render(notes, bpm: float, b0: float, b1: float,
           feats=None, names=None, label: str = "", idioms: bool = False,
           elems=None, onsets=None, offset: float = 0.0) -> list[str]:
    """★`elems` adds the THREE ELEMENTS NOTHING COULD READ (see agent_mapper/elements.py).

    Without it this sheet shows a `[FULL]` map -- 89 walls, 90 arcs, 16 chains -- as if
    it were notes-only, which is exactly how "is FULL less empty than V2?" stayed
    unanswerable. The `lanes` column is the player's dodge decision (`██··` = blocked),
    and `arc`/`chn` mark the gestures.
    """
    par, _card = _parity_map(notes, bpm)
    idi = _idiom_map(notes) if idioms else {}
    # ★★PER-NOTE ALIGNMENT — "is this note ON a sound I can hear?"
    # The skill's own first warning is that a PASS does not mean the notes are on the
    # music and that `onset_precision` must be read directly -- but that is an
    # AGGREGATE, and a listener perceives alignment note by note. This column is that
    # perception: signed ms to the nearest detected onset.
    # ⚠️Note time must include `_songTimeOffset`; it carries the grid phase, and
    # omitting it once made a phase shift look like it moved nothing at all.
    align = {}
    if onsets is not None and len(onsets):
        import numpy as _np
        _o = _np.sort(_np.asarray(onsets, dtype=float))
        for n in notes:
            tsec = offset + n.beat * 60.0 / bpm
            i = int(_np.clip(_np.searchsorted(_o, tsec), 1, len(_o) - 1))
            lo, hi = _o[i - 1], _o[i]
            near = lo if abs(tsec - lo) <= abs(tsec - hi) else hi
            align[round(n.beat, 3)] = (tsec - near) * 1000.0
    by_slot: dict[tuple[int, int], list] = {}
    for n in notes:
        if b0 <= n.beat < b1:
            by_slot.setdefault((int(round(n.beat * SUB)), n.color), []).append(n)

    idx = {nm: i for i, nm in enumerate(names or [])}
    has_audio = feats is not None
    w = 19 if idioms else 10

    head = f"{'bar':>4s} {'beat':>7s} │ {'L':<{w}s} │ {'R':<{w}s}"
    if align:
        head += " │  ±ms"
    if elems:
        head += " │ lanes │ gest"
    if has_audio:
        head += " │ K S H │ bass lead"
    lines = []
    if label:
        lines.append(label)
    lines.append(head)
    lines.append("─" * len(head))

    s0, s1 = int(round(b0 * SUB)), int(round(b1 * SUB))
    for s in range(s0, s1):
        beat = s / SUB
        bar = int(beat // BEATS_PER_BAR) + 1
        cells = []
        for color in (0, 1):
            ns = by_slot.get((s, color), [])
            if not ns:
                cells.append(" " * w)
                continue
            n = ns[0]
            tag = par.get((round(n.beat, 3), color), "")
            extra = f"+{len(ns) - 1}" if len(ns) > 1 else ""
            cell = f"{n.x},{n.y} {ARROW.get(n.direction, '?')} {tag:<3s}{extra}"
            if idioms:
                r, pm = idi.get((round(n.beat, 3), color), (None, None))
                # "#12 4.1‰" = the 12th most common human idiom, used 4.1 per
                # mille of the time. "OOV" = not in the human vocabulary at all.
                cell += f" {'OOV' if r is None else f'#{r} {pm:.1f}'}"[:9]
            cells.append(cell[:w].ljust(w))
        # only print empty rows on the beat, to keep the score compact
        if not any(c.strip() for c in cells) and s % SUB != 0:
            continue
        row = f"{bar:>4d} {beat:>7.2f} │ {cells[0]} │ {cells[1]}"
        if align:
            d = align.get(round(beat, 3))
            if d is None:
                row += " │      "
            else:
                # ● inside the axis' 50 ms tolerance · ○ 50-120 ms (the near-miss band
                # the onset snap was built for) · ✗ beyond it: nothing there to hit.
                mk = "●" if abs(d) <= 50 else ("○" if abs(d) <= 120 else "✗")
                row += f" │{mk}{d:>5.0f}"
        if elems:
            # ★The player's dodge surface: which of the 4 columns is blocked NOW.
            row += f" │ {_EL.lane_map(elems['walls'], beat)}  │ "
            g = []
            for arc in elems["arcs"]:
                if abs(float(arc.get("b", -9)) - beat) < 1e-6:
                    g.append("⌒" + ("L" if int(arc.get("c", 0)) == 0 else "R"))
            for ch in elems["chains"]:
                if abs(float(ch.get("b", -9)) - beat) < 1e-6:
                    g.append(f"╞{int(ch.get('sc', 0))}")
            row += " ".join(g)[:6].ljust(6)
        if has_audio and s < len(feats):
            f = feats[s]
            def blk(nm):
                v = float(f[idx[nm]]) if nm in idx else 0.0
                return BLOCKS[min(int(v * (len(BLOCKS) - 1)), len(BLOCKS) - 1)]
            bass = float(f[idx["bass_pitch"]]) if "bass_pitch" in idx else 0.0
            lead = float(f[idx["lead_pitch"]]) if "lead_pitch" in idx else 0.0
            row += (f" │ {blk('kick_density')} {blk('snare_density')} {blk('hat_density')}"
                    f" │ {bass:4.2f} {lead:4.2f}")
        lines.append(row)
    return lines


def sections(notes, bpm: float) -> list[str]:
    """Coarse section view: notes, density, and HAND ROLE per 8 bars.

    The `lead` column is the axis-A6 view: which hand carries the passage and by
    how much. Human maps show one hand leading with a mild margin and the lead
    rotating; ours show `--` almost everywhere, because both hands play every
    beat. That pattern is what the whole hand-role discovery came from, so the
    reading channel that found it should display it.
    """
    max_beat = max((n.beat for n in notes), default=0.0)
    span = 8 * BEATS_PER_BAR
    out = [f"{'bars':>10s} {'beat':>13s} {'notes':>6s} {'nps':>6s} {'lead':>7s}  density"]
    b = 0.0
    while b < max_beat:
        seg = [n for n in notes if b <= n.beat < b + span]
        secs = span * 60.0 / bpm
        nps = len(seg) / secs if secs else 0.0
        bar = BLOCKS[min(int(nps / 12.0 * 8), 8)] * max(1, int(nps))
        if seg:
            left = sum(1 for n in seg if n.color == 0)
            asym = abs(2 * left - len(seg)) / len(seg)
            lead = "--" if asym < 0.02 else f"{'L' if left * 2 > len(seg) else 'R'}{asym:.2f}"
        else:
            lead = "--"
        out.append(f"{int(b // BEATS_PER_BAR) + 1:>4d}-{int((b + span) // BEATS_PER_BAR):<5d} "
                   f"{b:>6.0f}-{b + span:<6.0f} {len(seg):>6d} {nps:>6.2f} {lead:>7s}  {bar}")
        b += span
    return out


def find(notes, bpm: float, what: str, context: float = 2.0,
         limit: int = 8) -> list[str]:
    """Locate every occurrence of something, with surrounding context.

    `what` is one of:
      violations  — swing-simulator wrist-breaks
      oov         — transitions outside the human idiom vocabulary
      doubles     — beats where both hands fire together
    """
    hits: list[float] = []
    if what == "violations":
        _par, card = _parity_map(notes, bpm)
        hits = sorted(card.violation_beats)
    elif what == "oov":
        idi = _idiom_map(notes)
        hits = sorted({b for (b, _c), (r, _p) in idi.items() if r is None})
    elif what == "doubles":
        by: dict[float, set] = {}
        for n in notes:
            by.setdefault(round(n.beat, 3), set()).add(n.color)
        hits = sorted(b for b, cs in by.items() if len(cs) == 2)
    else:
        return [f"unknown --find target: {what}"]

    out = [f"# {what}: {len(hits)} occurrence(s)"]
    if not hits:
        return out
    # collapse hits that fall inside the same context window
    shown, last = 0, -1e9
    for b in hits:
        if b - last < context or shown >= limit:
            continue
        last, shown = b, shown + 1
        out.append("")
        out.extend(render(notes, bpm, b - context, b + context,
                          label=f"— around beat {b:.2f} —", idioms=(what == "oov")))
    if len(hits) > shown:
        out.append(f"\n(+{len(hits) - shown} more, showing first {shown})")
    return out


def vs(a_notes, a_bpm, b_notes, b_bpm, t0: float, t1: float,
       labels=("A", "B")) -> list[str]:
    """Two maps side by side aligned in SECONDS, not bars.

    Bar numbers do NOT align between our maps and the human ones: tempo detection
    is wrong on 30% of the eval set, so the same bar index is a different moment
    in the song. Aligning on wall-clock time is the only honest comparison.
    """
    left = render(a_notes, a_bpm, t0 * a_bpm / 60.0, t1 * a_bpm / 60.0,
                  label=f"{labels[0]} @ {a_bpm:.0f}bpm  {t0:.0f}-{t1:.0f}s")
    right = render(b_notes, b_bpm, t0 * b_bpm / 60.0, t1 * b_bpm / 60.0,
                   label=f"{labels[1]} @ {b_bpm:.0f}bpm  {t0:.0f}-{t1:.0f}s")
    w = max((len(x) for x in left), default=0) + 3
    out = []
    for i in range(max(len(left), len(right))):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        out.append(f"{l:<{w}}{r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map")
    ap.add_argument("--bars", help="bar range, e.g. 33-40")
    ap.add_argument("--audio", help="song file, to add per-stem lanes")
    ap.add_argument("--sections", action="store_true", help="whole-song overview")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="two bar ranges to print side by side")
    ap.add_argument("--align", action="store_true",
                    help="per-note distance to the nearest detected onset, in ms "
                         "(● within 50ms · ○ 50-120ms · ✗ beyond). Answers 'is this "
                         "note on a sound I can hear?' note by note, which the "
                         "aggregate onset_precision cannot. Uses the cached onsets for "
                         "a corpus song, or computes them from --audio")
    ap.add_argument("--elements", action="store_true",
                    help="show WALLS, ARCS and CHAINS in the sheet — the three "
                         "elements no reading tool could see until 2026-08-24, and "
                         "the reason 'is FULL less empty than V2?' was unanswerable. "
                         "Adds a `lanes` column (██·· = blocked columns, the player's "
                         "dodge decision) and arc/chain gesture marks, then prints an "
                         "element audit including notes trapped inside walls")
    ap.add_argument("--idioms", action="store_true",
                    help="annotate each note with the rank + corpus frequency of the "
                         "human idiom it completes, or OOV if out of vocabulary")
    ap.add_argument("--find", choices=("violations", "oov", "doubles"),
                    help="show every occurrence of something, with context")
    ap.add_argument("--vs", metavar="OTHER_MAP",
                    help="second map to read alongside, aligned in SECONDS "
                         "(bar numbers do not align across different tempi)")
    ap.add_argument("--secs", default="60-75",
                    help="time range for --vs, e.g. 60-75")
    a = ap.parse_args()

    notes, bpm = load(pathlib.Path(a.map))
    print(f"# {a.map}  —  {len(notes)} notes @ {bpm:.1f} bpm\n")

    if a.sections:
        print("\n".join(sections(notes, bpm)))
        return

    if a.find:
        print("\n".join(find(notes, bpm, a.find)))
        return

    if a.vs:
        o_notes, o_bpm = load(pathlib.Path(a.vs))
        s0, _, s1 = a.secs.partition("-")
        print("\n".join(vs(notes, bpm, o_notes, o_bpm, float(s0), float(s1),
                           labels=(pathlib.Path(a.map).stem, pathlib.Path(a.vs).stem))))
        return

    if a.compare:
        r0, r1 = _bar_range(a.compare[0]), _bar_range(a.compare[1])
        feats = names = None
        if a.audio:
            feats, names = _stem_lanes(pathlib.Path(a.audio), bpm,
                                       max(r0[1], r1[1]))
        left = render(notes, bpm, *r0, feats, names, f"bars {a.compare[0]}")
        right = render(notes, bpm, *r1, feats, names, f"bars {a.compare[1]}")
        w = max(len(x) for x in left) + 3
        for i in range(max(len(left), len(right))):
            l = left[i] if i < len(left) else ""
            r = right[i] if i < len(right) else ""
            print(f"{l:<{w}}{r}")
        return

    b0, b1 = _bar_range(a.bars) if a.bars else (0.0, 8 * BEATS_PER_BAR)
    feats = names = None
    if a.audio:
        feats, names = _stem_lanes(pathlib.Path(a.audio), bpm, b1)
    el = _EL.load_elements(pathlib.Path(a.map)) if a.elements else None
    ons = off = None
    _align_all = None
    if a.align:
        sys.path.insert(0, str(REPO))
        from agent_mapper import refonsets as _RO
        # `song_id` is the map filename's own id; refonsets prefers the cached corpus
        # entry and falls back to computing from audio (content-hashed), so this works
        # on a song the corpus has never seen.
        sid = pathlib.Path(a.map).stem.split("__")[-1].split("_")[0]
        ons = _RO.reference_onsets(sid, audio=a.audio, compute=bool(a.audio))
        if ons is None:
            print("⚠️--align: no onsets for this song and no --audio to compute them; "
                  "alignment column omitted")
        off = _EL.load_elements(pathlib.Path(a.map))["offset"] if ons is not None else 0.0
    print("\n".join(render(notes, bpm, b0, b1, feats, names, idioms=a.idioms,
                           elems=el, onsets=ons, offset=off or 0.0)))
    if ons is not None and len(ons):
        # ★Whole-map alignment, so the page and the cohort agree. The bars on screen
        # are a sample; this is the map. `READING.md` rule 0: the page proposes, the
        # cohort disposes -- and a passage can easily look worse than the map is.
        import numpy as _np
        _o = _np.sort(_np.asarray(ons, dtype=float))
        ts = _np.array([(off or 0.0) + n.beat * 60.0 / bpm for n in notes])
        i = _np.clip(_np.searchsorted(_o, ts), 1, len(_o) - 1)
        d = _np.minimum(_np.abs(ts - _o[i - 1]), _np.abs(ts - _o[i])) * 1000.0
        on, near, miss = (d <= 50).mean(), ((d > 50) & (d <= 120)).mean(), (d > 120).mean()
        print(f"\nALIGNMENT (whole map, {len(notes)} notes)")
        print(f"  ● on a sound (≤50ms)  {on:6.1%}   ← this is `onset_precision`")
        print(f"  ○ near-miss (50-120)  {near:6.1%}")
        print(f"  ✗ nothing there       {miss:6.1%}   ← notes the player hears as unmotivated")
        # ⚠️Signed offset must use the NEARER onset. Measuring against `_o[i-1]`
        # always takes the lower neighbour, which biases the median positive by
        # roughly half a gap -- it read +32ms where the true value is +2ms.
        _sgn = _np.where(_np.abs(ts - _o[i - 1]) <= _np.abs(ts - _o[i]),
                         ts - _o[i - 1], ts - _o[i]) * 1000.0
        print(f"  median |offset| {_np.median(d):.0f}ms · "
              f"signed median {_np.median(_sgn):+.0f}ms")
    if el:
        s = _EL.summary(el)
        print(f"\nELEMENTS  walls {s['walls']} · arcs {s['arcs']} · chains "
              f"{s['chains']} ({s['chain_segments']} segments) · bombs {s['bombs']}")
        print(f"  wall duty {s['wall_duty']:.1%} of the song · "
              f"{s['walls_per_min']}/min · arcs on {s['arc_share_of_notes']:.1%} of notes")
        # 🔴A note inside a wall is unplayable and HAS shipped: wiring walls before
        # idiomize once put 12 notes inside walls, caught only by a collision check.
        flag = "🔴" if s["notes_in_walls"] else "✅"
        print(f"  {flag} notes trapped inside walls: {s['notes_in_walls']}")
        if s["min_dodge_s"] is not None:
            print(f"  dodge windows: {s['tight_dodges_lt_0p5s']} under 0.5 s, "
                  f"tightest {s['min_dodge_s']}s "
                  f"(how long the player had to leave the blocked lane)")
        # ★Reading is not judging. Place every quantity against 2 688 human maps, so
        # the agent gets "18th percentile" rather than "0.042".
        print("\n".join(_EL.format_judgement(_EL.judge(el))))


if __name__ == "__main__":
    main()
