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
           feats=None, names=None, label: str = "") -> list[str]:
    par, _card = _parity_map(notes, bpm)
    by_slot: dict[tuple[int, int], list] = {}
    for n in notes:
        if b0 <= n.beat < b1:
            by_slot.setdefault((int(round(n.beat * SUB)), n.color), []).append(n)

    idx = {nm: i for i, nm in enumerate(names or [])}
    has_audio = feats is not None

    head = f"{'bar':>4s} {'beat':>7s} │ {'L':<10s} │ {'R':<10s}"
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
                cells.append(" " * 10)
                continue
            n = ns[0]
            tag = par.get((round(n.beat, 3), color), "")
            extra = f"+{len(ns) - 1}" if len(ns) > 1 else ""
            cells.append(f"{n.x},{n.y} {ARROW.get(n.direction, '?')} {tag:<3s}{extra}"[:10].ljust(10))
        # only print empty rows on the beat, to keep the score compact
        if not any(c.strip() for c in cells) and s % SUB != 0:
            continue
        row = f"{bar:>4d} {beat:>7.2f} │ {cells[0]} │ {cells[1]}"
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
    """Coarse section view: notes and density per 8 bars."""
    max_beat = max((n.beat for n in notes), default=0.0)
    span = 8 * BEATS_PER_BAR
    out = [f"{'bars':>10s} {'beat':>13s} {'notes':>6s} {'nps':>6s}  density"]
    b = 0.0
    while b < max_beat:
        seg = [n for n in notes if b <= n.beat < b + span]
        secs = span * 60.0 / bpm
        nps = len(seg) / secs if secs else 0.0
        bar = BLOCKS[min(int(nps / 12.0 * 8), 8)] * max(1, int(nps))
        out.append(f"{int(b // BEATS_PER_BAR) + 1:>4d}-{int((b + span) // BEATS_PER_BAR):<5d} "
                   f"{b:>6.0f}-{b + span:<6.0f} {len(seg):>6d} {nps:>6.2f}  {bar}")
        b += span
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
    a = ap.parse_args()

    notes, bpm = load(pathlib.Path(a.map))
    print(f"# {a.map}  —  {len(notes)} notes @ {bpm:.1f} bpm\n")

    if a.sections:
        print("\n".join(sections(notes, bpm)))
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
    print("\n".join(render(notes, bpm, b0, b1, feats, names)))


if __name__ == "__main__":
    main()
