#!/usr/bin/env python
"""Does a style TRANSFER? Replicate leg 3's ordering checks across many songs.

Leg 3 was demonstrated on **one song, one seed** — which, after this session watched
three separate 2-song readings reverse at n=23, is not a result. Kyle's third goal is
*"map to whatever style you want"*, so the question is whether a style preset moves the
map in the asked-for direction **on songs it was never tuned on**.

★**Scored as ORDERINGS, not distances.** A style is a target percentile band, and the
honest test is whether `dense` really is denser than `calm` on the same song — a
pairwise comparison that cannot be satisfied by a lucky absolute value, and that does
not care about the per-song offsets which make absolute percentiles noisy.
🔴**`rank_score` is never used here**: minimising distance-from-typical Goodharts
toward the average map, and a style is deliberately NOT the average map.
⚠️Each style is also judged: **hitting a style is not passing the gate**, and a preset
that only reaches its target by making defective maps has not succeeded.

Usage:
    python scripts/sweep_style.py --songs 1f767 1f8d6 ... --styles calm human dense
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import subprocess
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AM))
sys.path.insert(0, str(REPO / "scripts"))

# The orderings a preset CLAIMS, read straight off style.py's presets.
ORDERINGS = [
    ("nps", "calm", "dense"),
    ("peak_nps", "calm", "dense"),
    ("angle_change", "flowing", "technical"),
    ("crossover", "flowing", "technical"),
    ("travel", "calm", "technical"),
    ("ebpm_burst", "calm", "dense"),
]


def onsets_for(sid):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--styles", nargs="+",
                    default=["calm", "human", "dense", "flowing", "technical"])
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()

    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))][: a.n]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="style_"))
    vals: dict[tuple, float] = {}
    verdicts: dict[str, list] = {s: [] for s in a.styles}
    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        on = onsets_for(sid)
        for st in a.styles:
            out = tmp / f"{st}__{sid}.zip"
            subprocess.run(
                [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
                 "--lead-bias", "0.3", "--style", st,
                 "--name", f"st_{sid}_{st}", "--out", str(out)],
                capture_output=True, text=True, cwd=REPO)
            if not out.exists():
                continue
            try:
                r = mj.judge_zip(out, onsets=on, reference=ref)
            except Exception:  # noqa: BLE001
                continue
            verdicts[st].append(r.verdict())
            for m in r.metrics:
                vals[(sid, st, m.name)] = m.value

    print(f"\nsongs: {len(sids)}   styles: {', '.join(a.styles)}")
    print(f"\n{'ordering':<38}{'holds':>10}{'median gap':>14}")
    print("-" * 64)
    n_ok = 0
    for metric, lo_s, hi_s in ORDERINGS:
        if lo_s not in a.styles or hi_s not in a.styles:
            continue
        gaps = []
        for sid in sids:
            lo = vals.get((sid, lo_s, metric))
            hi = vals.get((sid, hi_s, metric))
            if lo is None or hi is None:
                continue
            gaps.append(hi - lo)
        if not gaps:
            continue
        held = sum(1 for g in gaps if g > 0)
        ok = held >= 0.8 * len(gaps)
        n_ok += ok
        print(f"{metric + ': ' + lo_s + ' < ' + hi_s:<38}"
              f"{held:>4}/{len(gaps):<5}{statistics.median(gaps):>14.3f}"
              f"{'  ✅' if ok else '  🔴'}")
    print(f"\nDoD: each ordering holds on >=8 of 10 songs.  met by {n_ok} of "
          f"{len([o for o in ORDERINGS if o[1] in a.styles and o[2] in a.styles])}")
    print("\nPASS rate per style (hitting a style is NOT passing the gate):")
    for st in a.styles:
        v = verdicts[st]
        if v:
            print(f"  {st:<12}{sum(1 for x in v if x == 'PASS')}/{len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
