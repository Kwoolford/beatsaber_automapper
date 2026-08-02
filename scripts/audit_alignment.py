#!/usr/bin/env python
"""Control battery for axis A8 (audio alignment) — run BEFORE it steers anything.

`scripts/audit_eval_suite.py` is the gate every v2 axis has to clear: score real
human maps and deliberately-degenerate control maps, and keep the axis only if it
ranks human above the controls. That gate is the only reason the rest of the suite
is trustworthy, and A8 does not get an exemption for having been born out of
Kyle's ear.

A8 cannot run inside the main battery because it needs AUDIO: the battery samples
random maps from `data/raw`, and onsets have to be cached for each one
(`scripts/build_onset_cache.py --from-raw N`). So it lives here, but reuses the
SAME control constructors — one definition of "degenerate", not two.

**WHAT A8 CAN AND CANNOT CATCH — read this before reading the table.**
Three of the six controls (`random`, `shuffled`, `zigzag`) rewrite x/y/direction
and leave every note TIME untouched. Their alignment is therefore *identical to the
human map's, by construction* — not approximately, exactly. An axis that scores
notes against audio onsets cannot possibly separate them, and an axis that claimed
to would be measuring something other than what it says. They are caught by A1
flow, A3 idiom and A6 hand-role, which is the division of labour working as
intended.

The controls A8 must catch are the ones that destroy timing:
  metronome      constant interval, ignores the music entirely -> must FAIL hard
  timing_random  human patterns at random times -> must FAIL hardest
  timing_jitter  human map nudged +-0.04 beats -> must FAIL, and this is the
                 control that most resembles OUR OWN defect ("many notes just have
                 their own slightly off timings" — Kyle, 2026-08-01)

Everything is scored against a reference built from a DISJOINT half of the human
cohort, so no cohort here is scored against a reference containing itself.

Usage:
  python scripts/build_onset_cache.py --from-raw 80
  python scripts/audit_alignment.py --arms prod,ds055,hl014_ds055,b1_e17_ds055
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from audit_eval_suite import CONTROLS  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402

RAW = REPO / "data" / "raw"
CACHE = REPO / "outputs" / "eval_sweep_cache"

# Controls that leave note times untouched — A8 is blind to them BY CONSTRUCTION.
TIMING_PRESERVING = {"random", "shuffled", "zigzag"}


class _BM:
    def __init__(self, notes):
        self.color_notes = notes
        self.bomb_notes = []


def _metrics(notes, bpm, onsets) -> dict:
    return alignment.alignment_metrics(_BM(sorted(notes, key=lambda n: n.beat)),
                                       bpm=bpm, onsets=onsets).metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="prod,ds055,hl014_ds055,b1_e17_ds055",
                    help="comma-separated eval_sweep_cache arm names to rank")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    ids = sorted(p.stem for p in scorecard.ONSET_CACHE.glob("*.npz"))
    human = []
    for sid in ids:
        zp = RAW / f"{sid}.zip"
        if not zp.exists():
            continue
        try:
            loaded = scorecard._load_any(zp)
        except Exception:  # noqa: BLE001
            continue
        if not loaded or loaded[2] is None or len(loaded[0].color_notes) < 100:
            continue
        human.append((sid, loaded[0].color_notes, loaded[1], loaded[2]))

    if len(human) < 10:
        print(f"only {len(human)} human maps have cached onsets — run "
              "`python scripts/build_onset_cache.py --from-raw 80` first")
        raise SystemExit(2)

    half = len(human) // 2
    ref_recs = [_metrics(n, bpm, on) for _sid, n, bpm, on in human[:half]]
    ref = {k: (v["median"], v["mad"]) for k, v in alignment.calibrate(ref_recs).items()}
    test = human[half:]
    print(f"=== A8 CONTROL BATTERY — reference from {half} human maps, "
          f"cohorts of {len(test)} ===\n")

    cohorts: dict[str, list[dict]] = {"human": [_metrics(n, b, o) for _s, n, b, o in test]}
    for name, fn in CONTROLS.items():
        cohorts[name] = [_metrics(fn(n, rng), b, o) for _s, n, b, o in test]

    for arm in [x for x in a.arms.split(",") if x]:
        recs = []
        for zp in sorted(CACHE.glob(f"{arm}__*.zip")):
            try:
                loaded = scorecard._load_any(zp)
            except Exception:  # noqa: BLE001
                continue
            if loaded and loaded[2] is not None:
                m = _metrics(loaded[0].color_notes, loaded[1], loaded[2])
                if m.get("onset_precision") == m.get("onset_precision"):
                    recs.append(m)
        if recs:
            cohorts[f"{arm}(ours)"] = recs

    print(f"{'cohort':22s}{'n':>4s}{'gap':>8s}{'spread':>8s}{'prec':>8s}"
          f"{'mad_ms':>8s}{'recall':>8s}  verdict")
    print("-" * 88)
    rows = {}
    for name, recs in cohorts.items():
        if len(recs) < 3:
            continue
        cc = alignment.cohort_comparison(recs, ref)
        s = cc["_summary"]
        gap, spread = s["alignment_gap"], s["min_spread"]
        med = {k: cc[k]["median"] for k in alignment.KEYS if k in cc}
        rows[name] = gap
        if name in TIMING_PRESERVING:
            verdict = "n/a — timing-preserving, A8 is blind by construction"
        elif gap <= scorecard.ALIGN_GAP_BAR and spread >= scorecard.ALIGN_SPREAD_BAR:
            verdict = "PASS"
        else:
            bits = []
            if gap > scorecard.ALIGN_GAP_BAR:
                bits.append(f"gap {gap:.2f} > {scorecard.ALIGN_GAP_BAR:.2f}")
            if spread < scorecard.ALIGN_SPREAD_BAR:
                bits.append(f"spread {spread:.2f} collapsed")
            verdict = "FAIL — " + ", ".join(bits)
        print(f"{name:22s}{len(recs):4d}{gap:8.2f}{spread:8.2f}"
              f"{med.get('onset_precision', float('nan')):8.3f}"
              f"{med.get('offset_mad_ms', float('nan')):8.1f}"
              f"{med.get('onset_recall', float('nan')):8.3f}  {verdict}")

    print("\n--- GATE ---")
    timing_controls = ["metronome", "timing_random", "timing_jitter"]
    ok = True
    for c in timing_controls:
        if c not in rows:
            print(f"  {c:14s} NOT SCORED — gate incomplete")
            ok = False
            continue
        passed_gate = rows[c] > rows.get("human", 0.0) and rows[c] > scorecard.ALIGN_GAP_BAR
        print(f"  {c:14s} gap {rows[c]:6.2f} vs human {rows.get('human', float('nan')):.2f}"
              f"  -> {'caught' if passed_gate else 'NOT CAUGHT — A8 IS BLIND TO IT'}")
        ok = ok and passed_gate
    print(f"\n  battery: {'PASS — A8 may steer the generator' if ok else 'FAIL — do NOT use A8 to steer anything yet'}")
    print("\n  (random/shuffled/zigzag are timing-preserving: identical alignment to")
    print("   the human map by construction. A1/A3/A6 are what catch those.)")

    # --- Does the axis move the way Kyle's judgement moves? ---
    # New process as of 2026-08-01: the control battery can only prove an axis
    # separates good from degenerate. It CANNOT prove the axis ranks two plausible
    # maps the way a player would — and that is precisely the failure that let five
    # configurations pass while sounding wrong. Kyle ranked `hl014_ds055` ("a
    # noticeable step up") above `b1_e17_ds055` ("a lot wrong"); no existing axis
    # reproduced that. Any axis added from here on gets checked against his ordering.
    print("\n--- KYLE-ORDERING CHECK (his ear, 2026-08-01) ---")
    a_key, b_key = "hl014_ds055(ours)", "b1_e17_ds055(ours)"
    if a_key in rows and b_key in rows:
        okk = rows[a_key] < rows[b_key]
        print(f"  hl014_ds055 gap {rows[a_key]:.2f} {'<' if okk else '>='} "
              f"b1_e17_ds055 gap {rows[b_key]:.2f}  -> "
              f"{'AGREES with Kyle' if okk else 'DISAGREES with Kyle — do not ship this axis'}")
    else:
        print("  both arms not scored — pass --arms hl014_ds055,b1_e17_ds055")
    ours = {k: v for k, v in rows.items() if k.endswith("(ours)")}
    if ours:
        print("  full ordering, best alignment first: "
              + " < ".join(k.replace("(ours)", "") for k in sorted(ours, key=ours.get)))


if __name__ == "__main__":
    main()
