#!/usr/bin/env python
"""K2 — do diagonal cuts rise or fall with local note speed?

Kyle: broad outside-in diagonal swings are *wanted* in slow sections and on drops
— *"they get the player moving and feel like they are playing a grand
orchestra"* — and only become a problem in fast passages, where they are
*"difficult but possible, and not preferred"*. So the claim is not "too many
diagonals" flat; it is that the **correlation with speed has the wrong sign**.

★ WHY THIS SCRIPT EXISTS. K2's evidence in TODO.md is **one song** (1f333:
0.516 / 0.477 / 0.530 / 0.653 across speed bands). The landmine list warns that
single-song probes have already killed two hypotheses in this project — 1f333 in
particular is a half-tempo song where beat-domain metrics lie. Nobody should
build a decode lever on one song's four numbers, so this measures the whole
corpus and, critically, measures the **human** cohort the same way: if humans
also trend upward, the target is the level, not the slope.

Local speed is notes per second in a centred window, computed on the union of
both hands (that is what the player experiences). Bands are chosen to match the
numbers already recorded for 1f333 so the old evidence stays comparable.

Usage:
    python scripts/eval_diagonal_vs_speed.py --maps 'outputs/eval_sweep_cache/arm#s0__*.zip'
    python scripts/eval_diagonal_vs_speed.py --human --n 60
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from beatsaber_automapper.evaluation import scorecard  # noqa: E402
from beatsaber_automapper.evaluation.playfeel import DIAGONAL  # noqa: E402

BANDS = [(0, 4), (4, 7), (7, 10), (10, 1e9)]
BAND_LABELS = ["0-4", "4-7", "7-10", "10+"]
WIN_SEC = 2.0


def per_map(beatmap, bpm: float) -> dict | None:
    """Diagonal share within each local-speed band, for one map."""
    notes = list(beatmap.color_notes)
    if bpm <= 0 or len(notes) < 100:
        return None
    spb = 60.0 / bpm
    t = np.array([n.beat * spb for n in notes], dtype=np.float64)
    order = np.argsort(t)
    t = t[order]
    d = np.array([notes[i].direction for i in order], dtype=int)

    # Local nps: count of notes within +-WIN/2 of each note, over the window.
    lo = np.searchsorted(t, t - WIN_SEC / 2.0, side="left")
    hi = np.searchsorted(t, t + WIN_SEC / 2.0, side="right")
    nps = (hi - lo) / WIN_SEC

    is_diag = np.isin(d, DIAGONAL)
    out: dict = {"n_notes": len(t)}
    shares = []
    for (a, b), lab in zip(BANDS, BAND_LABELS):
        m = (nps >= a) & (nps < b)
        if m.sum() < 20:          # too few notes in this band to be meaningful
            out[lab] = None
            shares.append(None)
            continue
        v = float(is_diag[m].mean())
        out[lab] = round(v, 4)
        shares.append(v)

    # Slope of diagonal share against local nps, over notes rather than bands,
    # so it does not depend on where the band edges fall.
    if len(t) > 50 and nps.std() > 1e-9:
        out["slope"] = round(float(np.polyfit(nps, is_diag.astype(float), 1)[0]), 5)
    else:
        out["slope"] = None
    ok = [s for s in shares if s is not None]
    out["overall"] = round(float(is_diag.mean()), 4)
    out["band_span"] = round(ok[-1] - ok[0], 4) if len(ok) >= 2 else None
    return out


def _score(paths: list[pathlib.Path], label: str) -> list[dict]:
    rows = []
    for p in paths:
        try:
            L = scorecard._load_any(p)
        except Exception:  # noqa: BLE001
            continue
        if not L:
            continue
        r = per_map(L[0], L[1])
        if r:
            r["map"] = p.name
            rows.append(r)
    print(f"{label}: {len(rows)} maps scored")
    return rows


def report(rows: list[dict], label: str) -> dict:
    if not rows:
        return {}
    print(f"\n=== {label} (n={len(rows)}) ===")
    print("diagonal share by LOCAL nps band (median across maps)")
    print("  " + "".join(f"{b:>10s}" for b in BAND_LABELS)
          + f"{'overall':>10s}{'slope':>10s}{'span':>9s}")
    cells = []
    for lab in BAND_LABELS:
        v = [r[lab] for r in rows if r.get(lab) is not None]
        cells.append(st.median(v) if v else float("nan"))
    sl = [r["slope"] for r in rows if r.get("slope") is not None]
    sp = [r["band_span"] for r in rows if r.get("band_span") is not None]
    ov = [r["overall"] for r in rows if r.get("overall") is not None]
    print("  " + "".join(f"{c:>10.3f}" for c in cells)
          + f"{st.median(ov):>10.3f}{st.median(sl):>10.5f}"
          + (f"{st.median(sp):>9.3f}" if sp else f"{'--':>9s}"))
    n_up = sum(1 for s in sl if s > 0)
    print(f"  maps whose diagonal share RISES with speed: {n_up}/{len(sl)} "
          f"({n_up/len(sl):.0%})")
    return {"bands": [round(c, 4) for c in cells],
            "overall": round(st.median(ov), 4),
            "slope_median": round(st.median(sl), 5),
            "share_rising": round(n_up / len(sl), 4),
            "n": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--maps", default=None)
    ap.add_argument("--label", default="ours")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out = {}
    if a.human:
        zips = sorted((REPO / "data" / "raw").glob("*.zip"))[: a.n]
        out["human"] = report(_score(zips, "human"), "HUMAN corpus")
    if a.maps:
        paths = [pathlib.Path(p) for p in sorted(glob.glob(a.maps))]
        out[a.label] = report(_score(paths, a.label), a.label.upper())

    if "human" in out and a.label in out and out["human"] and out[a.label]:
        h, g = out["human"], out[a.label]
        print("\n=== VERDICT ===")
        print(f"  human slope {h['slope_median']:+.5f}   ours {g['slope_median']:+.5f}")
        print(f"  human rising {h['share_rising']:.0%}    ours {g['share_rising']:.0%}")
        print(f"  human overall {h['overall']:.3f}  ours {g['overall']:.3f}")
        same_sign = (h["slope_median"] > 0) == (g["slope_median"] > 0)
        print()
        if same_sign:
            print("  ★ SAME SIGN as the human corpus. K2's premise -- that the")
            print("    correlation should be negative -- does NOT hold for humans")
            print("    either. The target is then the LEVEL, not the slope, and a")
            print("    lever that inverts the correlation would push us AWAY from")
            print("    human behaviour.")
        else:
            print("  ★ OPPOSITE SIGN to the human corpus. K2 is confirmed as a")
            print("    slope defect and a speed-dependent lever is justified.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
