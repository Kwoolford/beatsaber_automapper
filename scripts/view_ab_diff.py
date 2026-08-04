#!/usr/bin/env python
"""Visual EDA (3) — A/B DIFF: show only what a lever actually changed.

Kyle's spare-time ask, item 3. Most sweeps move very little; drawing both maps in
full hides the change among the ~95 % that is identical. This draws the notes that
**differ** and answers the question a lever sweep actually poses:

    when a lever spends more notes (or fewer), does it spend them on the
    musically important moments, or on filler?

That is answered by bucketing every added / removed note by the **coincidence
order k** of the nearest stem-onset event (from `eval_coincidence.py`). Humans map
a k>=3 event 72-85 % of the time and a k=1 event 41 %, so:

    added notes concentrated at HIGH k  -> the budget is buying real musical events
    added notes concentrated at k=1     -> the budget is buying filler, and the
                                           nps cost is being paid for nothing

Built to read the `BEAT_NOTE_BUDGET` arms (W2) and `BEAT_END_RESOLVE` (W7), but it
is lever-agnostic — any two maps of the same song.

Usage:
    python scripts/view_ab_diff.py --song 1f8d6 \
        --a 'outputs/eval_sweep_cache/tf_trim_ev03_rc05#s0__1f8d6.zip' \
        --b 'outputs/eval_sweep_cache/nb130#s0__1f8d6.zip'
    # text summary only, no PNG:
    python scripts/view_ab_diff.py --song 1f8d6 --a A.zip --b B.zip --no-plot
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from eval_coincidence import events_for  # noqa: E402

TOL = 0.050
PANEL_SECS = 30.0


def k_of(times, ks, t):
    """Coincidence order of the event nearest `t`, or 0 if none within TOL."""
    if len(times) == 0:
        return 0
    i = int(np.searchsorted(times, t))
    best, bk = np.inf, 0
    for j in (i - 1, i):
        if 0 <= j < len(times):
            d = abs(t - times[j])
            if d < best:
                best, bk = d, int(ks[j])
    return bk if best <= TOL else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True)
    ap.add_argument("--a", required=True, help="control map")
    ap.add_argument("--b", required=True, help="arm map")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    LA = scorecard._load_any(pathlib.Path(a.a))
    LB = scorecard._load_any(pathlib.Path(a.b))
    if not LA or not LB:
        sys.exit("could not load one of the maps")
    ta = np.asarray(alignment.note_times(LA[0], LA[1]), dtype=float)
    tb = np.asarray(alignment.note_times(LB[0], LB[1]), dtype=float)
    bpm = float(LA[1])

    sa, sb = set(np.round(ta, 4)), set(np.round(tb, 4))
    added = np.array(sorted(sb - sa))
    removed = np.array(sorted(sa - sb))
    kept = np.array(sorted(sa & sb))

    times, ks = events_for(a.song, 0.030)

    print(f"=== {a.song}: {pathlib.Path(a.a).name}  ->  {pathlib.Path(a.b).name} ===")
    print(f"  A {len(sa)} distinct note times   B {len(sb)}   "
          f"kept {len(kept)}  added {len(added)}  removed {len(removed)}")

    def bucket(arr, label):
        if len(arr) == 0:
            print(f"  {label}: none")
            return
        c = collections.Counter(k_of(times, ks, t) for t in arr)
        tot = sum(c.values())
        cells = "  ".join(f"k={k}: {c.get(k,0):4d} ({c.get(k,0)/tot:5.1%})"
                          for k in (0, 1, 2, 3, 4))
        print(f"  {label} by coincidence order:  {cells}")
        hi = sum(c.get(k, 0) for k in (3, 4)) / tot
        print(f"    -> {hi:.1%} landed on a k>=3 multi-instrument event")

    bucket(added, "ADDED  ")
    bucket(removed, "REMOVED")
    # the baseline to judge those shares against: what fraction of ALL events are k>=3
    base = float(np.mean(ks >= 3))
    print(f"  (for reference: {base:.1%} of all events in this song are k>=3, and")
    print("   humans map a k>=3 event 72-85% of the time vs 41% for k=1)")

    if a.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end = float(max(ta.max(), tb.max())) + 1.0
    panels = int(np.ceil(end / PANEL_SECS))
    fig, axes = plt.subplots(panels, 1, figsize=(19, 1.9 * panels), squeeze=False)
    axes = axes[:, 0]
    for pi, ax in enumerate(axes):
        t0, t1 = pi * PANEL_SECS, min((pi + 1) * PANEL_SECS, end)
        m = (times >= t0) & (times <= t1)
        for t, k in zip(times[m], ks[m]):
            ax.vlines(t, 0, 0.18 * k, color="#222", alpha=0.85 if k >= 3 else 0.25,
                      lw=1.4 if k >= 3 else 0.7)
        for arr, y, col, lab in ((kept, 1.15, "0.72", "kept"),
                                 (added, 1.55, "#2a9d5c", "added"),
                                 (removed, 1.55, "#c9002b", "removed")):
            v = arr[(arr >= t0) & (arr <= t1)] if len(arr) else arr
            if len(v):
                ax.vlines(v, y, y + 0.32, color=col, lw=1.7)
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, 2.05)
        ax.set_yticks([0.4, 1.3, 1.7])
        ax.set_yticklabels(["k", "kept", "±"], fontsize=8)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_title(
        f"{a.song}   {pathlib.Path(a.a).stem} → {pathlib.Path(a.b).stem}   (bpm {bpm:g})\n"
        f"green = added ({len(added)})   red = removed ({len(removed)})   "
        "tall black = k≥3 multi-instrument event",
        fontsize=11, loc="left")
    fig.tight_layout()
    out = pathlib.Path(a.out) if a.out else REPO / "outputs" / f"diff_{a.song}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
