#!/usr/bin/env python
"""P1.2's DoD: can `width` buy back the diagonals without giving back recurrence?

`scripts/diag_diagonal_leak.py` located the leak (n=23): the candidate pool carries
diagonals at the HUMAN rate (0.404 vs 0.415) and `_pick`'s truncation to the `width`
(=3) highest-COUNT candidates strips it to 0.261. Human idiom frequency is
vertical-dominated, so a top-N-by-frequency cut removes diagonals by construction.

⚠️**But `width=3` is not arbitrary.** It became the default on 2026-08-21 on
RECURRENCE evidence -- top-5 cell share 0.342 -> 0.492 (human 0.577),
recurrence-within-8 0.319 -> 0.434 (human 0.496), because reading two maps side by
side showed the human playing a small recurring vocabulary while we played a scatter.
**So this is a trade, and the sweep has to price both sides at once.**

★**ISOLATES `width` EXACTLY.** Each song is BUILT ONCE with `--no-idiomize`, then the
same base zip is re-dressed at every width. Note times, budget, hand assignment and
section plan are therefore byte-identical across arms -- the only thing that varies is
the sampler's truncation. Building per arm instead would let the density search and
the pulse lattice drift between arms and confound the comparison (and would cost 5x
the builds).

**DoD (from TODO P1.2)**: `diagonal_share` inside the human-human spread
(|Δ| <= 0.125) **without `viol` rising** -- and, added here because it is the thing
`width=3` was bought with, without recurrence returning to its pre-2026-08-21 value.

⚠️**This is NOT the refuted "width mapping"**, which concerned mapping `width` to
STYLES for `travel`/`angle_change` and has been measured three times. This asks a
different question -- what `width` does to CUT DIRECTION -- and it has never been asked.

Usage:
    python scripts/sweep_width_cutdir.py --json outputs/width_cutdir.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import subprocess
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AM))

WIDTHS = (0, 3, 5, 8, 12)   # 0 = no truncation at all (pre-2026-08-21 behaviour)

# Human values for the axes this trade is between.
HUMAN = {"diagonal_share": 0.415, "vertical_share": 0.480}
# The human-human spread P1.2's DoD is written against.
SPREAD_DIAG = 0.125

# ⚠️There is NO `recurrence` metric in the judge -- the 2026-08-21 recurrence evidence
# (top-5 cell share, recurrence-within-8) came from a READING tool, not the suite. The
# judge's proxy for the same property is `idiom_local` (distinct idioms per
# 16-transition window), which `width=3` drove to the 15th human percentile and which
# TODO records as CONVERGENCE rather than a cost. So the trade is priced here on
# `idiom_local` + `idiom_top50`, and percentiles are printed because the DoD is about
# sitting inside a human range, not about a raw value.
WATCH = ("diagonal_share", "vertical_share", "idiom_local", "idiom_top50",
         "idiom_coverage", "travel", "angle_change")


def onsets_for(sid):
    f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[list(z.keys())[0]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", default=None)
    ap.add_argument("--json", default="")
    ap.add_argument("--widths", nargs="*", type=int, default=list(WIDTHS))
    # ⚠️Cut direction is a GEOMETRY property, and geometry is exactly where seeds
    # matter on the agent path (the 10 time-domain metrics are seed-invariant by
    # construction, but this is not one of them). A single-seed width recommendation
    # would be the "identical config scored 4, 4, 2" trap.
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    a = ap.parse_args()
    from beatsaber_automapper.evaluation import mapjudge as mj

    import idiomize as I
    ref = mj.load_reference()

    sids = a.songs or [p.stem for p in
                       sorted((REPO / "data" / "eval_songset").glob("*.ogg"))]
    widths = tuple(a.widths)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="widthsweep_"))
    res: dict[int, list] = {w: [] for w in widths}
    # per (width, seed) so the seed SPREAD can be reported, not just pooled medians
    per_seed: dict[tuple[int, int], list] = {}

    for sid in sids:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        base = tmp / f"BASE__{sid}.zip"
        # ★Build ONCE, un-dressed. Every arm below re-dresses this same map.
        subprocess.run(
            [sys.executable, str(AM / "autobuild.py"), str(audio), "--pulse",
             "--lead-bias", "0.2", "--name", f"wd_{sid}", "--out", str(base),
             "--no-idiomize"],
            capture_output=True, text=True, cwd=REPO)
        if not base.exists():
            print(f"{sid}: build failed")
            continue
        on = onsets_for(sid)
        for w in widths:
            for sd in a.seeds:
                out = tmp / f"W{w}s{sd}__{sid}.zip"
                try:
                    I.idiomize_zip(base, out, seed=sd, width=w)
                except Exception as exc:  # noqa: BLE001
                    print(f"{sid} w={w} seed={sd}: idiomize failed ({exc})")
                    continue
                try:
                    r = mj.judge_zip(out, onsets=on, reference=ref)
                except Exception:  # noqa: BLE001
                    continue
                res[w].append(r)
                per_seed.setdefault((w, sd), []).append(r)
        print(f"  {sid} done", flush=True)

    def med(rs, name):
        vals = [m.value for r in rs for m in r.metrics if m.name == name]
        return st.median(vals) if vals else float("nan")

    def medpct(rs, name):
        vals = [m.pct for r in rs for m in r.metrics
                if m.name == name and m.pct is not None]
        return st.median(vals) if vals else float("nan")

    print(f"\n{'width':<7}{'PASS':>8}{'viol':>7}" + "".join(f"{k[:13]:>16}" for k in WATCH))
    print("-" * 130)
    rows = []
    for w in widths:
        rs = res[w]
        if not rs:
            continue
        npass = sum(1 for r in rs if r.verdict() == "PASS")
        viol = sum(getattr(r, "n_violations", 0) or 0 for r in rs)
        cells = {k: med(rs, k) for k in WATCH}
        pcts = {k: medpct(rs, k) for k in WATCH}
        star = "  <-- today" if w == 3 else ""
        print(f"{w:<7}{npass:>4}/{len(rs):<3}{viol:>7}"
              + "".join(f"{cells[k]:>9.3f}({100 * pcts[k]:>3.0f}%)" for k in WATCH)
              + star)
        rows.append(dict(width=w, npass=npass, n=len(rs), viol=viol, **cells,
                         **{f"pct_{k}": pcts[k] for k in WATCH}))

    print(f"\nhuman: diagonal_share {HUMAN['diagonal_share']:.3f} · "
          f"vertical_share {HUMAN['vertical_share']:.3f}   "
          f"(P1.2 DoD: |Δdiagonal| ≤ {SPREAD_DIAG})")

    # ★★An arm gap only means something if it clears the SEED spread. This repo has
    # recorded a "cost" that turned out to sit inside it (role_swap_rate 58 -> 81 %,
    # gap 8.8 against a spread of 10.1) and an identical config scoring npass 4, 4, 2.
    if len(a.seeds) > 1:
        print(f"\nSEED SPREAD (n={len(a.seeds)} seeds, {len(sids)} songs each)")
        print(f"  {'width':<7}" + "".join(f"{k[:13]:>16}" for k in
                                          ("diagonal_share", "idiom_local",
                                           "idiom_top50", "idiom_coverage")))
        for w in widths:
            cells = []
            for k in ("diagonal_share", "idiom_local", "idiom_top50",
                      "idiom_coverage"):
                per = [med(per_seed[(w, sd)], k) for sd in a.seeds
                       if per_seed.get((w, sd))]
                if len(per) > 1:
                    cells.append(f"{st.median(per):8.3f}±{st.pstdev(per):<6.3f}")
                else:
                    cells.append(f"{(per[0] if per else float('nan')):>15.3f}")
            print(f"  {w:<7}" + "".join(f"{c:>16}" for c in cells))
        print("  ⇒ read every arm gap above against these ± values before believing it.")

    print("\nDoD CHECK")
    base_row = next((r for r in rows if r["width"] == 3), None)
    for r in rows:
        d = abs(r["diagonal_share"] - HUMAN["diagonal_share"])
        ok = "✅" if d <= SPREAD_DIAG else "  "
        note = ""
        if base_row and r["width"] != 3:
            note = (f"   idiom_local {base_row['idiom_local']:.3f}"
                    f"(p{100 * base_row['pct_idiom_local']:.0f}) -> "
                    f"{r['idiom_local']:.3f}(p{100 * r['pct_idiom_local']:.0f})")
        print(f"  width {r['width']:<3} |Δdiagonal| {d:.3f} {ok}"
              f"   viol {r['viol']}{note}")

    print("\nHOW TO READ IT")
    print("  A width that meets |Δdiagonal| ≤ 0.125 with viol unchanged AND idiom_local")
    print("     still low (the CONVERGENCE width=3 bought) is a real fix for P1.2.")
    print("  If every width that fixes the diagonals also sends idiom_local back up")
    print("     toward the 98th percentile it sat at before 2026-08-21, then width is a")
    print("     TRADE, and the fix has to come from HOW the pool is truncated rather")
    print("     than how much -- e.g. truncate by frequency but preserve the pool's")
    print("     direction mix, which would keep the small local vocabulary AND the")
    print("     diagonals.")
    print("  ⚠️`p` is NOT a ranking statistic here: high p means blander.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
