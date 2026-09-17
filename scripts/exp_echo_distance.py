#!/usr/bin/env python
"""Where does a human's FIRST-OCCURRENCE echo come from? (2026-09-16)

2026-09-13r: on blocks the song has NOT played before we echo 0.418 against his 0.597, and his echo
is the same in both classes, so section-driven copying (`repeat.py`) cannot reach it. The untried
mechanism in TODO is *"return to a figure AT A DISTANCE rather than repeating it locally"*. Before
building that, measure what it would be copying toward:

  * the DISTANCE (bars) from a first-occurrence block to the earlier block it echoes most;
  * whether that source block is itself a first occurrence or a returning one;
  * the echo a first-occurrence block reaches using only NEAR sources (≤ 8 bars back) vs only FAR.

Human side: every structure-cache song with a human map in `data/raw` (the human is read as the map).
Ours: the songset's best builds (`outputs/best_2026-09-13b/`).
Block = 4 bars, `queries._echo`'s bag of `(hand, x, y, dir)`; "returning" = ≥ half its bars are in
`repeat.section_repeats`.

Run: python scripts/exp_echo_distance.py
"""
from __future__ import annotations

import pathlib
import sys
from collections import Counter

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "agent_mapper"))
import queries as Q  # noqa: E402
from repeat import section_repeats  # noqa: E402

B, MIN_NOTES, NEAR = 4, 6, 8


def blocks_of(arr, bar):
    nb = int(bar.max())
    out = {}
    for b0 in range(1, nb + 1, B):
        f = []
        for h in (0, 1):
            f += [(h,) + x for x in Q.figures(arr, bar, b0, b0 + B - 1, h)]
        if len(f) >= MIN_NOTES:
            out[b0] = Counter(f)
    return out


def ov(A, C):
    return sum((A & C).values()) / max(sum(A.values()), sum(C.values()))


def read(arr, bar, rep_bars):
    """Per scored block: (kind, echo, best source distance, source kind, near echo, far echo)."""
    bl = blocks_of(arr, bar)
    ks = sorted(bl)
    kind = {b: ("ret" if sum((b + k) in rep_bars for k in range(B)) >= B / 2 else "first")
            for b in ks}
    rows = []
    for i, b in enumerate(ks[1:], 1):
        sc = [(ov(bl[b], bl[c]), c) for c in ks[:i]]
        e, src = max(sc)
        near = max((v for v, c in sc if b - c <= NEAR), default=0.0)
        far = max((v for v, c in sc if b - c > NEAR), default=0.0)
        rows.append((kind[b], e, b - src, kind[src], near, far))
    return rows


def summarise(label, rows):
    f = [r for r in rows if r[0] == "first"]
    if not f:
        return
    d = np.array([r[2] for r in f])
    print(f"{label:<10} first-occ blocks {len(f):4d}  echo {np.mean([r[1] for r in f]):.3f}"
          f"  | best source: median {np.median(d):4.0f} bars, >{NEAR} bars {np.mean(d > NEAR):5.1%},"
          f" is a returning block {np.mean([r[3] == 'ret' for r in f]):5.1%}"
          f"  | near-only {np.mean([r[4] for r in f]):.3f}  far-only {np.mean([r[5] for r in f]):.3f}")


def main() -> int:
    from verdict import load_arrays
    hum_all, n = [], 0
    for sj in sorted((REPO / "outputs" / "structure_cache").glob("*.json")):
        sid = sj.stem
        zp = REPO / "data" / "raw" / f"{sid}.zip"
        if not zp.exists():
            continue
        try:
            arrs, _, _ = load_arrays(zp, sid, "auto")
        except Exception:  # noqa: BLE001
            continue
        rep = {b for b, _ in section_repeats(sid)}
        if not rep:
            continue
        rows = read(arrs["map"], arrs["bar"], rep)
        hum_all += rows
        n += 1
        if sid in ("1f333", "1f767", "1f8d6", "1f913"):
            summarise(f"H {sid}", rows)
    summarise(f"H all({n})", hum_all)
    print()
    for zp in sorted((REPO / "outputs" / "best_2026-09-13b").glob("*.zip")):
        sid = zp.stem.split("__")[-1]
        try:
            arrs, _, _ = load_arrays(zp, sid, "auto")
        except Exception:  # noqa: BLE001
            continue
        rep = {b for b, _ in section_repeats(sid)}
        summarise(f"O {sid}", read(arrs["map"], arrs["bar"], rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
