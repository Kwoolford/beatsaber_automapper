#!/usr/bin/env python
"""How big is a human mapper's FIGURE VOCABULARY, and does it explain the echo?

★**The question this answers.** `q_scatter` reads *"does a cell come back?"* as the mean 4-bar
block echo, and `agent_mapper/repeat.py` answered it by bringing a figure back **when the song's
own section labels say the song came back**. Measured 2026-09-13r on the songset, that worked and
then stopped: on blocks inside a returning section we now sit at **0.613 against the human's
0.601**, but on first-occurrence blocks we sit at **0.418 against his 0.597**.

⇒**The human's echo does not come from song structure at all** — his is 0.601 on returns and
0.597 on music the song has never played before, a difference of 0.004. Ours falls by 0.195. So
whatever he is doing in new music is not repetition of the song, and no amount of structure-driven
copying can reach it.

The hypothesis this script tests is that he is drawing from a **small vocabulary of figures**
(a figure is `(hand, x, y, direction)` — where the hand goes and which way it swings, time thrown
away, exactly as `queries.figures` defines it). On the songset the human of `1f333` plays 1434
notes from **43 distinct figures**, with the top 20 covering **94.7 %** of them; ours plays 1414
notes from **102**, top 20 covering 60.7 %.

Run:
    python scripts/exp_vocabulary.py --n 500
    python scripts/exp_vocabulary.py --n 500 --out outputs/vocabulary_<date>.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from audit_eval_suite import _load_human  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
BEATS_PER_BAR = 4
BLOCK_BARS = 4
MIN_NOTES = 6            # both from queries._echo, so the number means the same thing


def stats(notes) -> dict | None:
    """Vocabulary + block echo for one map, read the way `q_scatter` reads them."""
    if len(notes) < 100:
        return None
    figs = [(n.color, n.x, n.y, n.direction) for n in notes]
    c = Counter(figs)
    v = np.array(sorted(c.values(), reverse=True), dtype=float)
    n = v.sum()
    p = v / n
    # blocks of BLOCK_BARS bars, keyed the same way _echo keys them
    span = BLOCK_BARS * BEATS_PER_BAR
    blocks: dict[int, Counter] = {}
    for f, note in zip(figs, notes):
        blocks.setdefault(int(note.beat // span), Counter())[f] += 1
    ks = sorted(b for b in blocks if sum(blocks[b].values()) >= MIN_NOTES)
    echoes = []
    for i, b in enumerate(ks[1:], 1):
        A = blocks[b]
        echoes.append(max(sum((A & blocks[e]).values())
                          / max(sum(A.values()), sum(blocks[e].values())) for e in ks[:i]))
    if len(echoes) < 8:
        return None
    return dict(notes=int(n), distinct=len(c),
                top8=float(v[:8].sum() / n), top20=float(v[:20].sum() / n),
                entropy=float(-(p * np.log2(p)).sum()),
                perplexity=float(2 ** -(p * np.log2(p)).sum()),
                echo=float(np.mean(echoes)), blocks=len(echoes))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=500, help="how many corpus maps to read")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the per-map records here as json")
    a = ap.parse_args()

    paths = sorted(RAW.glob("*.zip"))
    random.Random(a.seed).shuffle(paths)
    rows, skipped = [], 0
    for zp in paths:
        if len(rows) >= a.n:
            break
        try:
            loaded = _load_human(zp)
        except Exception:  # noqa: BLE001
            loaded = None
        if loaded is None:
            skipped += 1
            continue
        s = stats(loaded[0])
        if s is None:
            skipped += 1
            continue
        s["map"] = zp.stem
        rows.append(s)

    print(f"read {len(rows)} human maps ({skipped} skipped)\n")
    keys = ("notes", "distinct", "top8", "top20", "entropy", "perplexity", "echo")
    print(f"{'metric':<11s} {'p10':>8s} {'median':>8s} {'p90':>8s} {'sd':>8s}")
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        print(f"{k:<11s} {np.percentile(v, 10):8.3f} {np.median(v):8.3f} "
              f"{np.percentile(v, 90):8.3f} {v.std():8.3f}")

    echo = np.array([r["echo"] for r in rows])
    print(f"\ncorrelation with block echo (n={len(rows)}):")
    for k in ("distinct", "top8", "top20", "entropy", "perplexity", "notes"):
        v = np.array([r[k] for r in rows], dtype=float)
        r = float(np.corrcoef(v, echo)[0, 1])
        print(f"  {k:<11s} r = {r:+.3f}   r2 = {r * r:.3f}")

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
