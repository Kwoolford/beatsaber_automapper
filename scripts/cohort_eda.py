"""Render a comparative table of cohort reference stats.

Scans ``data/cohorts/*/reference.json`` and prints one row per cohort with the
columns that matter for deciding which cohort to train next: n_maps, NPS,
parity baseline, color balance, and top-2 directions.

Usage:
    python scripts/cohort_eda.py
    python scripts/cohort_eda.py --sort nps
    python scripts/cohort_eda.py --sort n_maps --desc
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("cohort_eda")

DIR_NAMES = {0: "U", 1: "D", 2: "L", 3: "R", 4: "UL", 5: "UR", 6: "DL", 7: "DR", 8: ".."}


def _top_dirs(hist: dict, k: int = 2) -> str:
    pairs = sorted(((int(d), p) for d, p in hist.items()), key=lambda x: -x[1])[:k]
    return " ".join(f"{DIR_NAMES[d]}={p:.0%}" for d, p in pairs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohorts-root", type=Path, default=Path("data/cohorts"))
    p.add_argument(
        "--sort",
        choices=("n_maps", "nps", "parity", "notes"),
        default="n_maps",
    )
    p.add_argument("--desc", action="store_true", help="Sort descending (default for n_maps/notes)")
    args = p.parse_args()

    rows: list[dict] = []
    for ref_path in sorted(args.cohorts_root.glob("*/reference.json")):
        data = json.loads(ref_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "slug": ref_path.parent.name,
                "n_maps": data.get("n_maps", 0),
                "n_notes": data.get("n_notes_total", 0),
                "nps": data.get("mean_nps", 0.0),
                "parity": data.get("parity_rate_baseline", 0.0),
                "color_balance": data.get("color_balance", 0.5),
                "top_dirs": _top_dirs(data.get("direction_hist", {})),
            }
        )

    if not rows:
        logger.info("No cohort reference files found under %s", args.cohorts_root)
        return 1

    key_map = {"n_maps": "n_maps", "nps": "nps", "parity": "parity", "notes": "n_notes"}
    desc = args.desc or args.sort in ("n_maps", "notes")
    rows.sort(key=lambda r: r[key_map[args.sort]], reverse=desc)

    hdr = f"{'slug':<18} {'maps':>5} {'notes':>8} {'nps':>5} {'parity':>7} {'r/b':>5}  top2-dirs"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['slug']:<18} "
            f"{r['n_maps']:>5} "
            f"{r['n_notes']:>8} "
            f"{r['nps']:>5.2f} "
            f"{r['parity']:>7.1%} "
            f"{r['color_balance']:>5.2f}  "
            f"{r['top_dirs']}"
        )
    print(sep)
    print(f"{len(rows)} cohorts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
