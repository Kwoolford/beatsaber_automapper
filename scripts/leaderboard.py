"""Render the experiments leaderboard as a table.

Usage:
    python scripts/leaderboard.py
    python scripts/leaderboard.py --sort-by parity_rate --asc
    python scripts/leaderboard.py --cohort joetastic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beatsaber_automapper.research.leaderboard import load_leaderboard, rank  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--path",
        type=Path,
        default=Path("experiments/leaderboard.jsonl"),
    )
    p.add_argument("--sort-by", default="composite_score")
    p.add_argument("--asc", action="store_true", help="Ascending sort")
    p.add_argument("--cohort", help="Filter by cohort slug")
    p.add_argument("--bucket", help="Filter by bucket id")
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args()

    rows = load_leaderboard(args.path)
    if args.cohort:
        rows = [r for r in rows if r.get("data_source") == f"cohort:{args.cohort}"]
    if args.bucket:
        rows = [r for r in rows if r.get("data_source") == f"bucket:{args.bucket}"]

    sort_key = args.sort_by
    ranked = rank(rows, key=sort_key, reverse=not args.asc)[: args.limit]

    if not ranked:
        print(f"No rows in {args.path}")
        return 0

    cols = [
        ("rank", 4),
        ("experiment_id", 14),
        ("name", 28),
        ("data_source", 28),
        ("composite_score", 8),
        ("parity_rate", 8),
        ("n_notes", 7),
        ("wall_clock_sec", 10),
    ]
    header = " ".join(f"{c[0]:<{c[1]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(ranked, 1):
        p_rate = row.get("playability", {}).get("parity_rate", 0.0)
        n_notes = row.get("playability", {}).get("n_notes", 0)
        vals = [
            str(i),
            str(row.get("experiment_id", ""))[:14],
            str(row.get("name", ""))[:28],
            str(row.get("data_source", ""))[:28],
            f"{row.get('composite_score', 0):.3f}",
            f"{p_rate:.3f}",
            str(n_notes),
            f"{row.get('wall_clock_sec', 0):.0f}",
        ]
        print(" ".join(f"{v:<{c[1]}}" for v, c in zip(vals, cols)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
