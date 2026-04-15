"""Append-only JSONL leaderboard for experiment runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append a single experiment result to the leaderboard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_leaderboard(path: Path) -> list[dict[str, Any]]:
    """Read all rows; latest row per experiment_id wins if duplicates."""
    if not path.exists():
        return []
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eid = row.get("experiment_id", row.get("name", ""))
            rows[eid] = row
    return list(rows.values())


def rank(
    rows: list[dict[str, Any]],
    *,
    key: str = "composite_score",
    reverse: bool = True,
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: r.get(key, -math.inf), reverse=reverse)
