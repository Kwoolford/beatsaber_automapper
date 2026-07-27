"""Shared distribution-scoring core for the v2 evaluation suite.

Every v2 axis scores the same way, and this module is the single implementation of
that method (it is also the first step of the "one scoring system" consolidation —
see docs/eval_suite_v2.md §1 Finding 4).

The method, and why it is this and not something simpler:

* Score against the human **distribution**, not a point target. A point target
  invites Goodharting: you tune until you match the number and the metric stops
  being able to order anything (that is how `h_dist` saturated — it now ranks our
  maps as *more human than human*).
* Rank cohorts by **median shift AND spread**, never by a mean of per-map distances
  to the human median. A mode-collapsed generator sits closer to the median than
  typical human maps do, so a per-map distance score rewards collapse. Shift is
  blind to collapse; spread exposes it. (Learned building axis A1 — the first
  version of the flow metric reproduced the h_dist failure exactly.)
* Use median/MAD rather than mean/sd, so a few pathological human maps cannot
  widen the reference enough to admit anything.
"""
from __future__ import annotations

import math
import statistics


def isnan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def calibrate(records: list[dict], keys: list[str]) -> dict[str, dict]:
    """Human reference: median + MAD per key, from raw per-map metric dicts."""
    out: dict[str, dict] = {}
    for k in keys:
        vals = [r[k] for r in records if r.get(k) is not None and not isnan(r[k])]
        if len(vals) < 5:
            continue
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])
        if mad <= 0:  # degenerate; fall back to sd so the key stays usable
            mad = statistics.pstdev(vals) or 1e-6
        out[k] = {"median": med, "mad": mad, "n": len(vals)}
    return out


def score_map(metrics: dict, reference: dict, sequence_keys: list[str]) -> float:
    """Per-map mean |robust z| over the sequence-aware keys.

    A sanity/outlier check ONLY — do not rank generators by this. See the module
    docstring: use `cohort_comparison` for that.
    """
    if not reference:
        return float("nan")
    zs = []
    for k in sequence_keys:
        if k not in reference:
            continue
        med, mad = reference[k]
        v = metrics.get(k)
        if v is None or isnan(v) or mad <= 0:
            continue
        zs.append(abs(v - med) / mad)
    return statistics.fmean(zs) if zs else float("nan")


def cohort_comparison(records: list[dict], reference: dict, keys: list[str],
                      sequence_keys: list[str], gap_name: str = "gap") -> dict:
    """Compare a COHORT of maps against the human distribution.

    Per key: `shift` = (cohort median - human median) / human MAD, and
    `spread` = cohort MAD / human MAD (< 1 means under-dispersed / collapsed).
    Summary carries the mean |shift| over `sequence_keys` and the worst spread.
    """
    out: dict[str, dict] = {}
    for k in keys:
        if k not in reference:
            continue
        hmed, hmad = reference[k]
        vals = [r[k] for r in records if r.get(k) is not None and not isnan(r[k])]
        if len(vals) < 3 or hmad <= 0:
            continue
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])
        out[k] = {"median": med, "shift": (med - hmed) / hmad,
                  "spread": mad / hmad, "n": len(vals)}
    seq = [abs(out[k]["shift"]) for k in sequence_keys if k in out]
    out["_summary"] = {
        gap_name: statistics.fmean(seq) if seq else float("nan"),
        "min_spread": min((out[k]["spread"] for k in sequence_keys if k in out),
                          default=float("nan")),
    }
    return out
