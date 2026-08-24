#!/usr/bin/env python
"""EFFORT — what the player's wrists are doing, transition by transition.

★**Why this is not another metric.** `mapjudge` already scores `travel` and
`angle_change` as AGGREGATES over the map, and Kyle's own framing is that difficulty
has two axes: *"difficulty isn't always just NPS, it's how hard are the notes to get to
from the last note as well."* What no tool could do is **point at the transition**. An
aggregate says the map is at the 42nd percentile on travel; it never says *"bar 47,
right hand, 3.2 grid-units in 130 ms, and the wrist has to reverse 135°."*

⇒**This is a LOCATOR, not a score.** It ranks the individual transitions a player would
feel and gives their timestamps, which is the same shape as `flowview`'s founding DoD:
*"given his complaint, we can point at the bar."*

**What a player feels between two notes of one hand, and what is computed here:**
  * **reach speed** -- grid-units per second. The hand has to physically get there.
  * **wrist rotation** -- degrees between the two cut angles, taken from `swing_sim`'s
    PARITY-AWARE tables, so an up-cut on a backhand is not confused with an up-cut on a
    forehand. ⚠️Using raw cut directions here would be wrong: the same arrow is a
    different wrist position depending on which parity the hand is in.
    ★★**A MEDIAN ROTATION OF 0° IS CORRECT, NOT A BUG.** Down-forehand -> up-backhand is
    the NATURAL swing cycle and costs no rotation at all, so this measures rotation
    **beyond** the natural alternation -- which is exactly the part a wrist feels. A
    tool reporting "median 0" here is reporting that most transitions flow.
  * **reset** -- the same swing direction twice with no time to re-seat. This is the
    silent wrist reset `READING.md` looks for by eye.

⚠️**"Hard" only means anything against humans**, so `hard_transitions` scores against
percentiles built by `scripts/calibrate_effort.py`. Without that reference this module
reports physics, not judgement.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from beatsaber_automapper.evaluation import swing_sim as ss  # noqa: E402

REFERENCE_PATH = REPO / "outputs" / "effort_reference.json"


def _angle(direction: int, parity_forehand: bool) -> float:
    """Wrist angle for this cut, given which parity the hand is in."""
    tbl = ss.FOREHAND_ANGLE if parity_forehand else ss.BACKHAND_ANGLE
    return float(tbl.get(int(direction), 0.0))


def transitions(elems: dict) -> list[dict]:
    """Every consecutive same-hand pair, with the effort the player spends on it."""
    spb = 60.0 / max(elems["bpm"], 1e-6)
    out = []
    for color in (0, 1):
        ns = sorted((n for n in elems["notes"] if int(n.get("c", 0)) == color),
                    key=lambda n: float(n.get("b", 0.0)))
        for a, b in zip(ns, ns[1:]):
            dt = (float(b.get("b", 0)) - float(a.get("b", 0))) * spb
            if dt <= 1e-6 or dt > 4.0:
                continue
            dx = int(b.get("x", 0)) - int(a.get("x", 0))
            dy = int(b.get("y", 0)) - int(a.get("y", 0))
            dist = float((dx * dx + dy * dy) ** 0.5)
            da, db = int(a.get("d", 8)), int(b.get("d", 8))
            # Parity alternates on a normal swing, so the second cut is read on the
            # OPPOSITE parity from the first -- that is what makes the rotation real.
            rot = abs(ss._angle_delta(_angle(da, True), _angle(db, False)))
            out.append({
                "t": elems["offset"] + float(a.get("b", 0)) * spb,
                "beat": float(a.get("b", 0)), "color": color,
                "dt": dt, "dist": dist,
                "speed": dist / dt,
                "rotation": rot,
                # 🔴Same direction twice: the wrist has to re-seat silently, and it is
                # one of the reads READING.md does by eye ("long runs of the same
                # direction mean the wrist is resetting").
                "reset": bool(da == db and da != 8),
                "from": (int(a.get("x", 0)), int(a.get("y", 0)), da),
                "to": (int(b.get("x", 0)), int(b.get("y", 0)), db),
            })
    return sorted(out, key=lambda r: r["t"])


def load_reference() -> dict | None:
    if not REFERENCE_PATH.exists():
        return None
    return json.loads(REFERENCE_PATH.read_text())


def hard_transitions(elems: dict, n: int = 8,
                     reference: dict | None = None) -> list[dict]:
    """The transitions a player would actually notice, hardest first.

    ★Ranked against the HUMAN per-transition distribution, so "hard" means *harder than
    human mappers ask for*, not merely "large in this map". A map whose worst transition
    sits at the human 60th percentile has no hard transitions, and should say so.
    """
    reference = reference or load_reference()
    tr = transitions(elems)
    if not tr:
        return []
    if not reference:
        return sorted(tr, key=lambda r: -r["speed"])[:n]
    sp, ro = reference["speed"], reference["rotation"]
    for r in tr:
        # Percentile against humans on each axis; the transition is as hard as its
        # worst axis, because either one alone is what the wrist feels.
        r["speed_pct"] = _pct(sp, r["speed"])
        r["rot_pct"] = _pct(ro, r["rotation"])
        r["hardness"] = max(r["speed_pct"], r["rot_pct"])
    return sorted(tr, key=lambda r: -r["hardness"])[:n]


def _pct(d: dict, v: float) -> float:
    pts = [(0.05, d["p5"]), (0.25, d["p25"]), (0.50, d["median"]),
           (0.75, d["p75"]), (0.95, d["p95"]), (0.99, d["p99"])]
    if v <= pts[0][1]:
        return 0.05
    if v >= pts[-1][1]:
        return 0.99
    for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
        if v0 <= v <= v1:
            return p1 if v1 == v0 else p0 + (p1 - p0) * (v - v0) / (v1 - v0)
    return 0.5


def summary(elems: dict, reference: dict | None = None) -> dict:
    tr = transitions(elems)
    if not tr:
        return {}
    sp = np.array([r["speed"] for r in tr])
    ro = np.array([r["rotation"] for r in tr])
    reference = reference or load_reference()
    out = {
        "n_transitions": len(tr),
        "speed_median": round(float(np.median(sp)), 3),
        "speed_p95": round(float(np.percentile(sp, 95)), 3),
        "rotation_median": round(float(np.median(ro)), 1),
        "reset_share": round(float(np.mean([r["reset"] for r in tr])), 4),
    }
    if reference:
        # Share of transitions above the human 95th percentile -- humans put ~5 % of
        # their own transitions there BY DEFINITION, so a map well above 0.05 is asking
        # for more than human mappers do.
        out["share_above_human_p95_speed"] = round(
            float(np.mean(sp > reference["speed"]["p95"])), 4)
        out["share_above_human_p95_rotation"] = round(
            float(np.mean(ro > reference["rotation"]["p95"])), 4)
    return out
