"""Auto-researcher harness.

Drives rapid experiment iteration over cohorts × model-sizes × loss-weights.
Each experiment is a deterministic spec → short training → generation →
evaluation → leaderboard row.
"""

from beatsaber_automapper.research.leaderboard import append_row, load_leaderboard
from beatsaber_automapper.research.spec import ExperimentSpec, load_queue

__all__ = [
    "ExperimentSpec",
    "append_row",
    "load_leaderboard",
    "load_queue",
]
