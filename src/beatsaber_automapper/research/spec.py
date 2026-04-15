"""Experiment specification — the atomic unit of auto-research.

A spec deterministically defines: what data (cohort or bucket), what model
size, what loss weights, what caps, what seed. Its hash is the experiment ID.
Two specs with the same hash produce the same run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSpec:
    """A single experiment's configuration.

    Exactly one of ``cohort`` or ``bucket`` must be set (not both).

    Attributes:
        name: Human-readable label (does NOT affect hash).
        cohort: Mapper slug (e.g. "joetastic"). Mutually exclusive with ``bucket``.
        bucket: Bucket id from mappers.json. Mutually exclusive with ``cohort``.
        stage: Which training stage — "sequence", "note_pred", "onset".
        model_preset: Hydra model config group — "sequence", "sequence_small", etc.
        max_epochs: Hard cap on epochs.
        max_wall_clock_min: Hard cap on wall-clock minutes. Runner kills the run at this mark.
        batch_size: Training batch size.
        learning_rate: AdamW lr.
        max_samples_per_epoch: Subsampling cap per epoch (None = no cap).
        loss_weights: Extra kwargs merged into model config (flow_loss_alpha, etc).
        seed: RNG seed.
        notes: Free-text (does NOT affect hash).
    """

    name: str
    cohort: str | None = None
    bucket: str | None = None
    stage: str = "sequence"
    model_preset: str = "sequence_small"
    max_epochs: int = 10
    max_wall_clock_min: int = 45
    batch_size: int = 256
    learning_rate: float = 1e-4
    max_samples_per_epoch: int | None = 50_000
    loss_weights: dict[str, float] = field(default_factory=dict)
    seed: int = 42
    notes: str = ""

    def __post_init__(self) -> None:
        if (self.cohort is None) == (self.bucket is None):
            raise ValueError("Exactly one of cohort or bucket must be set")

    @property
    def data_source(self) -> str:
        """"cohort:<slug>" or "bucket:<id>"."""
        if self.cohort:
            return f"cohort:{self.cohort}"
        return f"bucket:{self.bucket}"

    def content_dict(self) -> dict[str, Any]:
        """Fields that contribute to the hash (excludes name + notes)."""
        d = asdict(self)
        d.pop("name", None)
        d.pop("notes", None)
        return d

    def experiment_id(self) -> str:
        """Deterministic 12-char hash of the hash-relevant content."""
        blob = json.dumps(self.content_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["experiment_id"] = self.experiment_id()
        return d


def load_queue(path: Path) -> list[ExperimentSpec]:
    """Load a YAML queue file: either a list of specs or {experiments: [...]}.

    Supports per-entry default inheritance via an optional top-level
    ``defaults`` mapping; each experiment's fields override defaults.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
        defaults: dict[str, Any] = {}
    else:
        items = data.get("experiments", [])
        defaults = data.get("defaults", {}) or {}

    specs: list[ExperimentSpec] = []
    for entry in items:
        merged = {**defaults, **entry}
        specs.append(ExperimentSpec(**merged))
    return specs
