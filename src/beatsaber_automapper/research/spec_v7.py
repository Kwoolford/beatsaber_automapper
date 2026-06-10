"""V7 layout experiment spec — derivative of ExperimentSpec for V7-5b.

V6 used `ExperimentSpec` (cohort × hyperparam over `scripts/train.py`). V7
has different training scripts (`train_layout.py`, `train_beats.py`) with
argparse signatures, different metrics (`val_token_acc` not `val_loss`), and
needs a two-stage generate+align eval. Rather than overload the V6 spec,
this file mirrors the same interface for the V7 layout stage so the V6
queue runner keeps working unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class V7LayoutSpec:
    """Single V7 Stage 2 layout experiment.

    Fields covered by the hash define experiment identity (same hash → same
    run, dedup on resume). `name`, `notes`, and the `test_*` knobs are
    excluded from the hash so changing the eval song doesn't invalidate
    every prior result.
    """

    name: str

    # Training caps
    max_epochs: int = 30
    max_wall_clock_min: int = 240
    batch_size: int = 32
    learning_rate: float = 1e-4
    seed: int = 42

    # Model
    d_model: int = 384
    n_heads: int = 6
    n_enc_layers: int = 3
    n_dec_layers: int = 4
    dim_feedforward: int = 1536
    dropout: float = 0.1
    x_role_weight: float = 2.0
    max_layout_len: int = 384
    max_phrase_slots: int = 96
    ctx_len: int = 16
    max_song_phrases: int = 150
    sched_sampling_start: float = 0.0
    sched_sampling_end: float = 0.0
    sched_sampling_epochs: int = 20
    patience: int = 12
    difficulties: tuple[str, ...] = ("Expert", "ExpertPlus")

    # Generation + eval (NOT hashed — these can change without re-training)
    beat_ckpt: str = (
        "logs/beat_classifier/version_4/checkpoints/"
        "beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
    )
    test_audio: str = "data/test_songs/SO TIRED ROCK - NUEKI.mp3"
    test_difficulty: str = "ExpertPlus"
    tolerance_ms: float = 50.0

    notes: str = ""

    _NON_HASH = ("name", "notes", "beat_ckpt", "test_audio", "test_difficulty", "tolerance_ms")

    def content_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in self._NON_HASH:
            d.pop(k, None)
        return d

    def experiment_id(self) -> str:
        blob = json.dumps(self.content_dict(), sort_keys=True, default=list).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["experiment_id"] = self.experiment_id()
        return d


def load_v7_queue(path: Path) -> list[V7LayoutSpec]:
    """Load YAML queue with optional top-level `defaults` mapping."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
        defaults: dict[str, Any] = {}
    else:
        items = data.get("experiments", [])
        defaults = data.get("defaults", {}) or {}

    specs: list[V7LayoutSpec] = []
    for entry in items:
        merged = {**defaults, **entry}
        if "difficulties" in merged and isinstance(merged["difficulties"], list):
            merged["difficulties"] = tuple(merged["difficulties"])
        specs.append(V7LayoutSpec(**merged))
    return specs
