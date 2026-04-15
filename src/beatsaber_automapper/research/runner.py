"""Experiment runner — orchestrates train → generate → evaluate → log.

Each experiment gets an isolated directory:
    experiments/runs/{experiment_id}/
        spec.yaml           — frozen spec
        train.log           — training stdout/stderr
        generate.log        — generation stdout/stderr
        checkpoints/        — best checkpoint produced by training
        generated/          — generated test map .zip + raw .dat
        metrics.json        — evaluated metrics
        status.json         — state: queued|training|generating|evaluating|done|failed

The runner shells out to ``scripts/train.py`` and ``scripts/generate.py`` so it
stays loosely coupled to the Hydra/Lightning training path.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from beatsaber_automapper.research.leaderboard import append_row
from beatsaber_automapper.research.metrics import (
    CohortReference,
    analyze_generated_zip,
    composite_score,
    style_closeness,
)
from beatsaber_automapper.research.spec import ExperimentSpec

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    experiment_id: str
    status: str
    run_dir: Path
    metrics: dict[str, Any]


def _write_status(run_dir: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(extra)
    (run_dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _freeze_spec(run_dir: Path, spec: ExperimentSpec) -> None:
    (run_dir / "spec.yaml").write_text(
        yaml.safe_dump(dataclasses.asdict(spec), sort_keys=False),
        encoding="utf-8",
    )


def _build_train_cmd(spec: ExperimentSpec, run_dir: Path, project_root: Path) -> list[str]:
    """Construct `python scripts/train.py ...` with Hydra overrides.

    Uses the top-level `cohort=`/`bucket=` keys added to configs/train.yaml.
    """
    python = sys.executable
    # Hydra composes model configs via defaults list. To swap in a preset like
    # sequence_small.yaml, override the specific stage's config group.
    model_override = f"model/{spec.stage}={spec.model_preset}"
    base = [
        python,
        str(project_root / "scripts" / "train.py"),
        f"stage={spec.stage}",
        model_override,
        f"output_dir={run_dir}",
        f"max_epochs={spec.max_epochs}",
        f"data.dataset.batch_size={spec.batch_size}",
    ]
    if spec.max_samples_per_epoch is not None:
        base.append(f"max_samples_per_epoch={spec.max_samples_per_epoch}")
    if spec.cohort:
        base.append(f"cohort={spec.cohort}")
    if spec.bucket:
        base.append(f"bucket={spec.bucket}")
    base.append(f"seed={spec.seed}")
    for k, v in spec.loss_weights.items():
        base.append(f"model.{spec.stage}.{k}={v}")
    base.append(f"model.{spec.stage}.learning_rate={spec.learning_rate}")
    return base


def _run_with_timeout(
    cmd: list[str], log_path: Path, timeout_s: int, cwd: Path
) -> tuple[int, bool]:
    """Run cmd, streaming to log_path. Kill at timeout_s. Returns (rc, timed_out)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
            return rc, False
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            return -1, True


_OOM_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "torch.cuda.OutOfMemoryError",
    "CUBLAS_STATUS_ALLOC_FAILED",
)


def _classify_failure(log_path: Path, timed_out: bool) -> str:
    """Classify a failed run by scanning its log for known markers.

    Returns one of: ``oom`` (OOM marker found), ``timeout`` (wall-clock hit),
    ``error`` (other non-zero exit). A retry policy can key off this string.
    """
    if timed_out:
        return "timeout"
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "error"
    return "oom" if any(m in text for m in _OOM_MARKERS) else "error"


def _find_best_checkpoint(run_dir: Path) -> Path | None:
    """Walk run_dir for best Lightning checkpoint (lowest val_loss in filename)."""
    candidates = list(run_dir.rglob("*.ckpt"))
    if not candidates:
        return None
    def score(p: Path) -> float:
        name = p.stem
        for tok in name.split("-"):
            if tok.startswith("val_loss="):
                try:
                    return float(tok.split("=", 1)[1])
                except ValueError:
                    pass
        return float("inf")
    return min(candidates, key=score)


def _build_generate_cmd(
    spec: ExperimentSpec,
    seq_ckpt: Path,
    test_audio: Path,
    out_zip: Path,
    onset_ckpt: Path | None,
    project_root: Path,
) -> list[str]:
    python = sys.executable
    cmd = [
        python,
        str(project_root / "scripts" / "generate.py"),
        str(test_audio),
        "--seq-ckpt",
        str(seq_ckpt),
        "--output",
        str(out_zip),
        "--difficulty",
        "Expert",
    ]
    if onset_ckpt is not None:
        cmd += ["--onset-ckpt", str(onset_ckpt)]
    return cmd


def _load_cohort_reference(cohort_slug: str, cohort_root: Path) -> CohortReference | None:
    """Load precomputed cohort reference stats if available."""
    ref_path = cohort_root / cohort_slug / "reference.json"
    if not ref_path.exists():
        return None
    data = json.loads(ref_path.read_text(encoding="utf-8"))
    return CohortReference(
        n_maps=data.get("n_maps", 0),
        mean_nps=data.get("mean_nps", 6.0),
        direction_hist={int(k): v for k, v in data.get("direction_hist", {}).items()},
        parity_rate_baseline=data.get("parity_rate_baseline", 0.05),
        color_balance=data.get("color_balance", 0.5),
    )


def _audio_duration_sec(path: Path) -> float:
    """Read duration from audio file header. Falls back to 60.0 on failure."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception as e:
        logger.warning("audio duration probe failed for %s: %s — using 60s fallback", path, e)
        return 60.0


def run_experiment(
    spec: ExperimentSpec,
    *,
    project_root: Path,
    experiments_root: Path,
    test_audio: Path,
    test_duration_sec: float | None,
    onset_ckpt: Path | None,
    cohort_root: Path,
    leaderboard_path: Path,
) -> RunResult:
    """Execute a single experiment end-to-end.

    If ``test_duration_sec`` is None, the duration is auto-detected from the
    audio file header. Correct duration matters for NPS-based style metrics.
    """
    if test_duration_sec is None:
        test_duration_sec = _audio_duration_sec(test_audio)
    exp_id = spec.experiment_id()
    run_dir = experiments_root / "runs" / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _freeze_spec(run_dir, spec)
    _write_status(run_dir, "queued", {"spec_name": spec.name})

    t_start = time.time()

    # ---- TRAIN ----
    _write_status(run_dir, "training")
    train_cmd = _build_train_cmd(spec, run_dir, project_root)
    train_log = run_dir / "train.log"
    logger.info("[%s] train: %s", exp_id, " ".join(train_cmd))
    rc, timed_out = _run_with_timeout(
        train_cmd, train_log, timeout_s=spec.max_wall_clock_min * 60, cwd=project_root
    )
    if rc != 0:
        reason = _classify_failure(train_log, timed_out)
        _write_status(
            run_dir,
            "failed",
            {
                "phase": "train",
                "rc": rc,
                "timed_out": timed_out,
                "failure_reason": reason,
            },
        )
        return RunResult(exp_id, f"failed_train_{reason}", run_dir, {})

    # ---- CHECKPOINT ----
    best_ckpt = _find_best_checkpoint(run_dir)
    if best_ckpt is None:
        _write_status(run_dir, "failed", {"phase": "no_checkpoint"})
        return RunResult(exp_id, "failed_no_ckpt", run_dir, {})

    # ---- GENERATE ----
    _write_status(run_dir, "generating")
    out_zip = run_dir / "generated" / "test_map.zip"
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    gen_cmd = _build_generate_cmd(
        spec, best_ckpt, test_audio, out_zip, onset_ckpt, project_root
    )
    gen_log = run_dir / "generate.log"
    logger.info("[%s] generate: %s", exp_id, " ".join(gen_cmd))
    rc, _ = _run_with_timeout(
        gen_cmd, gen_log, timeout_s=10 * 60, cwd=project_root
    )
    if rc != 0 or not out_zip.exists():
        _write_status(run_dir, "failed", {"phase": "generate", "rc": rc})
        return RunResult(exp_id, "failed_generate", run_dir, {})

    # ---- EVALUATE ----
    _write_status(run_dir, "evaluating")
    playability = analyze_generated_zip(out_zip, test_duration_sec)

    style: dict[str, float] | None = None
    if spec.cohort:
        ref = _load_cohort_reference(spec.cohort, cohort_root)
        if ref is not None:
            style = style_closeness(playability, ref)

    score = composite_score(playability, style)

    wall_clock = time.time() - t_start
    metrics = {
        "experiment_id": exp_id,
        "name": spec.name,
        "data_source": spec.data_source,
        "stage": spec.stage,
        "model_preset": spec.model_preset,
        "seed": spec.seed,
        "wall_clock_sec": wall_clock,
        "best_checkpoint": str(best_ckpt),
        "composite_score": score,
        "playability": playability.as_dict(),
        "style": style or {},
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    append_row(leaderboard_path, metrics)

    _write_status(run_dir, "done", {"composite_score": score})
    logger.info("[%s] done score=%.3f wall=%.1fs", exp_id, score, wall_clock)
    return RunResult(exp_id, "done", run_dir, metrics)
