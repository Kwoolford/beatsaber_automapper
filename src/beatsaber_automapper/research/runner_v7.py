"""V7 layout experiment runner — train → generate → eval_alignment → leaderboard.

Each run drops into ``experiments/runs/{id}/``:
    spec.yaml          frozen spec
    train.log          stdout/stderr of train_layout.py
    generate.log       stdout/stderr of generate.py --v7
    alignment.log      stdout/stderr of eval_alignment.py
    alignment.json     parsed alignment report
    generated/test_map.zip
    metrics.json       aggregated metrics (best val + alignment + playability)
    status.json        queued|training|generating|evaluating|done|failed

Layout training writes its Lightning ckpts to ``logs/layout_phrase/version_N/``
(not into the run dir). The runner snapshots the set of version dirs before
training and after, then claims the new one as this run's checkpoint dir.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from beatsaber_automapper.research.leaderboard import append_row
from beatsaber_automapper.research.metrics import analyze_generated_zip
from beatsaber_automapper.research.spec_v7 import V7LayoutSpec

logger = logging.getLogger(__name__)


@dataclass
class V7RunResult:
    experiment_id: str
    status: str
    run_dir: Path
    metrics: dict[str, Any]


def _write_status(run_dir: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(extra)
    (run_dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _freeze_spec(run_dir: Path, spec: V7LayoutSpec) -> None:
    (run_dir / "spec.yaml").write_text(
        yaml.safe_dump(dataclasses.asdict(spec), sort_keys=False),
        encoding="utf-8",
    )


def _layout_version_dirs(project_root: Path) -> set[Path]:
    root = project_root / "logs" / "layout_phrase"
    if not root.exists():
        return set()
    return {p for p in root.iterdir() if p.is_dir() and p.name.startswith("version_")}


def _build_layout_train_cmd(spec: V7LayoutSpec, project_root: Path) -> list[str]:
    py = sys.executable
    cmd = [
        py, str(project_root / "scripts" / "train_layout.py"),
        "--max-epochs", str(spec.max_epochs),
        "--batch-size", str(spec.batch_size),
        "--lr", str(spec.learning_rate),
        "--d-model", str(spec.d_model),
        "--n-heads", str(spec.n_heads),
        "--n-enc-layers", str(spec.n_enc_layers),
        "--n-dec-layers", str(spec.n_dec_layers),
        "--dim-feedforward", str(spec.dim_feedforward),
        "--dropout", str(spec.dropout),
        "--max-layout-len", str(spec.max_layout_len),
        "--max-phrase-slots", str(spec.max_phrase_slots),
        "--x-role-weight", str(spec.x_role_weight),
        "--ctx-len", str(spec.ctx_len),
        "--max-song-phrases", str(spec.max_song_phrases),
        "--sched-sampling-start", str(spec.sched_sampling_start),
        "--sched-sampling-end", str(spec.sched_sampling_end),
        "--sched-sampling-epochs", str(spec.sched_sampling_epochs),
        "--patience", str(spec.patience),
        "--difficulties", *spec.difficulties,
    ]
    return cmd


def _build_generate_cmd(
    spec: V7LayoutSpec,
    layout_ckpt: Path,
    out_zip: Path,
    project_root: Path,
) -> list[str]:
    py = sys.executable
    return [
        py, str(project_root / "scripts" / "generate.py"),
        spec.test_audio,
        "--output", str(out_zip),
        "--difficulty", spec.test_difficulty,
        "--v7",
        "--beat-ckpt", spec.beat_ckpt,
        "--layout-ckpt", str(layout_ckpt),
    ]


def _build_eval_cmd(
    spec: V7LayoutSpec,
    map_zip: Path,
    out_json: Path,
    project_root: Path,
) -> list[str]:
    py = sys.executable
    return [
        py, str(project_root / "scripts" / "eval_alignment.py"),
        "--audio", spec.test_audio,
        "--map", str(map_zip),
        "--difficulty", spec.test_difficulty,
        "--tolerance-ms", str(spec.tolerance_ms),
        "--json", str(out_json),
    ]


def _run(cmd: list[str], log_path: Path, timeout_s: int, cwd: Path) -> tuple[int, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, text=True)
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


_CKPT_FILENAME_RE = re.compile(r"val_token_acc=(\d+\.\d+)")


def _best_layout_ckpt(version_dir: Path) -> tuple[Path | None, float | None]:
    """Highest val_token_acc among version_dir/checkpoints/*.ckpt (skip `last.ckpt`)."""
    ckpts = list((version_dir / "checkpoints").glob("*.ckpt"))
    scored: list[tuple[float, Path]] = []
    for p in ckpts:
        if p.name.startswith("last"):
            continue
        m = _CKPT_FILENAME_RE.search(p.name)
        if m:
            scored.append((float(m.group(1)), p))
    if not scored:
        return None, None
    score, path = max(scored, key=lambda t: t[0])
    return path, score


def _audio_duration_sec(path: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception as e:
        logger.warning("audio duration probe failed for %s: %s — using 60s fallback", path, e)
        return 60.0


def run_v7_layout_experiment(
    spec: V7LayoutSpec,
    *,
    project_root: Path,
    experiments_root: Path,
    leaderboard_path: Path,
) -> V7RunResult:
    eid = spec.experiment_id()
    run_dir = experiments_root / "runs" / eid
    run_dir.mkdir(parents=True, exist_ok=True)
    _freeze_spec(run_dir, spec)
    _write_status(run_dir, "queued", {"spec_name": spec.name})
    t_start = time.time()

    # ---- TRAIN ----
    _write_status(run_dir, "training")
    pre_versions = _layout_version_dirs(project_root)
    train_cmd = _build_layout_train_cmd(spec, project_root)
    train_log = run_dir / "train.log"
    logger.info("[%s] train: %s", eid, " ".join(train_cmd))
    rc, timed_out = _run(train_cmd, train_log, timeout_s=spec.max_wall_clock_min * 60, cwd=project_root)
    if rc != 0 and not timed_out:
        _write_status(run_dir, "failed", {"phase": "train", "rc": rc})
        return V7RunResult(eid, "failed_train", run_dir, {})

    post_versions = _layout_version_dirs(project_root)
    new_versions = sorted(post_versions - pre_versions, key=lambda p: p.stat().st_mtime)
    if not new_versions:
        _write_status(run_dir, "failed", {"phase": "no_new_version_dir"})
        return V7RunResult(eid, "failed_no_version", run_dir, {})
    version_dir = new_versions[-1]
    (run_dir / "version_dir.txt").write_text(str(version_dir), encoding="utf-8")

    best_ckpt, best_val = _best_layout_ckpt(version_dir)
    if best_ckpt is None:
        _write_status(run_dir, "failed", {"phase": "no_checkpoint", "version_dir": str(version_dir)})
        return V7RunResult(eid, "failed_no_ckpt", run_dir, {})
    logger.info("[%s] best ckpt %s (val_token_acc=%.3f)", eid, best_ckpt, best_val)

    # ---- GENERATE ----
    _write_status(run_dir, "generating")
    out_zip = run_dir / "generated" / "test_map.zip"
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    gen_cmd = _build_generate_cmd(spec, best_ckpt, out_zip, project_root)
    gen_log = run_dir / "generate.log"
    logger.info("[%s] generate: %s", eid, " ".join(gen_cmd))
    rc, _ = _run(gen_cmd, gen_log, timeout_s=15 * 60, cwd=project_root)
    if rc != 0 or not out_zip.exists():
        _write_status(run_dir, "failed", {"phase": "generate", "rc": rc})
        return V7RunResult(eid, "failed_generate", run_dir, {"best_val_token_acc": best_val})

    # ---- EVAL ALIGNMENT ----
    _write_status(run_dir, "evaluating")
    align_json = run_dir / "alignment.json"
    align_cmd = _build_eval_cmd(spec, out_zip, align_json, project_root)
    align_log = run_dir / "alignment.log"
    logger.info("[%s] eval_alignment: %s", eid, " ".join(align_cmd))
    rc, _ = _run(align_cmd, align_log, timeout_s=15 * 60, cwd=project_root)
    if rc != 0 or not align_json.exists():
        logger.warning("[%s] alignment eval failed rc=%s — leaderboard row will lack alignment metrics", eid, rc)
        align: dict[str, Any] = {}
    else:
        try:
            align = json.loads(align_json.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[%s] alignment json parse failed: %s", eid, e)
            align = {}

    # ---- PLAYABILITY (existing analyzer) ----
    duration = _audio_duration_sec(Path(spec.test_audio))
    try:
        playability = analyze_generated_zip(out_zip, duration).as_dict()
    except Exception as e:
        logger.warning("[%s] playability analyze failed: %s", eid, e)
        playability = {}

    # ---- LEADERBOARD ROW ----
    wall_clock = time.time() - t_start
    align_combined = align.get("overall_combined", {}) if isinstance(align, dict) else {}
    align_drums = align.get("overall_drums", {}) if isinstance(align, dict) else {}
    metrics = {
        "experiment_id": eid,
        "name": spec.name,
        "stage": "v7_layout",
        "seed": spec.seed,
        "wall_clock_sec": wall_clock,
        "version_dir": str(version_dir),
        "best_checkpoint": str(best_ckpt),
        "best_val_token_acc": best_val,
        "alignment_f1_combined": align_combined.get("f1"),
        "alignment_p_combined": align_combined.get("precision"),
        "alignment_r_combined": align_combined.get("recall"),
        "alignment_f1_drums": align_drums.get("f1"),
        "n_notes": playability.get("n_notes"),
        "notes_per_sec": playability.get("notes_per_sec"),
        "parity_rate": playability.get("parity_rate"),
        "collision_rate": playability.get("collision_rate"),
        "ctx_len": spec.ctx_len,
        "max_song_phrases": spec.max_song_phrases,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    append_row(leaderboard_path, metrics)
    _write_status(run_dir, "done", {"best_val_token_acc": best_val})
    logger.info("[%s] done val=%.3f align_f1=%s wall=%.1fs",
                eid, best_val or -1, metrics.get("alignment_f1_combined"), wall_clock)
    return V7RunResult(eid, "done", run_dir, metrics)
