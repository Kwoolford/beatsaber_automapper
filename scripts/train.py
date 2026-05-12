"""CLI: Train a model stage.

Usage:
    python scripts/train.py stage=onset
    python scripts/train.py stage=onset data_dir=data/processed max_epochs=50
"""

from __future__ import annotations

import gc
import logging
from datetime import UTC
from pathlib import Path

import hydra
import lightning
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig

from beatsaber_automapper.data.dataset import (
    OnsetDataset,
    SequenceDataset,
    SwingSequenceDataset,
    create_dataloader,
    swing_collate_fn,
)
from beatsaber_automapper.training.onset_module import OnsetLitModule
from beatsaber_automapper.training.seq_module import SequenceLitModule

logger = logging.getLogger(__name__)


class _GarbageCollectCallback(Callback):
    """Force garbage collection after each validation epoch to prevent memory creep."""

    def on_validation_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _HeartbeatCallback(Callback):
    """Write a heartbeat file after each epoch for remote monitoring.

    Writes JSON with timestamp, stage, epoch, metric, and status
    to $OUTPUT_DIR/heartbeat.json. Allows detecting hung/frozen training
    processes during multi-day unattended runs.
    """

    def __init__(self, output_dir: str, stage: str) -> None:
        self.heartbeat_path = Path(output_dir) / "heartbeat.json"
        self.stage = stage

    def on_train_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        import json
        from datetime import datetime

        metrics = trainer.callback_metrics
        # Pick the most relevant metric per stage
        if self.stage == "onset":
            metric_val = f"val_f1={metrics.get('val_f1', 'N/A')}"
        else:
            metric_val = f"val_loss={metrics.get('val_loss', 'N/A')}"

        heartbeat = {
            "timestamp": datetime.now(UTC).isoformat(),
            "stage": self.stage,
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "metric": metric_val,
            "status": "training",
        }
        try:
            self.heartbeat_path.write_text(json.dumps(heartbeat, indent=2))
        except OSError:
            pass  # Don't crash training over a heartbeat write failure


def _build_onset(cfg: DictConfig) -> tuple[lightning.LightningModule, lightning.Trainer]:
    """Build onset training components from Hydra config."""
    ac = cfg.model.audio_encoder
    oc = cfg.model.onset

    module = OnsetLitModule(
        n_mels=ac.n_mels,
        encoder_d_model=ac.d_model,
        encoder_nhead=ac.nhead,
        encoder_num_layers=ac.num_layers,
        encoder_dim_feedforward=ac.dim_feedforward,
        encoder_dropout=ac.dropout,
        onset_d_model=oc.d_model,
        onset_nhead=oc.nhead,
        onset_num_layers=oc.num_layers,
        onset_num_difficulties=oc.num_difficulties,
        onset_num_genres=oc.get("num_genres", 11),
        onset_dropout=oc.dropout,
        # TCN parameters
        tcn_channels=oc.get("tcn_channels", 128),
        tcn_num_blocks=oc.get("tcn_num_blocks", 6),
        tcn_kernel_size=oc.get("tcn_kernel_size", 3),
        # Conditioning dropout for CFG
        conditioning_dropout=oc.get("conditioning_dropout", 0.0),
        pos_weight=oc.get("pos_weight", 1.0),
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=cfg.scheduler.warmup_steps,
        onset_threshold=oc.onset_threshold,
        min_onset_distance=oc.get("min_onset_distance_frames", 5),
        use_gradient_checkpointing=oc.get("gradient_checkpointing", False),
        n_structure_features=ac.get("n_structure_features", 8),
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            filename="onset-{epoch:02d}-{val_f1:.3f}",
        ),
        EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=cfg.get("early_stopping_patience", 15),
        ),
        LearningRateMonitor(logging_interval="step"),
        _GarbageCollectCallback(),
        _HeartbeatCallback(cfg.output_dir, "onset"),
    ]

    # Logger — stage subdir keeps onset and sequence checkpoints separate
    if cfg.logger.name == "wandb":
        tb_logger = lightning.pytorch.loggers.WandbLogger(project=cfg.logger.project)
    else:
        tb_logger = lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=cfg.output_dir, name=f"{cfg.logger.project}/onset"
        )

    trainer = lightning.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=cfg.output_dir,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    return module, trainer


def _build_sequence(cfg: DictConfig) -> tuple[lightning.LightningModule, lightning.Trainer]:
    """Build sequence training components from Hydra config."""
    ac = cfg.model.audio_encoder
    sc = cfg.model.sequence

    module = SequenceLitModule(
        n_mels=ac.n_mels,
        encoder_d_model=ac.d_model,
        encoder_nhead=ac.nhead,
        encoder_num_layers=ac.num_layers,
        encoder_dim_feedforward=ac.dim_feedforward,
        encoder_dropout=ac.dropout,
        vocab_size=sc.vocab_size,
        seq_d_model=sc.d_model,
        seq_nhead=sc.nhead,
        seq_num_layers=sc.num_layers,
        seq_dim_feedforward=sc.dim_feedforward,
        seq_num_difficulties=sc.num_difficulties,
        seq_num_genres=sc.get("num_genres", 11),
        seq_num_mappers=sc.get("num_mappers", 0),
        seq_dropout=sc.dropout,
        # V6 conditioning
        bos_token_id=sc.get("bos_token_id", 1),
        conditioning_dropout=sc.get("conditioning_dropout", 0.0),
        # Training loss params
        label_smoothing=sc.get("label_smoothing", 0.1),
        rhythm_weight=sc.get("rhythm_weight", 3.0),
        eos_weight=sc.get("eos_weight", 1.0),
        # Per-stage LR/schedule overrides (fall back to global optimizer config)
        learning_rate=sc.get("learning_rate", cfg.optimizer.learning_rate),
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=sc.get("warmup_steps", cfg.scheduler.warmup_steps),
        lr_min_ratio=sc.get("lr_min_ratio", 0.01),
        token_dropout=sc.get("token_dropout", 0.0),
        freeze_encoder=sc.get("freeze_encoder", False),
        # V6 aux losses
        phrase_energy_alpha=sc.get("phrase_energy_alpha", 0.0),
        # Structure features
        n_structure_features=ac.get("n_structure_features", 8),
        # Legacy V5 compat
        prev_context_k=sc.get("prev_context_k", 0),
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            filename="sequence-{epoch:02d}-{val_loss:.3f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.get("early_stopping_patience", 15),
        ),
        LearningRateMonitor(logging_interval="step"),
        _GarbageCollectCallback(),
        _HeartbeatCallback(cfg.output_dir, "sequence"),
    ]

    # Logger — stage subdir keeps onset and sequence checkpoints separate
    if cfg.logger.name == "wandb":
        tb_logger = lightning.pytorch.loggers.WandbLogger(project=cfg.logger.project)
    else:
        tb_logger = lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=cfg.output_dir, name=f"{cfg.logger.project}/sequence"
        )

    trainer = lightning.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=cfg.output_dir,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=cfg.get("limit_val_batches", 1.0),
    )

    return module, trainer


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for the training CLI."""
    import torch

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    # Suppress "triton not found" spam that fires once per DataLoader worker process.
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

    # Gaming mode: run at below-normal CPU priority so the OS schedules games first.
    # DataLoader workers inherit this priority, preventing rhythm-game micro-stutters.
    if cfg.get("low_priority", False):
        import sys as _sys

        if _sys.platform == "win32":
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, 0x4000)  # BELOW_NORMAL
            logger.info("Process priority set to BELOW_NORMAL (gaming mode)")

    seed = cfg.get("seed", None)
    if seed is not None:
        lightning.seed_everything(int(seed), workers=True)
        logger.info("Seeded Lightning with seed=%s", seed)

    stage = cfg.stage
    ckpt_path = cfg.get("ckpt_path", None)

    # Cohort / bucket routing — overrides data_dir when set.
    cohort = cfg.get("cohort", None)
    bucket = cfg.get("bucket", None)
    if cohort and bucket:
        raise ValueError("cohort and bucket are mutually exclusive")
    if cohort:
        cfg.data_dir = str(Path(cfg.cohort_root) / cohort / "processed")
        logger.info("Cohort mode: cohort=%s data_dir=%s", cohort, cfg.data_dir)
    elif bucket:
        cfg.data_dir = str(Path(cfg.cohort_root) / "_buckets" / bucket)
        logger.info("Bucket mode: bucket=%s data_dir=%s", bucket, cfg.data_dir)
    logger.info("Training stage: %s", stage)
    if ckpt_path:
        logger.info("Resuming from checkpoint: %s", ckpt_path)

    if stage == "onset":
        module, trainer = _build_onset(cfg)

        data_dir = Path(cfg.data_dir)
        ds_cfg = cfg.data.dataset
        window_size = cfg.model.onset.get("window_size", 1024)
        hop = cfg.model.onset.get("hop", 128)

        # Filter to specific difficulties if configured (e.g. Expert/ExpertPlus only)
        onset_diffs = cfg.model.onset.get("difficulties", None)
        if onset_diffs is not None:
            onset_diffs = list(onset_diffs)
            logger.info("Filtering to difficulties: %s", onset_diffs)

        train_ds = OnsetDataset(
            data_dir, split="train", window_size=window_size, hop=hop,
            difficulties=onset_diffs,
        )
        val_ds = OnsetDataset(
            data_dir, split="val", window_size=window_size, hop=hop,
            difficulties=onset_diffs,
        )

        logger.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))

        train_dl = create_dataloader(
            train_ds,
            batch_size=ds_cfg.batch_size,
            shuffle=True,
            num_workers=ds_cfg.num_workers,
            pin_memory=ds_cfg.pin_memory,
        )
        val_dl = create_dataloader(
            val_ds,
            batch_size=ds_cfg.batch_size,
            shuffle=False,
            num_workers=ds_cfg.num_workers,
            pin_memory=ds_cfg.pin_memory,
        )

        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path)
    elif stage == "sequence":
        module, trainer = _build_sequence(cfg)

        data_dir = Path(cfg.data_dir)
        ds_cfg = cfg.data.dataset
        sc = cfg.model.sequence

        seq_diffs = sc.get("difficulties", None)
        if seq_diffs is not None:
            seq_diffs = list(seq_diffs)
            logger.info("Filtering to difficulties: %s", seq_diffs)

        mirror_augment = sc.get("mirror_augment", False)
        # dataset_format: "swing" = V6 SwingSequenceDataset; anything else = V5 SequenceDataset
        dataset_format = cfg.get("dataset_format", "v5")
        mapper_id = cfg.get("mapper_id", 0)

        if dataset_format == "swing":
            logger.info("Using V6 SwingSequenceDataset (dataset_format=swing)")
            train_ds = SwingSequenceDataset(
                data_dir,
                split="train",
                window_events=sc.get("window_events", 128),
                window_hop=sc.get("window_hop", 64),
                max_swing_len=sc.get("max_swing_len", 512),
                context_frames=sc.get("context_frames", 256),
                phrase_frames=sc.get("phrase_frames", 1024),
                difficulties=seq_diffs,
                mapper_id=mapper_id,
                mirror_augment=mirror_augment,
            )
            val_ds = SwingSequenceDataset(
                data_dir,
                split="val",
                window_events=sc.get("window_events", 128),
                window_hop=sc.get("window_hop", 64),
                max_swing_len=sc.get("max_swing_len", 512),
                context_frames=sc.get("context_frames", 256),
                phrase_frames=sc.get("phrase_frames", 1024),
                difficulties=seq_diffs,
                mapper_id=mapper_id,
                mirror_augment=False,
            )
        else:
            prev_context_k = sc.get("prev_context_k", 0)
            train_ds = SequenceDataset(
                data_dir,
                split="train",
                context_frames=sc.get("context_frames", 128),
                max_token_len=sc.get("max_seq_length", 64),
                difficulties=seq_diffs,
                prev_context_k=prev_context_k,
                mirror_augment=mirror_augment,
            )
            val_ds = SequenceDataset(
                data_dir,
                split="val",
                context_frames=sc.get("context_frames", 128),
                max_token_len=sc.get("max_seq_length", 64),
                difficulties=seq_diffs,
                prev_context_k=prev_context_k,
                mirror_augment=False,
            )

        max_samples = cfg.get("max_samples_per_epoch", 500_000)
        logger.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))
        if max_samples is not None and max_samples < len(train_ds):
            logger.info("Epoch subsampling: %d/%d per epoch (full coverage in ~%d epochs)",
                        max_samples, len(train_ds), len(train_ds) // max_samples)
        elif max_samples is None:
            logger.info("Epoch subsampling disabled (max_samples_per_epoch=null) — full epochs")

        collate = swing_collate_fn if dataset_format == "swing" else None
        train_dl = create_dataloader(
            train_ds,
            batch_size=ds_cfg.batch_size,
            shuffle=True,
            num_workers=ds_cfg.num_workers,
            pin_memory=ds_cfg.pin_memory,
            max_samples_per_epoch=max_samples,
            collate_fn=collate,
        )
        val_dl = create_dataloader(
            val_ds,
            batch_size=ds_cfg.batch_size,
            shuffle=False,
            num_workers=ds_cfg.num_workers,
            pin_memory=ds_cfg.pin_memory,
            collate_fn=collate,
        )

        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"Unknown stage: {stage!r}. Must be one of: onset, sequence")


if __name__ == "__main__":
    main()
