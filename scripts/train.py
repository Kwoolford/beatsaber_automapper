"""CLI: Train a model stage.

Usage:
    python scripts/train.py stage=onset
    python scripts/train.py stage=onset data_dir=data/processed max_epochs=50
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig

from beatsaber_automapper.data.dataset import (
    LightingDataset,
    OnsetDataset,
    SequenceDataset,
    create_dataloader,
)
from beatsaber_automapper.training.light_module import LightingLitModule
from beatsaber_automapper.training.onset_module import OnsetLitModule
from beatsaber_automapper.training.seq_module import SequenceLitModule

logger = logging.getLogger(__name__)


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
        pos_weight=oc.get("pos_weight", 5.0),
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=cfg.scheduler.warmup_steps,
        onset_threshold=oc.onset_threshold,
        min_onset_distance=oc.get("min_onset_distance_frames", 5),
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
            patience=10,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    if cfg.logger.name == "wandb":
        tb_logger = lightning.pytorch.loggers.WandbLogger(project=cfg.logger.project)
    else:
        tb_logger = lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=cfg.output_dir, name=cfg.logger.project
        )

    trainer = lightning.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=cfg.output_dir,
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
        seq_dropout=sc.dropout,
        label_smoothing=sc.get("label_smoothing", 0.1),
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=cfg.scheduler.warmup_steps,
        freeze_encoder=sc.get("freeze_encoder", False),
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
            patience=10,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    if cfg.logger.name == "wandb":
        tb_logger = lightning.pytorch.loggers.WandbLogger(project=cfg.logger.project)
    else:
        tb_logger = lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=cfg.output_dir, name=cfg.logger.project
        )

    trainer = lightning.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=cfg.output_dir,
    )

    return module, trainer


def _build_lighting(cfg: DictConfig) -> tuple[lightning.LightningModule, lightning.Trainer]:
    """Build lighting training components from Hydra config."""
    ac = cfg.model.audio_encoder
    lc = cfg.model.lighting

    module = LightingLitModule(
        n_mels=ac.n_mels,
        encoder_d_model=ac.d_model,
        encoder_nhead=ac.nhead,
        encoder_num_layers=ac.num_layers,
        encoder_dim_feedforward=ac.dim_feedforward,
        encoder_dropout=ac.dropout,
        light_vocab_size=lc.get("light_vocab_size", 35),
        note_vocab_size=lc.get("note_vocab_size", 167),
        light_d_model=lc.d_model,
        light_nhead=lc.nhead,
        light_num_layers=lc.num_layers,
        light_dim_feedforward=lc.dim_feedforward,
        light_num_genres=lc.get("num_genres", 11),
        light_dropout=lc.dropout,
        label_smoothing=lc.get("label_smoothing", 0.1),
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=cfg.scheduler.warmup_steps,
        freeze_encoder=lc.get("freeze_encoder", False),
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            filename="lighting-{epoch:02d}-{val_loss:.3f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if cfg.logger.name == "wandb":
        tb_logger = lightning.pytorch.loggers.WandbLogger(project=cfg.logger.project)
    else:
        tb_logger = lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=cfg.output_dir, name=cfg.logger.project
        )

    trainer = lightning.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=cfg.output_dir,
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

    stage = cfg.stage
    logger.info("Training stage: %s", stage)

    if stage == "onset":
        module, trainer = _build_onset(cfg)

        data_dir = Path(cfg.data_dir)
        ds_cfg = cfg.data.dataset
        window_size = cfg.model.onset.get("window_size", 256)
        hop = cfg.model.onset.get("hop", 128)

        train_ds = OnsetDataset(data_dir, split="train", window_size=window_size, hop=hop)
        val_ds = OnsetDataset(data_dir, split="val", window_size=window_size, hop=hop)

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

        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    elif stage == "sequence":
        module, trainer = _build_sequence(cfg)

        data_dir = Path(cfg.data_dir)
        ds_cfg = cfg.data.dataset
        sc = cfg.model.sequence

        train_ds = SequenceDataset(
            data_dir,
            split="train",
            context_frames=sc.get("context_frames", 128),
            max_token_len=sc.get("max_seq_length", 64),
        )
        val_ds = SequenceDataset(
            data_dir,
            split="val",
            context_frames=sc.get("context_frames", 128),
            max_token_len=sc.get("max_seq_length", 64),
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

        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    elif stage == "lighting":
        module, trainer = _build_lighting(cfg)

        data_dir = Path(cfg.data_dir)
        ds_cfg = cfg.data.dataset
        lc = cfg.model.lighting

        train_ds = LightingDataset(
            data_dir,
            split="train",
            context_frames=lc.get("context_frames", 128),
            max_note_len=lc.get("max_note_len", 64),
            max_light_len=lc.get("max_light_len", 32),
        )
        val_ds = LightingDataset(
            data_dir,
            split="val",
            context_frames=lc.get("context_frames", 128),
            max_note_len=lc.get("max_note_len", 64),
            max_light_len=lc.get("max_light_len", 32),
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

        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be one of: onset, sequence, lighting")


if __name__ == "__main__":
    main()
