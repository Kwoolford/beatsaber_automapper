"""V7-3: Train Stage 1 BeatClassifier.

Trains on drum-MERT features from preprocessed .pt files (requires
scripts/preprocess_v7.py to have been run first).

Usage:
    python scripts/train_beats.py [overrides]

Key overrides (Hydra-style):
    max_epochs=20
    data.dataset.batch_size=64
    model.d_model=256
    model.n_layers=2
    model.pos_weight=6.0
    learning_rate=3e-4
    data_dir=data/processed
    difficulties=[Expert,ExpertPlus]
    limit_val_batches=50
"""

from __future__ import annotations

import logging
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train Stage 1 BeatClassifier")
    parser.add_argument("--data-dir",       default="data/processed")
    parser.add_argument("--max-epochs",     type=int,   default=20)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--d-model",        type=int,   default=256)
    parser.add_argument("--n-layers",       type=int,   default=2)
    parser.add_argument("--n-heads",        type=int,   default=4)
    parser.add_argument("--pos-weight",     type=float, default=3.6,
                        help="BCE positive class weight (default 3.6 ≈ 78.2/21.8)")
    parser.add_argument("--mix-dim",        type=int,   default=768,
                        help="Mix-stem MERT dim. 0 disables mix features.")
    parser.add_argument("--window-size",    type=int,   default=128)
    parser.add_argument("--num-workers",    type=int,   default=8)
    parser.add_argument("--limit-val",      type=int,   default=200)
    parser.add_argument("--patience",       type=int,   default=8,
                        help="EarlyStopping patience on val_f1_avg")
    parser.add_argument("--difficulties",   nargs="+",  default=["Expert", "ExpertPlus"])
    parser.add_argument("--tolerance-slots", type=int,  default=1,
                        help="±slot match window for val_f1_avg_tol (1 ≈ ±125 ms @ 120 BPM)")
    parser.add_argument("--monitor",        default="val_f1_avg_tol",
                        help="Metric to monitor for checkpointing/early-stopping "
                             "(default: tolerance F1)")
    parser.add_argument("--device",         default="auto")
    args = parser.parse_args()

    from torch.utils.data import DataLoader, RandomSampler
    import lightning
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger

    from beatsaber_automapper.data.beat_dataset import BeatDataset
    from beatsaber_automapper.training.beat_module import BeatLitModule

    data_dir = REPO_ROOT / args.data_dir

    log.info("Building datasets …")
    train_ds = BeatDataset(
        data_dir=data_dir,
        split="train",
        window_size=args.window_size,
        hop=args.window_size // 2,
        difficulties=args.difficulties,
        exclude_categories=["noodle", "mapping_extensions"],
    )
    val_ds = BeatDataset(
        data_dir=data_dir,
        split="val",
        window_size=args.window_size,
        hop=args.window_size,
        difficulties=args.difficulties,
        exclude_categories=["noodle", "mapping_extensions"],
    )

    log.info("Train: %d windows | Val: %d windows", len(train_ds), len(val_ds))

    if len(train_ds) == 0:
        log.error("No training samples found. Have you run scripts/preprocess_v7.py?")
        sys.exit(1)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    module = BeatLitModule(
        d_model=args.d_model,
        mix_dim=args.mix_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        learning_rate=args.lr,
        pos_weight=args.pos_weight,
        warmup_steps=500,
        max_len=args.window_size + 32,
        tolerance_slots=args.tolerance_slots,
    )

    tb_logger = TensorBoardLogger("logs", name="beat_classifier")

    callbacks = [
        ModelCheckpoint(
            monitor=args.monitor,
            mode="max",
            save_top_k=3,
            save_last=True,
            filename="beat-{epoch:02d}-{" + args.monitor + ":.3f}",
        ),
        EarlyStopping(monitor=args.monitor, mode="max", patience=args.patience, verbose=True),
    ]

    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto" if args.device == "auto" else args.device,
        precision="bf16-mixed",
        logger=tb_logger,
        callbacks=callbacks,
        limit_val_batches=args.limit_val,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )

    log.info("Starting Stage 1 training …")
    trainer.fit(module, train_dl, val_dl)

    best = callbacks[0].best_model_path
    log.info("Best checkpoint: %s", best)
    log.info("Best %s: %.3f", args.monitor, callbacks[0].best_model_score)


if __name__ == "__main__":
    main()
