"""V7-5: Train Stage 2 LayoutModel.

Trains the spatial token generator on per-onset conditioning from preprocessed
.pt files. Requires V7 preprocessing (drum_beat_features, mix_beat_features,
phrase_fingerprints) to be present in the .pt files.

Usage:
    python scripts/train_layout.py [options]

Options:
    --data-dir PATH       data/processed (default)
    --max-epochs N        30 (default)
    --batch-size N        128 (default)
    --lr FLOAT            1e-4 (default)
    --d-model N           512 (default)
    --n-layers N          4 (default)
    --difficulties DIFF…  Expert ExpertPlus (default)
    --num-workers N       8 (default)
    --limit-val N         200 (default)
    --max-samples N       cap training samples per epoch (default: unlimited)
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",      default="data/processed")
    parser.add_argument("--max-epochs",    type=int,   default=30)
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--d-model",       type=int,   default=512)
    parser.add_argument("--n-layers",      type=int,   default=4)
    parser.add_argument("--n-heads",       type=int,   default=8)
    parser.add_argument("--num-workers",   type=int,   default=8)
    parser.add_argument("--limit-val",     type=int,   default=200)
    parser.add_argument("--max-samples",   type=int,   default=None)
    parser.add_argument("--difficulties",  nargs="+",  default=["Expert", "ExpertPlus"])
    parser.add_argument("--device",        default="auto")
    args = parser.parse_args()

    from torch.utils.data import DataLoader, RandomSampler
    import lightning
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger

    from beatsaber_automapper.data.layout_dataset import LayoutDataset
    from beatsaber_automapper.training.layout_module import LayoutLitModule

    data_dir = REPO_ROOT / args.data_dir

    log.info("Building datasets …")
    train_ds = LayoutDataset(
        data_dir=data_dir,
        split="train",
        difficulties=args.difficulties,
        exclude_categories=["noodle", "mapping_extensions"],
    )
    val_ds = LayoutDataset(
        data_dir=data_dir,
        split="val",
        difficulties=args.difficulties,
        exclude_categories=["noodle", "mapping_extensions"],
    )

    log.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))
    if len(train_ds) == 0:
        log.error("No training samples. Have you run scripts/preprocess_v7.py?")
        sys.exit(1)

    sampler = None
    shuffle = True
    if args.max_samples and args.max_samples < len(train_ds):
        sampler = RandomSampler(train_ds, replacement=False, num_samples=args.max_samples)
        shuffle = False

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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

    module = LayoutLitModule(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        learning_rate=args.lr,
        warmup_steps=1000,
    )

    tb_logger = TensorBoardLogger("logs", name="layout_model")

    callbacks = [
        ModelCheckpoint(
            monitor="val_token_acc",
            mode="max",
            save_top_k=3,
            save_last=True,
            filename="layout-{epoch:02d}-acc={val_token_acc:.3f}",
        ),
        EarlyStopping(monitor="val_token_acc", mode="max", patience=8, verbose=True),
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

    log.info("Starting Stage 2 training …")
    trainer.fit(module, train_dl, val_dl)
    log.info("Best val_token_acc: %.3f", callbacks[0].best_model_score)
    log.info("Best checkpoint: %s", callbacks[0].best_model_path)


if __name__ == "__main__":
    main()
