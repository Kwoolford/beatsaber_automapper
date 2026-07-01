#!/usr/bin/env python3
"""Fine-tune the v10 layout model with the anti-collapse entropy bonus (2026-06-30).

Loads v10 WEIGHTS (not full Lightning state — avoids the saturated early-stop
baggage), runs a few short epochs with LAYOUT_ENT_REG>0 (set via env, read in
layout_module._forward_batch) to flatten the over-confident X/Y position logits
that cause the row0×{col0,col2} mode-collapse. Saves every epoch so we can eval
the row_conc/density tradeoff per epoch and pick the best.

Usage:
  LAYOUT_ENT_REG=0.5 python scripts/finetune_layout_diversity.py --epochs 4 --tag ent0.5
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

V10_CKPT = REPO / "logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"
# v10 architecture/data config (from logs/layout_phrase/version_10/hparams.yaml)
CTX_LEN = 16
MAX_SONG_PHRASES = 150
MAX_LAYOUT_LEN = 384
MAX_PHRASE_SLOTS = 96
DIFFS = ["Expert", "ExpertPlus"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-val", type=int, default=200)
    ap.add_argument("--tag", default="entreg")
    ap.add_argument("--data-dir", default="data/processed")
    args = ap.parse_args()

    print(f"[finetune] LAYOUT_ENT_REG={os.environ.get('LAYOUT_ENT_REG', '0')}  "
          f"epochs={args.epochs} lr={args.lr} tag={args.tag}")

    from torch.utils.data import DataLoader
    import lightning
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from beatsaber_automapper.data.layout_dataset import LayoutPhraseDataset
    from beatsaber_automapper.training.layout_module import LayoutPhraseLitModule

    data_dir = REPO / args.data_dir
    ds_kw = dict(
        data_dir=data_dir, difficulties=DIFFS,
        exclude_categories=["noodle", "mapping_extensions"],
        max_layout_len=MAX_LAYOUT_LEN, max_phrase_slots=MAX_PHRASE_SLOTS,
        ctx_len=CTX_LEN, max_song_phrases=MAX_SONG_PHRASES,
    )
    train_ds = LayoutPhraseDataset(split="train", **ds_kw)
    val_ds = LayoutPhraseDataset(split="val", **ds_kw)
    print(f"[finetune] train={len(train_ds)} val={len(val_ds)} phrases")

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 pin_memory=True, persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kw)

    # weights-only load; override LR + short warmup for a gentle fine-tune
    module = LayoutPhraseLitModule.load_from_checkpoint(
        str(V10_CKPT), learning_rate=args.lr, warmup_steps=50,
    )

    ckpt_cb = ModelCheckpoint(
        save_top_k=-1, every_n_epochs=1,
        filename=f"layoutft_{args.tag}" + "-{epoch:02d}-{val_token_acc:.3f}",
    )
    trainer = lightning.Trainer(
        max_epochs=args.epochs, accelerator="auto", precision="bf16-mixed",
        logger=TensorBoardLogger("logs", name=f"layout_ft_{args.tag}"),
        callbacks=[ckpt_cb], limit_val_batches=args.limit_val,
        log_every_n_steps=20, gradient_clip_val=1.0,
    )
    trainer.fit(module, train_dl, val_dl)
    print(f"[finetune] done. checkpoints in {trainer.logger.log_dir}/../checkpoints "
          f"(or the ModelCheckpoint dir). last val_token_acc logged above.")
    print(f"[finetune] ckpt dir: {ckpt_cb.dirpath}")


if __name__ == "__main__":
    main()
