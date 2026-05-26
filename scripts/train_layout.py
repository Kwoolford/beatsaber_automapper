"""V7-5b: Train phrase-level Stage 2 LayoutPhraseModel.

Each training sample is one phrase (16-beat / 64-slot window) of one
(song, difficulty). The decoder emits the full spatial token sequence for
all notes in the phrase, with cross-attention to phrase MERT.

Usage:
    python scripts/train_layout.py [options]

Options:
    --data-dir PATH         data/processed (default)
    --max-epochs N          30 (default)
    --batch-size N          32 (default — sequences are longer than per-onset)
    --lr FLOAT              1e-4 (default)
    --d-model N             384 (default)
    --n-enc-layers N        3
    --n-dec-layers N        4
    --difficulties D…       Expert ExpertPlus (default)
    --max-layout-len N      384 (default — typical phrase ≤ 250 tokens)
    --max-phrase-slots N    96  (default — observed phrase length is 64)
    --num-workers N         8
    --limit-val N           200
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

torch.set_float32_matmul_precision("high")  # use Tensor Cores on Ampere+

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",         default="data/processed")
    parser.add_argument("--max-epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--d-model",          type=int,   default=384)
    parser.add_argument("--n-heads",          type=int,   default=6)
    parser.add_argument("--n-enc-layers",     type=int,   default=3)
    parser.add_argument("--n-dec-layers",     type=int,   default=4)
    parser.add_argument("--dim-feedforward",  type=int,   default=1536)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--max-layout-len",   type=int,   default=384)
    parser.add_argument("--max-phrase-slots", type=int,   default=96)
    parser.add_argument("--difficulties",     nargs="+",  default=["Expert", "ExpertPlus"])
    parser.add_argument("--num-workers",      type=int,   default=8)
    parser.add_argument("--limit-val",        type=int,   default=200)
    parser.add_argument("--patience",         type=int,   default=8)
    parser.add_argument("--monitor",          default="val_token_acc")
    parser.add_argument("--max-samples",      type=int,   default=None,
                        help="Cap samples per epoch (random sampler).")
    parser.add_argument("--device",           default="auto")
    parser.add_argument("--x-role-weight",    type=float, default=2.0,
                        help="Loss weight on ROLE_X positions to push the weakest "
                             "role harder. 1.0 = uniform loss.")
    parser.add_argument("--max-song-phrases",  type=int,   default=150,
                        help="Max phrase fingerprints per song in the song-memory "
                             "encoder. 0 = disable song-memory (legacy behaviour).")
    parser.add_argument("--ctx-len",          type=int,   default=0,
                        help="Cross-phrase context prefix length. 0 = disabled. "
                             "N > 0 prepends last N tokens from the prior phrase as "
                             "read-only context (ROLE_CONTEXT, loss masked).")
    parser.add_argument("--sched-sampling-start",  type=float, default=0.0,
                        help="Scheduled-sampling probability at epoch 0 (0 = pure TF).")
    parser.add_argument("--sched-sampling-end",    type=float, default=0.0,
                        help="Scheduled-sampling probability at final ramp epoch.")
    parser.add_argument("--sched-sampling-epochs", type=int,   default=20,
                        help="Epochs over which to ramp scheduled sampling.")
    parser.add_argument("--resume-from",      default=None,
                        help="Checkpoint path to resume from (model + optimizer).")
    args = parser.parse_args()

    from torch.utils.data import DataLoader, RandomSampler
    import lightning
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger

    from beatsaber_automapper.data.layout_dataset import LayoutPhraseDataset
    from beatsaber_automapper.training.layout_module import LayoutPhraseLitModule

    data_dir = REPO_ROOT / args.data_dir

    log.info("Building datasets …")
    train_ds = LayoutPhraseDataset(
        data_dir=data_dir,
        split="train",
        difficulties=args.difficulties,
        exclude_categories=["noodle", "mapping_extensions"],
        max_layout_len=args.max_layout_len,
        max_phrase_slots=args.max_phrase_slots,
        ctx_len=args.ctx_len,
        max_song_phrases=args.max_song_phrases,
    )
    val_ds = LayoutPhraseDataset(
        data_dir=data_dir,
        split="val",
        difficulties=args.difficulties,
        exclude_categories=["noodle", "mapping_extensions"],
        max_layout_len=args.max_layout_len,
        max_phrase_slots=args.max_phrase_slots,
        ctx_len=args.ctx_len,
        max_song_phrases=args.max_song_phrases,
    )

    log.info("Train: %d phrases | Val: %d phrases", len(train_ds), len(val_ds))
    if len(train_ds) == 0:
        log.error("No training samples. Have you run scripts/preprocess_v7.py?")
        sys.exit(1)

    sampler = None
    shuffle = True
    if args.max_samples and args.max_samples < len(train_ds):
        sampler = RandomSampler(train_ds, replacement=False, num_samples=args.max_samples)
        shuffle = False

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    module = LayoutPhraseLitModule(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        dim_feedforward=args.dim_feedforward,
        max_layout_len=args.max_layout_len,
        max_phrase_slots=args.max_phrase_slots,
        learning_rate=args.lr,
        warmup_steps=1000,
        dropout=args.dropout,
        x_role_weight=args.x_role_weight,
        ctx_len=args.ctx_len,
        max_song_phrases=args.max_song_phrases,
        sched_sampling_start=args.sched_sampling_start,
        sched_sampling_end=args.sched_sampling_end,
        sched_sampling_epochs=args.sched_sampling_epochs,
    )

    tb_logger = TensorBoardLogger("logs", name="layout_phrase")

    callbacks = [
        ModelCheckpoint(
            monitor=args.monitor, mode="max", save_top_k=3, save_last=True,
            filename="layout-{epoch:02d}-{" + args.monitor + ":.3f}",
        ),
        EarlyStopping(monitor=args.monitor, mode="max",
                      patience=args.patience, verbose=True),
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

    log.info("Starting V7-5b phrase-level Stage 2 training …")
    trainer.fit(module, train_dl, val_dl, ckpt_path=args.resume_from)
    log.info("Best %s: %.3f", args.monitor, callbacks[0].best_model_score)
    log.info("Best checkpoint: %s", callbacks[0].best_model_path)


if __name__ == "__main__":
    main()
