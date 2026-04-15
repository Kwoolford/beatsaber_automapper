"""Assemble bucket "virtual cohorts" from member-mapper cohorts.

A bucket is the union of its member mappers' maps, with unified splits and
a unified frame_index. Because our Dataset classes only scan ``data_dir`` for
``*.pt``, we materialize buckets as a directory of **hard links** to member
cohort .pt files. This keeps zero disk overhead (NTFS hard links on the same
volume) while looking identical to a normal cohort to downstream code.

Usage:
    python scripts/build_bucket_manifests.py                    # all buckets
    python scripts/build_bucket_manifests.py --bucket anime_jpop_flow
    python scripts/build_bucket_manifests.py --force            # rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beatsaber_automapper.data.tokenizer import genre_from_tags  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_bucket_manifests")


def _slugify(name: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()


def _deterministic_split(
    song_ids: list[str], train_ratio: float = 0.85, val_ratio: float = 0.10
) -> dict[str, list[str]]:
    """Hash-based deterministic split identical in spirit to preprocess.py."""
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for sid in song_ids:
        h = int(hashlib.md5(sid.encode("utf-8")).hexdigest(), 16) % 10_000 / 10_000
        if h < train_ratio:
            train.append(sid)
        elif h < train_ratio + val_ratio:
            val.append(sid)
        else:
            test.append(sid)
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def build_bucket(
    bucket_id: str,
    member_slugs: list[str],
    cohorts_root: Path,
    *,
    force: bool,
) -> dict[str, int]:
    """Build one bucket dir with hard-linked .pt files + combined splits/index."""
    bucket_dir = cohorts_root / "_buckets" / bucket_id
    bucket_dir.mkdir(parents=True, exist_ok=True)

    all_song_ids: list[str] = []
    combined_index: dict[str, dict] = {}
    linked = 0
    skipped = 0
    missing_cohorts: list[str] = []

    for slug in member_slugs:
        member_proc = cohorts_root / slug / "processed"
        if not member_proc.exists():
            missing_cohorts.append(slug)
            continue

        member_idx_path = member_proc / "frame_index.json"
        if member_idx_path.exists():
            member_idx = json.loads(member_idx_path.read_text(encoding="utf-8"))
        else:
            member_idx = {}

        for pt in member_proc.glob("*.pt"):
            dest = bucket_dir / pt.name
            if dest.exists():
                if force:
                    dest.unlink()
                else:
                    skipped += 1
                    all_song_ids.append(pt.stem)
                    if pt.stem in member_idx:
                        combined_index[pt.stem] = member_idx[pt.stem]
                    continue
            try:
                os.link(pt, dest)
                linked += 1
            except OSError as e:
                logger.warning("hard-link failed %s → %s: %s", pt, dest, e)
                continue
            all_song_ids.append(pt.stem)
            if pt.stem in member_idx:
                combined_index[pt.stem] = member_idx[pt.stem]

    if missing_cohorts:
        logger.warning(
            "[%s] missing cohorts (skipped): %s",
            bucket_id,
            ", ".join(missing_cohorts),
        )

    # Splits
    splits = _deterministic_split(all_song_ids)
    (bucket_dir / "splits.json").write_text(json.dumps(splits), encoding="utf-8")

    # Frame index
    if combined_index:
        (bucket_dir / "frame_index.json").write_text(
            json.dumps(combined_index), encoding="utf-8"
        )

    # Bucket metadata for reference
    (bucket_dir / "_bucket_info.json").write_text(
        json.dumps(
            {
                "bucket_id": bucket_id,
                "members": member_slugs,
                "missing": missing_cohorts,
                "n_maps": len(all_song_ids),
                "n_splits": {k: len(v) for k, v in splits.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "[%s] %d maps (linked=%d, skipped_existing=%d), splits %s",
        bucket_id,
        len(all_song_ids),
        linked,
        skipped,
        {k: len(v) for k, v in splits.items()},
    )
    return {"n_maps": len(all_song_ids), "linked": linked, "skipped": skipped}


def main() -> int:
    # genre_from_tags imported for future metadata enrichment; silence linters
    _ = genre_from_tags

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mappers-json",
        type=Path,
        default=Path("data/reference/mappers.json"),
    )
    p.add_argument("--cohorts-root", type=Path, default=Path("data/cohorts"))
    p.add_argument("--bucket", help="Build only this bucket_id")
    p.add_argument("--force", action="store_true", help="Relink existing files")
    args = p.parse_args()

    data = json.loads(args.mappers_json.read_text(encoding="utf-8"))

    # Map display_name → slug (mirror download_cohorts.py slug convention)
    slug_by_name = {m["display_name"]: _slugify(m["display_name"]) for m in data["mappers"]}

    buckets = data["style_buckets"]
    if args.bucket:
        buckets = [b for b in buckets if b["bucket_id"] == args.bucket]
        if not buckets:
            logger.error("bucket %r not found", args.bucket)
            return 2

    for b in buckets:
        member_slugs = [slug_by_name[name] for name in b["mapper_ids"] if name in slug_by_name]
        build_bucket(b["bucket_id"], member_slugs, args.cohorts_root, force=args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
