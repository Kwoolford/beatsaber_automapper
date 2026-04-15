"""Download per-mapper cohort catalogs from BeatSaver.

Reads ``data/reference/mappers.json`` and downloads each mapper's full uploaded
catalog to ``data/cohorts/{mapper_slug}/raw/{map_id}.zip``. Maintains a
per-cohort ``manifest.json`` with map metadata (hash, uploaded date, diffs,
tags, score, category).

Usage:
    python scripts/download_cohorts.py                       # all mappers
    python scripts/download_cohorts.py --mapper Joetastic    # one mapper
    python scripts/download_cohorts.py --min-year 2019       # override filter
    python scripts/download_cohorts.py --dry-run             # count only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from tqdm import tqdm

# Reuse existing helpers where possible.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from beatsaber_automapper.data.download import (  # noqa: E402
    _classify_map_api,
    _classify_map_zip,
    _download_zip,
    _extract_genre_tags,
    _get_download_hash,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_cohorts")

API_BASE = "https://api.beatsaver.com"
UA = "beatsaber-automapper/0.1.0 (https://github.com/Kwoolford/beatsaber_automapper)"
HEADERS = {"User-Agent": UA}

EXPERT_DIFFS = {"Expert", "ExpertPlus"}

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def slugify(name: str) -> str:
    """Lowercase, strip non-alphanumerics → folder-safe."""
    return _SLUG_RE.sub("_", name).strip("_").lower()


def fetch_uploader_page(
    session: requests.Session, user_id: str, page: int, rate_limit: float
) -> list[dict]:
    """GET /maps/uploader/{user_id}/{page}. Returns list of map docs."""
    time.sleep(rate_limit)
    url = f"{API_BASE}/maps/uploader/{user_id}/{page}"
    try:
        r = session.get(url, timeout=30)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 10))
            logger.warning("429 rate limit — sleeping %ds", wait)
            time.sleep(wait)
            r = session.get(url, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json().get("docs", [])
    except requests.RequestException as e:
        logger.warning("uploader page %d failed: %s", page, e)
        return []


def passes_cohort_filters(m: dict, *, min_year: int, exclude_ai: bool) -> bool:
    """Permissive filter for cohort downloads — style > quality.

    - Must have at least one Standard characteristic diff
    - Must have at least one Expert or ExpertPlus diff
    - Exclude AI-declared maps (don't train AI on AI)
    - Uploaded >= min_year
    - No NPS cap: some mappers (helloimdaan) are legit speedmappers
    - No rating cap: their catalog defines their style, not community votes
    """
    if exclude_ai:
        if m.get("automapper"):
            return False
        ai = m.get("declaredAi", "None")
        if ai and ai != "None":
            return False

    up = m.get("uploaded", "")
    if up:
        try:
            if int(up[:4]) < min_year:
                return False
        except (ValueError, IndexError):
            pass

    standard_diffs = [
        d
        for v in m.get("versions", [])
        for d in v.get("diffs", [])
        if d.get("characteristic") == "Standard"
    ]
    if not standard_diffs:
        return False

    expert_present = any(d.get("difficulty") in EXPERT_DIFFS for d in standard_diffs)
    if not expert_present:
        return False

    return True


def save_manifest(manifest: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def download_cohort(
    mapper: dict,
    cohorts_root: Path,
    *,
    min_year: int,
    rate_limit: float,
    dry_run: bool,
    exclude_ai: bool,
) -> dict:
    """Download one mapper's full catalog. Returns stats dict."""
    name = mapper["display_name"]
    bid = mapper["beatsaver_id"]
    slug = slugify(name)
    cohort_dir = cohorts_root / slug
    raw_dir = cohort_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cohort_dir / "manifest.json"

    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    session = requests.Session()
    session.headers.update(HEADERS)

    stats = {
        "mapper": name,
        "slug": slug,
        "beatsaver_id": bid,
        "pages_fetched": 0,
        "maps_seen": 0,
        "maps_passed": 0,
        "maps_skipped_existing": 0,
        "maps_downloaded": 0,
        "maps_failed": 0,
    }

    page = 0
    saves_since = 0

    # tqdm placeholder (unknown total)
    pbar = tqdm(desc=f"{name:22s}", unit="map", leave=True)

    while True:
        docs = fetch_uploader_page(session, bid, page, rate_limit)
        if not docs:
            break
        stats["pages_fetched"] += 1
        stats["maps_seen"] += len(docs)

        for m in docs:
            map_id = m.get("id")
            if not map_id:
                continue
            if not passes_cohort_filters(m, min_year=min_year, exclude_ai=exclude_ai):
                continue
            stats["maps_passed"] += 1

            if map_id in manifest and (raw_dir / f"{map_id}.zip").exists():
                stats["maps_skipped_existing"] += 1
                pbar.update(1)
                continue

            if dry_run:
                stats["maps_downloaded"] += 1  # would download
                pbar.update(1)
                continue

            dl_hash = _get_download_hash(m)
            if not dl_hash:
                stats["maps_failed"] += 1
                continue

            dest = raw_dir / f"{map_id}.zip"
            try:
                _download_zip(session, dl_hash, dest, rate_limit)
            except Exception as e:
                logger.warning("download %s failed: %s", map_id, e)
                stats["maps_failed"] += 1
                continue

            # Classify + record metadata
            try:
                category, reqs, suggs = _classify_map_zip(dest)
            except Exception:
                category = _classify_map_api(m)
                reqs, suggs = [], []

            versions = m.get("versions", [])
            latest = versions[-1] if versions else {}
            diff_list = [
                {
                    "characteristic": d.get("characteristic"),
                    "difficulty": d.get("difficulty"),
                    "nps": d.get("nps"),
                    "nb": d.get("notes"),
                }
                for d in latest.get("diffs", [])
            ]

            manifest[map_id] = {
                "hash": dl_hash,
                "name": m.get("name", ""),
                "uploaded": m.get("uploaded", ""),
                "uploader": (m.get("uploader") or {}).get("name", ""),
                "score": (m.get("stats") or {}).get("score"),
                "ranked": m.get("ranked", False),
                "tags": _extract_genre_tags(m),
                "category": category,
                "requirements": reqs,
                "suggestions": suggs,
                "diffs": diff_list,
                "downloaded_at": datetime.now(UTC).isoformat(),
            }
            stats["maps_downloaded"] += 1
            pbar.update(1)
            saves_since += 1
            if saves_since >= 10:
                save_manifest(manifest, manifest_path)
                saves_since = 0

        page += 1

    pbar.close()
    if saves_since > 0 or not manifest_path.exists():
        save_manifest(manifest, manifest_path)

    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mappers-json",
        type=Path,
        default=Path("data/reference/mappers.json"),
    )
    p.add_argument(
        "--cohorts-root",
        type=Path,
        default=Path("data/cohorts"),
    )
    p.add_argument("--mapper", default=None, help="Only download this one display_name")
    p.add_argument("--min-year", type=int, default=2019)
    p.add_argument("--rate-limit", type=float, default=0.25, help="seconds between requests")
    p.add_argument("--no-exclude-ai", action="store_true", help="Do not skip AI-declared maps")
    p.add_argument("--dry-run", action="store_true", help="Count only; don't download zips")
    args = p.parse_args()

    data = json.loads(args.mappers_json.read_text(encoding="utf-8"))
    mappers = data["mappers"]
    if args.mapper:
        mappers = [m for m in mappers if m["display_name"] == args.mapper]
        if not mappers:
            logger.error("mapper %r not found in %s", args.mapper, args.mappers_json)
            return 2

    args.cohorts_root.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for m in mappers:
        s = download_cohort(
            m,
            args.cohorts_root,
            min_year=args.min_year,
            rate_limit=args.rate_limit,
            dry_run=args.dry_run,
            exclude_ai=not args.no_exclude_ai,
        )
        all_stats.append(s)
        logger.info(
            "[%s] seen=%d passed=%d dl=%d skipped=%d failed=%d",
            s["mapper"],
            s["maps_seen"],
            s["maps_passed"],
            s["maps_downloaded"],
            s["maps_skipped_existing"],
            s["maps_failed"],
        )

    # Global summary
    summary_path = args.cohorts_root / "_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_at": datetime.now(UTC).isoformat(),
                "min_year": args.min_year,
                "exclude_ai": not args.no_exclude_ai,
                "dry_run": args.dry_run,
                "cohorts": all_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s", summary_path)

    total_dl = sum(s["maps_downloaded"] for s in all_stats)
    total_passed = sum(s["maps_passed"] for s in all_stats)
    logger.info(
        "Done. total_passed=%d total_downloaded=%d cohorts=%d",
        total_passed,
        total_dl,
        len(all_stats),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
