"""BeatSaver API client for downloading Beat Saber maps.

Handles paginated API queries with rate limiting, quality filtering,
per-category quotas, and a persistent manifest tracking every downloaded map.

Key endpoints:
    - GET /search/text/{page}?sortOrder=Rating — search by rating
    - Download: https://r2cdn.beatsaver.com/{hash}.zip

Manifest (data/raw/manifest.json):
    Maps each map ID to its category, mod requirements, and download time.
    Used by the preprocessor to embed mod_requirements in .pt files and to
    skip categories during preprocessing.
"""

from __future__ import annotations

import json
import logging
import os
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import requests
from tqdm import tqdm

from beatsaber_automapper.data.tokenizer import genre_from_tags

logger = logging.getLogger(__name__)

API_BASE = "https://api.beatsaver.com"
CDN_BASE = "https://r2cdn.beatsaver.com"
USER_AGENT = "beatsaber-automapper/0.1.0 (https://github.com/Kwoolford/beatsaber_automapper)"

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SAVE_INTERVAL = 25  # Save manifest every N new downloads


def _classify_map_api(map_data: dict) -> str:
    """Classify a map from BeatSaver API diff booleans (pre-download).

    Uses ne/me/chroma boolean flags on each diff. Cannot detect Vivify —
    use _classify_map_zip for accurate classification post-download.

    Priority: noodle > mapping_extensions > chroma > vanilla

    Args:
        map_data: BeatSaver API map object.

    Returns:
        Category string: "noodle", "mapping_extensions", "chroma", or "vanilla".
    """
    has_ne = has_me = has_chroma = False
    for version in map_data.get("versions", []):
        for diff in version.get("diffs", []):
            if diff.get("ne"):
                has_ne = True
            if diff.get("me"):
                has_me = True
            if diff.get("chroma"):
                has_chroma = True
    if has_ne:
        return "noodle"
    if has_me:
        return "mapping_extensions"
    if has_chroma:
        return "chroma"
    return "vanilla"


def _extract_genre_tags(map_data: dict) -> list[str]:
    """Extract genre-relevant tags from a BeatSaver API map object.

    BeatSaver maps carry a free-form ``tags`` list (e.g. ``["electronic",
    "edm", "fast"]``). We return the raw list; callers use
    ``genre_from_tags()`` from the tokenizer to convert to a canonical genre.

    Args:
        map_data: BeatSaver API map object.

    Returns:
        List of tag strings (may be empty).
    """
    return list(map_data.get("tags", []))


def _classify_map_zip(zip_path: Path) -> tuple[str, list[str], list[str]]:
    """Classify a map by reading Info.dat inside the zip (post-download).

    Reads _customData._requirements and _customData._suggestions from both
    the top-level Info.dat and per-difficulty _customData entries.

    Priority: vivify > noodle > mapping_extensions > chroma > vanilla

    Args:
        zip_path: Path to the downloaded .zip file.

    Returns:
        Tuple of (category, requirements, suggestions). Requirements and
        suggestions are deduplicated sorted lists of mod name strings.
    """
    requirements: set[str] = set()
    suggestions: set[str] = set()

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find Info.dat case-insensitively
            info_name = None
            for name in zf.namelist():
                basename = name.rsplit("/", 1)[-1] if "/" in name else name
                if basename.lower() == "info.dat":
                    info_name = name
                    break

            if info_name is None:
                return "vanilla", [], []

            info_data = json.loads(zf.read(info_name).decode("utf-8"))

        # Top-level customData
        top_custom = info_data.get("_customData", {})
        requirements.update(top_custom.get("_requirements", []))
        suggestions.update(top_custom.get("_suggestions", []))

        # Per-difficulty customData
        for bms in info_data.get("_difficultyBeatmapSets", []):
            for diff in bms.get("_difficultyBeatmaps", []):
                custom = diff.get("_customData", {})
                requirements.update(custom.get("_requirements", []))
                suggestions.update(custom.get("_suggestions", []))

    except Exception as e:
        logger.debug("Could not classify %s: %s", zip_path.name, e)
        return "vanilla", [], []

    reqs = sorted(requirements)
    suggs = sorted(suggestions)
    all_mods = requirements | suggestions

    # Determine category by priority
    if any("vivify" in m.lower() for m in all_mods):
        return "vivify", reqs, suggs
    if any("noodle extensions" in m.lower() for m in requirements):
        return "noodle", reqs, suggs
    if any("mapping extensions" in m.lower() for m in requirements):
        return "mapping_extensions", reqs, suggs
    if any("chroma" in m.lower() for m in all_mods):
        return "chroma", reqs, suggs
    return "vanilla", reqs, suggs


def _load_manifest(output_dir: Path) -> dict:
    """Load the download manifest from output_dir/manifest.json.

    Args:
        output_dir: Directory containing manifest.json.

    Returns:
        Manifest dict mapping map ID -> {category, requirements, suggestions,
        downloaded_at}, or empty dict if file doesn't exist.
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load manifest: %s", e)
        return {}


def _save_manifest(manifest: dict, output_dir: Path) -> None:
    """Atomically save the manifest to output_dir/manifest.json.

    Writes to a .tmp file then renames to avoid corruption on crash.

    Args:
        manifest: Manifest dict to save.
        output_dir: Directory for manifest.json.
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    tmp_path = manifest_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, manifest_path)
    except Exception as e:
        logger.warning("Could not save manifest: %s", e)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def download_maps(
    output_dir: Path | str,
    *,
    quotas: dict[str, int | None] | None = None,
    count: int = 500,
    min_rating: float = 0.8,
    max_nps: float = 20.0,
    min_year: int = 2022,
    rate_limit: float = 0.5,
    exclude_ai: bool = True,
) -> list[Path]:
    """Download high-quality maps from BeatSaver with per-category quotas.

    If ``quotas`` is provided, downloads are tracked per category and stopped
    when each non-None quota is reached. Categories omitted from quotas (or
    with None values) are downloaded opportunistically with no cap. If
    ``quotas`` is None, falls back to ``count`` as a total cap across all
    categories.

    Maintains a ``manifest.json`` in ``output_dir`` tracking every map's
    category, requirements, and suggestions. Existing zips are backfilled into
    the manifest on the first run (one-time cost of opening ~N zips).

    Accepts maps of any difficulty level (Easy through ExpertPlus) as long as
    they have at least one Standard characteristic beatmap. The NPS cap is only
    enforced on Expert/ExpertPlus diffs to catch speed-map outliers.

    Args:
        output_dir: Directory to save downloaded .zip files.
        quotas: Per-category download targets, e.g.
            ``{"vanilla": 10000, "chroma": 2000, "noodle": 1000}``.
            A None value for a category means no cap (opportunistic).
            Omitted categories are also opportunistic.
        count: Total target when ``quotas`` is None. Ignored if ``quotas`` set.
        min_rating: Minimum upvote ratio (0.0–1.0).
        max_nps: Maximum notes-per-second cap for Expert/ExpertPlus diffs.
        min_year: Minimum upload year. Defaults to 2022 (v3 format era).
        rate_limit: Seconds between API requests.
        exclude_ai: If True, skip maps marked as AI-generated or automapped.

    Returns:
        List of paths to newly downloaded .zip files (excludes pre-existing).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest and backfill all existing zips (one-time cost)
    manifest = _load_manifest(output_dir)
    category_counts: Counter = Counter()
    manifest_dirty = False

    existing_zips = list(output_dir.glob("*.zip"))
    if existing_zips:
        logger.info("Backfilling %d existing zips into manifest...", len(existing_zips))
        for zip_path in tqdm(existing_zips, desc="Backfilling manifest", unit="map"):
            map_id = zip_path.stem
            if map_id not in manifest:
                cat, reqs, suggs = _classify_map_zip(zip_path)
                manifest[map_id] = {
                    "category": cat,
                    "requirements": reqs,
                    "suggestions": suggs,
                    "genre_tags": [],
                    "genre": "unknown",
                    "downloaded_at": datetime.now(UTC).isoformat(),
                }
                manifest_dirty = True
            category_counts[manifest[map_id]["category"]] += 1

    if manifest_dirty:
        _save_manifest(manifest, output_dir)
        manifest_dirty = False

    logger.info("Category counts from existing maps: %s", dict(category_counts))

    # Determine total target for progress bar
    if quotas is not None:
        capped = {cat: q for cat, q in quotas.items() if q is not None}
        total_target = sum(capped.values()) if capped else count
    else:
        capped = {}
        total_target = count

    def _quota_met(category: str) -> bool:
        """Return True if the quota for this category has been reached."""
        if quotas is None:
            return False
        quota = quotas.get(category)
        if quota is None:
            return False  # Opportunistic — no cap
        return category_counts[category] >= quota

    def _all_quotas_met() -> bool:
        """Return True when it's time to stop the download loop."""
        if not capped:
            # No capped categories — fall back to total count of new downloads
            return len(downloaded) >= count
        return all(category_counts[cat] >= quota for cat, quota in capped.items())

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    downloaded: list[Path] = []
    page = 0
    consecutive_empty = 0
    saves_since_write = 0

    pbar = tqdm(total=total_target, desc="Downloading maps", unit="map")
    # Pre-fill pbar with existing counts toward capped quotas
    if capped:
        existing_progress = sum(
            min(category_counts.get(cat, 0), quota) for cat, quota in capped.items()
        )
        pbar.update(existing_progress)

    while not _all_quotas_met():
        maps = _fetch_search_page(session, page, rate_limit)
        if not maps:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                logger.info("No more results after page %d, stopping", page)
                break
            page += 1
            continue
        consecutive_empty = 0

        for map_data in maps:
            if _all_quotas_met():
                break

            map_id = map_data.get("id", "")
            zip_path = output_dir / f"{map_id}.zip"

            # Skip already-downloaded maps (counted during backfill)
            if zip_path.exists():
                continue

            # Apply quality filters
            if not _passes_filters(
                map_data,
                min_rating=min_rating,
                max_nps=max_nps,
                min_year=min_year,
                exclude_ai=exclude_ai,
            ):
                continue

            # Pre-classify from API booleans to skip obvious over-quota categories
            api_category = _classify_map_api(map_data)
            if _quota_met(api_category):
                continue

            # Extract genre tags from API response (available pre-download)
            genre_tags = _extract_genre_tags(map_data)
            genre = genre_from_tags(genre_tags)

            # Download
            dl_hash = _get_download_hash(map_data)
            if not dl_hash:
                continue

            try:
                _download_zip(session, dl_hash, zip_path, rate_limit)
            except requests.RequestException as e:
                logger.warning("Failed to download %s: %s", map_id, e)
                continue

            # Accurate classification from the downloaded zip
            cat, reqs, suggs = _classify_map_zip(zip_path)

            # Re-check quota with the accurate category
            if _quota_met(cat):
                zip_path.unlink(missing_ok=True)
                logger.debug("Removed %s: %s quota already met", map_id, cat)
                continue

            category_counts[cat] += 1
            manifest[map_id] = {
                "category": cat,
                "requirements": reqs,
                "suggestions": suggs,
                "genre_tags": genre_tags,
                "genre": genre,
                "downloaded_at": datetime.now(UTC).isoformat(),
            }
            manifest_dirty = True
            downloaded.append(zip_path)
            pbar.update(1)
            saves_since_write += 1

            if saves_since_write >= MANIFEST_SAVE_INTERVAL:
                _save_manifest(manifest, output_dir)
                manifest_dirty = False
                saves_since_write = 0

        page += 1

    pbar.close()

    if manifest_dirty:
        _save_manifest(manifest, output_dir)

    logger.info("Downloaded %d new maps to %s", len(downloaded), output_dir)
    logger.info("Final category counts: %s", dict(category_counts))
    return downloaded


def _fetch_search_page(
    session: requests.Session,
    page: int,
    rate_limit: float,
) -> list[dict]:
    """Fetch one page of search results sorted by rating.

    Args:
        session: Requests session with headers.
        page: Page number (0-indexed).
        rate_limit: Seconds to wait before request.

    Returns:
        List of map data dicts, or empty list on error.
    """
    time.sleep(rate_limit)
    url = f"{API_BASE}/search/text/{page}"
    params = {"sortOrder": "Rating"}

    try:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 10))
            logger.warning("Rate limited, sleeping %ds", retry_after)
            time.sleep(retry_after)
            resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("docs", [])
    except requests.RequestException as e:
        logger.warning("Search request failed (page %d): %s", page, e)
        return []


_EXPERT_DIFFS = {"Expert", "ExpertPlus"}


def _passes_filters(
    map_data: dict,
    *,
    min_rating: float,
    max_nps: float,
    min_year: int,
    exclude_ai: bool = True,
) -> bool:
    """Check if a map passes all quality filters.

    Accepts any difficulty level (Easy–ExpertPlus). Requires at least one
    Standard characteristic diff so we don't train on 360Degree/OneSaber/etc.
    NPS cap is only enforced on Expert/ExpertPlus to catch speed-map outliers.
    """
    # Exclude AI/automapped maps from training data.
    # declaredAi is a string field: "None" means human-made; "Assisted"/"SemiAuto"/"FullAuto" = AI.
    if exclude_ai:
        if map_data.get("automapper"):
            return False
        declared_ai = map_data.get("declaredAi", "None")
        if declared_ai and declared_ai != "None":
            return False

    # Rating filter
    stats = map_data.get("stats", {})
    score = stats.get("score", 0)
    if score < min_rating:
        return False

    # Year filter
    uploaded = map_data.get("uploaded", "")
    if uploaded:
        try:
            year = int(uploaded[:4])
            if year < min_year:
                return False
        except (ValueError, IndexError):
            pass

    # Characteristic filter — must have at least one Standard diff.
    # Excludes 360Degree, OneSaber, Lightshow, Lawless, etc.
    standard_diffs = [
        diff
        for version in map_data.get("versions", [])
        for diff in version.get("diffs", [])
        if diff.get("characteristic") == "Standard"
    ]
    if not standard_diffs:
        return False

    # NPS filter — only cap Expert/ExpertPlus to catch speed maps.
    # Easy/Normal/Hard diffs are uncapped (low NPS by nature).
    for diff in standard_diffs:
        if diff.get("difficulty") in _EXPERT_DIFFS:
            if diff.get("nps", 0) > max_nps:
                return False

    return True


def _get_download_hash(map_data: dict) -> str | None:
    """Extract the download hash from map data."""
    versions = map_data.get("versions", [])
    if not versions:
        return None
    # Use the latest version
    return versions[-1].get("hash", None)


def _download_zip(
    session: requests.Session,
    dl_hash: str,
    dest: Path,
    rate_limit: float,
) -> None:
    """Download a map zip from CDN.

    Args:
        session: Requests session.
        dl_hash: Map version hash for CDN URL.
        dest: Destination file path.
        rate_limit: Seconds to wait before request.
    """
    time.sleep(rate_limit)
    url = f"{CDN_BASE}/{dl_hash}.zip"
    resp = session.get(url, timeout=60, stream=True)
    resp.raise_for_status()

    # Write to temp file then rename for atomicity
    tmp_dest = dest.with_suffix(".tmp")
    try:
        with open(tmp_dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        tmp_dest.rename(dest)
    except Exception:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise
