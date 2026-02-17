"""BeatSaver API client for downloading Beat Saber maps.

Handles paginated API queries with rate limiting and quality filtering.
Downloads map .zip files from BeatSaver's CDN.

Key endpoints:
    - GET /api/search/text/{page}?sortOrder=Rating — search by rating
    - Download: https://r2cdn.beatsaver.com/{hash}.zip
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

API_BASE = "https://api.beatsaver.com"
CDN_BASE = "https://r2cdn.beatsaver.com"
USER_AGENT = "beatsaber-automapper/0.1.0 (https://github.com/Kwoolford/beatsaber_automapper)"


def download_maps(
    output_dir: Path | str,
    *,
    count: int = 500,
    min_rating: float = 0.8,
    max_nps: float = 20.0,
    min_year: int = 2020,
    require_difficulties: list[str] | None = None,
    rate_limit: float = 0.5,
) -> list[Path]:
    """Download high-quality maps from BeatSaver.

    Args:
        output_dir: Directory to save downloaded .zip files.
        count: Target number of maps to download.
        min_rating: Minimum upvote ratio (0.0-1.0).
        max_nps: Maximum notes-per-second filter.
        min_year: Minimum upload year.
        require_difficulties: Required difficulties (e.g. ["Expert", "ExpertPlus"]).
        rate_limit: Seconds between API requests.

    Returns:
        List of paths to downloaded .zip files.
    """
    if require_difficulties is None:
        require_difficulties = ["Expert", "ExpertPlus"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    downloaded: list[Path] = []
    page = 0
    consecutive_empty = 0

    pbar = tqdm(total=count, desc="Downloading maps", unit="map")

    while len(downloaded) < count:
        # Fetch search results page
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
            if len(downloaded) >= count:
                break

            map_id = map_data.get("id", "")
            zip_path = output_dir / f"{map_id}.zip"

            # Resume support: skip already downloaded
            if zip_path.exists():
                downloaded.append(zip_path)
                pbar.update(1)
                continue

            # Apply filters
            if not _passes_filters(
                map_data,
                min_rating=min_rating,
                max_nps=max_nps,
                min_year=min_year,
                require_difficulties=require_difficulties,
            ):
                continue

            # Download the map
            dl_hash = _get_download_hash(map_data)
            if not dl_hash:
                continue

            try:
                _download_zip(session, dl_hash, zip_path, rate_limit)
                downloaded.append(zip_path)
                pbar.update(1)
            except requests.RequestException as e:
                logger.warning("Failed to download %s: %s", map_id, e)

        page += 1

    pbar.close()
    logger.info("Downloaded %d maps to %s", len(downloaded), output_dir)
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
    url = f"{API_BASE}/api/search/text/{page}"
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


def _passes_filters(
    map_data: dict,
    *,
    min_rating: float,
    max_nps: float,
    min_year: int,
    require_difficulties: list[str],
) -> bool:
    """Check if a map passes all quality filters."""
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

    # Difficulty filter — check all versions
    available_diffs: set[str] = set()
    for version in map_data.get("versions", []):
        for diff in version.get("diffs", []):
            available_diffs.add(diff.get("difficulty", ""))

    has_required = any(d in available_diffs for d in require_difficulties)
    if not has_required:
        return False

    # NPS filter
    for version in map_data.get("versions", []):
        for diff in version.get("diffs", []):
            nps = diff.get("nps", 0)
            if nps > max_nps:
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
