#!/usr/bin/env python3
"""Install generated map zips into a Beat Saber CustomLevels folder so they are
playable, with human-readable names in the in-game song browser.

Why this exists: generated zips carry `_songName: "1f333"` and no cover art, so in
the headset every map is an untitled grey tile named after a corpus id. This script
rewrites the display metadata (leaving all note data byte-identical) and drops the
result into CustomLevels.

    python scripts/deploy_maps.py outputs/kyle_review_2026-08-03/*_AFTER2_reach.zip
    python scripts/deploy_maps.py --list-dest          # just show where it would go
    python scripts/deploy_maps.py --dest /path/to/CustomLevels <zips...>

Nothing is deleted unless --replace is passed, and then only the target folder of a
map this script is currently installing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

# Corpus ids are opaque; Kyle asked for names. Keep this table in sync with
# docs/BASELINE_2026-08-03.md ("The standing song set").
SONGS: dict[str, tuple[str, str]] = {
    "1f333": ("Hunger", "Aether Realm"),
    "1f767": ("アリスブルー", "HoneyWorks"),
    "1f8d6": ("Fallen Kingdom (2022 Remap)", "CaptainSparklez"),
    "1f913": ("Digital Life Hacker", "Wotoha"),
    "SOTIREDROCK": ("SO TIRED ROCK", "NUEKI"),
}

# Distinct cover colours per variant so BEFORE/AFTER/AFTER2 are tellable apart on
# the song wall without reading the subtitle.
VARIANT_COLOURS: dict[str, tuple[int, int, int]] = {
    "BEFORE": (150, 60, 60),
    "AFTER": (150, 120, 50),
    "AFTER2": (60, 130, 90),
    "EXTRA": (80, 80, 140),
}

CANDIDATE_ROOTS = [
    Path.home() / ".local/share/bs-manager/BSInstances",
    Path.home() / ".local/share/bs-manager",
    # BSManager's installation folder is user-chosen at first-run setup. Kyle put
    # it on the big NVMe on 2026-08-04 because / had only 33 GB free against
    # /mnt/giga_speed's 825 GB, and BSManager keeps whole ~2 GB Beat Saber
    # versions plus the map library. Without this entry auto-detection silently
    # finds nothing and deploy_maps tells you to install Beat Saber first.
    Path("/mnt/giga_speed/BSManager"),
    Path("/mnt/giga_speed/BSManager/BSInstances"),
    Path("/mnt/giga_speed/SteamLibrary/steamapps/common/Beat Saber"),
    Path.home() / ".local/share/Steam/steamapps/common/Beat Saber",
]


def find_custom_levels() -> list[Path]:
    """Locate every `Beat Saber_Data/CustomLevels` on this machine, newest first."""
    found: list[Path] = []
    for root in CANDIDATE_ROOTS:
        if not root.exists():
            continue
        # A Beat Saber install is identified by its _Data dir, whether or not
        # CustomLevels has been created yet (a fresh modded install may not have it).
        for data_dir in sorted(root.glob("**/Beat Saber_Data")):
            found.append(data_dir / "CustomLevels")
    # De-duplicate, preserving order, then sort by mtime of the install.
    seen: set[Path] = set()
    uniq = [p for p in found if not (p in seen or seen.add(p))]
    uniq.sort(key=lambda p: p.parent.stat().st_mtime if p.parent.exists() else 0, reverse=True)
    return uniq


def corpus_title(song_id: str) -> tuple[str, str] | None:
    """Real title + artist from the CORPUS zip for this id, if we have it.

    Added 2026-08-05: the wide cohort is 149 corpus songs, and without this every
    one of them lands in the song browser as an opaque hex id like `236e7`. The
    human map for the same id already carries the real metadata.
    ⚠️Exact basename match on info.dat — "BPMInfo.dat" also ends with "info.dat"
    and 73 of 300 corpus zips list it FIRST.
    """
    z = Path(__file__).resolve().parents[1] / "data" / "raw" / f"{song_id}.zip"
    if not z.exists():
        return None
    try:
        with zipfile.ZipFile(z) as zf:
            name = next((n for n in zf.namelist()
                         if n.split("/")[-1].lower() == "info.dat"), None)
            if name is None:
                return None
            d = json.loads(zf.read(name).decode("utf-8-sig"))
        title = str(d.get("_songName") or "").strip()
        artist = str(d.get("_songAuthorName") or "").strip()
        return (title, artist or "Unknown Artist") if title else None
    except Exception:
        return None


def parse_stem(stem: str) -> tuple[str, str, str, str]:
    """`1f333_AFTER2_reach` -> (id, variant, rest, pretty title)."""
    parts = stem.split("_")
    song_id = parts[0]
    variant = parts[1] if len(parts) > 1 else ""
    rest = "_".join(parts[2:])
    if song_id in SONGS:
        title, artist = SONGS[song_id]
    else:
        got = corpus_title(song_id)
        title, artist = got if got else (song_id, "Unknown Artist")
    return song_id, variant, rest, title


def make_cover(path: Path, colour: tuple[int, int, int], label: str) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    img = Image.new("RGB", (256, 256), colour)
    draw = ImageDraw.Draw(img)
    draw.text((14, 14), "AUTO", fill=(235, 235, 235))
    draw.text((14, 34), label[:22], fill=(215, 215, 215))
    img.save(path, "PNG")
    return True


def safe_folder_name(name: str) -> str:
    """Beat Saber tolerates unicode, but not path separators or control chars."""
    cleaned = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "-", name)
    return cleaned.strip().rstrip(".")[:120]


def deploy(zip_path: Path, dest_root: Path, replace: bool, dry_run: bool) -> str | None:
    song_id, variant, rest, title = parse_stem(zip_path.stem)
    tag = " ".join(x for x in (variant, rest) if x)
    folder = safe_folder_name(f"AUTO {title} [{tag}]" if tag else f"AUTO {title}")
    target = dest_root / folder

    if target.exists():
        if not replace:
            return f"  SKIP  {folder}  (exists; pass --replace to overwrite)"
        if not dry_run:
            shutil.rmtree(target)

    if dry_run:
        return f"  would install -> {target}"

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)

    info_path = next(
        (p for p in target.iterdir() if p.name.lower() == "info.dat"), None
    )
    if info_path is None:
        shutil.rmtree(target)
        return f"  FAIL  {zip_path.name}: no Info.dat inside"

    info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    _, artist = SONGS.get(song_id, (title, "Unknown Artist"))
    info["_songName"] = title
    info["_songSubName"] = tag
    info["_songAuthorName"] = artist
    info["_levelAuthorName"] = f"automapper ({song_id})"

    cover_name = info.get("_coverImageFilename") or ""
    if not cover_name:
        colour = VARIANT_COLOURS.get(variant, (90, 90, 90))
        if make_cover(target / "cover.png", colour, title):
            info["_coverImageFilename"] = "cover.png"

    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"  OK    {folder}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zips", nargs="*", type=Path, help="generated map zip files")
    ap.add_argument("--dest", type=Path, help="CustomLevels dir (default: auto-detect)")
    ap.add_argument("--list-dest", action="store_true", help="show detected install(s) and exit")
    ap.add_argument("--replace", action="store_true", help="overwrite an already-installed map")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen")
    args = ap.parse_args()

    detected = find_custom_levels()
    if args.list_dest:
        if not detected:
            print("No Beat Saber install found. Looked in:")
            for r in CANDIDATE_ROOTS:
                print(f"  {r}   {'(exists)' if r.exists() else '(missing)'}")
            return 1
        print("Detected CustomLevels targets (newest first):")
        for p in detected:
            n = len(list(p.glob("*"))) if p.exists() else 0
            print(f"  {p}   [{n} maps installed]" if p.exists() else f"  {p}   [will be created]")
        return 0

    if not args.zips:
        ap.error("give at least one zip, or use --list-dest")

    dest = args.dest or (detected[0] if detected else None)
    if dest is None:
        print(
            "No Beat Saber install found and --dest not given.\n"
            "Install Beat Saber via BSManager first, then re-run.\n"
            "Checked: " + ", ".join(str(r) for r in CANDIDATE_ROOTS),
            file=sys.stderr,
        )
        return 1
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    print(f"Destination: {dest}")
    rc = 0
    for zp in args.zips:
        if not zp.exists():
            print(f"  FAIL  {zp}: not found")
            rc = 1
            continue
        msg = deploy(zp, dest, args.replace, args.dry_run)
        print(msg)
        if msg and msg.strip().startswith("FAIL"):
            rc = 1
    if not args.dry_run:
        print("\nRestart Beat Saber (or use SongCore's in-game refresh) to see new maps.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
