"""V8-0 PoC, part 3 — does per-instrument event activity read song STRUCTURE
(rise/drop/density) better than the current energy-percentile section detector?

The first gate (v8_poc_alignment) disproved the *timing-backbone* rebuild (the BPM
grid already represents 94-99% of human note timing). But the user's core argument
for V8 is different: feeding the model per-INSTRUMENT note events lets it interpret
the song's dynamic structure (builds/drops/breakdowns) — which is what should drive
WHEN/density, and what the hand-tuned section detector currently (mis)handles.

This script tests that claim directly. Per song, in 2 s windows:
  * human note density  (notes/sec the mapper actually placed)  -- the target
  * per-instrument event features from transcription:
        total event density, drum density, kick density, bass activity,
        lead activity, #active stems (the "layering" build/drop proxy)
  * baseline: the energy-percentile section detector's implied density rank
        (drop>chorus>verse>bridge>intro>outro) -- what V7 uses today.

For each feature we report the per-song Spearman correlation with human density,
averaged over songs. If instrument-activity features correlate with where humans map
notes substantially better than the section-detector rank does, the per-instrument
representation is a better structure signal than the current mechanism — the user's point.

Usage:
    python scripts/v8_poc_structure.py --songs 2d675 3d2f9 ... --out outputs/v8_poc/structure.json
    python scripts/v8_poc_structure.py --n 12
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import random
import sys
import tempfile
import warnings
import zipfile

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v8_struct")

from v8_poc import (  # noqa: E402
    PITCHED_STEMS, DRUM_STEM, separate_stems, transcribe_pitched, transcribe_drums,
)
from v8_poc_alignment import _human_times, _pick_difficulty, DIFF_PREFERENCE  # noqa: E402

# Section type -> implied density rank (what the V7 section-threshold stack encodes:
# drops dense, intros/outros sparse). This is the baseline structure signal.
_SECTION_RANK = {"drop": 5, "chorus": 4, "verse": 3, "bridge": 2, "intro": 1, "outro": 0}

WIN = 2.0  # window seconds


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(raw_zip: pathlib.Path) -> dict | None:
    import torch
    from beatsaber_automapper.data.audio import load_audio, detect_sections_energy_percentile

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="v8struct_"))
    with zipfile.ZipFile(raw_zip) as zf:
        zf.extractall(tmp)
    egg = next(tmp.glob("*.egg"), None) or next(tmp.glob("*.ogg"), None)
    if egg is None:
        return None
    diff = _pick_difficulty(tmp)
    if diff is None:
        return None
    human, _bpm = _human_times(tmp, diff)
    if human.size < 50:
        return None

    stems, sr = separate_stems(egg)
    duration = max(len(v) for v in stems.values()) / sr

    events = []
    for stem in PITCHED_STEMS:
        if stem in stems:
            tau = 0.10 if stem == "other" else 0.0
            merge = 40.0 if stem == "other" else 0.0
            events += transcribe_pitched(stems[stem], sr, stem, salience_tau=tau, chord_merge_ms=merge)
    if DRUM_STEM in stems:
        events += transcribe_drums(stems[DRUM_STEM], sr)

    n_win = int(duration // WIN)
    if n_win < 5:
        return None
    edges = np.arange(n_win + 1) * WIN

    def dens(times):
        h, _ = np.histogram(times, bins=edges)
        return h / WIN

    human_d = dens(human)
    onsets = np.array([e.onset_sec for e in events])
    drum_t = np.array([e.onset_sec for e in events if e.stem in ("kick", "snare", "hat")])
    kick_t = np.array([e.onset_sec for e in events if e.stem == "kick"])
    bass_t = np.array([e.onset_sec for e in events if e.stem == "bass"])
    lead_t = np.array([e.onset_sec for e in events if e.stem == "lead"])
    voc_t  = np.array([e.onset_sec for e in events if e.stem == "vocals"])

    # #active stems per window (layering -> build/drop).
    active = np.zeros(n_win)
    for ts in (drum_t, bass_t, lead_t, voc_t):
        active += (dens(ts) > 0).astype(float)

    feats = {
        "total_event_density": dens(onsets),
        "drum_density":        dens(drum_t),
        "kick_density":        dens(kick_t),
        "bass_activity":       dens(bass_t),
        "lead_activity":       dens(lead_t),
        "n_active_stems":      active,
    }

    # Baseline: section-detector implied density rank per window.
    sections = detect_sections_energy_percentile(
        load_audio(egg, target_sr=sr)[0], sample_rate=sr
    )
    sec_rank = np.zeros(n_win)
    win_centers = (edges[:-1] + edges[1:]) / 2
    for i, c in enumerate(win_centers):
        lab = next((t for (t, s, e) in sections if s <= c < e), "verse")
        sec_rank[i] = _SECTION_RANK.get(lab, 3)
    feats["section_detector_rank"] = sec_rank

    corrs = {k: _spearman(v, human_d) for k, v in feats.items()}
    log.info("%s (%s, %d win): %s", raw_zip.stem, diff, n_win,
             {k: round(v, 2) for k, v in corrs.items()})
    return {"song": raw_zip.stem, "difficulty": diff, "n_win": n_win, "spearman": corrs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "outputs/v8_poc/structure.json")
    args = ap.parse_args()

    raw_dir = REPO_ROOT / "data/raw"
    if args.songs:
        zips = [raw_dir / f"{s}.zip" for s in args.songs]
    else:
        zips = sorted(raw_dir.glob("*.zip"))
        random.Random(args.seed).shuffle(zips)
        zips = zips[: args.n * 3]

    results = []
    for z in zips:
        if len(results) >= args.n:
            break
        try:
            r = analyse(z)
            if r:
                results.append(r)
        except Exception as exc:
            log.warning("%s failed: %s", z.name, exc)

    if not results:
        log.error("nothing analysed")
        return

    keys = list(results[0]["spearman"].keys())
    means = {k: round(float(np.nanmean([r["spearman"][k] for r in results])), 3) for k in keys}
    summary = {"n_songs": len(results), "window_sec": WIN,
               "mean_spearman_vs_human_density": means, "per_song": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print(f"Structure signal vs human note density — {len(results)} songs, {WIN:.0f}s windows")
    print("mean per-song Spearman r (higher = better predicts where humans map):")
    print("=" * 70)
    for k in sorted(means, key=lambda x: -means[x]):
        tag = "  <- V7's current signal" if k == "section_detector_rank" else ""
        print(f"  {k:24s} r={means[k]:+.3f}{tag}")
    print("=" * 70)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
