"""V8-0 de-risk PoC — symbolic per-stem transcription backbone.

Proves (or disproves) the V8 thesis from ``docs/architecture_v8_plan.md`` BEFORE
committing to the rebuild. Runs the full transcription front-end on one song:

    audio -> Demucs (4 stems)
          -> drums  : multi-band librosa onset      -> unpitched events
          -> bass   : basic-pitch                   -> pitched note events
          -> vocals : basic-pitch                   -> pitched note events
          -> other  : basic-pitch (+salience gate + chord-merge) -> lead events
          -> merged & sorted NoteEvent stream

Then it answers the three gate questions:

  (a) Drop cluster.   Does the ~13-15 s drop produce a dense onset cluster that
      the V7 generated map misses?  (Pass a ``--v7-map`` to quantify the miss.)
  (b) Alignment F1.   Do transcribed onsets predict a *human* mapper's note times
      better than the current 0.41 baseline (and better than plain librosa onsets,
      the closest "what V7 has to work with" signal)?  Needs ``--human-map``.
  (c) Lead contour.   Does the ``other``-stem pitch contour visibly track the
      melody?  (Eyeball the saved piano-roll / contour PNG.)

This script is intentionally self-contained: the V8-0 phase is a HARD GATE, so we
do not add ``data/note_events.py`` etc. to the package until the gate is green.

Usage
-----
SO TIRED ROCK (validation a + c, no human map exists for this held-out song):
    python scripts/v8_poc.py \
        --audio "data/test_songs/SO TIRED ROCK - NUEKI.mp3" \
        --v7-map outputs/v7_section_aware.zip \
        --out-dir outputs/v8_poc/so_tired_rock

In-dataset song (validation b — has a human map):
    python scripts/v8_poc.py \
        --raw-zip data/raw/1ccca.zip --difficulty ExpertStandard \
        --out-dir outputs/v8_poc/1ccca
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import sys
import tempfile
import warnings
import zipfile

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger("v8_poc")

# Pitched stems get basic-pitch; drums get multi-band onset.
PITCHED_STEMS = ("bass", "vocals", "other")
DRUM_STEM = "drums"


@dataclasses.dataclass
class NoteEvent:
    """A single transcribed musical event. Continuous time, NOT a grid slot."""

    onset_sec: float
    dur_sec: float
    pitch: int | None        # MIDI pitch; None for unpitched drum hits
    stem: str                # kick|snare|hat|bass|vocal|lead|other
    salience: float          # transcription confidence x amplitude

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Stem separation
# ---------------------------------------------------------------------------
def separate_stems(audio_path: pathlib.Path) -> tuple[dict[str, np.ndarray], int]:
    """Demucs-separate into mono stems. Returns (dict stem->mono float32, sr)."""
    import torch
    from beatsaber_automapper.data.audio import load_audio
    from beatsaber_automapper.data.stem_separator import separate, DEMUCS_SR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    waveform, src_sr = load_audio(audio_path, target_sr=DEMUCS_SR)
    log.info("Separating stems (Demucs, device=%s) …", device)
    stems = separate(waveform, src_sr, device=device)
    out: dict[str, np.ndarray] = {}
    for name, stem in stems.items():
        arr = stem.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 2:           # [channels, samples] -> mono
            arr = arr.mean(axis=0)
        out[name] = arr
    return out, DEMUCS_SR


# ---------------------------------------------------------------------------
# Transcription — pitched stems (basic-pitch)
# ---------------------------------------------------------------------------
def transcribe_pitched(
    y: np.ndarray,
    sr: int,
    stem_name: str,
    salience_tau: float = 0.0,
    chord_merge_ms: float = 0.0,
) -> list[NoteEvent]:
    """basic-pitch transcription of a single pitched stem.

    ``salience_tau``  : drop events whose amplitude is below this fraction of the
                        max amplitude (the ``other``-stem distorted-guitar gate).
    ``chord_merge_ms``: collapse events whose onsets fall within this window into a
                        single onset (keeps the highest-salience pitch) so a strummed
                        chord becomes one onset, not a smear.
    """
    import soundfile as sf
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_path = tf.name
    try:
        sf.write(tmp_path, y, sr)
        _, _, note_events = predict(tmp_path, ICASSP_2022_MODEL_PATH)
    finally:
        os.unlink(tmp_path)

    # note_events tuple: (start_sec, end_sec, pitch_midi, amplitude, pitch_bends)
    events: list[NoteEvent] = []
    for start, end, pitch, amp, _bends in note_events:
        events.append(
            NoteEvent(
                onset_sec=float(start),
                dur_sec=float(end) - float(start),
                pitch=int(pitch),
                stem="lead" if stem_name == "other" else stem_name,
                salience=float(amp),
            )
        )
    events.sort(key=lambda e: e.onset_sec)

    if salience_tau > 0.0 and events:
        max_sal = max(e.salience for e in events)
        thr = salience_tau * max_sal
        events = [e for e in events if e.salience >= thr]

    if chord_merge_ms > 0.0 and events:
        events = _chord_merge(events, chord_merge_ms / 1000.0)

    return events


def _chord_merge(events: list[NoteEvent], window_sec: float) -> list[NoteEvent]:
    """Collapse onsets within ``window_sec`` into the single highest-salience event."""
    merged: list[NoteEvent] = []
    cluster: list[NoteEvent] = []
    for ev in events:                       # events are onset-sorted
        if cluster and ev.onset_sec - cluster[0].onset_sec <= window_sec:
            cluster.append(ev)
        else:
            if cluster:
                merged.append(max(cluster, key=lambda e: e.salience))
            cluster = [ev]
    if cluster:
        merged.append(max(cluster, key=lambda e: e.salience))
    return merged


# ---------------------------------------------------------------------------
# Transcription — drum stem (multi-band librosa onset)
# ---------------------------------------------------------------------------
def transcribe_drums(y: np.ndarray, sr: int) -> list[NoteEvent]:
    """Multi-band onset detection on the drum stem -> kick/snare/hat events."""
    import librosa
    from scipy.signal import butter, sosfiltfilt

    bands = {
        "kick":  ("low", 0.0, 150.0),
        "snare": ("band", 150.0, 2000.0),
        "hat":   ("high", 6000.0, 0.0),
    }
    events: list[NoteEvent] = []
    nyq = sr / 2.0
    for name, (kind, lo, hi) in bands.items():
        if kind == "low":
            sos = butter(4, hi / nyq, btype="low", output="sos")
        elif kind == "high":
            sos = butter(4, lo / nyq, btype="high", output="sos")
        else:
            sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        yb = sosfiltfilt(sos, y).astype(np.float32)
        env = librosa.onset.onset_strength(y=yb, sr=sr, hop_length=512)
        onsets = librosa.onset.onset_detect(
            onset_envelope=env, sr=sr, hop_length=512, units="time", backtrack=True
        )
        # Salience = onset-envelope value at the detected frame, normalised per band.
        frames = librosa.time_to_frames(onsets, sr=sr, hop_length=512)
        frames = np.clip(frames, 0, len(env) - 1)
        env_max = float(env.max()) if env.size else 1.0
        for t, fr in zip(onsets, frames):
            events.append(
                NoteEvent(
                    onset_sec=float(t),
                    dur_sec=0.0,
                    pitch=None,
                    stem=name,
                    salience=float(env[fr]) / (env_max + 1e-9),
                )
            )
    events.sort(key=lambda e: e.onset_sec)
    return events


# ---------------------------------------------------------------------------
# Alignment scoring (reused from eval_alignment)
# ---------------------------------------------------------------------------
def alignment_f1(generated_times, reference_times, tol_sec: float = 0.05):
    from eval_alignment import alignment_score

    return alignment_score(list(generated_times), list(reference_times), tol_sec)


def librosa_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return np.asarray(
        librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, units="time", backtrack=True),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Human / V7 map note times
# ---------------------------------------------------------------------------
def map_note_times(map_path: pathlib.Path, difficulty: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (all_note_times_sec, drop_window_count placeholder) for a map zip/dir.

    Second element is unused; kept simple — returns sorted onset times in seconds
    and the bpm-derived count is computed by the caller.
    """
    from beatsaber_automapper.data.beatmap import parse_info_dat, parse_difficulty_dat

    if map_path.suffix == ".zip":
        tmp = tempfile.mkdtemp(prefix="v8poc_map_")
        with zipfile.ZipFile(map_path) as zf:
            zf.extractall(tmp)
        map_dir = pathlib.Path(tmp)
    else:
        map_dir = map_path

    info_path = next(map_dir.glob("[Ii]nfo.dat"), None)
    if info_path is None:
        raise FileNotFoundError(f"No Info.dat in {map_dir}")
    info = parse_info_dat(info_path)
    bpm = float(info.bpm)

    diff_path = None
    for f in sorted(map_dir.glob("*.dat")):
        if f.name.lower().startswith(difficulty.lower()):
            diff_path = f
            break
    if diff_path is None:                       # loose fallback
        for cand in (difficulty, "ExpertPlus", "Expert", "Hard", "Normal", "Easy"):
            for f in sorted(map_dir.glob("*.dat")):
                if cand.lower() in f.name.lower():
                    diff_path = f
                    break
            if diff_path:
                break
    if diff_path is None:
        raise FileNotFoundError(f"No difficulty .dat matching {difficulty} in {map_dir}")
    log.info("Human/V7 map difficulty file: %s (bpm=%.1f)", diff_path.name, bpm)

    bm = parse_difficulty_dat(diff_path)
    times = np.array(sorted(n.beat * 60.0 / bpm for n in bm.color_notes), dtype=np.float64)
    return times, np.array([bpm])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pianoroll(
    events: list[NoteEvent],
    duration: float,
    out_png: pathlib.Path,
    drop_window: tuple[float, float] | None,
    human_times: np.ndarray | None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pitched = [e for e in events if e.pitch is not None]
    drums = [e for e in events if e.pitch is None]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8), height_ratios=[3, 1], sharex=True
    )

    stem_color = {"bass": "tab:blue", "vocals": "tab:green", "lead": "tab:red"}
    for stem, color in stem_color.items():
        evs = [e for e in pitched if e.stem == stem]
        if not evs:
            continue
        ax1.hlines(
            [e.pitch for e in evs],
            [e.onset_sec for e in evs],
            [e.onset_sec + max(e.dur_sec, 0.05) for e in evs],
            color=color, lw=2, alpha=0.7, label=stem,
        )
    ax1.set_ylabel("MIDI pitch")
    ax1.set_title(f"V8 PoC piano-roll — {out_png.parent.name}  ({len(pitched)} pitched events)")
    ax1.legend(loc="upper right")

    drum_row = {"kick": 0, "snare": 1, "hat": 2}
    drum_color = {"kick": "black", "snare": "tab:orange", "hat": "tab:cyan"}
    for e in drums:
        ax2.vlines(e.onset_sec, drum_row[e.stem] - 0.4, drum_row[e.stem] + 0.4,
                   color=drum_color[e.stem], lw=1.0, alpha=0.6)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["kick", "snare", "hat"])
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("drums")

    if human_times is not None:
        ax2.vlines(human_times, -0.5, 2.5, color="magenta", lw=0.5, alpha=0.4)

    if drop_window is not None:
        for ax in (ax1, ax2):
            ax.axvspan(drop_window[0], drop_window[1], color="yellow", alpha=0.2, zorder=0)

    ax1.set_xlim(0, duration)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    log.info("Wrote %s", out_png)


def plot_lead_contour(events: list[NoteEvent], duration: float, out_png: pathlib.Path) -> None:
    """Validation (c): plot the lead-stem pitch over time to eyeball melody-tracking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lead = sorted([e for e in events if e.stem == "lead"], key=lambda e: e.onset_sec)
    fig, ax = plt.subplots(figsize=(16, 4))
    if lead:
        ax.plot([e.onset_sec for e in lead], [e.pitch for e in lead],
                "-o", ms=3, lw=1, color="tab:red", alpha=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("lead MIDI pitch")
    ax.set_title(f"Lead (other-stem) pitch contour — {len(lead)} events")
    ax.set_xlim(0, duration)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    log.info("Wrote %s", out_png)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="V8-0 transcription de-risk PoC")
    ap.add_argument("--audio", type=pathlib.Path, help="Audio file (mp3/egg/wav)")
    ap.add_argument("--raw-zip", type=pathlib.Path,
                    help="Beat Saber map zip (contains Song.egg + human .dat); "
                         "used as both audio source and human-map reference")
    ap.add_argument("--human-map", type=pathlib.Path,
                    help="Human map zip/dir for validation (b) (if separate from audio)")
    ap.add_argument("--v7-map", type=pathlib.Path,
                    help="V7-generated map zip to quantify the silent-drop miss")
    ap.add_argument("--difficulty", default="ExpertStandard")
    ap.add_argument("--drop-start", type=float, default=12.0)
    ap.add_argument("--drop-end", type=float, default=16.0)
    ap.add_argument("--salience-tau", type=float, default=0.10,
                    help="other-stem salience gate (fraction of max amplitude)")
    ap.add_argument("--chord-merge-ms", type=float, default=40.0,
                    help="other-stem chord-merge window")
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tol = args.tolerance_ms / 1000.0

    # Resolve audio + optional human map from --raw-zip.
    audio_path = args.audio
    human_map = args.human_map
    tmp_extract = None
    if args.raw_zip is not None:
        tmp_extract = pathlib.Path(tempfile.mkdtemp(prefix="v8poc_raw_"))
        with zipfile.ZipFile(args.raw_zip) as zf:
            zf.extractall(tmp_extract)
        egg = next(tmp_extract.glob("*.egg"), None) or next(tmp_extract.glob("*.ogg"), None)
        if egg is None:
            raise FileNotFoundError(f"No Song.egg in {args.raw_zip}")
        audio_path = egg
        human_map = human_map or tmp_extract
    if audio_path is None:
        ap.error("Provide --audio or --raw-zip")

    # 1. Separate.
    stems, sr = separate_stems(audio_path)
    duration = max(len(v) for v in stems.values()) / sr
    log.info("Duration %.1fs at %d Hz. stems=%s", duration, sr, list(stems))

    # 2. Transcribe.
    events: list[NoteEvent] = []
    for stem in PITCHED_STEMS:
        if stem not in stems:
            continue
        tau = args.salience_tau if stem == "other" else 0.0
        merge = args.chord_merge_ms if stem == "other" else 0.0
        evs = transcribe_pitched(stems[stem], sr, stem, salience_tau=tau, chord_merge_ms=merge)
        log.info("  %-7s basic-pitch -> %4d events", stem, len(evs))
        events.extend(evs)
    if DRUM_STEM in stems:
        drum_evs = transcribe_drums(stems[DRUM_STEM], sr)
        log.info("  %-7s multi-band  -> %4d events", DRUM_STEM, len(drum_evs))
        events.extend(drum_evs)
    events.sort(key=lambda e: e.onset_sec)
    log.info("Total NoteEvents: %d (%.2f events/sec)", len(events), len(events) / duration)

    onset_times = np.array([e.onset_sec for e in events], dtype=np.float64)
    onset_times = np.unique(np.round(onset_times, 4))  # dedupe exact-equal across stems

    # Human/V7 note times.
    human_times = None
    if human_map is not None:
        human_times, _ = map_note_times(pathlib.Path(human_map), args.difficulty)
        log.info("Human map: %d notes", len(human_times))
    v7_times = None
    if args.v7_map is not None:
        v7_times, _ = map_note_times(args.v7_map, args.difficulty if args.human_map else "ExpertPlus")
        log.info("V7 map: %d notes", len(v7_times))

    # ---- Validation (a): drop cluster ----
    ds, de = args.drop_start, args.drop_end
    win = de - ds
    n_trans_drop = int(((onset_times >= ds) & (onset_times < de)).sum())
    trans_rate = n_trans_drop / win
    global_rate = len(onset_times) / duration
    drop_a = {
        "drop_window": [ds, de],
        "transcribed_onsets_in_drop": n_trans_drop,
        "transcribed_drop_rate_per_sec": round(trans_rate, 3),
        "transcribed_global_rate_per_sec": round(global_rate, 3),
        "drop_vs_global_density_ratio": round(trans_rate / (global_rate + 1e-9), 2),
    }
    if v7_times is not None:
        n_v7_drop = int(((v7_times >= ds) & (v7_times < de)).sum())
        drop_a["v7_notes_in_drop"] = n_v7_drop
        drop_a["v7_global_notes"] = int(len(v7_times))
    log.info("(a) drop %.0f-%.0fs: transcribed=%d (%.1f/s) vs global %.1f/s%s",
             ds, de, n_trans_drop, trans_rate, global_rate,
             f"  | V7 notes in drop={drop_a.get('v7_notes_in_drop','?')}" if v7_times is not None else "")

    # ---- Validation (b): alignment vs human map ----
    align_b = None
    if human_times is not None and len(human_times):
        trans_vs_human = alignment_f1(onset_times, human_times, tol)
        libro = librosa_onsets(
            stems.get("other", next(iter(stems.values()))), sr
        )
        libro_drum = librosa_onsets(stems.get("drums", next(iter(stems.values()))), sr)
        libro_union = np.unique(np.concatenate([libro, libro_drum]))
        libro_vs_human = alignment_f1(libro_union, human_times, tol)
        align_b = {
            "transcribed_vs_human": dataclasses.asdict(trans_vs_human),
            "librosa_union_vs_human": dataclasses.asdict(libro_vs_human),
            "baseline_v7_generated_f1": 0.41,
        }
        log.info("(b) transcribed->human F1=%.4f  | librosa->human F1=%.4f  | V7-gen baseline 0.41",
                 trans_vs_human.f1, libro_vs_human.f1)

    # ---- Validation (c): lead contour plot ----
    plot_lead_contour(events, duration, args.out_dir / "lead_contour.png")
    plot_pianoroll(events, duration, args.out_dir / "pianoroll.png",
                   drop_window=(ds, de), human_times=human_times)

    # Persist everything.
    report = {
        "audio": str(audio_path),
        "duration_sec": duration,
        "sr": sr,
        "n_events": len(events),
        "events_per_sec": round(len(events) / duration, 3),
        "per_stem_counts": {
            s: sum(1 for e in events if e.stem == s)
            for s in sorted({e.stem for e in events})
        },
        "validation_a_drop_cluster": drop_a,
        "validation_b_alignment": align_b,
        "params": {
            "salience_tau": args.salience_tau,
            "chord_merge_ms": args.chord_merge_ms,
            "tolerance_ms": args.tolerance_ms,
        },
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    np.save(args.out_dir / "onset_times.npy", onset_times)
    (args.out_dir / "note_events.json").write_text(
        json.dumps([e.as_dict() for e in events], indent=1)
    )
    log.info("Wrote %s", args.out_dir / "report.json")

    # ---- Gate verdict (printed, also in report) ----
    print("\n" + "=" * 70)
    print(f"V8-0 PoC verdict — {args.out_dir.name}")
    print("=" * 70)
    pass_a = drop_a["drop_vs_global_density_ratio"] >= 1.0 and n_trans_drop >= 5
    if v7_times is not None:
        pass_a = pass_a and drop_a["v7_notes_in_drop"] <= 2
    print(f"(a) drop cluster:   {'PASS' if pass_a else 'FAIL'}  "
          f"({n_trans_drop} onsets in {ds:.0f}-{de:.0f}s, "
          f"{drop_a['drop_vs_global_density_ratio']}x global density"
          + (f", V7 placed {drop_a['v7_notes_in_drop']} there)" if v7_times is not None else ")"))
    if align_b is not None:
        f1 = align_b["transcribed_vs_human"]["f1"]
        lib = align_b["librosa_union_vs_human"]["f1"]
        pass_b = f1 > 0.41 and f1 >= lib
        print(f"(b) alignment F1:   {'PASS' if pass_b else 'FAIL'}  "
              f"(transcribed->human {f1:.3f} vs baseline 0.41 and librosa {lib:.3f})")
    else:
        pass_b = None
        print("(b) alignment F1:   SKIPPED (no --human-map / --raw-zip)")
    print("(c) lead contour:   see lead_contour.png (eyeball melody-tracking)")
    report["verdict"] = {"pass_a": pass_a, "pass_b": pass_b}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
