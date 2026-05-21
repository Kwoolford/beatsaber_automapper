"""V7-0 proof of concept: Demucs → MERT → sklearn beat classifier.

Run this before building the full V7 pipeline. Validates:
  1. Demucs separates the test song cleanly
  2. MERT-v1-95M produces frame-level embeddings (shape, framerate)
  3. A trivial sklearn logistic regression on drum MERT features achieves
     F1 > 0.70 on beat slot prediction for at least one held-out processed song

Usage:
    python scripts/v7_poc.py

Downloads MERT model (~400 MB) to HuggingFace cache on first run.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).parent.parent
TEST_SONG  = REPO_ROOT / "data/test_songs/SO TIRED ROCK - NUEKI.mp3"
DATA_DIR   = REPO_ROOT / "data/processed"
SAMPLE_RATE = 44100
MERT_SR     = 24000   # MERT expects 24 kHz
MERT_HZ     = 75      # MERT output frame rate
BEAT_SUBDIV = 4       # 1/4-note slots per beat

# ---------------------------------------------------------------------------
# Step 1 — Demucs separation
# ---------------------------------------------------------------------------

def separate_stems(audio_path: pathlib.Path, device: str = "cuda") -> tuple[dict[str, torch.Tensor], int]:
    """Run htdemucs on audio_path; return (stems dict, sample_rate).

    stems values are [2, T] stereo tensors at the model's native sample rate.
    """
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    log.info("Loading Demucs htdemucs …")
    model = get_model("htdemucs")
    model.to(device)
    model.eval()
    sr = model.samplerate  # 44100

    log.info("Loading audio %s …", audio_path.name)
    from beatsaber_automapper.data.audio import load_audio
    wav, file_sr = load_audio(audio_path, target_sr=sr)  # returns [C, T] at sr
    if file_sr != sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # mono → stereo
    wav = wav.to(device)

    log.info("Separating …")
    with torch.no_grad():
        sources = apply_model(model, wav.unsqueeze(0), device=device, progress=True)
    # sources: [1, 4, 2, T]  in order: model.sources = [drums, bass, other, vocals]
    sources = sources.squeeze(0).cpu()  # [4, 2, T]
    stems = {name: sources[i] for i, name in enumerate(model.sources)}

    log.info("Demucs done. stems: %s, sr=%d, shapes: %s",
             list(stems.keys()), sr, {k: tuple(v.shape) for k, v in stems.items()})
    return stems, sr


# ---------------------------------------------------------------------------
# Step 2 — MERT feature extraction
# ---------------------------------------------------------------------------

def extract_mert_features(
    audio: torch.Tensor,
    in_sr: int,
    device: str = "cuda",
    layer: int = -1,
) -> torch.Tensor:
    """Encode audio with MERT-v1-95M; return hidden states [T, 768].

    Args:
        audio: waveform [C, T] or [T] at in_sr Hz.
        in_sr: input sample rate.
        layer: which transformer layer to use (-1 = last).
    """
    import torchaudio
    from transformers import Wav2Vec2FeatureExtractor, AutoModel

    model_name = "m-a-p/MERT-v1-95M"
    log.info("Loading MERT feature extractor …")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name, trust_remote_code=True,
    )
    log.info("Loading MERT model …")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    # Resample to MERT_SR (24 kHz)
    if audio.ndim == 2:
        audio = audio.mean(0)  # stereo → mono
    if in_sr != MERT_SR:
        resampler = torchaudio.transforms.Resample(in_sr, MERT_SR)
        audio = resampler(audio)

    audio_np = audio.numpy()
    inputs = processor(audio_np, sampling_rate=MERT_SR, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)

    log.info("Running MERT forward pass …")
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    # outputs.hidden_states: tuple of [1, T, 768] for each layer
    hidden = outputs.hidden_states[layer]  # [1, T, 768]
    features = hidden.squeeze(0).cpu()     # [T, 768]
    log.info("MERT output shape: %s  (~%.1f Hz)", tuple(features.shape),
             features.shape[0] / (audio.shape[-1] / MERT_SR))
    return features


# ---------------------------------------------------------------------------
# Step 3 — Beat-grid pooling
# ---------------------------------------------------------------------------

def pool_to_beat_grid(
    mert_features: torch.Tensor,
    bpm: float,
    total_beats: float,
    subdiv: int = BEAT_SUBDIV,
) -> torch.Tensor:
    """Pool MERT frame features to 1/subdiv-note beat slots.

    Returns: [N_slots, 768]
    """
    mert_hz = MERT_HZ
    frames_per_slot = mert_hz * 60.0 / bpm / subdiv
    n_slots = int(total_beats * subdiv)
    T = mert_features.shape[0]
    d = mert_features.shape[1]

    beat_features = torch.zeros(n_slots, d)
    for slot in range(n_slots):
        start_frame = int(slot * frames_per_slot)
        end_frame   = min(T, int((slot + 1) * frames_per_slot))
        if end_frame > start_frame:
            beat_features[slot] = mert_features[start_frame:end_frame].mean(0)
        elif start_frame < T:
            beat_features[slot] = mert_features[start_frame]

    log.info("Beat grid: %d slots @ 1/%d note (%.1f frames/slot)",
             n_slots, subdiv, frames_per_slot)
    return beat_features


# ---------------------------------------------------------------------------
# Step 4 — Beat labels from swing_tokens
# ---------------------------------------------------------------------------

def extract_beat_labels(
    swing_tokens: list[int],
    bpm: float,
    n_slots: int,
    subdiv: int = BEAT_SUBDIV,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive binary left/right note presence per 1/subdiv beat slot.

    Returns: (left_labels [n_slots], right_labels [n_slots]) — binary int arrays.
    """
    from beatsaber_automapper.data.swing_tokenizer import (
        HAND_LEFT, HAND_RIGHT, HAND_NONE,
        DT_BASE, DT_COUNT, _DT_BINS,
        EOS, BOS, PAD,
    )

    left  = np.zeros(n_slots, dtype=np.int32)
    right = np.zeros(n_slots, dtype=np.int32)

    tokens = swing_tokens
    i, current_beat = 0, 0.0
    while i < len(tokens):
        tok = tokens[i]
        if tok in (PAD, BOS):
            i += 1; continue
        if tok == EOS:
            break
        if tok not in (HAND_LEFT, HAND_RIGHT, HAND_NONE):
            i += 1; continue
        hand = tok
        if i + 1 >= len(tokens):
            break
        dt_tok = tokens[i + 1]
        if not (DT_BASE <= dt_tok < DT_BASE + DT_COUNT):
            i += 1; continue
        dt = _DT_BINS[dt_tok - DT_BASE]
        current_beat += dt
        slot = int(round(current_beat * subdiv))
        if 0 <= slot < n_slots:
            if hand == HAND_LEFT:
                left[slot] = 1
            elif hand == HAND_RIGHT:
                right[slot] = 1
        # Skip rest of event tokens (at least 3 more: KIND X Y …)
        i += 2
        while i < len(tokens) and tokens[i] not in (HAND_LEFT, HAND_RIGHT, HAND_NONE, EOS, BOS, PAD):
            i += 1

    n_left  = left.sum()
    n_right = right.sum()
    log.info("Beat labels: %d left notes, %d right notes / %d slots",
             n_left, n_right, n_slots)
    return left, right


# ---------------------------------------------------------------------------
# Step 5 — sklearn classifier PoC
# ---------------------------------------------------------------------------

def run_classifier_poc(
    beat_features: np.ndarray,
    left_labels:   np.ndarray,
    right_labels:  np.ndarray,
) -> dict[str, float]:
    """Train / eval logistic regression on a single song (80/20 split).

    Returns dict with f1_left, f1_right, precision_left, recall_left, …
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.preprocessing import StandardScaler

    n = len(beat_features)
    split = int(n * 0.8)

    X_train, X_test = beat_features[:split], beat_features[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = {}
    for hand, labels in [("left", left_labels), ("right", right_labels)]:
        y_train, y_test = labels[:split], labels[split:]
        pos = y_train.sum()
        neg = (y_train == 0).sum()
        cw  = {0: 1.0, 1: neg / max(pos, 1)}

        clf = LogisticRegression(class_weight=cw, max_iter=500, C=1.0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        log.info("\n=== %s hand ===\n%s", hand,
                 classification_report(y_test, y_pred, zero_division=0))
        results[f"f1_{hand}"] = float(f1)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ---- 1. Demucs on test song ----
    if not TEST_SONG.exists():
        log.error("Test song not found: %s", TEST_SONG)
        sys.exit(1)

    stems, stem_sr = separate_stems(TEST_SONG, device=device)

    drum_audio  = stems["drums"]   # [2, T]
    mix_audio   = stems["other"]   # melody stem (no drums/bass)

    # ---- 2. MERT on drum stem ----
    log.info("Extracting MERT features from drum stem …")
    drum_features = extract_mert_features(drum_audio, stem_sr, device=device)

    # ---- 3. Beat grid ----
    # Test song: 123 BPM, ~361 beats
    bpm = 123.0
    duration_secs = drum_audio.shape[-1] / stem_sr
    total_beats   = duration_secs * bpm / 60.0
    n_slots       = int(total_beats * BEAT_SUBDIV)

    beat_feats = pool_to_beat_grid(drum_features, bpm, total_beats)
    log.info("Beat features shape: %s", tuple(beat_feats.shape))

    # ---- 4. Labels from a processed .pt file that matches this song ----
    # Fall back to the first available .pt file with Expert swing_tokens;
    # we only need it to validate that the label extraction and classifier work.
    log.info("Loading a processed .pt for label extraction …")
    pt_files = sorted(DATA_DIR.glob("*.pt"))
    target_pt = None
    for pf in pt_files:
        d = torch.load(pf, weights_only=False)
        diffs = d.get("difficulties", {})
        if "Expert" in diffs and diffs["Expert"].get("swing_tokens"):
            target_pt = pf
            target_data = d
            break

    if target_pt is None:
        log.error("No processed .pt with Expert swing_tokens found in %s", DATA_DIR)
        sys.exit(1)

    # ---- Find matching audio for the same song (same-song eval) ----
    song_id = target_pt.stem
    raw_zip = REPO_ROOT / "data/raw" / f"{song_id}.zip"
    song_audio_path = None

    if raw_zip.exists():
        import zipfile
        with zipfile.ZipFile(raw_zip) as zf:
            audio_names = [f for f in zf.namelist()
                           if f.lower().endswith((".mp3", ".ogg", ".wav", ".egg"))]
        if audio_names:
            # Extract to a temp file for loading
            import tempfile, os
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(raw_zip) as zf:
                extracted = zf.extract(audio_names[0], tmpdir)
            song_audio_path = pathlib.Path(extracted)
            log.info("Using same-song audio: %s (from %s)", audio_names[0], raw_zip.name)

    if song_audio_path is None:
        log.warning("No raw audio found for %s — falling back to cross-song eval", song_id)
        song_audio_path = TEST_SONG

    # ---- 2b. Demucs + MERT on the matched song ----
    log.info("Separating matched song for same-song evaluation …")
    song_stems, song_sr = separate_stems(song_audio_path, device=device)
    log.info("Extracting MERT features from matched song drum stem …")
    song_drum_feats = extract_mert_features(song_stems["drums"], song_sr, device=device)

    pt_bpm   = float(target_data.get("bpm", 120.0))
    pt_mel   = target_data["mel_spectrogram"]
    pt_dur   = pt_mel.shape[1] / (SAMPLE_RATE / 512)
    pt_beats = pt_dur * pt_bpm / 60.0
    pt_slots = int(pt_beats * BEAT_SUBDIV)

    log.info("Using %s (bpm=%.1f, dur=%.1fs, %d slots)", target_pt.name,
             pt_bpm, pt_dur, pt_slots)

    beat_feats_song = pool_to_beat_grid(song_drum_feats, pt_bpm, pt_beats)

    swing_tokens = target_data["difficulties"]["Expert"]["swing_tokens"]
    left_labels, right_labels = extract_beat_labels(swing_tokens, pt_bpm, pt_slots)

    min_len = min(len(beat_feats_song), pt_slots)
    X = beat_feats_song[:min_len].numpy()
    y_left  = left_labels[:min_len]
    y_right = right_labels[:min_len]

    # ---- 5. Sklearn PoC ----
    log.info("Running sklearn logistic regression PoC (SAME-SONG) …")
    results = run_classifier_poc(X, y_left, y_right)

    # ---- Summary ----
    print("\n" + "="*60)
    print("V7-0 PROOF OF CONCEPT RESULTS")
    print("="*60)
    print(f"MERT drum features shape:  {tuple(song_drum_feats.shape)}")
    print(f"Test song beat slots:      {n_slots}  @ {BEAT_SUBDIV} subdiv")
    print(f"Eval song:                 {song_id}  (bpm={pt_bpm}, {pt_slots} slots)")
    print(f"F1 left  hand (logreg):   {results['f1_left']:.3f}")
    print(f"F1 right hand (logreg):   {results['f1_right']:.3f}")
    avg_f1 = (results["f1_left"] + results["f1_right"]) / 2
    verdict = "PASS ✓" if avg_f1 >= 0.50 else "NEEDS INVESTIGATION"
    print(f"Average F1:                {avg_f1:.3f}  → {verdict}")
    print("="*60)
    print()

    if avg_f1 >= 0.70:
        print("✓  Strong signal. Proceed directly to V7-1.")
    elif avg_f1 >= 0.50:
        print("✓  Usable signal. Proceed to V7-1; tune MERT layer if needed.")
    else:
        print("⚠  Weak signal. Before building V7-1, investigate:")
        print("   - Try a different MERT layer (layer=6 or layer=8)")
        print("   - Check Demucs drum quality for this song")
        print("   - Try using full-mix MERT instead of drum stem")


if __name__ == "__main__":
    main()
