# V8 Architecture Plan: Symbolic Per-Instrument Transcription Backbone

**Status:** DESIGN — not yet built. Supersedes the V7 inference representation.
**Author of finding:** user-flagged 2026-06-02 ("the way we digest the song is wrong"),
root-caused in code the same day.
**One-line thesis:** V7's note timing is decided by an audio-blind section-threshold gate
sitting on top of BPM-blurred MERT embeddings, with no per-instrument structure. V8 replaces
the *input representation* with an explicit symbolic note-event layer (a piano-roll / "DAW
view"): per-stem onset+pitch transcription drives WHEN and conditions WHAT. MERT is retained as
a secondary timbre/energy feature, not the timing backbone.

---

## Why V7 Produces Atrocious Maps (Root Cause, Code-Confirmed)

Three independent "false-signal" layers, all on the input side. Each is confirmed in the
current code, not hypothesized.

### Layer 1 — Timing is audio-blind (the silent-drop bug)
`generation/generate.py::generate_v7_level` computes Stage 1 onset probabilities, then
**overrides them** with six hand-tuned per-section thresholds:

```python
_SECTION_THRESHOLDS = {"drop":0.38,"chorus":0.44,"verse":0.52,
                       "bridge":0.58,"intro":0.68,"outro":0.72}
```

Sections come from `detect_sections_energy_percentile`, which snaps boundaries to round-second
windows. On the test song (BPM 123) it produced `intro 0–16s, drop 16–32s, …`. The real drop
hits ~13s — inside the "intro" window — so it is gated at **0.68** and silenced. Every song
with a pre-16s drop hits this. **One mislabeled boundary = a silent drop.** The "pause at drop"
reported in May was never fixed; it moved from the clustering detector to the energy detector.

### Layer 2 — Onsets are blurred into a BPM grid
`data/mert_encoder.py::pool_to_beat_grid` mean-pools the 75 Hz MERT frames into one 768-vec per
1/4-note slot (`frames_per_slot = 75 * 60 / bpm / subdiv` ≈ 9 frames at 123 BPM). Individual
note onsets *inside* a slot are averaged away. The grid is anchored to a single global BPM with
**no downbeat/phase lock** — any BPM/phase error drifts the entire song. Training labels
(`data/beat_grid.extract_beat_labels`) live on the same grid, so the whole pipeline is trapped
in BPM-quantized space and can never place a note off-grid even when the music demands it.

### Layer 3 — No per-instrument structure
Demucs yields 4 stems (drums, bass, other, vocals). Stage 2 sees only `other` — lead guitar,
rhythm, synths, and pads blended into one embedding. The layout model has **no individual
instrument line to follow**, so swing directions have nothing to be coherent *with*. Result:
the "diagonal swings with no cohesion, merely for sport" the user describes. The X-column 70%
ceiling is partly mapper subjectivity, but the *incoherence* is this missing melodic anchor.

**Net:** V7 cannot, even in principle, (a) place a note at the true drop time, (b) align notes
to off-grid events, or (c) make directions follow a melody. These are not tuning problems.

---

## V8 Thesis: A Symbolic Note-Event Backbone

Give the generator the same thing a human mapper sees in a DAW: **a piano-roll of discrete note
events per instrument**, with explicit onset times and pitch, and *no* grid restriction on when
a note may be placed. Keep MERT for what it is genuinely good at (timbre, energy, section feel)
as a *secondary* conditioning feature.

### Core representation: the NoteEvent stream
Per song, a list of events:

```
NoteEvent = (onset_sec: float,       # continuous time, NOT a grid slot
             dur_sec:   float,
             pitch:     int | None,  # MIDI pitch; None for unpitched drums
             stem:      enum,        # kick|snare|hat|bass|vocal|lead|other
             salience:  float)       # transcription confidence × energy
```

This is produced offline per stem (see Preprocessing) and is the **single source of truth for
WHEN**. WHAT is conditioned on the local pitch contour of the lead/melodic events.

### What this fixes, mechanically
- **Silent-drop bug → impossible.** The drop is a dense cluster of NoteEvents at t≈13s
  regardless of any section label. Notes are placed *on events*; there is no section gate.
- **Off-grid alignment.** Onset times are continuous; snapping (if any) is to a phase-locked
  1/8 or 1/16 grid, not a global-BPM 1/4 grid.
- **Directional cohesion.** Stage 2 conditions each note's direction on the pitch contour of the
  lead stem around its onset (ascending pitch → up/right, descending → down/left, sustained →
  hold), the actual convention human mappers use.
- **Density self-regulates.** Verse = few events, drop = many. The NPS density-curve and
  adaptive-threshold hacks (`_apply_density_curve`, `_compute_adaptive_threshold`) are deleted.

---

## New Dependencies

```bash
uv pip install basic-pitch        # Spotify polyphonic note transcription (onset+pitch+dur)
uv pip install pretty_midi        # MIDI/event manipulation, piano-roll rendering
# already present: demucs, librosa (drum-band onset detection), transformers (MERT)
```

**basic-pitch** (`spotify/basic-pitch`): lightweight CNN, audio → note events
(onset, offset, pitch, amplitude). Best on monophonic-ish sources — which is exactly why we run
it **per Demucs stem** rather than on the mix. Vocals and bass transcribe cleanly; `other` is
the hard case (mitigation below). **Drums do not need pitch transcription** — multi-band
`librosa.onset` on the drum stem (kick/snare/hat bands) is more robust than forcing a pitch
tracker onto unpitched audio.

---

## Data Flow

### Preprocessing (offline, re-run once on full dataset → new .pt keys, non-destructive)

```
audio.mp3
  │
  ▼ Demucs htdemucs (already in pipeline)
  ├── drums.wav  ─▶ multi-band librosa onset  ─▶ drum events {kick,snare,hat}
  ├── bass.wav   ─▶ basic-pitch               ─▶ bass note events (onset,pitch,dur)
  ├── vocals.wav ─▶ basic-pitch               ─▶ vocal note events
  └── other.wav  ─▶ basic-pitch (+ salience gate) ─▶ lead/other note events
                                                     │
            all merged & sorted by onset ─▶ NoteEvent stream  [E events]
                                                     │
   (retained from V7, secondary) MERT per stem ─▶ pooled energy/timbre features
                                                     │
   stored as new .pt keys:  note_events, lead_contour, + existing *_beat_features
```

`other` mitigation: gate basic-pitch output by `salience > τ` and merge notes within a short
window so distorted-guitar over-transcription collapses to chord onsets rather than a smear.
This is a tunable in the PoC.

### Training labels — alignment to NoteEvents
Today, labels are binary presence on the BPM grid (`extract_beat_labels`). V8 changes the label
*target space*: for each ground-truth Beat Saber note, find the nearest NoteEvent within ±ε ms
and mark that event "selected" + hand + spatial tokens. Notes with no nearby event become
**negative-but-eligible** (the model may still place there, but it's rare). This makes the
WHEN problem "select from real events" instead of "classify every grid slot," collapsing a
huge negative class and removing the metronome failure mode.

### Stage 1 (V8) — Event Selector + Hand Assigner
- Input per event: `[basic-pitch features, stem one-hot, salience, local MERT energy, diff]`
- Output per event: `P(note)`, `P(hand=L)`, `P(hand=R)`
- Sequence model over the event stream (events are sparse — typically 2–6/sec — so the full
  song fits without windowing). Replaces the grid-classifier BeatClassifier.

### Stage 2 (V8) — Layout, conditioned on pitch contour
- Largely the existing `LayoutPhraseModel` encoder-decoder, BUT the per-note conditioning gains
  the **lead pitch contour** around the onset (Δpitch over the preceding/following events).
- Direction/X/Y tokens learn to follow the contour → cohesion. KIND/Y/FIELD_D unchanged.
- PhraseIndex retrieval keyed on contour-segment similarity rather than mean-MERT — repeats of
  the same riff now match on melody, not timbre average.

### Inference
```
audio → Demucs → per-stem transcription → NoteEvent stream
      → Stage 1 selects events + hands  (NO section-threshold gate)
      → Stage 2 lays out spatial tokens, contour-conditioned
      → existing swing-event assembly → postprocess → export
```
Deleted from `generate_v7_level`: `_SECTION_THRESHOLDS`, the per-slot threshold vector,
`_apply_density_curve`, `_compute_adaptive_threshold`, and the whole `detect_sections_*` →
threshold path. Sections may still be computed for *lighting*, not for note gating.

---

## Orthogonal Data-Quality Fix (do regardless of V8)

User-confirmed: Expert+ maps teach ergonomically hard "for-sport" swings. Filter the training
cohort to **Expert only, or all difficulties capped at NPS 4–8**. This is independent of the
representation change and is cheap — a cohort filter in the dataset builder. Recommend doing it
in the V8 retrain anyway so the model isn't learning Expert+ pattern density.

---

## File Map

### New
- `data/transcribe.py` — per-stem basic-pitch + drum-band onset → `NoteEvent` list; salience
  gate + chord-merge for `other`.
- `data/note_events.py` — `NoteEvent` dataclass, serialization to/from .pt, piano-roll render.
- `data/event_dataset.py` — Stage 1 V8 dataset (event stream + selected/hand/spatial labels via
  nearest-event matching to ground-truth notes).
- `models/event_selector.py` — Stage 1 V8 (sequence over events).
- `docs/architecture_v8_plan.md` — this file.

### Modified
- `data/preprocess*.py` — add transcription pass, write `note_events` + `lead_contour` keys.
- `data/layout_dataset.py` — add pitch-contour conditioning channel.
- `models/layout_model.py` — accept contour conditioning.
- `generation/generate.py` — new `generate_v8_level`; delete section-threshold gating.
- `generation/phrase_index.py` — contour-segment keys.

### Kept unchanged
- Demucs separation, MERT extraction (now secondary), swing-event grammar/tokenizer,
  postprocessor, lighting rules, export, Lightning/training infra, leaderboard + V7 harness
  (a `runner_v8` mirrors `runner_v7`).

### Retired
- `_SECTION_THRESHOLDS`, `_apply_density_curve`, `_compute_adaptive_threshold`,
  `detect_sections_*`-as-note-gate, `BeatClassifier` (replaced by `event_selector`), the
  BPM-grid label path (`extract_beat_labels`) for Stage 1.

---

## Staging / Iteration Plan

| Phase | Goal | DoD |
|-------|------|-----|
| **V8-0 PoC (de-risk)** | Install basic-pitch; transcribe SO TIRED ROCK stems. | (a) Drop @13–15s yields a dense onset cluster V7 misses; (b) transcribed-onset→human-map alignment F1 **beats current 0.41**; (c) lead-stem contour visibly tracks melody. **Green-light gate.** |
| V8-1 | `transcribe.py` + `note_events.py`; batch-transcribe full dataset → new .pt keys. | 5320 songs transcribed; events sane (median 2–6/sec); stored. |
| V8-2 | `event_dataset.py` — nearest-event label matching. | ≥X% of GT notes match an event within ±ε; report the unmatched residual. |
| V8-3 | Stage 1 `event_selector` train. | Event-selection F1 reported; sanity: drop sections dense, breakdowns sparse. |
| V8-4 | Stage 2 contour conditioning. | Directional-cohesion metric (contour-follow rate) up vs V7. |
| V8-5 | `generate_v8_level` end-to-end; ArcViewer. | Human play: drop has notes, swings cohere, NPS in band. The real DoD. |

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Polyphonic transcription noisy on distorted/dense audio → new false signals | **High** | Transcribe per-stem (cleaner sources); salience gate + chord-merge on `other`; keep MERT energy as backstop; **V8-0 PoC measures this before any rebuild**. |
| basic-pitch misses fast drum-adjacent melodic onsets | Med | Drums use librosa onset, not basic-pitch; union of stems covers gaps. |
| Nearest-event label matching drops GT notes with no event | Med | Report unmatched residual in V8-2; widen ε or add a fallback grid-candidate channel if residual is large. |
| Doesn't fix WHAT-subjectivity (X 70% / beat 0.60 ceilings) | Low (expected) | V8 targets timing + cohesion, which are objective failures; subjectivity ceiling is out of scope and accepted. |
| Big rebuild, weeks of work | Med | V8-0 gate prevents committing on a hunch; phases are independently shippable; V7 harness reused. |

---

## Open Questions for the build

- Snap NoteEvents to a phase-locked 1/8 grid, or keep fully continuous and let postprocess
  quantize? (PoC should look at human maps' off-grid fraction.)
- Is `bass`+`drums` enough for WHEN and `lead`(from `other`) enough for WHAT, i.e. do we even
  need vocals events for non-vocal-led genres? (Genre-dependent; make it a weighting.)
- Contour conditioning: relative Δpitch sequence vs absolute MIDI — relative is key/transpose
  invariant and almost certainly the right call.
