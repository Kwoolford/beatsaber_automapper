# Beat Saber Map Format Reference

> Analysis of the 14,492-map dataset downloaded from BeatSaver (post-2022, Standard
> characteristic, ≥80% rating, no AI maps). Last updated: 2026-02-20.

---

## Overview

| Category | Count | Info.dat version | Difficulty .dat version |
|----------|-------|-----------------|------------------------|
| vanilla | 10,432 | v2.0.0 | v2.2.0 |
| chroma | 3,122 | v2.0.0 | v2.2.0 |
| noodle | 777 | v2.0.0 | v2.2.0 |
| mapping_extensions | 112 | v2.0.0 | v2.2.0 |
| vivify | 49 | v2.1.0 | **v3.2.0 / v3.3.0** |
| **Total** | **14,492** | | |

**The overwhelming majority of community maps use v2 format.** Only the 49 vivify maps use v3
difficulty files. Our pipeline must parse v2.

---

## Info.dat Format (All Categories)

Info.dat is always v2.0.0 or v2.1.0. Uses underscore-prefixed keys.

```json
{
  "_version": "2.0.0",
  "_songName": "Song Title",
  "_songSubName": "",
  "_songAuthorName": "Artist",
  "_levelAuthorName": "Mapper",
  "_beatsPerMinute": 135,
  "_shuffle": 0,
  "_shufflePeriod": 0.5,
  "_previewStartTime": 12,
  "_previewDuration": 10,
  "_songFilename": "Song.egg",
  "_coverImageFilename": "cover.jpg",
  "_environmentName": "DefaultEnvironment",
  "_songTimeOffset": 0,
  "_difficultyBeatmapSets": [
    {
      "_beatmapCharacteristicName": "Standard",
      "_difficultyBeatmaps": [
        {
          "_difficulty": "Expert",
          "_difficultyRank": 7,
          "_beatmapFilename": "ExpertStandard.dat",
          "_noteJumpMovementSpeed": 16,
          "_noteJumpStartBeatOffset": 0
        }
      ]
    }
  ]
}
```

### Key observations
- `_songFilename` almost always has `.egg` extension (OGG Vorbis, renamed). Soundfile reads it transparently.
- `_songTimeOffset` is 0 in virtually all maps in our dataset.
- Multiple `_beatmapCharacteristicName` values appear: `Standard`, `OneSaber`, `360Degree`,
  `90Degree`, `Lightshow`, `Lawless`. **We only care about `Standard`.**
- `_difficultyRank`: Easy=1, Normal=3, Hard=5, Expert=7, ExpertPlus=9.

---

## V2 Difficulty Format (99.7% of dataset)

File version: `"_version": "2.2.0"`. All keys are underscore-prefixed.

```json
{
  "_version": "2.2.0",
  "_customData": { ... },
  "_events": [ ... ],
  "_notes": [ ... ],
  "_obstacles": [ ... ],
  "_waypoints": []
}
```

### V2 Notes (`_notes`)

Bombs are embedded in the notes array as `_type: 3`.

```json
{"_time": 12.0, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1}
```

| Field | Description | Values |
|-------|-------------|--------|
| `_time` | Beat position | float |
| `_lineIndex` | Column | 0–3 (can be out-of-range in mapping_extensions) |
| `_lineLayer` | Row | 0–2 (can be out-of-range in mapping_extensions) |
| `_type` | Note type | 0=red, 1=blue, **3=bomb** (no type 2) |
| `_cutDirection` | Cut direction | 0=up, 1=down, 2=left, 3=right, 4=upLeft, 5=upRight, 6=downLeft, 7=downRight, 8=any |

**V2→V3 mapping:**
- `_type 0/1` → `colorNotes` with `color = _type`
- `_type 3` → `bombNotes` (no color/direction fields)
- `_lineIndex` → `x`, `_lineLayer` → `y`

### V2 Obstacles (`_obstacles`)

No `_lineLayer` or `_height`; those are derived from `_type`.

```json
{"_time": 4.0, "_lineIndex": 0, "_type": 1, "_duration": 0.03, "_width": 1}
```

| Field | Description |
|-------|-------------|
| `_time` | Beat position |
| `_lineIndex` | Starting column (can be negative in mapping_extensions) |
| `_type` | 0=full-height wall (y=0, h=5), 1=crouch wall (y=2, h=3) |
| `_duration` | Beat duration |
| `_width` | Column width |

### V2 Events (`_events`)

```json
{"_time": 4.0, "_type": 4, "_value": 5}
```

| Field | Description |
|-------|-------------|
| `_time` | Beat position |
| `_type` | Event type (0–14, same as v3) |
| `_value` | Integer value (0–7) |

No `_floatValue` in v2 — defaults to 1.0 in our parser.

**No arcs (`sliders`) or chains (`burstSliders`) in v2.** These are v3-only features.

### V2 BPM Changes (`_customData._BPMChanges`)

**49% of maps** have BPM changes in `_customData`. Critical for correct beat→frame alignment.

```json
"_customData": {
  "_BPMChanges": [
    {"_BPM": 67.5, "_time": 44, "_beatsPerBar": 4, "_metronomeOffset": 4}
  ]
}
```

| Field | Description |
|-------|-------------|
| `_BPM` | New BPM value after this change |
| `_time` | Beat position (in base BPM beats) where this change takes effect |
| `_beatsPerBar` | Metadata only, not needed for timing |
| `_metronomeOffset` | Metadata only, not needed for timing |

**Without handling BPM changes, beat→frame conversion will drift for half the dataset.**

### Other `_customData` Fields (non-mod)

| Field | Meaning | Safe to ignore? |
|-------|---------|-----------------|
| `_bookmarks` | Editor bookmarks | Yes |
| `_time` | Editor cursor position | Yes |
| `_BPMChanges` | BPM change events | **No — affects timing** |
| `_environment` | Custom environment | Yes |

---

## Mod-Specific Extensions

### Vanilla

Standard v2 format with no per-note `_customData`. BPM changes may still be present. No special
handling needed beyond v2 parsing and BPM change support.

### Chroma

Adds color and lighting customization. Chroma data appears primarily on **events**, not notes.

**Per-event custom data:**
```json
{
  "_time": 4.0, "_type": 4, "_value": 5,
  "_customData": {
    "_color": [1.0, 0.718, 0.773, 1.0],
    "_lightID": [1, 2]
  }
}
```

**Per-note custom data (less common):**
```json
{"_customData": {"_color": [1.0, 0.0, 0.0, 1.0]}}
```

**Impact on pipeline:** None. We ignore `_customData` on notes and events. Chroma notes have
standard `_lineIndex`, `_lineLayer`, `_type`, `_cutDirection` — parse identically to vanilla.

### Noodle Extensions

Adds per-note/obstacle animation, custom positions, fake notes, and precision placement.

**Fake notes (must be skipped):**
```json
{"_customData": {"_fake": true, "_interactable": false}}
```
Fake notes are decorative only — not gameplay. **Including them corrupts onset labels.**

**Custom position notes:**
```json
{"_customData": {"_position": [0.5, 1.2], "_rotation": 45}}
```
We use the standard `_lineIndex/_lineLayer` for position. Custom `_position` is sub-grid precision
that our 4×3 grid tokenizer cannot represent — ignore it.

**Animated obstacles:**
Obstacles may have `_animation`, `_scale`, `_localRotation`, `_interactable: false` — these are
visual only, not gameplay. Parse the standard position/duration fields and ignore `_customData`.

**Impact on pipeline:**
- Skip fake notes (`_customData._fake == True`)
- Ignore custom positions, animations, rotation
- Standard note grid fields remain intact

### Mapping Extensions

Extends the note grid beyond 4 columns × 3 rows.

**Out-of-bounds notes:**
```json
{"_lineIndex": -1, "_lineLayer": 0, "_type": 0, "_cutDirection": 6}
{"_lineIndex": 4, "_lineLayer": 3, ...}
```

Negative and > 3 column indices, row > 2 are common. Our tokenizer only handles 0–3 × 0–2.

**Impact on pipeline:** Clamp OOB coordinates: `x = clamp(x, 0, 3)`, `y = clamp(y, 0, 2)`.
This loses precision but keeps the note in the vocab. Mapping extensions maps may produce
slightly distorted patterns. Consider excluding them during training (small dataset: 112 maps).

**OOB obstacles:**
```json
{"_lineIndex": -1, "_type": 1, "_width": 1}
```
Same clamping applies to obstacle `_lineIndex`.

### Vivify

Uses v3 difficulty format (3.2.0 / 3.3.0) — the only category with v3 diffs.
Info.dat is still v2.1.0.

```json
{
  "version": "3.3.0",
  "colorNotes": [{"b": 36, "x": 2, "y": 0, "c": 1, "d": 1, "a": 0, "customData": {...}}],
  "bombNotes": [...],
  "burstSliders": [...],
  "basicBeatmapEvents": [...],
  "bpmEvents": [{"b": 0, "m": 128}],
  "customData": {
    "customEvents": [...],
    "materials": {...},
    "environment": [...]
  }
}
```

**V3 BPM events use `bpmEvents` array** (not `_customData._BPMChanges`):
```json
{"b": 0, "m": 128}
```

**Per-note customData (Vivify/Noodle v3 style):**
```json
{
  "customData": {
    "disableNoteGravity": true,
    "spawnEffect": false,
    "coordinates": [0.5, 0],
    "noteJumpMovementSpeed": 0.0001,
    "track": ["floatingNote_Z1"]
  }
}
```

**Impact on pipeline:** Vivify maps are only 49 in our dataset. Our v3 parser already handles
them. Per-note `customData` is ignored. `bpmEvents` should be parsed for timing (similar to
v2 BPM changes). `lightTranslationEventBoxGroups` is a v3.3 addition our lighting tokenizer
does not handle — these events are silently skipped.

---

## Audio Format

| Extension | Format | Prevalence | Soundfile support |
|-----------|--------|------------|-------------------|
| `.egg` | OGG Vorbis (renamed) | ~99.8% | ✅ Reads transparently |
| `.wav` | PCM WAV | ~0.2% | ✅ Native |
| `.ogg` | OGG Vorbis | Rare | ✅ Native |

`.egg` is the Beat Saber native audio format — OGG Vorbis with a renamed extension.
Soundfile identifies by magic bytes, not extension. No special handling required.

---

## Filtering Decisions for Training

| Category | Include? | Reason |
|----------|----------|--------|
| vanilla | ✅ Yes | Clean data, largest category |
| chroma | ✅ Yes | Identical note format to vanilla |
| noodle | ✅ Yes (with fake note filter) | Valid note data, just skip fakes |
| mapping_extensions | ⚠️ Optional | OOB coords clamped; 112 maps is small anyway |
| vivify | ✅ Yes | V3 format, already supported |

**Recommended first run:** `--exclude-categories mapping_extensions` (112 maps, OOB distortion
risk not worth it for a tiny slice of data).

---

## Pipeline Compatibility Summary

| Issue | Severity | Status |
|-------|----------|--------|
| V2 format not parsed | 🔴 Critical | **Fixed** — v2 parser added to `beatmap.py` |
| BPM changes cause frame drift | 🟠 Serious | **Fixed** — variable BPM `beat_to_frame` in `audio.py` |
| Non-Standard characteristics included | 🟠 Serious | **Fixed** — `preprocess.py` filters to Standard |
| Mapping extensions OOB coords | 🟡 Moderate | **Fixed** — clamped in v2 parser |
| Noodle fake notes included | 🟡 Moderate | **Fixed** — skipped in v2 parser |
| `.egg` audio format | ✅ OK | Soundfile reads it transparently |
| `_songTimeOffset` non-zero | ✅ OK | Always 0 in dataset |
| V3 `bpmEvents` in vivify | 🟡 Moderate | Parsed (trivially: index 0 = base BPM) |
