"""Audio alignment — axis A8 of the v2 evaluation suite. **The suite's blind spot.**

Built 2026-08-01, immediately after Kyle played the first two maps in the project's
history to pass all five existing axes and said:

    "It's painfully obvious the notes are off beat. The consistent beat of the song
     is not where the notes are played... many just have their own slightly off
     timings."

He was right, and **the suite was structurally incapable of seeing it: not one of
the five existing axes ever loads the audio.** `rhythm.py` — the axis whose whole
job is timing — scores note times against the map's DECLARED BPM GRID. A map can
therefore have a perfectly human interval distribution, human hand roles, human
flow and human difficulty *while sitting off the song's actual beat*, and score
5/5. That is the complete explanation for five different configurations passing
while sharing one obviously audible defect.

The lesson is not that a metric saturated — the audit battery is designed to catch
that. It is that **an axis nobody thought to add is invisible in exactly the same
way a saturated metric is.** The battery checks whether the existing axes
discriminate; it never checked whether the SET of axes was complete.

    measurement (1f767, 50ms tol)   human    ours (best / worst)
    onset_precision                 0.966    0.817 / 0.753
    offset_mad (ms)                   8.0     11.7 / 23.2

A human mapper puts ~97% of notes on a real audio onset; we manage 75-82%. **One
note in five of ours lands where there is no musical event at all**, against one in
thirty for a human, and the notes that do land are 1.5-3x more scattered in time.

Metrics:
  onset_precision  share of our note TIMES (deduplicated, so a double counts once)
                   that land within TOL of a detected audio onset.
                   RULE: a note should mark something that happens in the music.
  offset_mad_ms    MAD of the signed note-to-onset offsets of matched notes — the
                   timing SCATTER. Distinguishes "wrong notes" (low precision) from
                   "right notes, sloppily placed" (high scatter). Kyle heard both.
  onset_lag_ms     median signed offset. A constant lag is a different defect from
                   scatter (it is a sync problem, not a musicality one), so it is
                   measured separately and deliberately kept OUT of the gap.
  onset_recall     share of detected onsets that got a note. **Deliberately NOT in
                   SEQUENCE_KEYS.** Humans ignore most onsets on purpose — that is
                   what makes a map a map and not a transcription — so low recall is
                   not a defect and gating on it would push us toward note spam.
                   It is reported because precision alone is gameable by emitting
                   very few notes; `playfeel.nps` is the real density gate.

Scored like every other axis: cohort median shift + spread against the human
distribution (`_dist`), never per-map distance to the human median. See
`_dist.__doc__` for why that distinction is the difference between a metric that
ranks and one that saturates.

The onsets themselves come from `scripts/build_onset_cache.py` (union over Demucs
stems), so human and generated maps of the same song are scored against
byte-identical references. **That shared footing is the entire point** — the reason
this gap survived so long is that `scripts/eval_alignment.py`'s loader silently
returned 0 notes for human zips, so the control that would have exposed it was
never run.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from dataclasses import dataclass, field

import numpy as np

from beatsaber_automapper.evaluation import _dist

KEYS = ["onset_precision", "offset_mad_ms", "onset_lag_ms", "onset_recall"]
# Precision and scatter only. Recall is excluded on purpose (see module docstring);
# lag is excluded because a constant offset is a sync defect, not a musical one.
SEQUENCE_KEYS = ["onset_precision", "offset_mad_ms"]

# Matching tolerance. 50ms is the value every number in TODO.md was measured at,
# and it is roughly the window inside which a hit reads as "on" the sound.
TOL_S = 0.05

# A map needs enough notes for the medians to mean anything (same guard as A7).
MIN_NOTES = 40

REFERENCE_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "outputs" / "alignment_human_reference.json"
)


@dataclass(slots=True)
class AlignmentReport:
    metrics: dict[str, float] = field(default_factory=dict)
    n_notes: int = 0
    n_onsets: int = 0

    def as_dict(self) -> dict:
        return {"metrics": {k: round(v, 4) for k, v in self.metrics.items()},
                "n_notes": self.n_notes, "n_onsets": self.n_onsets}


def note_times(beatmap, bpm: float) -> list[float]:
    """Distinct note onset times in SECONDS.

    Deduplicated: a double (two hands on the same beat) is ONE musical event, and
    counting it twice would let the 4x-too-many-doubles defect quietly inflate this
    axis.
    """
    if bpm <= 0:
        return []
    spb = 60.0 / bpm
    return sorted({round(n.beat * spb, 4) for n in beatmap.color_notes})


def match_offsets(times: list[float], onsets: np.ndarray,
                  tol: float = TOL_S) -> tuple[int, list[float]]:
    """Greedy nearest-onset matching within `tol`. Returns (n_matched, offsets).

    Each onset can absorb at most one note, so note spam cannot manufacture
    precision by stacking hits on one loud event.
    """
    if len(times) == 0 or len(onsets) == 0:
        return 0, []
    ref = np.sort(np.asarray(onsets, dtype=np.float64))
    used = np.zeros(len(ref), dtype=bool)
    matched, offsets = 0, []
    for t in times:
        i = int(np.searchsorted(ref, t))
        best, bestd = -1, tol + 1.0
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(ref) and not used[j]:
                d = abs(ref[j] - t)
                if d < bestd:
                    best, bestd = j, d
        if best >= 0 and bestd <= tol:
            used[best] = True
            matched += 1
            offsets.append(float(t - ref[best]))
    return matched, offsets


def alignment_metrics(beatmap, *, bpm: float, onsets) -> AlignmentReport:
    """Score one map against one song's detected onsets."""
    rep = AlignmentReport()
    nan = {k: float("nan") for k in KEYS}
    if onsets is None or len(onsets) == 0:
        rep.metrics = nan
        return rep
    times = note_times(beatmap, bpm)
    rep.n_notes, rep.n_onsets = len(times), int(len(onsets))
    if len(times) < MIN_NOTES:
        rep.metrics = nan
        return rep

    matched, offsets = match_offsets(times, onsets)
    if not offsets:
        # Genuinely zero alignment: report it as the worst possible score rather
        # than NaN, so a map that hits nothing FAILS instead of going "not scored".
        rep.metrics = {"onset_precision": 0.0, "offset_mad_ms": float(TOL_S * 1000),
                       "onset_lag_ms": 0.0, "onset_recall": 0.0}
        return rep

    med = statistics.median(offsets)
    mad = statistics.median([abs(o - med) for o in offsets])
    rep.metrics = {
        "onset_precision": matched / len(times),
        "offset_mad_ms": mad * 1000.0,
        "onset_lag_ms": med * 1000.0,
        "onset_recall": matched / float(len(onsets)),
    }
    return rep


def load_reference() -> dict[str, tuple[float, float]]:
    if REFERENCE_PATH.exists():
        try:
            raw = json.loads(REFERENCE_PATH.read_text())
            return {k: (float(v["median"]), float(v["mad"])) for k, v in raw.items()}
        except Exception:  # noqa: BLE001
            pass
    return {}


def cohort_comparison(records: list[dict], reference: dict | None = None) -> dict:
    ref = reference if reference is not None else load_reference()
    return _dist.cohort_comparison(records, ref, KEYS, SEQUENCE_KEYS,
                                   gap_name="alignment_gap")


def calibrate(records: list[dict]) -> dict[str, dict]:
    return _dist.calibrate(records, KEYS)
