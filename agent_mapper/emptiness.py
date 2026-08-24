#!/usr/bin/env python
"""EMPTINESS — where the song asks and the map does not answer.

🔴🔴**DO NOT RENAME THIS BACK TO `coverage.py`.** It was called that for one commit and
broke **8 tests** across four unrelated files. `agent_mapper/` goes on `sys.path`, so
`agent_mapper/coverage.py` SHADOWS the installed `coverage` package for every importer
in the process -- numba's `coverage_support` did `import coverage`, got this module,
and died on `module 'coverage' has no attribute 'types'`. The traceback pointed at
numba and named no file of ours.
★**Never give a module on an importable path the name of a widely-installed package**
(`coverage`, `types`, `json`, `parser`, `test`...). The failure surfaces far away.

★**The complaint this exists for.** Kyle on Fallen Kingdom: *"it feels really empty."*
**Five separate note-based instruments failed to explain it**, and every one of them
asked the same question: *was this NOTE motivated?* (`overlay.py`'s HIT/MISSED/WASTED).
**"Empty" is the OTHER direction** -- *was this musical MOMENT answered?* -- and nothing
measured it, because a note that is never placed raises no metric. ★`READING.md` says
it outright: *"look for what is ABSENT, not what is wrong. Absence does not raise a
number."*

**What a player perceives, and what this reports:**
  * **`answered`** -- of the song's onsets in this window, what share got a note within
    the axis' 50 ms tolerance. Low `answered` while the music is busy IS the feeling.
  * **`gaps`** -- the longest stretches with NO note while the song keeps going, with
    timestamps. ⇒**this is the part that can point at the bar**, which is what the
    complaint always lacked.
  * **density contour** -- the song's onset rate against the map's note rate, so a
    section that is quiet *because the song is quiet* is not mistaken for a hole.

⚠️**A low `answered` is NOT automatically a defect.** Human maps answer a minority of
onsets -- a song offers far more events than any map should play, and `C1` measured
**4.5 onsets available per note we emit**. That is why `scripts/calibrate_coverage.py`
builds the human reference: the question is never "did we answer everything" but "did
we answer as much as a human would, HERE."
"""
from __future__ import annotations

import json
import pathlib

import numpy as np


def note_times(elems: dict) -> np.ndarray:
    """Note times in seconds. ⚠️Includes `_songTimeOffset`, which carries the grid
    phase -- omitting it once made a phase shift look like it moved nothing."""
    spb = 60.0 / max(elems["bpm"], 1e-6)
    return np.sort(np.array([elems["offset"] + float(n.get("b", 0.0)) * spb
                             for n in elems["notes"]], dtype=float))


def answered_share(nt: np.ndarray, onsets: np.ndarray, tol: float = 0.050) -> float:
    """Share of ONSETS that have a note within `tol`.

    ⚠️Note the direction: this is onset->note, the inverse of `onset_precision`
    (note->onset). A map can score high precision while answering very little, by
    placing few notes and placing them well. **That combination is exactly "empty".**
    """
    if not len(onsets) or not len(nt):
        return 0.0
    i = np.clip(np.searchsorted(nt, onsets), 1, len(nt) - 1)
    d = np.minimum(np.abs(onsets - nt[i - 1]), np.abs(onsets - nt[i]))
    return float((d <= tol).mean())


def gaps(elems: dict, onsets: np.ndarray, min_gap_s: float = 2.0,
         busy_onsets: int = 4) -> list[dict]:
    """Stretches with no note while the song keeps playing.

    A gap only counts as EMPTY if the song is still going -- `busy_onsets` is how many
    detected onsets must fall inside it. ★Otherwise a quiet outro reads as a hole, and
    letting the player breathe is something Kyle named as worth PROTECTING:
    *"when there is a slow spot we let the player breathe."*
    """
    nt = note_times(elems)
    if len(nt) < 2:
        return []
    out = []
    for a, b in zip(nt, nt[1:]):
        if b - a < min_gap_s:
            continue
        n_on = int(((onsets > a) & (onsets < b)).sum())
        if n_on >= busy_onsets:
            out.append({"t0": float(a), "t1": float(b), "dur": float(b - a),
                        "onsets_inside": n_on,
                        "onsets_per_s": round(n_on / max(b - a, 1e-6), 2)})
    return sorted(out, key=lambda g: -g["dur"])


def contour(elems: dict, onsets: np.ndarray, win_s: float = 8.0) -> list[dict]:
    """Per window: what the song offers, what the map plays, and what it answered."""
    nt = note_times(elems)
    if not len(nt) or not len(onsets):
        return []
    end = max(float(nt[-1]), float(onsets[-1]))
    out = []
    for t0 in np.arange(0.0, end, win_s):
        t1 = t0 + win_s
        o = onsets[(onsets >= t0) & (onsets < t1)]
        n = nt[(nt >= t0) & (nt < t1)]
        out.append({"t0": float(t0), "t1": float(t1),
                    "song_ops": len(o) / win_s, "map_nps": len(n) / win_s,
                    "answered": answered_share(nt, o) if len(o) else None})
    return out


def summary(elems: dict, onsets) -> dict:
    onsets = np.sort(np.asarray(onsets, dtype=float))
    nt = note_times(elems)
    g = gaps(elems, onsets)
    c = [w for w in contour(elems, onsets) if w["answered"] is not None]
    # ⚠️Reported over windows the SONG is active in, so a silent intro cannot flatter
    # or damn the number.
    busy = [w for w in c if w["song_ops"] > 0.5]
    return {
        "answered_overall": round(answered_share(nt, onsets), 4),
        "answered_busy_windows": round(
            float(np.mean([w["answered"] for w in busy])), 4) if busy else None,
        "n_gaps_over_2s": len(g),
        "longest_gap_s": round(g[0]["dur"], 2) if g else 0.0,
        "gap_seconds_total": round(sum(x["dur"] for x in g), 1),
        "gap_share_of_song": round(
            sum(x["dur"] for x in g) / max(float(nt[-1] - nt[0]), 1e-6), 4)
        if len(nt) > 1 else 0.0,
        "worst_gaps": g[:5],
    }


REFERENCE_PATH = (pathlib.Path(__file__).resolve().parents[1]
                  / "outputs" / "coverage_reference.json")

# Direction of harm per key. ★`answered` is the one that matters and it is
# TWO-SIDED: answering too little is "empty", answering too much is a wall of notes
# with no shape -- and human maps sit at a MEDIAN of ~49 %, not near 100 %.
_TOO_LOW_IS_BAD = {"answered_overall", "answered_busy_windows"}
_TOO_HIGH_IS_BAD = {"n_gaps_over_2s", "longest_gap_s", "gap_share_of_song"}


def load_reference() -> dict | None:
    if not REFERENCE_PATH.exists():
        return None
    return json.loads(REFERENCE_PATH.read_text())


def _pct_of(d: dict, v: float) -> float:
    pts = [(0.05, d["p5"]), (0.25, d["p25"]), (0.50, d["median"]),
           (0.75, d["p75"]), (0.95, d["p95"])]
    if v <= pts[0][1]:
        return 0.05
    if v >= pts[-1][1]:
        return 0.95
    for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
        if v0 <= v <= v1:
            return p1 if v1 == v0 else p0 + (p1 - p0) * (v - v0) / (v1 - v0)
    return 0.5


def judge(elems: dict, onsets, reference: dict | None = None) -> dict:
    """Place this map's coverage against human mappers.

    ★**This is what makes "empty" answerable.** Before it, a coverage number had no
    yardstick and the complaint could not be confirmed OR refuted.
    """
    reference = reference or load_reference()
    s = summary(elems, onsets)
    if not reference:
        return {"summary": s, "lines": [], "note": "no coverage reference; run "
                "scripts/calibrate_coverage.py"}
    lines = []
    for k, d in reference.get("dist", {}).items():
        if s.get(k) is None:
            continue
        v, pct = float(s[k]), _pct_of(d, float(s[k]))
        bad = ((k in _TOO_LOW_IS_BAD and pct <= 0.05)
               or (k in _TOO_HIGH_IS_BAD and pct >= 0.95))
        warn = ((k in _TOO_LOW_IS_BAD and pct <= 0.25)
                or (k in _TOO_HIGH_IS_BAD and pct >= 0.75))
        lines.append({"key": k, "value": v, "pct": pct, "median": d["median"],
                      "flag": "🔴" if bad else ("🟡" if warn else "✅")})
    return {"summary": s, "lines": lines}
