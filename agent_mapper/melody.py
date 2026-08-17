#!/usr/bin/env python
"""THE MELODY — pitch, the perception axis this project has never had.

★**Why this is the highest-value tool in `agent_mapper/`.** The single biggest finding
of 2026-08-14 is that W1, W4 and `follow_vocals` are *one* defect: Stage-1's
representation does not carry the melodic instruments — `version_4` has `drum_proj` +
`mix_proj` and nothing else. But `brief.py` had exactly the same blindness in a
different form: it reports that a vocal onset *happened*, never what NOTE it was. An
onset count cannot tell you the chorus lifts a fifth; a human mapper hears that
immediately and puts the hand higher.

**And pitch is not decoration — it is how a human picks the grid cell.** Beat Saber has
3 rows × 4 columns; a mapper walks them with the melody, up on a rising line, down on a
falling one. Our measured `travel` gap (**4.60 vs a human 12.53** — our hands barely
move) is exactly what you get when nothing tells the placer *where to go*. Onsets say
WHEN. Pitch says WHERE.

## What it gives you

- **note events** per melodic stem — onset, MIDI pitch, name, duration
- **a pitch level 0-9** per note, its rank inside the song's own range, in the same
  16-cell-per-bar format `brief.py --bars` uses, so the two views line up
- **contour** per phrase: rising / falling / arch / flat, and the interval sizes
- **key and mode**, so "this note is the tonic" is answerable
- **register shifts** — where the melody jumps octave or the chorus lifts

⚠️**Two stems, two different confidences.** `vocals` is near-monophonic and pYIN is
reliable there. `other` (guitar/synth/lead) is polyphonic, so its "melody" is a
salience peak of the CQT — treat it as *the top line you'd hum*, not as transcription.
The confidence is printed; below ~0.4 do not place notes off it.

Usage:
    python agent_mapper/melody.py <audio.ogg>                 # overview + contour
    python agent_mapper/melody.py <audio.ogg> --bars 57-72    # the note-by-note detail
    python agent_mapper/melody.py <audio.ogg> --stem other
    python agent_mapper/melody.py <audio.ogg> --json out.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = REPO / "outputs" / "melody_cache"
HOP = 256                       # 11.6 ms at 22 050 Hz — finer than a 1/16 at any bpm
MELODIC = ("vocals", "other")
NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl-Schmuckler key profiles: the standard correlation template for
# "which of the 24 keys does this pitch-class histogram look like".
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def note_name(midi: float) -> str:
    m = int(round(midi))
    return f"{NAMES[m % 12]}{m // 12 - 1}"


def _track_vocals(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pYIN on the near-monophonic vocals stem. Returns (times, midi, voiced).

    ⚠️**Gate on `voiced_flag`, never on `voiced_prob`.** librosa's returned posterior
    on these stems sits at **0.01-0.16 even where the flag is True and the pitch is
    right** — a first version gated at prob>=0.30 and threw away 95 % of a correctly
    tracked vocal line. Measured on four songs: voiced-flag coverage on loud frames is
    0.91-0.99 wherever the singing is pitched.
    """
    import librosa

    f0, voiced, _prob = librosa.pyin(
        y, sr=sr, hop_length=HOP,
        fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C6")),
    )
    t = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
    midi = librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))
    return t, midi, np.asarray(voiced) & np.isfinite(f0)


def _track_salience(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The top line of a POLYPHONIC stem, by harmonic-summed CQT salience.

    pYIN assumes one voice and returns confident nonsense on a guitar chord, so for
    `other` we ask a different question: at each frame, which pitch has the most
    energy once its harmonics are folded in? That is the line a listener hums.

    ⚠️It agrees with pYIN on only ~0.6 of voiced vocal frames (measured, 4 songs, and
    the disagreements are **not** octave errors), so the two are not interchangeable:
    pYIN is the truth for `vocals`, this is the only option for `other`.
    """
    import librosa

    fmin = float(librosa.note_to_hz("C2"))
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP, fmin=fmin,
                           n_bins=12 * 4, bins_per_octave=12))          # C2..C6
    # Fold harmonics 2f (+12 semitones) and 3f (+19) down onto the fundamental, so
    # the true root out-ranks its own overtones — a raw CQT argmax gets that wrong on
    # anything with a bright timbre.
    sal = C.copy()
    for shift, w in ((12, 0.5), (19, 0.33)):
        sal[:-shift] += w * C[shift:]
    idx = sal.argmax(axis=0)
    peak = sal[idx, np.arange(sal.shape[1])]
    conf = np.clip(np.log1p(peak / (np.median(sal, axis=0) + 1e-9)) / 4.0, 0, 1)
    midi = librosa.hz_to_midi(fmin) + idx.astype(float)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0][:len(idx)]
    voiced = (conf > 0.35) & (rms > np.percentile(rms, 40))   # silence has an argmax too
    t = librosa.frames_to_time(np.arange(len(idx)), sr=sr, hop_length=HOP)
    return t, midi, voiced


def _fix_octaves(ev: list[dict], jump: float = 7.0) -> int:
    """Snap notes that are an octave off their neighbours back onto the line.

    Both trackers occasionally lock onto a partial for a note or two. A melody moves
    in small steps, so a note >7 semitones from its local median that lands *within*
    that window once shifted by ±12 is an octave error rather than a real leap — and a
    real leap is left alone, because shifting it would not help.
    """
    if len(ev) < 5:
        return 0
    m = np.array([e["midi"] for e in ev], dtype=float)
    fixed = 0
    for i in range(len(m)):
        lo, hi = max(0, i - 4), min(len(m), i + 5)
        local = float(np.median(np.delete(m[lo:hi], i - lo)))
        d = m[i] - local
        if abs(d) > jump:
            for shift in (-12, 12, -24, 24):
                if abs(d + shift) < jump:
                    ev[i]["midi"] += shift
                    ev[i]["name"] = note_name(ev[i]["midi"])
                    m[i] += shift
                    fixed += 1
                    break
    return fixed


def pitch_at_onsets(onsets: np.ndarray, t: np.ndarray, midi: np.ndarray,
                    voiced: np.ndarray, max_win: float = 0.40) -> list[dict]:
    """★The core: ONE PITCH PER ONSET, which is one pitch per placeable note.

    An earlier version segmented the f0 track into notes independently and then hoped
    they lined up with the onsets we place notes on. They do not — a singer's vibrato
    flips the rounded semitone every ~35 ms, so grouping runs of equal semitone gave
    **48 "notes" for a 343-word song**.

    Anchoring on the onsets we already trust inverts the problem: the question stops
    being "where are the notes" (hard, and already answered by the onset detector) and
    becomes "what pitch is *this* note" (easy — the median over the frames it owns).
    Every returned event is therefore something you can actually place, and `coverage`
    reports honestly how many onsets got an answer.
    """
    out: list[dict] = []
    if len(onsets) == 0 or not voiced.any():
        return out
    step = float(np.median(np.diff(t))) if len(t) > 1 else HOP / 22050.0
    for k, on in enumerate(onsets):
        nxt = onsets[k + 1] if k + 1 < len(onsets) else on + max_win
        a, b = on + 0.010, min(nxt - 0.005, on + max_win)
        if b <= a:
            continue
        sl = (t >= a) & (t < b) & voiced
        n_v = int(sl.sum())
        if n_v < 3:
            continue
        p = float(np.median(midi[sl]))
        out.append({"t": float(on), "dur": round(float(n_v * step), 3),
                    "midi": int(round(p)), "name": note_name(p),
                    "conf": round(min(1.0, n_v * step / max(b - a, 1e-6)), 2)})
    return out


def analyse(audio: pathlib.Path, force: bool = False) -> dict:
    """Pitch for every melodic onset in the song. Cached (pYIN is ~20 s a song)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{audio.stem}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())

    import brief as _brief
    from stemcache import stems, SR

    s_ = stems(audio)
    onsets = _brief.analyse(audio)["onsets"]
    out: dict = {"stems": {}, "meta": {}}
    for name in MELODIC:
        t, midi, voiced = (_track_vocals if name == "vocals" else _track_salience)(s_[name], SR)
        on = np.asarray(onsets[name], dtype=float)
        ev = pitch_at_onsets(on, t, midi, voiced)
        n_oct = _fix_octaves(ev)
        out["stems"][name] = ev
        out["meta"][name] = {
            "onsets": int(len(on)),
            "coverage": round(len(ev) / max(len(on), 1), 3),
            "octaves_fixed": n_oct,
            "step": round(float(np.median(np.abs(np.diff(
                [e["midi"] for e in ev])))), 2) if len(ev) > 2 else None,
        }
    f.write_text(json.dumps(out))
    return out


def key_of(events: list[dict]) -> tuple[str, float]:
    """Key and mode by Krumhansl correlation, weighted by note DURATION.

    Duration-weighted because a passing sixteenth and a held tonic are not equal
    evidence, and a raw note-count histogram lets fast ornamental runs outvote the
    note the phrase actually rests on.
    """
    if not events:
        return "?", 0.0
    hist = np.zeros(12)
    for e in events:
        hist[e["midi"] % 12] += e["dur"]
    if hist.sum() <= 0:
        return "?", 0.0
    hist /= hist.sum()
    best, bestr = "?", -2.0
    for tonic in range(12):
        for prof, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            p = np.roll(prof, tonic)
            r = float(np.corrcoef(hist, p)[0, 1])
            if r > bestr:
                bestr, best = r, f"{NAMES[tonic]} {mode}"
    return best, bestr


def levels(events: list[dict], lo: float | None = None, hi: float | None = None) -> None:
    """Attach `level` 0-9: where this note sits in the song's own pitch range.

    ★This is the number that becomes a grid ROW. Absolute MIDI is not usable for
    placement — C4 is high for a bass and low for a soprano — but "8th decile of
    what this singer actually does in this song" transfers directly.
    """
    if not events:
        return
    ms = np.array([e["midi"] for e in events], dtype=float)
    lo = float(np.percentile(ms, 2)) if lo is None else lo
    hi = float(np.percentile(ms, 98)) if hi is None else hi
    span = max(hi - lo, 1e-6)
    for e in events:
        e["level"] = int(np.clip(round((e["midi"] - lo) / span * 9), 0, 9))


def contour(events: list[dict]) -> str:
    """Rising / falling / arch / valley / flat, for a run of notes."""
    if len(events) < 3:
        return "-"
    m = np.array([e["midi"] for e in events], dtype=float)
    h = len(m) // 2
    a, b = m[:h].mean(), m[h:].mean()
    peak = m.argmax() / max(len(m) - 1, 1)
    if abs(b - a) < 1.0:
        if m.max() - m.min() < 2.0:
            return "flat"
        return "arch" if 0.25 < peak < 0.75 else "wave"
    return "RISE" if b > a else "FALL"


def _mmss(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60)}:{t % 60:05.2f}"


def pitch_row(events: list[dict], t0: float, t1: float, slot: float, n: int = 16) -> str:
    """One bar as 16 cells, each the pitch LEVEL digit — lines up with brief.py."""
    cells = ["."] * n
    for e in events:
        if t0 <= e["t"] < t1:
            i = int(round((e["t"] - t0) / slot))
            if 0 <= i < n:
                cells[i] = str(e.get("level", 0))
    return "".join(cells)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--bars", default=None, help="note-by-note detail for bars N-M")
    ap.add_argument("--stem", default=None, choices=MELODIC, help="restrict to one stem")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a_ = ap.parse_args()
    if not a_.audio.exists():
        print(f"no such audio: {a_.audio}", file=sys.stderr)
        return 2

    import brief as _brief

    res = analyse(a_.audio, a_.force)
    ev, meta = res["stems"], res["meta"]
    want = [a_.stem] if a_.stem else list(MELODIC)
    for s_ in want:
        levels(ev[s_])

    a = _brief.analyse(a_.audio)
    g = _brief.grid(a)

    print(f"SONG {a_.audio.stem}   {g['bpm']:.2f} bpm   bar = {g['bar_s']:.3f}s")
    for s_ in want:
        e, m = ev[s_], meta[s_]
        if not e:
            print(f"{s_.upper():>7}: NO PITCHED LINE — {m['onsets']} onsets, none trackable.")
            continue
        ms = np.array([x["midi"] for x in e])
        k, r = key_of(e)
        print(f"{s_.upper():>7}: {len(e)}/{m['onsets']} onsets pitched "
              f"(coverage {m['coverage']:.2f})   range {note_name(ms.min())}-"
              f"{note_name(ms.max())}   key {k} (r={r:.2f})   "
              f"median step {m['step']} semitones   {m['octaves_fixed']} octave fixes")
        # Two honesty gates, both measured rather than asserted.
        if m["coverage"] < 0.45:
            print(f"{'':>9}⚠️LOW COVERAGE — most onsets have no trackable pitch. Screamed "
                  "or heavily distorted vocals genuinely have no f0; this is the song, "
                  "not a bug. Map this stem on RHYTHM, not contour.")
        if m["step"] is not None and m["step"] > 4:
            print(f"{'':>9}⚠️median step {m['step']} semitones is too large for a sung "
                  "melody (real ones step by ~1-3) — this line is probably tracking "
                  "chords or bleed. Treat the contour as unreliable.")

    if a_.bars:
        b0, b1 = (int(x) for x in a_.bars.split("-"))
        print(f"\nBARS {b0}-{b1}   digit = pitch level 0-9 within this song's range "
              "(★ this is your grid ROW)")
        print(f"{'bar':>4} {'time':>8}  {'|1e+a2e+a3e+a4e+a':<17} stem  notes")
        for bar in range(b0, b1 + 1):
            t0 = _brief.bar_time(g, bar)
            t1 = t0 + g["bar_s"]
            first = True
            for s_ in want:
                inbar = [e for e in ev[s_] if t0 <= e["t"] < t1]
                if not inbar:
                    continue
                row = pitch_row(ev[s_], t0, t1, g["slot"])
                head = f"{bar:>4} {_mmss(t0):>8}" if first else " " * 13
                first = False
                print(f"{head}  |{row}| {s_[0].upper()}     "
                      + " ".join(e["name"] for e in inbar[:8]))
            if first:
                print(f"{bar:>4} {_mmss(t0):>8}  |{'.'*16}| -     (no melody)")
    else:
        print(f"\nMELODIC CONTOUR (8-bar phrases)")
        print(f"{'bars':>9} {'time':>8}  " + "  ".join(
            f"{s_[:4]:>4} {'reg':>4} {'rng':>4} {'shape':>5}" for s_ in want))
        step = 8
        for b0 in range(1, g["n_bars"] + 1, step):
            t0 = _brief.bar_time(g, b0)
            t1 = min(t0 + g["bar_s"] * step, a["dur"])
            if t0 >= a["dur"]:
                break
            cells = []
            for s_ in want:
                inb = [e for e in ev[s_] if t0 <= e["t"] < t1]
                if not inb:
                    cells.append(f"{'-':>4} {'-':>4} {'-':>4} {'-':>5}")
                    continue
                ms = np.array([e["midi"] for e in inb], dtype=float)
                cells.append(f"{len(inb):>4} {note_name(ms.mean()):>4} "
                             f"{int(ms.max()-ms.min()):>4} {contour(inb):>5}")
            print(f"{b0:>4}-{b0+step-1:<4} {_mmss(t0):>8}  " + "  ".join(cells))
        print("\n  reg = mean pitch of the phrase (register) · rng = semitone span")
        print("  ★A phrase that lifts register is a phrase a human maps HIGHER on the grid.")

    if a_.json:
        a_.json.write_text(json.dumps(
            {"stems": {s_: ev[s_] for s_ in want}, "meta": meta}, indent=1))
        print(f"\nwrote {a_.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
