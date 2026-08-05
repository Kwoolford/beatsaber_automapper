#!/usr/bin/env python
"""Shared structure layer for the "masterpiece" axes (2026-08-04 night).

Kyle: *"We created a model to create a playable map but now need a model to start
producing masterpieces which we are far off from."*

Everything the suite measured before this file is a **defect axis**: is a note on
an onset, is the ending orphaned, does the map drift. A map can score perfectly on
all of them and still be a lifeless map, because none of them ask the question a
mapper asks — **is this placement INTENTIONAL?** Intent is not visible in a single
note; it is visible in the RELATION between what the music does twice and what the
map does twice.

This module provides the substrate for those relational axes:

    bars(song, bpm, end)        the bar grid, from the main beat (NOT from bpm/4 —
                                the main beat is the level the song is *stated* on)
    audio_features(song)        cached frame-level chroma / mfcc / onset-envelope
    bar_audio_matrix(...)       per-bar audio descriptors -> a self-similarity map
    map_bar_vectors(...)        per-bar rhythm + placement descriptors of a MAP
    bar_map_similarity(...)     map-side self-similarity map

★**THE DESIGN RULE THAT MAKES THESE AXES DIFFERENT.** Every metric this project
built that rewards REGULARITY turned out to be metronome-gameable (`halfbeat_rate`
0.036 vs a human 0.084; `share_over_1s` 0.200 vs 0.250). The fix is not a better
regularity metric — it is to stop measuring levels and measure **contrast**:

    score = (what the map does where the music says X)
          - (what the map does where the music says NOT-X)

A metronome is *identical everywhere*, so every contrast of this shape is **0 for a
metronome by construction**, and so is a uniform-random map. Degenerate-proof by
design rather than by audit — but still audited (see `audit_masterpiece.py`).

⚠️THE CONFOUND THIS MODULE EXISTS TO CONTROL: bars near each other in time are both
more audio-similar and more map-similar, for reasons that have nothing to do with
intent. Every consumer must stratify by |i - j| (`lag_strata`) before differencing.
"""

from __future__ import annotations

import hashlib
import pathlib
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

CACHE = REPO / "outputs" / "structure_cache"
SONGSET = REPO / "data" / "eval_songset"
RAW = REPO / "data" / "raw"

SR = 22050
HOP = 512
BEATS_PER_BAR = 4
SLOTS_PER_BAR = 16


# --------------------------------------------------------------------------- audio

def audio_path(song_id: str) -> pathlib.Path | None:
    """Prefer the songset copy; fall back to the audio inside the map zip.

    ⚠️Verified 2026-08-02 that songset audio is byte-identical to the zip payload
    (md5 + 0.0 ms cross-correlation), so the two sources are interchangeable.
    """
    for ext in (".ogg", ".mp3", ".egg", ".wav"):
        p = SONGSET / f"{song_id}{ext}"
        if p.exists():
            return p
    for p in SONGSET.glob(f"{song_id}*"):
        if p.suffix.lower() in (".ogg", ".mp3", ".egg", ".wav"):
            return p
    z = RAW / f"{song_id}.zip"
    if z.exists():
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="struct_"))
        try:
            with zipfile.ZipFile(z) as zf:
                name = next((n for n in zf.namelist()
                             if n.lower().endswith((".egg", ".ogg", ".mp3", ".wav"))), None)
                if name is None:
                    return None
                out = tmp / pathlib.Path(name).name
                out.write_bytes(zf.read(name))
                return out
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
    return None


def audio_features(song_id: str, rebuild: bool = False) -> dict | None:
    """Frame-level chroma / mfcc / onset-envelope, cached per song.

    Cached at the FRAME level, not the bar level, so the cache survives a change to
    the bar grid — the bar grid is the thing most likely to be revised.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{song_id}.npz"
    if f.exists() and not rebuild:
        d = np.load(f, allow_pickle=True)
        return {k: d[k] for k in d.files}

    import librosa
    p = audio_path(song_id)
    if p is None:
        return None
    y, sr = librosa.load(str(p), sr=SR, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=20)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP)
    out = {"chroma": chroma.astype(np.float32), "mfcc": mfcc.astype(np.float32),
           "onset_env": onset_env.astype(np.float32), "rms": rms.astype(np.float32),
           "times": times.astype(np.float32)}
    np.savez_compressed(f, **out)
    return out


# --------------------------------------------------------------------------- bars

@dataclass(slots=True)
class Bars:
    edges: np.ndarray            # (n+1,) bar boundaries in seconds
    period: float                # main-beat period
    ratio: float                 # main-beat ratio vs the fitted beat
    beats_per_bar: int
    confidence: str = ""
    f1: float = 0.0

    @property
    def n(self) -> int:
        return max(0, len(self.edges) - 1)

    @property
    def dur(self) -> float:
        return self.period * self.beats_per_bar

    @property
    def starts(self) -> np.ndarray:
        return self.edges[:-1]


TARGET_BAR_S = 2.0
BAR_MULTIPLES = (2, 4, 8, 16)


def song_end(song_id: str, fallback: float = 0.0) -> float:
    """Song duration from the AUDIO.

    ⚠️Callers used to pass the last note time of whichever map they were scoring,
    which makes the bar grid depend on the map — so our map and the human's were
    graded on grids of different lengths, and `hands_x_downbeat` read 0.18 in one
    script and 0.30 in another for the same cohort. The grid must be a property of
    the song alone.
    """
    A = audio_features(song_id)
    if A is not None and len(A["times"]):
        return float(A["times"][-1])
    return float(fallback)


def bars(song_id: str, bpm: float, end: float,
         beats_per_bar: int | None = None) -> Bars | None:
    """Bar grid anchored on the MAIN BEAT, not on bpm.

    ⚠️Using `60/bpm * 4` would put the bar on whichever metrical level librosa
    happened to fit; `main_beat` picks the level the song is *stated* on by
    two-sided support x capture scoring, which is the level a mapper hears.

    ★TWO CORRECTIONS THE FIRST VERSION NEEDED, both found on the first smoke test:

    1. **`main_beat * 4` is not a bar.** On Hunger the main beat comes back at the
       eighth level, so four of them is 0.638 s — half a bar, and a window that
       short holds ~2 notes, far too few for "is this pattern the same pattern".
       The multiple is therefore chosen (a power of two, so downbeats stay on main
       beats) to land the bar nearest `TARGET_BAR_S`.
    2. **The bar PHASE matters.** Bar 0 starting on beat 3 of the real bar shifts
       every descriptor. Nothing here detects downbeats, so the phase is picked as
       the offset whose bar starts carry the most onset energy — accented
       downbeats being the near-universal case. Both cohorts get the SAME grid, so
       a residual phase error costs sensitivity, never a cohort difference.
    """
    from main_beat import find_main_beat

    mb = find_main_beat(song_id, bpm, end)
    if mb is None or len(mb.grid) < 16:
        return None

    if beats_per_bar is None:
        beats_per_bar = min(BAR_MULTIPLES,
                            key=lambda k: abs(np.log(mb.period * k / TARGET_BAR_S)))
    if len(mb.grid) < beats_per_bar * 4:
        beats_per_bar = max(2, beats_per_bar // 2)
    if len(mb.grid) < beats_per_bar * 4:
        return None

    off = 0
    A = audio_features(song_id)
    if A is not None and len(A["onset_env"]):
        env, t = A["onset_env"], A["times"]
        best = None
        for o in range(beats_per_bar):
            starts = mb.grid[o::beats_per_bar]
            idx = np.clip(np.searchsorted(t, starts), 0, len(env) - 1)
            score = float(env[idx].mean())
            if best is None or score > best[0]:
                best, off = (score, o), o

    g = mb.grid[off::beats_per_bar]
    if len(g) < 5:
        return None
    edges = np.append(g, g[-1] + mb.period * beats_per_bar)
    return Bars(edges=edges, period=mb.period, ratio=mb.ratio,
                beats_per_bar=beats_per_bar, confidence=mb.confidence, f1=mb.f1)


# ------------------------------------------------------------------ audio per bar

def bar_audio_matrix(song_id: str, B: Bars) -> dict | None:
    """Per-bar audio descriptors + the self-similarity matrices they induce.

    Three separate views, kept apart deliberately — "the music repeats" can mean
    the harmony repeats, the timbre repeats, or the rhythm repeats, and a map may
    follow one and ignore another:

        harm    chroma mean (key/harmony)
        timb    mfcc 1-13 mean (instrumentation/texture)
        rhy     onset-envelope resampled to SLOTS_PER_BAR (the groove)
    """
    A = audio_features(song_id)
    if A is None:
        return None
    t = A["times"]
    if len(t) < 16:
        return None
    harm, timb, rhy, energy = [], [], [], []
    for i in range(B.n):
        s, e = B.edges[i], B.edges[i + 1]
        lo, hi = int(np.searchsorted(t, s)), int(np.searchsorted(t, e))
        if hi - lo < 3:
            harm.append(None)
            timb.append(None)
            rhy.append(None)
            energy.append(0.0)
            continue
        harm.append(A["chroma"][:, lo:hi].mean(axis=1))
        timb.append(A["mfcc"][1:14, lo:hi].mean(axis=1))
        env = A["onset_env"][lo:hi]
        idx = np.linspace(0, len(env) - 1, SLOTS_PER_BAR)
        rhy.append(np.interp(idx, np.arange(len(env)), env))
        energy.append(float(A["rms"][lo:hi].mean()))
    valid = np.array([h is not None for h in harm])

    def ssm(vecs):
        M = np.full((B.n, B.n), np.nan)
        V = [None if v is None else np.asarray(v, dtype=float) for v in vecs]
        V = [None if v is None else (v - v.mean()) for v in V]
        norms = [None if v is None else float(np.linalg.norm(v)) for v in V]
        for i in range(B.n):
            if V[i] is None or not norms[i]:
                continue
            for j in range(B.n):
                if V[j] is None or not norms[j]:
                    continue
                M[i, j] = float(V[i] @ V[j] / (norms[i] * norms[j]))
        return M

    return {"valid": valid, "energy": np.asarray(energy),
            "harm": ssm(harm), "timb": ssm(timb), "rhy": ssm(rhy)}


# -------------------------------------------------------------------- map per bar

def map_bar_vectors(notes_xydc: list[tuple], B: Bars,
                    slots: int = SLOTS_PER_BAR) -> dict:
    """Describe each bar of a MAP as rhythm + placement vectors.

    `notes_xydc` = (time_s, x, y, direction, color).

    Two descriptors, because "the map repeats" is also two different claims:
        rhythm    which of the `slots` subdivisions carry a note (binary)
        place     mean column / row / cut-direction-as-unit-vector per slot,
                  which is what makes a *pattern* recognisable as the same pattern
    """
    n = B.n
    rhythm = np.zeros((n, slots), dtype=float)
    place = np.zeros((n, slots, 4), dtype=float)
    cnt = np.zeros((n, slots), dtype=float)
    dur = B.dur
    # cut direction -> unit vector (0 up, 1 down, 2 left, 3 right, 4-7 diagonals,
    # 8 = any/dot, which carries no direction and is left at zero).
    DIRV = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
            4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1), 8: (0, 0)}
    for (t, x, y, d, _c) in notes_xydc:
        if t < B.edges[0] or t >= B.edges[-1]:
            continue
        bi = int((t - B.edges[0]) // dur)
        if not (0 <= bi < n):
            continue
        frac = (t - (B.edges[0] + bi * dur)) / dur
        si = int(round(frac * slots)) % slots
        if si == 0 and frac > 0.5:          # rounded forward into the next bar
            bi = min(bi + 1, n - 1)
        rhythm[bi, si] = 1.0
        dv = DIRV.get(int(d), (0, 0))
        place[bi, si] += (x / 3.0, y / 2.0, dv[0], dv[1])
        cnt[bi, si] += 1
    with np.errstate(invalid="ignore"):
        place = np.where(cnt[..., None] > 0, place / np.maximum(cnt, 1)[..., None], 0.0)
    return {"rhythm": rhythm, "place": place, "count": cnt.sum(axis=1)}


def bar_map_similarity(V: dict, min_notes: int = 3) -> dict:
    """Self-similarity of the MAP, in the two senses above.

    rhythm similarity = **Cohen's kappa** on the binary slot vectors: agreement
    corrected for the agreement expected from the two bars' densities alone.
    place similarity = agreement of (col, row, dir) over the slots BOTH bars play —
    undefined when they share no slot, which is the honest answer: two bars with no
    common rhythm have no common pattern.

    🔴**WHY KAPPA AND NOT COSINE — a confound caught on the first run.** With
    cosine, our maps scored a *higher* motif contrast than the humans on Hunger.
    The mechanism is not motif reuse: `DENSITY_SELECT` makes our note count track
    loudness hard, so two bars that sound alike hold a similar NUMBER of notes, and
    two binary vectors of similar density overlap more by chance. Cosine reads that
    as "the same pattern". Kappa subtracts exactly that chance term, so matching
    the density of a repeat earns nothing and only matching its SLOTS does.
    """
    R, P, C = V["rhythm"], V["place"], V["count"]
    n, slots = R.shape
    S_r = np.full((n, n), np.nan)
    S_p = np.full((n, n), np.nan)
    norms = np.linalg.norm(R, axis=1)
    dens = R.mean(axis=1)
    for i in range(n):
        if C[i] < min_notes or norms[i] == 0:
            continue
        for j in range(n):
            if C[j] < min_notes or norms[j] == 0:
                continue
            po = float((R[i] == R[j]).mean())
            pe = dens[i] * dens[j] + (1 - dens[i]) * (1 - dens[j])
            S_r[i, j] = 0.0 if pe >= 1.0 else float((po - pe) / (1 - pe))
            both = (R[i] > 0) & (R[j] > 0)
            if both.sum() >= 2:
                a, b = P[i][both], P[j][both]
                # mean per-slot agreement: 1 - normalised L1 over the 4 channels
                diff = np.abs(a - b).mean(axis=1)
                S_p[i, j] = float(np.clip(1.0 - diff, 0, 1).mean())
    return {"rhythm": S_r, "place": S_p}


# --------------------------------------------------------------- the lag control

def lag_strata(n: int, edges=(1, 2, 4, 8, 16, 32, 64, 10 ** 6)) -> list[tuple[int, int]]:
    """Bar-distance buckets used to control the proximity confound."""
    out, lo = [], 1
    for hi in edges:
        if hi > lo:
            out.append((lo, hi))
        lo = hi
    return out


def stratified_contrast(A: np.ndarray, M: np.ndarray, *,
                        hi_q: float = 0.75, lo_q: float = 0.25,
                        min_pairs: int = 8) -> dict:
    """★THE CORE ESTIMATOR. Mean of M where A is high, minus where A is low —
    computed WITHIN bar-distance strata and then averaged over strata.

    A = audio similarity matrix, M = map similarity matrix (both bar x bar, NaN
    where undefined). Returns the contrast, the two levels, and the pair count.

    ⚠️Without the stratification this measures mostly proximity: adjacent bars are
    both more audio-similar and more map-similar for reasons unrelated to intent.
    The unstratified value is returned too, as `contrast_raw`, so the size of that
    confound is visible rather than assumed.
    """
    n = A.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    a, m = A[ii, jj], M[ii, jj]
    lag = jj - ii
    ok = np.isfinite(a) & np.isfinite(m)
    a, m, lag = a[ok], m[ok], lag[ok]
    if len(a) < min_pairs * 2:
        return {}

    def contrast(a_, m_):
        if len(a_) < min_pairs * 2:
            return None
        hi_t, lo_t = np.quantile(a_, hi_q), np.quantile(a_, lo_q)
        hi, lo = m_[a_ >= hi_t], m_[a_ <= lo_t]
        if len(hi) < 3 or len(lo) < 3:
            return None
        return float(hi.mean()), float(lo.mean()), len(hi), len(lo)

    raw = contrast(a, m)
    parts, w = [], []
    for lo_l, hi_l in lag_strata(n):
        sel = (lag >= lo_l) & (lag < hi_l)
        if sel.sum() < min_pairs * 2:
            continue
        c = contrast(a[sel], m[sel])
        if c is None:
            continue
        parts.append(c[0] - c[1])
        w.append(c[2] + c[3])
    if not parts:
        return {}
    w = np.asarray(w, dtype=float)
    return {"contrast": float(np.average(parts, weights=w)),
            "contrast_raw": None if raw is None else float(raw[0] - raw[1]),
            "level_hi": None if raw is None else round(raw[0], 4),
            "level_lo": None if raw is None else round(raw[1], 4),
            "n_strata": len(parts), "n_pairs": int(len(a))}


def paired_delta(rows: list[dict], key: str, a: str = "ours", b: str = "human",
                 min_n: int = 6) -> dict:
    """Paired difference on identical songs — the ONLY cohort statistic quoted here.

    ⚠️This project's most repeated mistake is comparing across populations (our 24
    songs against a 200-map corpus median, added-note k against an event base rate,
    W3's 6.5-vs-5.5). Only ~13 of the 24 eval songs ship a human Expert map, so an
    unpaired median of ours against an unpaired median of theirs is two different
    song sets.

    Reports the MEAN delta with its standard error *and* the MEDIAN delta: on
    `hands_x_downbeat` they came out −0.344 and −0.111, which is a one-song outlier
    telling you not to quote the mean alone.
    """
    d = [r[a][key] - r[b][key] for r in rows
         if r.get(a) and r.get(b)
         and r[a].get(key) is not None and r[b].get(key) is not None]
    if len(d) < min_n:
        return {}
    m = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(len(d))
    return {"n": len(d), "delta": round(m, 4), "delta_median": round(float(np.median(d)), 4),
            "sd": round(sd, 4), "se": round(se, 4),
            "resolvable": bool(abs(m) > 2 * se),
            "sign_consistent": bool(abs(sum(np.sign(d))) >= len(d) - 2)}


def song_hash(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:8]
