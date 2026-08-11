"""M-E — STRUCTURE-CONDITIONED DECODE: when the music repeats, reuse the map.

★WHY THIS EXISTS, AND WHY IT IS NOT ANOTHER DECODE KNOB.

Every lever this project tried between 2026-06 and 2026-08-05 changed *how a slot is
scored* (density, gamma, probability floor, IOI prior, phase bonus, hand dealing) or
*how well the slot is scored* (the v8 instrument model). None of them moved a single
masterpiece axis, and the reason is now stated as C1's sixth direction:

    every slot is decided ON ITS OWN, so no amount of better per-slot evidence can
    make bar 74 relate to bar 42.

A human mapper does not re-decide a repeated section. They copy it and vary it.
Fallen Kingdom, the bar at 2:25 against the bar at 1:43 (audio similarity 0.83):

    human   X.X...X...X.X...   X.X...X...X.X...    identical
    ours    X.X.X.X.....X...   X.......X.X.....    unrelated

⚠️AND THE OBVIOUS READING OF THAT IS WRONG — checked before this file was written.
"We never repeat" is FALSE: 50 % of our bars (57/114) are exact copies of *some*
earlier bar, against the human's 74 %. We repeat plenty. **Our repeats do not land
where the music repeats.** So the fix is not "repeat more", it is "repeat THERE",
which is why this module keys every copy to the audio's own self-similarity.

★THE MECHANISM CLAIM: this is the one structural idea C1 does not block, because it
does not need a better probability field. It copies a decision that was already made.

---------------------------------------------------------------------------- modes

`place` (default when enabled)  — copy POSITION and CUT DIRECTION only, matched slot
    for slot. **No note is added, removed, or moved in time.** Every time-domain axis
    (alignment, rhythm/A2, density, nps, onset precision) is therefore bit-identical
    to the control BY CONSTRUCTION, not by measurement — the same property that made
    `BEAT_SPEED_DIAG` safe to reason about. This is the conservative arm.

`full` — also copy the RHYTHM (which slots carry notes), i.e. replace the bar. This is
    the arm that can move `rhy_rhythm`, and it is the risky one: a copied note lands
    where the SOURCE bar had an onset, which is not necessarily where THIS bar has one.
    That is exactly the wall `BEAT_HAND_DEAL` hit — the marginal note is much worse
    than the average note — so it is a separate arm and it is expected to cost onset
    precision. Measure, do not assume.

------------------------------------------------------------------ how to read a win

🔴**PRE-REGISTERED, BEFORE THE FIRST RUN — `harm_place` IS A MANIPULATION CHECK HERE,
NOT EVIDENCE OF QUALITY.** `harm_place` scores placement reuse on musical repeats and
this module copies placement on musical repeats, so a rise in it says only "the lever
did the thing it says on the tin". It cannot be cited as proof the map got better; that
would be fitting the metric, the error this project has already made under other names.

The claim "this made the map better" needs, in this order:
  1. **Kyle's ear** — the only success criterion the project has ever accepted.
  2. **Nothing else regresses**: the six-axis suite, `hard_rate` (reachability — a
     lever can pass every axis and still carry a defect no axis measures), and
     `follow_*` (a copied bar is right only if the repeat really is a repeat).
  3. The structure panel of `view_structure.py` growing the off-diagonal stripes the
     human panel has and ours does not.
⚠️And Kyle, 2026-08-10: *"the metrics still don't capture the full picture"* — so a
green table here is a licence to have him listen, nothing more.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SLOTS_PER_BAR = 16
TARGET_BAR_S = 2.0
BAR_MULTIPLES = (2, 4, 8, 16)

# Defaults chosen to be conservative: a copy only happens on a STRONG, ENERGY-MATCHED
# repeat that is far enough away to be a real section return rather than the local
# autocorrelation of adjacent bars.
DEFAULT_MIN_SIM = 0.60
DEFAULT_MIN_LAG = 4
DEFAULT_ENERGY_TOL = 1.5
DEFAULT_MIN_Z = 2.5
DEFAULT_MIN_RUN = 1


@dataclass(slots=True)
class ReusePlan:
    """Which bar copies which, plus everything needed to explain the decision."""

    edges: np.ndarray                 # (n+1,) bar boundaries in seconds
    source: dict[int, int]            # target bar -> source bar (already root-resolved)
    sim: dict[int, float]             # target bar -> the similarity that justified it
    n_bars: int

    @property
    def n_copied(self) -> int:
        return len(self.source)

    @property
    def share(self) -> float:
        return self.n_copied / self.n_bars if self.n_bars else 0.0


# --------------------------------------------------------------------------- grid

def bar_edges(carrier: np.ndarray, onset_env: np.ndarray, env_times: np.ndarray,
              bpm: float, end: float) -> np.ndarray | None:
    """Bar grid anchored on the MAIN BEAT, phase-locked to onset energy.

    Deliberately mirrors `scripts/song_structure.bars` so the generator and the
    evaluator cannot disagree about what a bar is. Two corrections that file records,
    both reproduced here because both were found the hard way:

      1. **`main_beat * 4` is not a bar.** On Hunger the main beat comes back at the
         eighth level, so four of them is 0.638 s — half a bar. The multiple is a
         power of two (so downbeats stay on main beats) chosen to land nearest 2 s.
      2. **The bar PHASE matters.** Bar 0 starting on beat 3 of the real bar shifts
         every descriptor, so the offset is picked as the one whose bar starts carry
         the most onset energy — accented downbeats being the near-universal case.
    """
    from beatsaber_automapper.generation.generate import _main_beat_grid

    mb = _main_beat_grid(carrier, bpm, end)
    if mb is None:
        return None
    grid, period = mb
    if len(grid) < 16 or period <= 0:
        return None

    bpb = min(BAR_MULTIPLES, key=lambda k: abs(np.log(period * k / TARGET_BAR_S)))
    while bpb > 2 and len(grid) < bpb * 4:
        bpb //= 2
    if len(grid) < bpb * 4:
        return None

    off = 0
    if onset_env is not None and len(onset_env):
        best = None
        for o in range(bpb):
            starts = grid[o::bpb]
            idx = np.clip(np.searchsorted(env_times, starts), 0, len(onset_env) - 1)
            score = float(onset_env[idx].mean())
            if best is None or score > best:
                best, off = score, o

    g = grid[off::bpb]
    if len(g) < 5:
        return None
    return np.append(g, g[-1] + period * bpb)


# ---------------------------------------------------------------------- audio SSM

def audio_bar_ssm(y: np.ndarray, sr: int, edges: np.ndarray) -> dict | None:
    """Per-bar harmony / timbre / rhythm self-similarity + per-bar energy.

    Three views kept apart on purpose — "the music repeats" can mean the harmony
    repeats, the timbre repeats, or the groove repeats, and a map may follow one and
    ignore another. A copy is only justified when the views AGREE (see `plan_reuse`).
    """
    import librosa

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=20)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    if len(t) < 16:
        return None

    n = len(edges) - 1
    harm, timb, rhy, energy = [], [], [], []
    for i in range(n):
        lo = int(np.searchsorted(t, edges[i]))
        hi = int(np.searchsorted(t, edges[i + 1]))
        if hi - lo < 3:
            harm.append(None), timb.append(None), rhy.append(None)
            energy.append(0.0)
            continue
        harm.append(chroma[:, lo:hi].mean(axis=1))
        timb.append(mfcc[1:14, lo:hi].mean(axis=1))
        e = env[lo:hi]
        idx = np.linspace(0, len(e) - 1, SLOTS_PER_BAR)
        rhy.append(np.interp(idx, np.arange(len(e)), e))
        energy.append(float(rms[lo:hi].mean()))

    def ssm(vecs):
        M = np.full((n, n), np.nan)
        V = [None if v is None else np.asarray(v, dtype=float) for v in vecs]
        V = [None if v is None else v - v.mean() for v in V]
        nm = [None if v is None else float(np.linalg.norm(v)) for v in V]
        for i in range(n):
            if V[i] is None or not nm[i]:
                continue
            for j in range(n):
                if V[j] is None or not nm[j]:
                    continue
                M[i, j] = float(V[i] @ V[j] / (nm[i] * nm[j]))
        return M

    return {"harm": ssm(harm), "timb": ssm(timb), "rhy": ssm(rhy),
            "energy": np.asarray(energy)}


# -------------------------------------------------------------------------- plan

def plan_reuse(S: dict, edges: np.ndarray, *,
               min_sim: float = DEFAULT_MIN_SIM,
               min_lag: int = DEFAULT_MIN_LAG,
               energy_tol: float = DEFAULT_ENERGY_TOL,
               min_z: float = DEFAULT_MIN_Z,
               min_run: int = DEFAULT_MIN_RUN) -> ReusePlan:
    """Decide, for each bar, whether it is a repeat of an earlier bar — and of which.

    ⚠️THE CONFOUND THIS FUNCTION EXISTS TO CONTROL: bars near each other in time are
    similar for reasons that have nothing to do with musical form (the local
    autocorrelation of any audio). `min_lag` is what keeps a "repeat" from meaning
    "the bar next door", and it is the single most important knob here.

    ★TWO VIEWS MUST AGREE. The score is `min(harm, rhy)`, not the mean: a section that
    shares a chord loop with an earlier one but has a different groove is a DIFFERENT
    section to a mapper, and a mean would let a strong harmony score carry it.

    ★ROOT RESOLUTION: if the best source is itself a copy, follow the chain to its
    origin. A whole repeated chorus then collapses onto ONE original rather than a
    chain of drifting copies — which is what makes the human's structure panel read as
    sharp discrete squares rather than a smear.

    🔴★**THE ABSOLUTE THRESHOLD IS NOT ENOUGH, AND THE FIRST SMOKE TEST PROVED IT.**
    A bare `sim >= 0.7` flagged **76–88 %** of bars as repeats on all four standing
    songs, collapsing 139 bars onto 13 roots — which is not "the map follows the form",
    it is the *uniform bright blob* the structure panel already caught us producing
    (`docs/eval_suite_v2.md`: ours is a smear where the human's is sharp squares).
    Most music sits in one key with a steady groove, so bar-to-bar cosine is high
    almost everywhere and a LEVEL cannot separate "the chorus is back" from "this is
    still the same song".

    So the match must prove it is **DISTINCTIVE**, not merely high — the same design
    rule that made the M-axes the first steer-safe metrics here (*score a contrast,
    not a level*). `min_z` requires the best candidate to stand clear of the bar's own
    similarity distribution by `min_z` robust sds (median/MAD over its candidates).
    A bar that resembles every earlier bar equally well resembles none of them
    *particularly*, and is left alone.
    """
    n = len(edges) - 1
    H, R, E = S["harm"], S["rhy"], S["energy"]
    source: dict[int, int] = {}
    sim: dict[int, float] = {}

    for i in range(min_lag, n):
        cand: list[tuple[int, float]] = []
        for j in range(0, i - min_lag + 1):
            h, r = H[i, j], R[i, j]
            if not np.isfinite(h) or not np.isfinite(r):
                continue
            cand.append((j, min(float(h), float(r))))
        if len(cand) < 4:
            continue
        best_j, best_s = max(cand, key=lambda c: c[1])
        if best_s < min_sim:
            continue
        # Distinctiveness: does this match stand out from what this bar resembles in
        # general? Median/MAD rather than mean/sd because the candidate list is itself
        # full of near-duplicates and a mean would be dragged toward the peak.
        vals = np.array([c[1] for c in cand], dtype=float)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826
        # ⚠️MAD FLOOR, and it is not cosmetic. A bar that matches exactly ONE earlier
        # bar and nothing else has MAD = 0 — the most distinctive match available —
        # and dividing by it would reject precisely the case this lever exists for.
        # The floor is in cosine units, so a spread narrower than 0.02 is treated as
        # 0.02 rather than as certainty.
        mad = max(mad, 0.02)
        if (best_s - med) / mad < min_z:
            continue
        # Energy guard: the same chords at a wildly different loudness is a different
        # section (a quiet intro vs the chorus built on it), and Kyle's own phrasing of
        # the idea was "transposed for the current section's energy".
        ei, ej = float(E[i]), float(E[best_j])
        if ei <= 0 or ej <= 0 or abs(np.log(ei / ej)) > np.log(energy_tol):
            continue
        root = best_j
        seen = {i}
        while root in source and root not in seen:
            seen.add(root)
            root = source[root]
        if root == i:
            continue
        source[i] = root
        sim[i] = best_s

    if min_run > 1:
        source, sim = _keep_runs(source, sim, min_run)

    return ReusePlan(edges=edges, source=source, sim=sim, n_bars=n)


def _keep_runs(source: dict[int, int], sim: dict[int, float],
               min_run: int) -> tuple[dict[int, int], dict[int, float]]:
    """Keep only copies that are part of a contiguous SECTION of length >= min_run.

    🔴★**WHY THIS EXISTS — the first arm's own result.** `me_z20` copied placement on
    every bar the audio called a distinctive repeat, and it **broke flow and idiom**
    against the 149-song control: flow 0.37 -> 0.75 and idiom 0.40 -> 1.07, both
    crossing their bars, while every rhythm-side axis stayed identical to 4 dp. The
    cause was then measured directly rather than guessed: **only 15.6 % of copied bars
    continued the previous bar's copy** (median 13.9 %; in 60/60 songs under half),
    because each bar picks its own best source independently. So the lever was not
    reusing a section — it was **shuffling ~29 bars per song in from two dozen
    different places**, and a bar's placement is not context-free: the positions were
    chosen for the run-up the SOURCE bar had, and dropped into a different
    neighbourhood they have no continuity with the notes on either side.

    A human copies a contiguous span and then varies it. Requiring the target and the
    source to advance together keeps the internal flow of the copied passage intact and
    leaves only the two seams new — and seams are what `fix_parity` and
    `enforce_reachability` are already good at repairing.
    """
    keep_src: dict[int, int] = {}
    keep_sim: dict[int, float] = {}
    run: list[int] = []

    def flush(run: list[int]) -> None:
        if len(run) >= min_run:
            for t in run:
                keep_src[t] = source[t]
                keep_sim[t] = sim[t]

    for t in sorted(source):
        if run and t == run[-1] + 1 and source[t] == source[run[-1]] + 1:
            run.append(t)
        else:
            flush(run)
            run = [t]
    flush(run)
    return keep_src, keep_sim


def plan_reuse_diagonal(S: dict, edges: np.ndarray, *,
                        min_sim: float = DEFAULT_MIN_SIM,
                        min_lag: int = DEFAULT_MIN_LAG,
                        energy_tol: float = DEFAULT_ENERGY_TOL,
                        min_run: int = 4,
                        smooth: int = 4) -> ReusePlan:
    """Find repeats as DIAGONAL STRIPES, which is what a repeated section actually is.

    🔴★**WHY THE PER-BAR VERSION HAD TO BE REPLACED, AND WHAT ITS FAILURE PROVED.**
    `plan_reuse` gives every bar its own independent argmax over earlier bars. The
    first arm (`me_z20`) shipped that and **broke flow 0.37 -> 0.75 and idiom
    0.40 -> 1.07** against the 149-song control while every rhythm-side axis stayed
    identical to 4 dp. Measuring the plan itself explained it: **only 15.6 % of copied
    bars continued the previous bar's copy.** The lever was shuffling ~29 bars per song
    in from two dozen places, and placement is not context-free — positions chosen for
    the source bar's run-up have no continuity with their new neighbours.

    ⚠️**AND THE OBVIOUS FIX WAS THE WRONG ONE.** Simply *requiring* contiguity
    (`min_run` on the per-bar plan) collapsed the copy share 0.297 -> 0.085 at run 2 and
    0.017 at run 4, keeping any copy at all on 16/60 songs. Read carelessly that says
    "songs do not contain contiguous repeats", which is plainly false of pop music. The
    real cause is **tie-breaking**: when a chorus returns four times, bar *i* has
    several near-equal sources and picks one, bar *i+1* independently picks another.
    The shuffle was an artifact of deciding each bar alone — the same disease as C1,
    one level up.

    ★So decide the whole stripe at once. A repeated section is a **diagonal** in the
    self-similarity matrix: bars *i..i+k* matching *j..j+k* at a constant lag. Smoothing
    along each lag's diagonal and taking runs above threshold finds sections directly,
    and every bar in a run shares one lag by construction — contiguity is a property of
    the representation rather than a filter applied afterwards.

    ⚠️Overlaps are resolved by mean similarity, strongest stripe first, so a bar is
    never claimed by two sections.
    """
    n = len(edges) - 1
    H, R, E = S["harm"], S["rhy"], S["energy"]
    combined = np.minimum(H, R)

    segs: list[tuple[float, int, int, int]] = []       # (score, start, end, lag)
    for lag in range(min_lag, n):
        idx = np.arange(lag, n)
        if len(idx) < min_run:
            continue
        d = np.array([combined[i, i - lag] for i in idx], dtype=float)
        d = np.where(np.isfinite(d), d, -1.0)
        if smooth > 1 and len(d) >= smooth:
            k = np.ones(smooth) / smooth
            ds = np.convolve(d, k, mode="same")
        else:
            ds = d
        good = ds >= min_sim
        start = None
        for pos in range(len(good) + 1):
            if pos < len(good) and good[pos]:
                start = pos if start is None else start
                continue
            if start is not None:
                if pos - start >= min_run:
                    segs.append((float(d[start:pos].mean()),
                                 int(idx[start]), int(idx[pos - 1]), lag))
                start = None

    source: dict[int, int] = {}
    sim: dict[int, float] = {}
    for score, a, b, lag in sorted(segs, key=lambda x: -x[0]):
        for t in range(a, b + 1):
            if t in source:
                continue
            src = t - lag
            if src < 0:
                continue
            ei, ej = float(E[t]), float(E[src])
            if ei <= 0 or ej <= 0 or abs(np.log(ei / ej)) > np.log(energy_tol):
                continue
            source[t] = src
            sim[t] = score

    # Resolve chains so a returning section points at its origin, not at the previous
    # return — what makes the structure panel read as discrete squares, not a smear.
    for t in sorted(source):
        root, seen = source[t], {t}
        while root in source and root not in seen:
            seen.add(root)
            root = source[root]
        if root != t:
            source[t] = root

    return ReusePlan(edges=edges, source=source, sim=sim, n_bars=n)


# -------------------------------------------------------------------------- apply

def _bar_of(t: float, edges: np.ndarray) -> int | None:
    if t < edges[0] or t >= edges[-1]:
        return None
    i = int(np.searchsorted(edges, t, side="right") - 1)
    return i if 0 <= i < len(edges) - 1 else None


def _slot_of(t: float, edges: np.ndarray, bi: int, slots: int = SLOTS_PER_BAR) -> int:
    dur = edges[bi + 1] - edges[bi]
    frac = (t - edges[bi]) / dur if dur > 0 else 0.0
    return int(min(slots - 1, max(0, round(frac * slots))))


def apply_reuse(beatmap, plan: ReusePlan, bpm: float, mode: str = "place") -> dict:
    """Rewrite the map so repeated music carries the repeated pattern.

    `place` copies (x, y, direction) only — note times are untouched, so no
    time-domain axis can move. `full` replaces the bar's notes outright.

    Returns a stats dict; the caller logs it. Mutates `beatmap.color_notes`.
    """
    notes = beatmap.color_notes
    if not notes or plan.n_copied == 0:
        return {"bars_copied": 0, "notes_changed": 0, "notes_added": 0,
                "notes_removed": 0, "share": 0.0}

    spb = 60.0 / bpm if bpm > 0 else 0.5
    edges = plan.edges

    # Index every note by (bar, slot, color). Ties inside a slot keep the first.
    by_bar: dict[int, list] = {}
    for nt in notes:
        bi = _bar_of(nt.beat * spb, edges)
        if bi is None:
            continue
        by_bar.setdefault(bi, []).append(nt)

    changed = added = removed = bars_done = 0

    if mode == "place":
        for tgt, src in sorted(plan.source.items()):
            srcmap: dict[tuple[int, int], object] = {}
            for nt in by_bar.get(src, []):
                srcmap.setdefault((_slot_of(nt.beat * spb, edges, src), nt.color), nt)
            if not srcmap:
                continue
            hit = False
            for nt in by_bar.get(tgt, []):
                s = _slot_of(nt.beat * spb, edges, tgt)
                ref = srcmap.get((s, nt.color))
                if ref is None:
                    # Nearest occupied slot of the same colour, within one slot. A
                    # repeat is rarely slot-exact after postprocess-free decoding, and
                    # refusing to bridge one slot would make the lever a near no-op.
                    for d in (-1, 1):
                        ref = srcmap.get((s + d, nt.color))
                        if ref is not None:
                            break
                if ref is None:
                    continue
                if (nt.x, nt.y, nt.direction) != (ref.x, ref.y, ref.direction):
                    nt.x, nt.y, nt.direction = ref.x, ref.y, ref.direction
                    changed += 1
                hit = True
            bars_done += 1 if hit else 0

    elif mode == "full":
        from beatsaber_automapper.data.beatmap import ColorNote

        drop: set[int] = set()
        new: list = []
        for tgt, src in sorted(plan.source.items()):
            s_notes = by_bar.get(src, [])
            if not s_notes:
                continue
            t0_s, t0_t = edges[src], edges[tgt]
            scale = (edges[tgt + 1] - t0_t) / max(edges[src + 1] - t0_s, 1e-9)
            for nt in by_bar.get(tgt, []):
                drop.add(id(nt))
                removed += 1
            for nt in s_notes:
                t = t0_t + (nt.beat * spb - t0_s) * scale
                if not (edges[tgt] <= t < edges[tgt + 1]):
                    continue
                new.append(ColorNote(beat=t / spb, x=nt.x, y=nt.y,
                                     color=nt.color, direction=nt.direction))
                added += 1
            bars_done += 1
        if drop or new:
            kept = [nt for nt in notes if id(nt) not in drop]
            beatmap.color_notes = sorted(kept + new, key=lambda n: n.beat)
    else:
        raise ValueError(f"unknown structure-reuse mode {mode!r}")

    return {"bars_copied": bars_done, "notes_changed": changed,
            "notes_added": added, "notes_removed": removed,
            "share": round(plan.share, 3)}


# ------------------------------------------------------------------- entry point

def maybe_apply(beatmap, waveform, sr: int, stems: dict, bpm: float, end: float):
    """Read `BEAT_STRUCTURE_REUSE` and apply the lever. Default OFF.

    Spec: ``<mode>[:<min_sim>[:<min_lag>[:<energy_tol>[:<min_z>[:<min_run>]]]]]`` —
    e.g. ``place``, ``place:0.7``, ``full:0.65:8``, ``place:0.6:4:1.5:2.0:4``.
    ⚠️`min_run` defaults to 1 = the original per-bar behaviour, which **broke flow and
    idiom** (see `_keep_runs`). Any new arm should set it. Empty/unset = untouched map, so production
    behaviour is unchanged unless the variable is set (Kyle's standing rule: isolated,
    tactical, default OFF, his ear decides).

    Every failure path is a silent no-op with a warning: a lever must never be able to
    take the generator down.
    """
    spec = os.environ.get("BEAT_STRUCTURE_REUSE", "").strip()
    if not spec:
        return None
    parts = spec.split(":")
    mode = parts[0] or "place"
    try:
        min_sim = float(parts[1]) if len(parts) > 1 and parts[1] else DEFAULT_MIN_SIM
        min_lag = int(parts[2]) if len(parts) > 2 and parts[2] else DEFAULT_MIN_LAG
        etol = float(parts[3]) if len(parts) > 3 and parts[3] else DEFAULT_ENERGY_TOL
        min_z = float(parts[4]) if len(parts) > 4 and parts[4] else DEFAULT_MIN_Z
        min_run = int(parts[5]) if len(parts) > 5 and parts[5] else DEFAULT_MIN_RUN
    except ValueError:
        logger.warning("BEAT_STRUCTURE_REUSE=%r unparseable — skipped", spec)
        return None

    try:
        import librosa

        y = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
        if y.ndim > 1:
            y = y.mean(axis=tuple(range(y.ndim - 1)))
        y = np.ascontiguousarray(y, dtype="float32")

        # Carrier = drums+bass onsets, the same signal BEAT_MAIN_BEAT_BONUS uses, so
        # the two levers cannot disagree about where the main beat is.
        car: list[float] = []
        for s in ("drums", "bass"):
            if s not in stems:
                continue
            a = stems[s].detach().cpu().numpy()
            if a.ndim > 1:
                a = a.mean(axis=tuple(range(a.ndim - 1)))
            car.extend(librosa.onset.onset_detect(y=a.astype("float32"), sr=sr,
                                                  units="time", backtrack=True).tolist())
        carrier = np.sort(np.asarray(car, dtype=float))

        hop = 512
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        et = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
        edges = bar_edges(carrier, env, et, bpm, end)
        if edges is None or len(edges) < 6:
            logger.warning("BEAT_STRUCTURE_REUSE: no usable bar grid — skipped")
            return None

        S = audio_bar_ssm(y, sr, edges)
        if S is None:
            logger.warning("BEAT_STRUCTURE_REUSE: no audio SSM — skipped")
            return None

        if mode.startswith("diag_"):
            mode = mode[len("diag_"):]
            plan = plan_reuse_diagonal(S, edges, min_sim=min_sim, min_lag=min_lag,
                                       energy_tol=etol,
                                       min_run=max(min_run, 2))
        else:
            plan = plan_reuse(S, edges, min_sim=min_sim, min_lag=min_lag,
                              energy_tol=etol, min_z=min_z, min_run=min_run)
        stats = apply_reuse(beatmap, plan, bpm, mode=mode)
        logger.info(
            "BEAT_STRUCTURE_REUSE=%s: %d/%d bars are musical repeats (%.0f%%), "
            "%d copied, %d notes re-placed, +%d/-%d notes",
            spec, plan.n_copied, plan.n_bars, 100 * plan.share,
            stats["bars_copied"], stats["notes_changed"],
            stats["notes_added"], stats["notes_removed"],
        )
        return stats
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("BEAT_STRUCTURE_REUSE failed (%s) — skipped", exc)
        return None
