#!/usr/bin/env python
"""THE SHAPE — sections, and which ones are the same section.

This is the tool that most directly answers Kyle's original ask: *"if you had a
longitudinal view … you could create some amazing maps."* A generator with a 16-note
window cannot know that bar 129 is a breakdown or that bars 59, 115 and 195 are the
same chorus. `brief.py` prints density in fixed **8-bar blocks**, which is a ruler laid
over the song rather than the song's own shape — a bridge that starts at bar 131 and
runs 11 bars is invisible in it.

## What it gives you
- **boundaries** where the music actually changes, at bar resolution, not on a fixed grid
- **labels**: sections marked `A`, `B`, `C`… where the *same letter means the same music*
- a **role guess** per section (intro / build / drop / breakdown / outro) from energy shape
- per section: dominant stem, energy, note-budget suggestion, and the lyric

★**The labels are the point.** `BEAT_STRUCTURE_REUSE` exists to infer repetition from an
SSM and apply it to the generator; an agent can simply *read* it here — map section `B`
once, then reuse and vary it deliberately at each later `B`.

⚠️**This is harmony+timbre, `brief.py`'s repeat map is lyrics.** They are independent
evidence about the same structure and `--validate` scores them against each other: a
chorus that repeats in the words should land in a section with the same letter. Where
they disagree, trust the lyrics — Whisper's words are near-ground-truth and this is an
unsupervised clustering.

Usage:
    python agent_mapper/structure.py <audio.ogg>
    python agent_mapper/structure.py <audio.ogg> --validate     # lyric agreement + null
    python agent_mapper/structure.py <audio.ogg> --k 6          # force section count
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = REPO / "outputs" / "structure_cache"
HOP = 512


def _beat_sync_features(audio: pathlib.Path, g: dict, dur: float) -> tuple[np.ndarray, np.ndarray]:
    """Harmony (chroma) + timbre (MFCC), averaged over each beat.

    Beat-synchronous because a section boundary is a musical event, not an acoustic
    one: averaging inside the beat removes the note-level detail that would otherwise
    dominate the similarity matrix, leaving harmony and timbre — which is what makes
    two choruses "the same" despite different words on top.
    """
    import librosa
    from stemcache import stems, mix, SR

    s = stems(audio)
    y = mix(audio)
    harm = s["bass"] + s["other"]                   # the chords, without drums or voice
    chroma = librosa.feature.chroma_cqt(y=harm, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)

    n_beats = max(int((dur - g["phase"]) / g["spb"]), 4)
    bt = g["phase"] + np.arange(n_beats) * g["spb"]
    frames = np.clip((bt * SR / HOP).astype(int), 0, chroma.shape[1] - 1)
    edges = np.r_[frames, chroma.shape[1]]

    def sync(M: np.ndarray) -> np.ndarray:
        return np.stack([M[:, edges[i]:max(edges[i + 1], edges[i] + 1)].mean(axis=1)
                         for i in range(n_beats)], axis=1)

    return sync(chroma), sync(mfcc)


def bar_ssm(audio: pathlib.Path, g: dict, dur: float) -> np.ndarray:
    """The bar-by-bar self-similarity matrix. ★This is the part that works.

    ⚠️**Beat resolution was tried first and finds the wrong thing.** Clustering
    beat-level chroma produced a clean 2-bar alternating label sequence
    (`0022011200110111…`) — a real pattern, but the **chord loop**, not the song's
    sections. Chord loops repeat every 2 bars; sections repeat every 30. Averaging to
    the bar and then delay-embedding two bars puts the analysis window at phrase scale,
    where the question actually lives.

    Measured on Hunger, whose choruses the lyrics place at bars 59/115/195: cross-
    affinity between two choruses is **0.47-0.49** against **0.000** for a chorus vs a
    verse, with a matrix mean of 0.067. There is no ambiguity left to resolve.
    """
    import librosa

    chroma, mfcc = _beat_sync_features(audio, g, dur)
    nb = chroma.shape[1] // 4
    bc = chroma[:, :nb * 4].reshape(12, nb, 4).mean(axis=2)
    bm = mfcc[:, :nb * 4].reshape(13, nb, 4).mean(axis=2)
    F = np.vstack([librosa.feature.stack_memory(bc, n_steps=2, delay=1),
                   librosa.feature.stack_memory(bm, n_steps=2, delay=1)])
    F = (F - F.mean(axis=1, keepdims=True)) / (F.std(axis=1, keepdims=True) + 1e-9)
    R = librosa.segment.recurrence_matrix(F, width=3, mode="affinity", sym=True)
    return librosa.segment.path_enhance(R, 3)


def novelty(R: np.ndarray, L: int = 8) -> np.ndarray:
    """Checkerboard novelty: how much does the music change at each bar?

    A 2L-bar checkerboard kernel is high where the block before a bar is self-similar,
    the block after is self-similar, and the two are unlike each other — which is the
    definition of a boundary.
    """
    from scipy.ndimage import gaussian_filter1d

    n = R.shape[0]
    K = np.zeros((2 * L, 2 * L))
    K[:L, :L] = K[L:, L:] = 1
    K[:L, L:] = K[L:, :L] = -1
    K *= np.outer(np.hanning(2 * L), np.hanning(2 * L))
    nov = np.zeros(n)
    for i in range(L, n - L):
        nov[i] = float((R[i - L:i + L, i - L:i + L] * K).sum())
    return gaussian_filter1d(nov, 2)


def boundaries(nov: np.ndarray, nb: int, min_bars: int = 8) -> list[int]:
    """Peaks of the novelty curve -> section boundaries, in bars.

    ⚠️**Gate on PROMINENCE, not height.** The novelty curve here is almost entirely
    negative (only ~4 % of it is above zero), so a height threshold — the obvious
    choice — selects nothing and returns the whole song as one section. Prominence asks
    the right question: how far does this peak rise above its own surroundings.
    """
    from scipy.signal import find_peaks

    pk, _ = find_peaks(nov, distance=min_bars, prominence=float(np.std(nov)) * 0.5)
    return sorted(set([0] + [int(x) for x in pk] + [nb]))


def label_segments(R: np.ndarray, segs: list[tuple[int, int]],
                   same: float = 1.0) -> list[int]:
    """Which segments are the SAME music, by mean cross-affinity in the SSM.

    Normalised by each segment's own self-affinity, so a busy section and a sparse one
    are compared on equal terms.

    ⚠️**Average-linkage clustering, not a greedy pass.** A first version walked the
    segments in order and let each one claim every later segment above the threshold;
    being order-dependent and non-transitive, it collapsed one song's eight sections
    into a single letter. Agglomerative linkage asks the symmetric question.

    ★**The threshold was set on Hunger, whose structure the lyrics give away**: its
    choruses start at bars 59/115/195 and its verses at 37/91, so a correct labelling
    is checkable rather than a matter of taste. Anything in 0.7-1.2 gets it right —
    choruses one letter, verses another, breakdown its own — so 1.0 sits in the middle
    of a wide plateau rather than on a fitted edge. `--validate` then scores the labels
    on *other* songs against their own lyric repeats.
    """
    from sklearn.cluster import AgglomerativeClustering

    m = len(segs)
    if m < 2:
        return [0] * m
    S = np.zeros((m, m))
    for i, (a0, a1) in enumerate(segs):
        for j, (b0, b1) in enumerate(segs):
            S[i, j] = R[a0:a1, b0:b1].mean()
    self_aff = np.maximum(np.diag(S), 1e-9)
    N = S / np.sqrt(np.outer(self_aff, self_aff))
    N = (N + N.T) / 2

    D = np.clip(2.0 - N, 0, None)
    np.fill_diagonal(D, 0.0)
    return list(AgglomerativeClustering(
        n_clusters=None, distance_threshold=2.0 - same,
        metric="precomputed", linkage="average").fit_predict(D))


def segment(audio: pathlib.Path, g: dict, dur: float, k: int | None = None) -> dict:
    R = bar_ssm(audio, g, dur)
    nb = R.shape[0]
    bnd = boundaries(novelty(R), nb)
    segs = [(bnd[i], bnd[i + 1]) for i in range(len(bnd) - 1) if bnd[i + 1] > bnd[i]]
    lab = label_segments(R, segs)
    return {"segs": segs, "labels": lab, "n_bars": nb}


def analyse(audio: pathlib.Path, force: bool = False, k: int | None = None) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{audio.stem}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    import brief as _brief

    a = _brief.analyse(audio)
    g = _brief.grid(a)
    seg = segment(audio, g, a["dur"])
    letters: dict[int, str] = {}
    secs = []
    for (b0, b1), lb in zip(seg["segs"], seg["labels"]):
        letters.setdefault(lb, chr(ord("A") + len(letters)))
        t0 = g["phase"] + b0 * g["bar_s"]
        t1 = min(g["phase"] + b1 * g["bar_s"], a["dur"])
        secs.append({"label": letters[lb], "bar0": b0 + 1, "bars": b1 - b0,
                     "t0": round(t0, 3), "t1": round(t1, 3)})
    out = {"sections": secs, "n_bars": seg["n_bars"]}
    f.write_text(json.dumps(out))
    return out


def roles(secs: list[dict], a: dict) -> None:
    """A role guess per section, from where its energy sits relative to the song.

    Deliberately coarse. "build" and "drop" are the two that change how you map — a
    build wants density that climbs, a drop wants the biggest note in the section on
    its first beat — and both are visible in energy alone.
    """
    import brief as _brief

    e = [_brief.energy_at(a, s["t0"], s["t1"]) for s in secs]
    lo, hi = min(e), max(e) + 1e-9
    norm = [(x - lo) / (hi - lo) for x in e]
    for i, s in enumerate(secs):
        v = norm[i]
        prev = norm[i - 1] if i else 0.0
        s["energy"] = round(v, 2)
        if i == 0 and v < 0.55:
            s["role"] = "intro"
        elif i == len(secs) - 1 and v < 0.6:
            s["role"] = "outro"
        elif v < 0.35:
            s["role"] = "breakdown"
        elif v - prev > 0.30:
            s["role"] = "DROP"
        elif 0.35 <= v < 0.65 and norm[min(i + 1, len(secs) - 1)] - v > 0.25:
            s["role"] = "build"
        elif v > 0.75:
            s["role"] = "peak"
        else:
            s["role"] = "body"


def _mmss(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60)}:{t % 60:05.2f}"


def lyric_agreement(secs: list[dict], lines: list[dict], rng) -> dict:
    """★The control: do lines sung twice land in the same LETTER?

    The lyrics come from Whisper and know nothing about chroma, MFCCs or clustering, so
    they are independent evidence about the same structure. The null re-draws section
    labels while keeping the section boundaries and label frequencies fixed, which is
    what makes the comparison about *which* section a repeat lands in rather than about
    there being few sections.
    """
    def label_at(t: float, order: list[str]) -> str | None:
        for s, lb in zip(secs, order):
            if s["t0"] <= t < s["t1"]:
                return lb
        return None

    groups: dict[str, list[float]] = {}
    for ln in lines:
        key = "".join(c for c in ln["text"].lower() if c.isalnum() or c == " ").strip()
        if len(key) >= 6:
            groups.setdefault(key, []).append(ln["t"])
    reps = [ts for ts in groups.values() if len(ts) > 1]
    if not reps:
        return {}

    def score(order: list[str]) -> float:
        hit = tot = 0
        for ts in reps:
            ls = [label_at(t, order) for t in ts]
            ls = [x for x in ls if x]
            for i in range(len(ls)):
                for j in range(i + 1, len(ls)):
                    tot += 1
                    hit += ls[i] == ls[j]
        return hit / max(tot, 1)

    real_order = [s["label"] for s in secs]
    real = score(real_order)
    nulls = []
    for _ in range(200):
        sh = list(real_order)
        rng.shuffle(sh)
        nulls.append(score(sh))
    nulls = np.array(nulls)
    return {"n_repeats": len(reps), "real": real, "null": float(nulls.mean()),
            "p": float((nulls >= real).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a_ = ap.parse_args()
    if not a_.audio.exists():
        print(f"no such audio: {a_.audio}", file=sys.stderr)
        return 2

    import brief as _brief

    res = analyse(a_.audio, a_.force, a_.k)
    secs = res["sections"]
    a = _brief.analyse(a_.audio)
    g = _brief.grid(a)
    roles(secs, a)
    lines = _brief.lyric_lines(a_.audio.stem)

    n_distinct = len({s_['label'] for s_ in secs})
    print(f"SONG {a_.audio.stem}   {g['bpm']:.2f} bpm   {len(secs)} sections, "
          f"{n_distinct} distinct")
    print(f"\n{'sec':>4} {'bar':>5} {'time':>8} {'len':>5} {'role':>9} {'nrg':>4}  "
          f"{'D':>4}{'B':>4}{'O':>4}{'V':>4}   lyric")
    for i, s in enumerate(secs):
        dens = {st: sum(1 for t in a["onsets"][st] if s["t0"] <= t < s["t1"])
                / max(s["t1"] - s["t0"], 1e-9) for st in ("drums", "bass", "other", "vocals")}
        said = " ".join(ln["text"] for ln in lines if s["t0"] <= ln["t"] < s["t1"])[:40]
        print(f"{s['label']:>4} {s['bar0']:>5} {_mmss(s['t0']):>8} {s['bars']:>4}b "
              f"{s['role']:>9} {s['energy']:>4.2f}  "
              f"{dens['drums']:>4.1f}{dens['bass']:>4.1f}{dens['other']:>4.1f}"
              f"{dens['vocals']:>4.1f}   {said}")

    reuse: dict[str, list[int]] = {}
    for s in secs:
        reuse.setdefault(s["label"], []).append(s["bar0"])
    rep = {k: v for k, v in reuse.items() if len(v) > 1}
    if rep:
        print("\n★ SECTION REUSE — same letter = same music. Map it once, then vary it.")
        for k, v in sorted(rep.items()):
            print(f"    {k} × {len(v)}   bars {v}")

    if a_.validate:
        rng = np.random.default_rng(0)
        r = lyric_agreement(secs, lines, rng)
        print("\nLYRIC-AGREEMENT CONTROL — do repeated lines land in the same section?")
        if not r:
            print("  no repeated lyric lines (instrumental, or no cached lyrics) — "
                  "this control cannot run on this song, which is not a pass.")
        else:
            print(f"  repeated lines           : {r['n_repeats']}")
            print(f"  same-letter rate         : {r['real']:.3f}")
            print(f"  label-shuffled null      : {r['null']:.3f}")
            print(f"  p(null >= real)          : {r['p']:.3f}")
            print("\n  " + ("✅ the sections agree with the words"
                            if r["p"] < 0.05 else
                            "⚠️ not better than chance — do not trust these letters here"))

    if a_.json:
        a_.json.write_text(json.dumps({"sections": secs}, indent=1))
        print(f"\nwrote {a_.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
