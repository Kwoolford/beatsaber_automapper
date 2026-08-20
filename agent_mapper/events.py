#!/usr/bin/env python
"""Every sound in the song as a typed EVENT — the "note types" layer.

**Kyle, 2026-08-20:** *"The current note sheet looks like it could use much more data
for you to work with. Like these electric songs have LOTS of different note types."*

He is describing a real hole. Until now the perception layer saw a song as **four
buckets** -- drums, bass, other, vocals -- and `other` is where every synth, lead,
pluck, pad, stab, riser and effect in the track ends up. On a metal song that is
tolerable. On an electronic song it is most of the music, collapsed into one
undifferentiated lane, so the mapper could not tell a lead hook from a background
pad, and therefore could not give them different treatment the way a human mapper
does. `percussion.py` already solved this shape of problem for the kit; this
generalises it to every stem.

**Two widenings, and they are independent.**

1. **Six stems, not four.** `htdemucs_6s` splits `guitar` and `piano` out of
   `other`. Free, and it is a different network, so it is cached separately
   (`stemcache.stems6`) -- the 6-source `other` is NOT the 4-source `other`, and
   silently mixing them would change what every existing number was measured on.
2. **A per-song timbre vocabulary inside each stem.** The remaining `other` is
   clustered against ITSELF and each cluster is named by physics. This is the part
   that finds "lots of different note types" in an electronic track.

★**Why the clustering is self-relative, always.** A landmine already paid for in
`docs/perception_scorecard.md`: *absolute spectral thresholds do not transfer between
mixes* -- two hand-tuned drum classifiers each broke somewhere different (one called
47 % of hits "tom", the other 96 % "snare"). Within one song the mix is fixed, so
hits can be compared to each other; across songs they cannot. Every feature here is
standardised within the song before clustering, and every loudness is quoted
relative to that stem's own median.

★**The names are a VOCABULARY, not a claim.** Exactly as `percussion.py` records: the
control proves the classes are *consistent and carry structure*; it does not prove
the cluster called "pad" is a pad. Trust the separation, treat the name as a handle.

**What each event carries**

    t          seconds (the ground truth -- 30 % of our maps have the wrong tempo,
               so bar indices are not comparable across maps of the same song)
    bar.slot   position on the fitted 1/16 grid
    stem       one of six
    cls        the song's own timbre class within that stem
    pitch      MIDI note + confidence, where the stem is pitched
    loud       dB relative to that stem's median hit -- the accent
    dur        seconds until this class next sounds (how long it rings)

Usage:
    python agent_mapper/events.py data/eval_songset/1f767.ogg --summary
    python agent_mapper/events.py data/eval_songset/1f767.ogg --bars 33-40
    python agent_mapper/events.py data/eval_songset/1f767.ogg --validate
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "agent_mapper"))

CACHE = REPO / "outputs" / "event_cache"
SR = 22050

# Which physics apply when naming a cluster. `percussive` stems get kit names,
# `pitched` stems get instrument-role names.
FAMILY = {"drums": "percussive", "bass": "pitched", "other": "pitched",
          "vocals": "pitched", "guitar": "pitched", "piano": "pitched"}

# Log-spaced bands. Same edges as percussion.py so the two agree on the kit.
BANDS = [(0, 120), (120, 400), (400, 1500), (1500, 5000), (5000, 11000)]

MAX_K = 5          # per stem; more classes than this stops being a vocabulary
MIN_CLUSTER = 8    # a "class" with fewer hits than this is noise, not a class


# --------------------------------------------------------------------------
# per-onset features
# --------------------------------------------------------------------------
def onsets_of(y: np.ndarray, sr: int = SR) -> np.ndarray:
    import librosa
    if not np.any(np.abs(y) > 1e-5):
        return np.zeros(0, dtype="float32")
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    idx = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=256,
                                     backtrack=False, units="frames")
    return librosa.frames_to_time(idx, sr=sr, hop_length=256).astype("float32")


def features(y: np.ndarray, sr: int, onsets: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Per-onset descriptors, all computed from the stem's own audio.

    Columns are chosen so that clusters correspond to things a listener would call
    different instruments: WHERE the energy is (bands, centroid), HOW NOISY it is
    (flatness), HOW IT STARTS (attack), and HOW LONG IT RINGS (decay) -- the last
    being the one that separates a pluck from a pad, and a hat from a crash.
    """
    import librosa

    n_fft, hop = 1024, 128
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bidx = [np.where((freqs >= lo) & (freqs < hi))[0] for lo, hi in BANDS]
    nframe = S.shape[1]

    def fr(t: float) -> int:
        return int(np.clip(round(t * sr / hop), 0, nframe - 1))

    cols = [f"band{i}" for i in range(len(BANDS))] + [
        "centroid", "flatness", "decay", "attack", "level"]
    out = np.full((len(onsets), len(cols)), np.nan, dtype="float32")

    for i, t in enumerate(onsets):
        a, b = fr(t), fr(t + 0.050)
        seg = S[:, a:max(b, a + 1)].sum(axis=1)
        tot = seg.sum() + 1e-12
        for j, ix in enumerate(bidx):
            out[i, j] = seg[ix].sum() / tot
        # spectral centroid in log Hz -- log because pitch and timbre are log-scaled
        out[i, len(BANDS)] = np.log(
            (seg * freqs).sum() / tot + 1e-6)
        gm = np.exp(np.log(seg + 1e-12).mean())
        out[i, len(BANDS) + 1] = gm / (seg.mean() + 1e-12)      # flatness: noisy vs tonal

        # ⚠️The decay window MUST stop at the next onset. percussion.py records a
        # first version that measured a fixed 220-320 ms tail and read decay ratios
        # ABOVE 1.0 -- energy apparently rising after the strike -- because at 160 bpm
        # an eighth note is 187 ms and the window was full of the NEXT hit. It was
        # measuring the groove, not the decay.
        nxt = onsets[i + 1] if i + 1 < len(onsets) else t + 1.0
        t0, t1 = t + 0.120, min(t + 0.400, nxt - 0.020)
        if t1 - t0 >= 0.040:
            e0 = S[:, a:fr(t + 0.030) + 1].sum()
            n0 = max(fr(t + 0.030) - a + 1, 1)
            e1 = S[:, fr(t0):fr(t1) + 1].sum()
            n1 = max(fr(t1) - fr(t0) + 1, 1)
            out[i, len(BANDS) + 2] = np.log((e1 / n1) / (e0 / n0 + 1e-12) + 1e-6)
        # attack: how fast it arrives (frames from onset to peak within 60 ms)
        w = S[:, a:fr(t + 0.060) + 1].sum(axis=0)
        out[i, len(BANDS) + 3] = float(np.argmax(w)) if len(w) else 0.0
        out[i, len(BANDS) + 4] = np.log(S[:, a:max(b, a + 1)].sum() + 1e-12)
    return out, cols


def pitches_at(y: np.ndarray, sr: int, onsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dominant MIDI pitch just after each onset, plus a 0-1 confidence.

    ⚠️One pitch per ONSET, never a free segmentation of the f0 track: segmenting f0
    independently of the onsets gave **48 "notes" for a 343-word song**, because
    vibrato flips the rounded semitone every ~35 ms (perception_scorecard landmine).
    """
    import librosa
    if len(onsets) == 0:
        return np.zeros(0, "float32"), np.zeros(0, "float32")
    hop = 256
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"),
                           n_bins=72, bins_per_octave=12))
    midi0 = librosa.note_to_midi("C1")
    nf = C.shape[1]
    pit = np.full(len(onsets), np.nan, dtype="float32")
    con = np.zeros(len(onsets), dtype="float32")
    for i, t in enumerate(onsets):
        a = int(np.clip(round(t * sr / hop), 0, nf - 1))
        b = int(np.clip(a + int(0.080 * sr / hop), a + 1, nf))
        col = C[:, a:b].mean(axis=1)
        s = col.sum() + 1e-12
        k = int(np.argmax(col))
        pit[i] = midi0 + k
        # confidence = how peaked the spectrum is on that bin's pitch class
        con[i] = float(col[k] / s)
    return pit, con


# --------------------------------------------------------------------------
# clustering + naming
# --------------------------------------------------------------------------
def cluster(F: np.ndarray, seed: int = 0) -> tuple[np.ndarray, int, float]:
    """Cluster a stem's hits against ITSELF; pick k by silhouette. (labels, k, sil)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # ⚠️Impute, do not drop. `decay` is NaN whenever the next onset leaves no room to
    # hear the tail, and on a dense stem that is a THIRD of the events -- the first
    # run put 476 of piano's 1020 hits and 135 of drums' 637 into an untyped "?"
    # bucket purely because the stem was busy. An unmeasurable decay is not missing
    # data, it is evidence the sound had no room to ring, so it takes the column's
    # low end. `percussion.py` reaches the same conclusion from the other side:
    # it treats an unmeasurable decay as "cannot be a crash" rather than guessing.
    F = F.copy()
    for j in range(F.shape[1]):
        col = F[:, j]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            col[bad] = np.nanmin(col[~bad])
    good = np.isfinite(F).all(axis=1)
    if good.sum() < MIN_CLUSTER * 2:
        return np.zeros(len(F), dtype=int), 1, float("nan")
    X = F[good]
    # ★**Winsorise before standardising.** Without this a handful of freak onsets
    # own the clustering: on 1f8d6's drums, k=2 split 441 hits into 438 + **3** at a
    # silhouette of 0.94 -- a perfect score for isolating three outliers and learning
    # nothing about the kit. Every k up to 4 did the same, and the real split (223 vs
    # 211, plainly the kick and the rest) only appeared at k=5, where it was thrown
    # out because the same run also produced 2-point clusters. Clipping each feature
    # to its 1st-99th percentile removes the outliers' leverage without removing the
    # events themselves.
    lo = np.percentile(X, 1, axis=0)
    hi = np.percentile(X, 99, axis=0)
    X = np.clip(X, lo, hi)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    best = (np.zeros(len(X), dtype=int), 1, -1.0)
    for k in range(2, min(MAX_K, len(X) // MIN_CLUSTER) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        lab = km.labels_.copy()
        # ⚠️Merge tiny clusters into their nearest surviving centroid rather than
        # rejecting the whole k. Rejecting was what hid the kit split above: one
        # 2-point cluster vetoed an otherwise clean 223/211 separation.
        sizes = np.bincount(lab, minlength=k)
        small = [c for c in range(k) if sizes[c] < MIN_CLUSTER]
        keep = [c for c in range(k) if sizes[c] >= MIN_CLUSTER]
        if not keep:
            continue
        for c in small:
            d = np.linalg.norm(km.cluster_centers_[keep] - km.cluster_centers_[c],
                               axis=1)
            lab[lab == c] = keep[int(np.argmin(d))]
        uniq = sorted(set(lab.tolist()))
        if len(uniq) < 2:
            continue
        remap = {c: i for i, c in enumerate(uniq)}
        lab = np.array([remap[c] for c in lab], dtype=int)
        try:
            sil = float(silhouette_score(X, lab))
        except Exception:  # noqa: BLE001
            continue
        if sil > best[2]:
            best = (lab, len(uniq), sil)
    labels = np.full(len(F), -1, dtype=int)
    labels[good] = best[0]
    return labels, best[1], best[2]


def name_clusters(stem: str, F: np.ndarray, labels: np.ndarray, cols: list[str],
                  pitch_conf: np.ndarray, pitch_midi: np.ndarray) -> dict[int, str]:
    """Name each cluster by its centroid's physics.

    ★These names are a **handle**, not a measurement. The control below proves the
    classes are consistent and repeat with the song's structure; nothing proves the
    cluster called "pad" is a pad. `percussion.py` states the same caveat and it is
    the honest reading of a self-relative clustering.
    """
    ci = {c: i for i, c in enumerate(cols)}
    ids = [c for c in sorted(set(labels.tolist())) if c >= 0]
    cent = {c: np.nanmean(F[labels == c], axis=0) for c in ids}
    fam = FAMILY.get(stem, "pitched")
    out: dict[int, str] = {}

    if fam == "percussive":
        # Physics, hardest-to-mistake first: the kick owns the bottom band; of the
        # bright clusters the one that RINGS is the crash, the one that stops is the
        # hat; what is left in the middle is the snare, then toms.
        by_low = sorted(ids, key=lambda c: -cent[c][ci["band0"]])
        by_hi = sorted(ids, key=lambda c: -(cent[c][ci["band3"]] + cent[c][ci["band4"]]))
        out[by_low[0]] = "kick"
        bright = [c for c in by_hi if c not in out]
        if bright:
            ring = sorted(bright, key=lambda c: -np.nan_to_num(cent[c][ci["decay"]], nan=-9))
            out[ring[0]] = "crash" if len(bright) > 1 else "hat"
            for c in ring[1:]:
                out.setdefault(c, "hat")
        for c in sorted(ids, key=lambda c: -cent[c][ci["band2"]]):
            out.setdefault(c, "snare")
        for c in ids:
            out.setdefault(c, "tom")
        # only one of each of the singular names
        seen: dict[str, int] = {}
        for c in ids:
            n = out[c]
            seen[n] = seen.get(n, 0) + 1
            if seen[n] > 1:
                out[c] = f"{n}{seen[n]}"
        return out

    # Pitched families. ★**Named by what was MEASURED, not by instrument identity.**
    # The first version guessed roles -- "lead", "pad", "pluck", "fx" -- and produced
    # `guitar/fx` for 328 hits at +2.2 dB, which is plainly the main guitar part and
    # not an effect. Nothing in a self-relative clustering can license the claim "this
    # is a pad", and `percussion.py` already records the honest reading: **trust the
    # separation, treat the name as a handle.** So the handle now states the two
    # things the features actually support -- REGISTER (from the cluster's median
    # pitch) and ENVELOPE (from its decay) -- plus a `~` mark when the cluster is
    # noisy enough that its pitch should not be relied on. That is more useful to a
    # mapper anyway: "hi-stab" says how to map it; "pluck2" does not.
    reg_edges = [(0, 40, "sub"), (40, 55, "low"), (55, 68, "mid"), (68, 128, "hi")]
    dec = {c: cent[c][ci["decay"]] for c in ids}
    flt = {c: cent[c][ci["flatness"]] for c in ids}
    med_dec = float(np.median(list(dec.values())))
    med_flt = float(np.median(list(flt.values())))
    for c in ids:
        sel = labels == c
        mp = pitch_midi[sel]
        mp = mp[np.isfinite(mp)]
        reg = "mid"
        if len(mp):
            m = float(np.median(mp))
            for lo, hi, nm in reg_edges:
                if lo <= m < hi:
                    reg = nm
                    break
        env = "ring" if dec[c] > med_dec else "stab"
        noisy = "~" if flt[c] > med_flt else ""
        out[c] = f"{reg}-{env}{noisy}"
    seen: dict[str, int] = {}
    for c in ids:
        n = out[c]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            out[c] = f"{n}{seen[n]}"
    return out


# --------------------------------------------------------------------------
# the event table
# --------------------------------------------------------------------------
def analyse(audio: pathlib.Path, force: bool = False, six: bool = True,
            seed: int = 0) -> dict:
    """Every onset in every stem, typed, timed and placed on the bar grid."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "6s" if six else "4s"
    f = CACHE / f"{audio.stem}.{tag}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())

    import brief as _brief
    import stemcache as _sc

    st = _sc.stems6(audio) if six else _sc.stems(audio)
    a = _brief.analyse(audio)
    g = _brief.grid(a)

    events, per_stem = [], {}
    for stem, y in st.items():
        on = onsets_of(y, SR)
        if len(on) < MIN_CLUSTER * 2:
            per_stem[stem] = {"n": int(len(on)), "k": 0, "silhouette": None,
                              "classes": {}, "note": "too few onsets to type"}
            for t in on:
                events.append({"t": float(t), "stem": stem, "cls": stem})
            continue
        F, cols = features(y, SR, on)
        lab, k, sil = cluster(F, seed=seed)
        # ⚠️k=1 means no split survived the silhouette + min-size test. Naming that
        # single group by physics would be an outright claim: on 1f8d6 it labelled
        # ALL 441 drum hits "kick", which is plainly false for a song with a beat.
        # An unsplit stem is reported as unsplit -- silent degradation is the failure
        # mode that hid the missing alignment axis for two nights.
        if k <= 1 or not np.isfinite(sil):
            names = {c: f"{stem}-all" for c in set(lab.tolist()) if c >= 0}
            lev = F[:, cols.index("level")]
            ref = float(np.nanmedian(lev)) if np.isfinite(lev).any() else 0.0
            for i, t in enumerate(on):
                events.append({"t": float(t), "stem": stem, "cls": f"{stem}-all",
                               "loud": round(float(4.343 * (lev[i] - ref)), 2),
                               "dur": -1.0})
            per_stem[stem] = {"n": int(len(on)), "k": 1, "silhouette": None,
                              "classes": {f"{stem}-all": int(len(on))},
                              "note": "NOT SEPARATED - no split passed the silhouette "
                                      "and minimum-size test; read as one lane"}
            continue
        pit, con = pitches_at(y, SR, on)
        if FAMILY.get(stem) != "pitched":
            pit = np.full(len(on), np.nan, dtype="float32")
        names = name_clusters(stem, F, lab, cols, con, pit)

        # loudness relative to this stem's own median hit -- absolute dB does not
        # transfer between mixes, and the accent is what a mapper actually reads.
        lev = F[:, cols.index("level")]
        ref = float(np.nanmedian(lev)) if np.isfinite(lev).any() else 0.0
        pitched = FAMILY.get(stem) == "pitched"

        for i, t in enumerate(on):
            c = int(lab[i])
            nm = names.get(c, "?") if c >= 0 else "?"
            same = np.where(lab == c)[0]
            nxt = same[same > i]
            ev = {
                "t": float(t), "stem": stem, "cls": nm,
                "loud": round(float(4.343 * (lev[i] - ref)), 2),   # ~dB, relative
                "dur": round(float(on[nxt[0]] - t) if len(nxt) else -1.0, 3),
            }
            if pitched and np.isfinite(pit[i]):
                ev["midi"] = int(pit[i])
                ev["pconf"] = round(float(con[i]), 3)
            events.append(ev)
        per_stem[stem] = {
            "n": int(len(on)), "k": int(k),
            "silhouette": None if not np.isfinite(sil) else round(float(sil), 3),
            "classes": {names.get(c, "?"): int((lab == c).sum())
                        for c in sorted(set(lab.tolist())) if c >= 0},
        }

    for ev in events:
        bar = (ev["t"] - g["phase"]) / g["bar_s"]
        ev["bar"] = int(np.floor(bar)) + 1
        ev["slot"] = int(round((bar - np.floor(bar)) * 16)) % 16
    events.sort(key=lambda e: e["t"])

    out = {"song": audio.stem, "model": tag, "bpm": g["bpm"], "phase": g["phase"],
           "bar_s": g["bar_s"], "n_bars": g["n_bars"], "grid_r": float(a["r"]),
           "stems": per_stem, "events": events}
    f.write_text(json.dumps(out))
    return out


# --------------------------------------------------------------------------
# the control: does the typing carry structure, or is it noise?
# --------------------------------------------------------------------------
# ★Stems this control does NOT apply to. A vocal line is not supposed to repeat
# bar-to-bar -- a singer sings different words on almost every bar -- so a
# bar-to-bar repetition null asks the wrong question of it, and answered "NOT
# TRUSTED" on 4 of the first 6 songs for a stem that was working. **A control the
# ground truth fails is not a control**: the same rule that retired the backbeat
# control (perception_scorecard) and the first `too_sparse` control (mapjudge).
# Vocals are validated instead by `lyrics.py` sung-coverage, which is the right
# question for them.
NO_REPEAT_CONTROL = {"vocals"}


def repetition_z(d: dict, stem: str, n_shuffle: int = 200, seed: int = 0) -> dict:
    """Do this stem's CLASS LABELS repeat bar-to-bar more than shuffled labels?

    ★This is the control `percussion.py` established and it is the only thing that
    makes the class names worth printing. A clustering always returns clusters; what
    has to be shown is that the resulting labelling **carries the song's structure**.
    If real labels agree between bars no better than labels shuffled among the same
    onsets, the typing is decoration and must be reported as untrustworthy rather
    than quietly used.

    ⚠️Shuffle the LABELS and keep the TIMES. Shuffling times would destroy the groove
    as well and the test would pass for the wrong reason.
    """
    if stem in NO_REPEAT_CONTROL:
        return {"stem": stem, "z": None, "na": True,
                "note": "n/a - a vocal line is not expected to repeat bar-to-bar; "
                        "validate this stem with lyrics.py sung-coverage"}
    rng = np.random.default_rng(seed)
    ev = [e for e in d["events"] if e["stem"] == stem and e.get("cls")]
    if len(ev) < 20:
        return {"stem": stem, "z": None, "note": "too few events"}
    bars = sorted({e["bar"] for e in ev})
    if len(bars) < 8:
        return {"stem": stem, "z": None, "note": "too few bars"}
    classes = sorted({e["cls"] for e in ev})
    if len(classes) < 2:
        return {"stem": stem, "z": None, "note": "only one class"}
    cidx = {c: i for i, c in enumerate(classes)}
    bidx = {b: i for i, b in enumerate(bars)}

    def agree(labels: list[int]) -> float:
        grid = np.full((len(bars), 16), -1, dtype=int)
        for e, c in zip(ev, labels):
            grid[bidx[e["bar"]], e["slot"]] = c
        tot = hit = 0
        for i in range(len(bars) - 1):
            m = (grid[i] >= 0) & (grid[i + 1] >= 0)
            tot += int(m.sum())
            hit += int((grid[i][m] == grid[i + 1][m]).sum())
        return hit / tot if tot else float("nan")

    # ⚠️When one class holds almost everything the control is at its CEILING and
    # cannot resolve: on 1f333's bass, 542 of 556 events share a class, and
    # bar-to-bar agreement is 0.953 real against 0.950 shuffled -- both pinned near
    # 1.0 because almost any two slots match. That is **not measurable**, not
    # refuted, and the difference matters (DOC CONVENTION: a null from a blunt ruler
    # is "not yet measurable").
    counts = {}
    for e in ev:
        counts[e["cls"]] = counts.get(e["cls"], 0) + 1
    dom = max(counts.values()) / len(ev)
    if dom > 0.90:
        return {"stem": stem, "z": None, "na": True, "dominance": round(dom, 3),
                "note": f"not resolvable - one class holds {dom:.0%} of events, so "
                        f"bar-to-bar agreement is at its ceiling in both arms"}

    real = agree([cidx[e["cls"]] for e in ev])
    null = []
    base = [cidx[e["cls"]] for e in ev]
    for _ in range(n_shuffle):
        null.append(agree(list(rng.permutation(base))))
    null = np.array([x for x in null if np.isfinite(x)])
    if not np.isfinite(real) or len(null) < 10 or null.std() < 1e-9:
        return {"stem": stem, "z": None, "note": "null degenerate"}
    z = (real - null.mean()) / null.std()
    return {"stem": stem, "z": round(float(z), 1), "agree": round(float(real), 3),
            "null": round(float(null.mean()), 3), "n": len(ev),
            "classes": len(classes)}


def _mmss(t: float) -> str:
    """m:ss.hh, sign-safe. ★A fitted downbeat can be NEGATIVE (Hunger's is -21 ms) and
    a naive divmod turned that into '-1:59.98' on the first row of every brief."""
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60)}:{t % 60:05.2f}"


def lanes(d: dict, b0: int, b1: int) -> str:
    """One row per (stem, class) over a bar range -- the score view, 16 cells a bar."""
    keys = sorted({(e["stem"], e["cls"]) for e in d["events"]
                   if b0 <= e["bar"] <= b1})
    if not keys:
        return "  (no events in that range)"
    rows = []
    width = max(len(f"{s}/{c}") for s, c in keys)
    labels = {(s, c): f"{s}/{c}".ljust(width) for s, c in keys}
    for stem, cls in keys:
        cells = {}
        for e in d["events"]:
            if e["stem"] == stem and e["cls"] == cls and b0 <= e["bar"] <= b1:
                cells[(e["bar"], e["slot"])] = "X" if e.get("loud", 0) > 3 else "x"
        line = []
        for b in range(b0, b1 + 1):
            line.append("".join(cells.get((b, s), ".") for s in range(16)))
        rows.append(f"  {labels[(stem, cls)]} |" + "|".join(line) + "|")
    hdr = "  " + " " * width + "  " + " ".join(
        f"{'bar ' + str(b):<16}" for b in range(b0, b1 + 1))
    return hdr + "\n" + "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--bars", help="bar range, e.g. 33-40")
    ap.add_argument("--validate", action="store_true",
                    help="does the typing carry structure? (shuffled-label null)")
    ap.add_argument("--four-stem", action="store_true",
                    help="use the 4-source model (default is htdemucs_6s)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", type=pathlib.Path)
    a = ap.parse_args()

    d = analyse(a.audio, force=a.force, six=not a.four_stem)
    ev = d["events"]
    print(f"{d['song']}  {d['model']}  bpm {d['bpm']:.2f} (grid fit r={d['grid_r']:.2f})"
          f"  {d['n_bars']} bars  {len(ev)} events")

    ntypes = len({(e['stem'], e['cls']) for e in ev})
    print(f"★ {ntypes} distinct note types across {len(d['stems'])} stems\n")
    print(f"  {'stem':<8} {'events':>7} {'k':>3} {'sil':>6}  classes")
    print("  " + "-" * 62)
    for stem, s in d["stems"].items():
        cls = ", ".join(f"{k}:{v}" for k, v in
                        sorted(s["classes"].items(), key=lambda kv: -kv[1]))
        sil = f"{s['silhouette']:.3f}" if s.get("silhouette") is not None else "-"
        print(f"  {stem:<8} {s['n']:>7} {s['k']:>3} {sil:>6}  {cls or s.get('note','')}")

    if a.summary:
        print("\n  loudest / most pitched classes")
        for stem, cls in sorted({(e["stem"], e["cls"]) for e in ev}):
            sel = [e for e in ev if e["stem"] == stem and e["cls"] == cls]
            lo = np.mean([e.get("loud", 0) for e in sel])
            mp = [e["midi"] for e in sel if "midi" in e]
            pit = (f"midi {int(np.median(mp))} +-{int(np.std(mp))}"
                   if len(mp) > 4 else "unpitched")
            print(f"    {stem}/{cls:<10} n={len(sel):<5} accent {lo:+5.1f} dB   {pit}")

    if a.validate:
        print("\n  CONTROL: do the class labels repeat bar-to-bar vs a shuffled null?")
        print("  (z > 3 = the typing carries structure; low z = names are decoration)")
        any_bad = False
        for stem in d["stems"]:
            r = repetition_z(d, stem)
            if r.get("z") is None:
                print(f"    {stem:<8} -       {r.get('note','')}")
                continue
            ok = "TRUSTED" if r["z"] >= 3 else "⚠️ NOT TRUSTED"
            any_bad |= r["z"] < 3
            print(f"    {stem:<8} z={r['z']:>6}  agree {r['agree']:.3f} vs null "
                  f"{r['null']:.3f}  ({r['classes']} classes, n={r['n']})  {ok}")
        if any_bad:
            print("    ⚠️ A stem that is not trusted should be read as ONE lane, not "
                  "as its classes.")

    if a.bars:
        b0, b1 = (int(x) for x in a.bars.split("-"))
        t0 = d["phase"] + (b0 - 1) * d["bar_s"]
        print(f"\n  bars {b0}-{b1}  (starts {_mmss(t0)})")
        print(lanes(d, b0, b1))

    if a.json:
        a.json.write_text(json.dumps(d, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
