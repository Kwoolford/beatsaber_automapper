#!/usr/bin/env python3
"""REWARD-SIGNAL DE-RISK GATE (2026-06-09)
================================================================================
The scoped-V8 stack is exhausted; every per-slot-F1 lever hit a subjectivity
ceiling. User pivoted to a WHOLE-MAP "feel" objective (human-preference / learned
reward). Before building the expensive preference/RL loop, this gate asks the
cheap, decisive question:

  Is there a LEARNABLE map-level "feel" signal such that
    (A) a cheap handcrafted-feature classifier separates HUMAN maps from
        feel-destroyed (corrupted) maps, and
    (B) that classifier scores our V7-GENERATED maps as sub-human?

  DoD-A: CV AUC(human vs corrupt) >= 0.80
  DoD-B: mean human P(human) - mean V7 P(human) >= 0.25

  A & B            -> green-light the preference/reward model + ranking objective.
  A, V7 ~= human   -> reframe: our maps are fine on feel; per-slot F1 was the wrong
                      metric all along.
  not A            -> cheap features insufficient; the reward direction needs a
                      learned map encoder (bigger build).

Uniform featurizer reads HUMAN maps from the cached .pt (decode_events on
swing_tokens) and GENERATED maps from .zip (the contour-eval loader), so both go
through the SAME feel-feature vector. GPU-free.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # for eval_* imports

from beatsaber_automapper.data.swing_tokenizer import SwingEventTokenizer
from beatsaber_automapper.data.beat_grid import BEAT_SUBDIV  # noqa: F401

# Vertical component of each Beat Saber cut direction (0-8): up=+1, down=-1.
_VERT = {0: +1, 1: -1, 2: 0, 3: 0, 4: +1, 5: +1, 6: -1, 7: -1, 8: 0}
_DIAGONAL = {4, 5, 6, 7}
_HORIZ_DOT = {2, 3, 8}

FEATURE_NAMES = [
    "nps",
    "density_cv",
    "density_corr_drum",
    "dir_entropy",
    "diagonal_frac",
    "horiz_dot_frac",
    "parity_viol_proxy",
    "x_spread",
    "y_spread",
    "ini_cv",
    "contour_follow",
]


def _notes_to_seconds(notes, bpm):
    """notes: list of (beat, x, y, dir). Returns sorted list of (t_sec, x, y, dir)."""
    spb = 60.0 / bpm
    out = [(b * spb, int(x), int(y), int(d)) for (b, x, y, d) in notes]
    out.sort(key=lambda r: r[0])
    return out


def _density_series(times, duration, win=2.0):
    if duration <= 0 or len(times) == 0:
        return np.zeros(1)
    nb = max(1, int(np.ceil(duration / win)))
    hist = np.zeros(nb)
    for t in times:
        i = min(nb - 1, int(t / win))
        hist[i] += 1
    return hist


def featurize(notes, bpm, duration, drum_density=None):
    """notes: list of (beat,x,y,dir). Returns feature vector aligned to FEATURE_NAMES."""
    if len(notes) < 4 or duration <= 0:
        return None
    rows = _notes_to_seconds(notes, bpm)
    times = np.array([r[0] for r in rows])
    xs = np.array([r[1] for r in rows], dtype=float)
    ys = np.array([r[2] for r in rows], dtype=float)
    dirs = np.array([r[3] for r in rows], dtype=int)

    nps = len(rows) / duration

    dens = _density_series(times, duration, win=2.0)
    density_cv = float(dens.std() / (dens.mean() + 1e-9))

    # density-corr vs drum-event density over the same 2s windows
    density_corr = 0.0
    if drum_density is not None and len(drum_density) >= 2 and len(dens) >= 2:
        n = min(len(dens), len(drum_density))
        a, b = dens[:n], np.asarray(drum_density)[:n]
        if a.std() > 1e-9 and b.std() > 1e-9:
            # Spearman via ranks
            ar = np.argsort(np.argsort(a)).astype(float)
            br = np.argsort(np.argsort(b)).astype(float)
            density_corr = float(np.corrcoef(ar, br)[0, 1])

    # direction entropy over 9 dirs
    counts = np.bincount(dirs, minlength=9).astype(float)
    p = counts / counts.sum()
    dir_entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())

    diagonal_frac = float(np.isin(dirs, list(_DIAGONAL)).mean())
    horiz_dot_frac = float(np.isin(dirs, list(_HORIZ_DOT)).mean())

    # parity-violation proxy: fraction of consecutive notes whose vertical swing
    # sign repeats instead of alternating (real maps mostly alternate up/down).
    vert = np.array([_VERT.get(int(d), 0) for d in dirs])
    nz = vert[vert != 0]
    if len(nz) >= 2:
        same = (nz[1:] == nz[:-1]).mean()
    else:
        same = 0.5
    parity_viol_proxy = float(same)

    x_spread = float(xs.std())
    y_spread = float(ys.std())

    ini = np.diff(times)
    ini_cv = float(ini.std() / (ini.mean() + 1e-9)) if len(ini) >= 2 else 0.0

    # crude contour-follow placeholder: alternation rate of vertical swings
    # (no audio); kept so generated/human use identical vector. Real audio
    # contour-follow is in eval_contour_follow; here we use swing alternation.
    if len(nz) >= 2:
        contour_follow = float((nz[1:] != nz[:-1]).mean())
    else:
        contour_follow = 0.5

    return np.array([
        nps, density_cv, density_corr, dir_entropy, diagonal_frac,
        horiz_dot_frac, parity_viol_proxy, x_spread, y_spread, ini_cv,
        contour_follow,
    ], dtype=float)


# ---- corruptions (feel-destroyers) -----------------------------------------
def corrupt(notes, bpm, duration, rng, mode):
    """Return a feel-destroyed copy of notes (list of (beat,x,y,dir))."""
    notes = list(notes)
    beats = np.array([n[0] for n in notes])
    if mode == "shuffle_time":
        # keep note count + spatial content, destroy timing structure: uniform
        # random beats across the same span.
        lo, hi = beats.min(), beats.max()
        newb = rng.uniform(lo, hi, size=len(notes))
        return [(float(newb[i]), n[1], n[2], n[3]) for i, n in enumerate(notes)]
    if mode == "rand_dir":
        return [(n[0], n[1], n[2], int(rng.integers(0, 9))) for n in notes]
    if mode == "flatten_density":
        # resample onto a uniform grid → metronome density, no structure.
        lo, hi = beats.min(), beats.max()
        grid = np.linspace(lo, hi, len(notes))
        return [(float(grid[i]), notes[i][1], notes[i][2], notes[i][3])
                for i in range(len(notes))]
    raise ValueError(mode)


CORRUPT_MODES = ["shuffle_time", "rand_dir", "flatten_density"]


def human_song_features(pt_path, difficulty, tok, rng):
    """Returns (human_feat, [corrupt_feats]) or None on failure."""
    try:
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    diffs = d.get("difficulties", {})
    if difficulty not in diffs:
        return None
    st = diffs[difficulty].get("swing_tokens")
    if not st:
        return None
    try:
        events = tok.decode_events(list(st))
    except Exception:
        return None
    notes = [(e.beat, e.x, e.y, e.direction) for e in events
             if getattr(e, "kind", 0) == 0 or True]  # keep all spatial events
    # keep only note-like with valid dir range
    notes = [(b, x, y, dr) for (b, x, y, dr) in notes if 0 <= dr <= 8]
    if len(notes) < 8:
        return None
    bpm = float(d.get("bpm", 120.0))
    # duration from mel frames if present else from beats
    spb = 60.0 / bpm
    duration = max(n[0] for n in notes) * spb + spb
    # drum density over 2s windows from instr_beat_features col 0 (kick) + drums
    drum_density = None
    instr = d.get("instr_beat_features")
    if instr is not None and hasattr(instr, "shape") and instr.shape[0] > 2:
        # instr rows are per 1/4-note slot; aggregate kick+snare+hat (cols 0-2) to 2s windows
        slot_sec = spb / 1.0  # instr_beat_features is on the 1/4-note grid → 1 slot per beat? approx
        # build per-2s drum activity proxy by summing first 3 cols then binning by slot index→time
        drums = instr[:, 0:3].sum(axis=1).numpy() if hasattr(instr, "numpy") else np.asarray(instr)[:, 0:3].sum(1)
        nslots = len(drums)
        slot_times = np.arange(nslots) * (duration / max(1, nslots))
        nb = max(1, int(np.ceil(duration / 2.0)))
        dd = np.zeros(nb)
        for i, t in enumerate(slot_times):
            dd[min(nb - 1, int(t / 2.0))] += drums[i]
        drum_density = dd
    hfeat = featurize(notes, bpm, duration, drum_density)
    if hfeat is None:
        return None
    cfeats = []
    for mode in CORRUPT_MODES:
        cn = corrupt(notes, bpm, duration, rng, mode)
        cf = featurize(cn, bpm, duration, drum_density)
        if cf is not None:
            cfeats.append(cf)
    return hfeat, cfeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="number of human songs")
    ap.add_argument("--difficulty", default="Expert")
    ap.add_argument("--pt-glob", default="data/processed/*.pt")
    ap.add_argument("--gen-glob", default="outputs/2026-06-07/*.zip")
    ap.add_argument("--gen-difficulty", default="Expert")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tok = SwingEventTokenizer()

    pts = sorted(glob.glob(args.pt_glob))[: args.n * 2]  # over-sample, some fail
    H, C = [], []
    used = 0
    for p in pts:
        if used >= args.n:
            break
        res = human_song_features(p, args.difficulty, tok, rng)
        if res is None:
            continue
        hf, cfs = res
        if not cfs:
            continue
        H.append(hf)
        C.extend(cfs)
        used += 1
    print(f"[data] human songs used={used}  human={len(H)}  corrupt={len(C)}")
    if used < 20:
        print("!! too few usable songs; aborting"); sys.exit(2)

    X = np.vstack([np.array(H), np.array(C)])
    y = np.concatenate([np.ones(len(H)), np.zeros(len(C))])
    # standardize (impute nan→col mean)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0, keepdims=True), X)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    proba = cross_val_predict(clf, Xs, y, cv=5, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    clf.fit(Xs, y)
    weights = dict(sorted(zip(FEATURE_NAMES, clf.coef_[0]),
                          key=lambda kv: -abs(kv[1])))

    print(f"\n[DoD-A] CV AUC(human vs corrupt) = {auc:.4f}   (>=0.80 PASS)")
    print("  top feature weights (signed; + → human-like):")
    for k, v in weights.items():
        print(f"    {k:20s} {v:+.3f}")

    # ---- probe generated maps ----
    from eval_contour_follow import _load_notes_with_direction
    gen_paths = [p for p in sorted(glob.glob(args.gen_glob))
                 if "_pre" not in pathlib.Path(p).stem]  # use post-processed gens
    gen_scores = []
    for gp in gen_paths:
        try:
            recs = _load_notes_with_direction(pathlib.Path(gp), args.gen_difficulty)
        except Exception:
            continue
        notes = [(b, x, y, dr) for (b, x, y, _c, dr) in recs]
        if len(notes) < 8:
            continue
        # crude bpm/duration from beats (assume bpm in name unknown → use 123 test song default)
        bpm = 123.0
        spb = 60.0 / bpm
        dur = max(n[0] for n in notes) * spb + spb
        gf = featurize(notes, bpm, dur, None)
        if gf is None:
            continue
        gf = np.where(np.isnan(gf), mu, gf)
        gs = float(clf.predict_proba(((gf - mu) / sd).reshape(1, -1))[:, 1][0])
        gen_scores.append((pathlib.Path(gp).name, gs))

    human_proba_mean = float(proba[y == 1].mean())
    gen_mean = float(np.mean([s for _, s in gen_scores])) if gen_scores else float("nan")
    print(f"\n[probe] mean human P(human) (CV) = {human_proba_mean:.3f}")
    print(f"[probe] V7 generated maps P(human):")
    for name, s in gen_scores:
        print(f"    {name:28s} {s:.3f}")
    delta = human_proba_mean - gen_mean
    print(f"[DoD-B] human - V7 mean P(human) = {delta:+.3f}   (>=0.25 → V7 sub-human, usable reward)")

    # ---- verdict ----
    print("\n=== VERDICT ===")
    a_pass = auc >= 0.80
    b_pass = delta >= 0.25
    if a_pass and b_pass:
        v = ("GREEN: a cheap map-level feel signal IS learnable (AUC>=0.80) AND it scores our V7 "
             "generator as sub-human (delta>=0.25). The reward direction is real → build the "
             "preference/reward model + ranking objective (rank human>generated; fine-tune Stage-2 "
             "to maximize reward, not per-slot F1).")
    elif a_pass and not b_pass:
        v = ("AMBER: feel signal IS learnable (AUC>=0.80) but our V7 maps already score ~human "
             "(delta<0.25). Implication: our maps may be FINE on feel and per-slot F1 was hiding it. "
             "Re-examine: is the user's quality complaint about feel or about something this "
             "featurizer misses? Inspect which features V7 maxes vs humans before committing RL.")
    else:
        v = ("RED: a cheap handcrafted-feature classifier can't cleanly separate human from "
             "feel-destroyed maps (AUC<0.80). The reward direction needs a LEARNED map encoder "
             "(deep features), not handcrafted ones — bigger build. Reconsider scope before "
             "committing.")
    print(v)

    out = {
        "n_songs": used, "auc_human_vs_corrupt": auc,
        "feature_weights": weights, "human_proba_mean": human_proba_mean,
        "gen_scores": gen_scores, "gen_mean": gen_mean, "dod_b_delta": delta,
        "dod_a_pass": bool(a_pass), "dod_b_pass": bool(b_pass), "verdict": v,
    }
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
