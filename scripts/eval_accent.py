#!/usr/bin/env python
"""M3 — IS THE MAP'S EMPHASIS SPENT WHERE THE MUSIC PUTS ITS EMPHASIS?

Kyle, 2026-08-04: *"significantly more intelligent and intentional placements of
notes."*

A map has a small budget of ways to say *this one matters*: play both hands, make
the player travel, change the direction of the swing, put it on the downbeat. A
mapper spends that budget on the moments the music emphasises. A generator that
spends it uniformly produces a map that is playable and says nothing — which is
precisely the gap between the map Kyle called playable and the map he wants.

**Method.** Group notes into EVENTS (a double is one event). Give each event an
emphasis score from the map alone, and a salience score from the audio alone, then
contrast:

    accent = mean(emphasis | salience in the top quartile)
           - mean(emphasis | salience in the bottom quartile)

Emphasis channels (each z-scored WITHIN the song, so a dense map cannot win by
being dense):
    hands     notes in the event (the double is the loudest thing a map can say)
    travel    distance the hand must move from the previous event
    turn      how much the cut direction changes
Salience channels (from the audio only):
    strength  onset-envelope value at the event time
    coincid   how many separated stems hit within 50 ms (Kyle's own hypothesis,
              already validated: humans map a 4-stem collision 84.5 % of the time)
    downbeat  metrical — treated separately, since it is a property of the grid
              rather than of the waveform

★Like M1/M2 this is a CONTRAST, so a metronome (constant emphasis) and a random map
(emphasis uncorrelated with the audio) both score 0 by construction. Everything
this project built that scored a LEVEL was metronome-gameable.

⚠️Expect `accent_hands` to read low for us for a known reason and check it against
C5 before treating it as new: our double share is 0.66 against a human 0.14, and an
emphasis spent on two thirds of all events cannot mark anything.

Usage:
    python scripts/eval_accent.py --arm tf_trim_ev03_rc05
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import song_structure as ss  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from eval_motif_rhyme import notes_xydc  # noqa: E402
from eval_rhythm_fidelity import STEMS, stem_onsets  # noqa: E402

EMPHASIS = ("hands", "travel", "turn")
SALIENCE = ("strength", "coincid")
COINC_TOL = 0.050
EVENT_TOL = 0.012

DIRV = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
        4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1), 8: (0, 0)}


def events(notes: list[tuple]) -> list[dict]:
    """Group notes into musical events; a double is ONE event with two hands."""
    by_t: dict[float, list] = collections.defaultdict(list)
    for (t, x, y, d, c) in sorted(notes):
        key = round(t / EVENT_TOL)
        by_t[key].append((t, x, y, d, c))
    out = []
    for key in sorted(by_t):
        grp = by_t[key]
        t = float(np.mean([g[0] for g in grp]))
        x = float(np.mean([g[1] for g in grp]))
        y = float(np.mean([g[2] for g in grp]))
        dv = np.mean([DIRV.get(int(g[3]), (0, 0)) for g in grp], axis=0)
        out.append({"t": t, "x": x, "y": y, "dv": dv, "hands": len(grp)})
    return out


def emphasis(evs: list[dict]) -> dict[str, np.ndarray]:
    hands = np.array([e["hands"] for e in evs], dtype=float)
    travel = np.zeros(len(evs))
    turn = np.zeros(len(evs))
    for i in range(1, len(evs)):
        a, b = evs[i - 1], evs[i]
        travel[i] = float(np.hypot(b["x"] - a["x"], b["y"] - a["y"]))
        na, nb = np.linalg.norm(a["dv"]), np.linalg.norm(b["dv"])
        if na > 0 and nb > 0:
            cos = float(np.clip(a["dv"] @ b["dv"] / (na * nb), -1, 1))
            turn[i] = (1 - cos) / 2.0
    def z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)
    return {"hands": z(hands), "travel": z(travel), "turn": z(turn)}


def salience(evs: list[dict], song: str) -> dict[str, np.ndarray] | None:
    A = ss.audio_features(song)
    if A is None:
        return None
    t = np.asarray(A["times"], dtype=float)
    env = np.asarray(A["onset_env"], dtype=float)
    et = np.array([e["t"] for e in evs])
    idx = np.clip(np.searchsorted(t, et), 0, len(env) - 1)
    strength = env[idx]

    stems = stem_onsets(song)
    k = np.zeros(len(evs))
    for s, on in stems.items():
        if len(on) == 0:
            continue
        pos = np.searchsorted(on, et)
        for i, p in enumerate(pos):
            cand = [on[j] for j in (p - 1, p) if 0 <= j < len(on)]
            if cand and min(abs(et[i] - c) for c in cand) <= COINC_TOL:
                k[i] += 1
    return {"strength": strength, "coincid": k}


BLOCK = 48


def contrast(emph: np.ndarray, sal: np.ndarray, min_n: int = 20) -> float | None:
    """Whole-song contrast. ⚠️Kept only for comparison — see `local_contrast`."""
    ok = np.isfinite(emph) & np.isfinite(sal)
    emph, sal = emph[ok], sal[ok]
    if len(emph) < min_n * 2:
        return None
    hi_t, lo_t = np.quantile(sal, 0.75), np.quantile(sal, 0.25)
    hi, lo = emph[sal >= hi_t], emph[sal <= lo_t]
    if len(hi) < min_n or len(lo) < min_n or hi_t <= lo_t:
        return None
    return float(hi.mean() - lo.mean())


def local_contrast(emph: np.ndarray, sal: np.ndarray,
                   block: int = BLOCK, min_n: int = 8) -> float | None:
    """★THE ESTIMATOR THAT SURVIVES THE BATTERY. Same contrast, computed inside
    blocks of `block` consecutive events and then pooled.

    🔴WHY. The whole-song version failed the control battery in a way worth
    recording: a map with its **bars rotated** scored 1.54x the human on
    `hands_x_coincid`, and **another song's map entirely** scored 0.77x on
    `hands_x_strength`. Neither map has any relation to the audio it was scored
    against. The mechanism is that emphasis and salience each carry slow structure
    — a chorus is loud and densely mapped, and *any* human map overlaid on *any*
    song will put its dense stretches somewhere loud often enough to correlate.

    ★**A correlation between two signals that each have slow structure is not
    evidence of correspondence.** Differencing inside a short block removes the
    slow component from both sides, which is the same fix M1 needed for proximity —
    the third time this project has been bitten by an autocorrelation confound.
    """
    ok = np.isfinite(emph) & np.isfinite(sal)
    emph, sal = emph[ok], sal[ok]
    n = len(emph)
    if n < block:
        return None
    parts, w = [], []
    for s in range(0, n - block // 2, block):
        e, v = emph[s:s + block], sal[s:s + block]
        if len(e) < block // 2:
            continue
        hi_t, lo_t = np.quantile(v, 0.75), np.quantile(v, 0.25)
        if hi_t <= lo_t:
            continue
        hi, lo = e[v >= hi_t], e[v <= lo_t]
        if len(hi) < min_n or len(lo) < min_n:
            continue
        parts.append(float(hi.mean() - lo.mean()))
        w.append(len(hi) + len(lo))
    if len(parts) < 4:
        return None
    return float(np.average(parts, weights=w))


def score_map(notes: list[tuple], song: str, B: ss.Bars | None) -> dict | None:
    evs = events(notes)
    if len(evs) < 120:
        return None
    E = emphasis(evs)
    S = salience(evs, song)
    if S is None:
        return None
    out = {}
    for e in EMPHASIS:
        for s in SALIENCE:
            c = local_contrast(E[e], S[s])
            if c is not None:
                out[f"{e}_x_{s}"] = round(c, 4)
            cg = contrast(E[e], S[s])
            if cg is not None:
                out[f"{e}_x_{s}_global"] = round(cg, 4)
    # metrical accent: emphasis on the downbeat vs off it
    if B is not None and B.n > 8:
        et = np.array([ev["t"] for ev in evs])
        rel = ((et - B.edges[0]) % B.dur) / B.dur
        slot = np.round(rel * ss.SLOTS_PER_BAR).astype(int) % ss.SLOTS_PER_BAR
        on_db = slot == 0
        if on_db.sum() >= 20 and (~on_db).sum() >= 20:
            for e in EMPHASIS:
                # local form: difference inside blocks, for the same reason
                parts, w = [], []
                for st_ in range(0, len(E[e]) - BLOCK // 2, BLOCK):
                    seg, m = E[e][st_:st_ + BLOCK], on_db[st_:st_ + BLOCK]
                    if m.sum() < 3 or (~m).sum() < 3:
                        continue
                    parts.append(float(seg[m].mean() - seg[~m].mean()))
                    w.append(len(seg))
                if len(parts) >= 4:
                    out[f"{e}_x_downbeat"] = round(float(np.average(parts, weights=w)), 4)
                out[f"{e}_x_downbeat_global"] = round(
                    float(E[e][on_db].mean() - E[e][~on_db].mean()), 4)
    out["double_share"] = round(float(np.mean([e["hands"] >= 2 for e in evs])), 4)
    out["n_events"] = len(evs)
    return out


def paired(rows, key) -> dict:
    """Delegates to the shared estimator (mean + median delta, se, resolvability)."""
    return ss.paired_delta(rows, key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="tf_trim_ev03_rc05")
    ap.add_argument("--seed", default="s0")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    files = sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{a.arm}#{a.seed}__*.zip"))) \
        or sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{a.arm}__*.zip")))
    rows = []
    for f in files:
        song = pathlib.Path(f).stem.split("__")[-1]
        L = scorecard._load_any(pathlib.Path(f))
        if not L:
            continue
        bm, bpm = L[0], float(L[1])
        ours_notes = notes_xydc(bm, bpm)
        t = np.asarray(alignment.note_times(bm, bpm), dtype=float)
        if len(t) < 100:
            continue
        B = ss.bars(song, bpm, ss.song_end(song, float(t.max())))
        ours = score_map(ours_notes, song, B)
        if ours is None:
            continue
        H = load_expert_only(REPO / "data" / "raw" / f"{song}.zip")
        human = score_map(notes_xydc(H[0], float(H[1])), song, B) if H else None
        rows.append({"song": song, "ours": ours, "human": human})
        print(f"  {song:22s} hands×strength ours {ours.get('hands_x_strength', float('nan')):+.3f}"
              + (f"  human {human.get('hands_x_strength', float('nan')):+.3f}" if human else "  (no human)"))

    keys = [f"{e}_x_{s}" for e in EMPHASIS for s in SALIENCE] + \
           [f"{e}_x_downbeat" for e in EMPHASIS] + ["double_share"]
    print(f"\n{'='*88}\nM3 ACCENT — arm {a.arm}, {len(rows)} songs (PAIRED subset)\n{'='*88}")
    print(f"{'metric':<22} {'n':>3} {'ours':>9} {'human':>9} {'paired Δ':>10} "
          f"{'Δ median':>10} {'resolvable':>11}")
    summary = {}
    for k in keys:
        both = [r for r in rows if r.get("human")
                and r["ours"].get(k) is not None and r["human"].get(k) is not None]
        if len(both) < 6:
            continue
        o = [r["ours"][k] for r in both]
        h = [r["human"][k] for r in both]
        p = paired(rows, k)
        summary[k] = {"n": len(both), "ours": round(st.median(o), 4),
                      "human": round(st.median(h), 4), "paired": p}
        print(f"{k:<22} {len(both):>3d} {st.median(o):>+9.4f} {st.median(h):>+9.4f} "
              f"{p.get('delta', float('nan')):>+10.4f} "
              f"{p.get('delta_median', float('nan')):>+10.4f} "
              f"{('YES' if p.get('resolvable') else 'no'):>11}")

    print("\nHOW TO READ: every emphasis channel is z-scored inside its own song, so")
    print("these are 'how many standard deviations louder is the map at a loud moment")
    print("than at a quiet one'. 0 = the map spends its emphasis without reference to")
    print("the music. A metronome and a random map both score 0 by construction.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"arm": a.arm, "rows": rows, "summary": summary}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
