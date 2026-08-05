#!/usr/bin/env python
"""M5 — ARE THE TWO HANDS DOING TWO DIFFERENT JOBS?

The thing a good mapper does that no axis here has ever looked at: **assign**. The
red hand takes the kick, the blue hand takes the melody, and for the length of a
passage that assignment holds — so the player's hands learn the song. A generator
that picks a hand per note by flow rules alone produces a map where both hands mean
the same thing, which is playable and says nothing.

`role_asymmetry` (A6) already asks whether the two hands *differ statistically*. This
asks the different and harder question: **do they differ ACCORDING TO THE MUSIC?**
A map can have a perfectly asymmetric hand distribution with no musical meaning at
all — that is what a flow rule produces.

**Method.** Attribute each note to a stem (nearest onset within 50 ms, winner takes
all — the same attribution A4 needed once it was found that 68 % of notes match more
than one stem). Then inside blocks of consecutive events, measure the association
between HAND and STEM with **Cramér's V**, and subtract the association obtained
after shuffling the hands within that same block:

    hand_stem = V(hand, stem) - mean V(hand_shuffled, stem)

The shuffle null holds the block's hand balance and stem mix exactly fixed, so the
number is the association that survives knowing both marginals. A metronome
(alternating hands, no relation to the stems) scores 0; so does any flow rule that
ignores the music.

Reported alongside:
    hand_stem_persist  P(the same stem keeps the same hand in the next block) minus
                       the chance rate — assignment that HOLDS is the point; a
                       mapping that re-decides every four bars teaches nothing.

⚠️Block-local by construction, for the reason M3 had to learn the hard way: two
signals that each drift slowly will correlate over a whole song without meaning
anything.

Usage:
    python scripts/eval_hand_intent.py --arm tf_trim_ev03_rc05
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

TOL = 0.050
BLOCK = 64
N_SHUFFLE = 12


def attribute(notes: list[tuple], stems: dict) -> tuple[np.ndarray, np.ndarray]:
    """(hand, stem index) per note, -1 where no stem is within TOL."""
    t = np.array([n[0] for n in notes])
    hand = np.array([int(n[4]) for n in notes])
    best = np.full(len(notes), -1)
    bestd = np.full(len(notes), TOL + 1.0)
    for si, s in enumerate(STEMS):
        on = stems.get(s)
        if on is None or len(on) == 0:
            continue
        pos = np.searchsorted(on, t)
        for i, p in enumerate(pos):
            cand = [on[j] for j in (p - 1, p) if 0 <= j < len(on)]
            if not cand:
                continue
            d = min(abs(t[i] - c) for c in cand)
            if d <= TOL and d < bestd[i]:
                bestd[i], best[i] = d, si
    return hand, best


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    ca = sorted(set(a.tolist()))
    cb = sorted(set(b.tolist()))
    if len(ca) < 2 or len(cb) < 2:
        return 0.0
    tab = np.zeros((len(ca), len(cb)))
    ia = {v: i for i, v in enumerate(ca)}
    ib = {v: i for i, v in enumerate(cb)}
    for x, y in zip(a.tolist(), b.tolist()):
        tab[ia[x], ib[y]] += 1
    n = tab.sum()
    if n < 8:
        return 0.0
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.nansum(np.where(exp > 0, (tab - exp) ** 2 / exp, 0.0))
    k = min(len(ca), len(cb)) - 1
    return float(np.sqrt(chi / (n * k))) if k > 0 else 0.0


def score_map(notes: list[tuple], stems: dict, seed: int = 0) -> dict | None:
    if len(notes) < 200 or len(stems) < 3:
        return None
    rng = np.random.default_rng(seed)
    notes = sorted(notes)
    hand, stem = attribute(notes, stems)
    ok = stem >= 0
    if ok.sum() < 200:
        return None
    h, s = hand[ok], stem[ok]

    gains, blocks = [], []
    for start in range(0, len(h) - BLOCK // 2, BLOCK):
        hb, sb = h[start:start + BLOCK], s[start:start + BLOCK]
        if len(hb) < BLOCK // 2 or len(set(hb.tolist())) < 2 or len(set(sb.tolist())) < 2:
            continue
        obs = cramers_v(hb, sb)
        # ★THE NULL MUST PRESERVE THE ALTERNATION, and the first one did not.
        # A free permutation of the hands destroys the strict left/right
        # alternation every Beat Saber map has (parity is a rule of the game, and
        # Kyle explicitly praised our hand-lead alternation). Against that null,
        # real maps came out NEGATIVE — ours −0.106, human −0.009 — because
        # alternation makes hand and instrument *less* associated than chance, and
        # the control battery duly ranked a hands-randomised map ABOVE both
        # cohorts. The metric was scoring alternation as a defect.
        # A circular ROTATION of the hand sequence keeps the alternation pattern
        # exactly and only breaks its alignment to the stems, which is the thing
        # under test.
        null = float(np.mean([
            cramers_v(np.roll(hb, int(k)), sb)
            for k in rng.integers(3, max(4, len(hb) - 3), N_SHUFFLE)]))
        gains.append(obs - null)
        # which hand this block gives to each stem (majority)
        assign = {}
        for st_ in set(sb.tolist()):
            m = sb == st_
            if m.sum() >= 4:
                assign[st_] = int(round(float(hb[m].mean())))
        blocks.append(assign)
    if len(gains) < 4:
        return None

    # does the assignment HOLD from block to block?
    keep = tot = 0
    for a, b in zip(blocks, blocks[1:]):
        for k in set(a) & set(b):
            tot += 1
            keep += int(a[k] == b[k])
    persist = keep / tot if tot >= 8 else None
    return {"hand_stem": round(float(np.mean(gains)), 4),
            "hand_stem_p90": round(float(np.quantile(gains, 0.9)), 4),
            "hand_stem_persist": round(persist, 4) if persist is not None else None,
            "attributed": round(float(ok.mean()), 4),
            "n_blocks": len(gains)}


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
        stems = stem_onsets(song)
        L = scorecard._load_any(pathlib.Path(f))
        if not L or len(stems) < 3:
            continue
        bm, bpm = L[0], float(L[1])
        ours = score_map(notes_xydc(bm, bpm), stems)
        H = load_expert_only(REPO / "data" / "raw" / f"{song}.zip")
        human = score_map(notes_xydc(H[0], float(H[1])), stems) if H else None
        if ours is None:
            continue
        rows.append({"song": song, "ours": ours, "human": human})
        print(f"  {song:22s} hand_stem ours {ours['hand_stem']:+.4f}"
              + (f"   human {human['hand_stem']:+.4f}" if human else "   (no human)"))

    print(f"\n{'='*88}\nM5 HAND INTENT — arm {a.arm}, {len(rows)} songs (PAIRED subset)\n{'='*88}")
    print(f"{'metric':<22} {'n':>3} {'ours':>9} {'human':>9} {'paired Δ':>10} "
          f"{'Δ med':>9} {'resolvable':>11}")
    summary = {}
    for k in ("hand_stem", "hand_stem_p90", "hand_stem_persist", "attributed"):
        both = [r for r in rows if r.get("human")
                and r["ours"].get(k) is not None and r["human"].get(k) is not None]
        if len(both) < 5:
            continue
        o = st.median([r["ours"][k] for r in both])
        h = st.median([r["human"][k] for r in both])
        p = ss.paired_delta(rows, k)
        summary[k] = {"n": len(both), "ours": round(o, 4), "human": round(h, 4),
                      "paired": p}
        print(f"{k:<22} {len(both):>3d} {o:>+9.4f} {h:>+9.4f} "
              f"{p.get('delta', float('nan')):>+10.4f} "
              f"{p.get('delta_median', float('nan')):>+9.4f} "
              f"{('YES' if p.get('resolvable') else 'no'):>11}")

    print("\nHOW TO READ: `hand_stem` is the hand↔instrument association that survives")
    print("shuffling the hands inside the same block, so a map whose hands differ")
    print("statistically but not MUSICALLY scores 0. That is the difference between")
    print("this and A6 role_asymmetry.")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"arm": a.arm, "rows": rows, "summary": summary}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
