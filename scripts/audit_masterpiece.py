#!/usr/bin/env python
"""THE CONTROL BATTERY FOR THE MASTERPIECE AXES (M1, M2, M3, M4).

Standing project rule: **no metric steers a lever until a battery of deliberately
degenerate maps has failed it.** Two metrics built on 2026-08-03/04 (`halfbeat_rate`,
`share_over_1s`) were caught here — a metronome beat a human on both — and the rule
exists because in this project a metric that *sounds* right has been wrong more
often than it has been right.

The M-axes are contrasts, so the claim under test is stronger than "human beats the
controls": it is that **every degenerate control scores ~0**, because a map that is
the same everywhere cannot correlate with a song that is not.

Controls (all scored on the SAME song, SAME bar grid, SAME estimator):

  human            the real Expert map — the top of the ranking, by assumption
  ours             the production arm
  metronome        one (x, y, dir) at a constant interval, human note count
  random_times     human attributes, times drawn uniformly over the song
                   ⇒ THE decisive control for a rhythm metric
  jitter_60ms      human map, every time jittered +-60ms (a whole slot at 16ths)
  shuffled_attrs   human map, (x, y, dir) permuted, TIMES UNTOUCHED
                   ⇒ isolates placement from rhythm: M2 must be blind to it and
                     M1's `place` view must not be
  bar_rotated      human map with its bars circularly rotated: same notes, same
                   internal patterns, wrong place in the song
                   ⇒ the sharpest control there is for "does this measure the
                     CORRESPONDENCE rather than the map"
  thinned_30       human map with 30 % of notes dropped at random
                   ⇒ THE DENSITY CONTROL. Every one of these axes could secretly
                     be a note-count metric; if thinning a human map moves the
                     score a lot, the axis is measuring density.
  human_wrong_song this song's audio scored against ANOTHER song's human map
                   ⇒ the cross-song null: a positive score here would mean the
                     axis reads something about human maps in general, not about
                     this map answering this song.

Usage:
    python scripts/audit_masterpiece.py --n 13 --json outputs/audit_masterpiece.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import eval_accent as m3  # noqa: E402
import eval_arrangement as m4  # noqa: E402
import eval_hand_intent as m5  # noqa: E402
import eval_motif_rhyme as m1  # noqa: E402
import eval_rhythm_fidelity as m2  # noqa: E402
import song_structure as ss  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402

ARM = "tf_trim_ev03_rc05"
M1_KEYS = ("rhy_rhythm", "harm_rhythm", "timb_rhythm", "harm_place")
M2_KEYS = ("follow_mean", "follow_best", "follow_drums", "follow_vocals")
M3_KEYS = ("hands_x_strength", "hands_x_coincid", "hands_x_downbeat",
           "travel_x_strength", "turn_x_strength")
M4_KEYS = ("arrange",)
M5_KEYS = ("hand_stem", "hand_stem_p90")
ALL_KEYS = M1_KEYS + M2_KEYS + M3_KEYS + M4_KEYS + M5_KEYS

# ★AXIS-AWARE VERDICTS. `shuffled_attrs` permutes (x, y, dir) and leaves every note
# TIME untouched, so a metric computed on times alone scores it EXACTLY equal to
# the human — and on the first run it duly "beat" every rhythm metric and marked
# them all diagnostic-only. That is blindness BY CONSTRUCTION, not a failure, and
# the same reasoning `audit_eval_suite.py` already applies to A2. A control only
# tests a metric if it perturbs the domain that metric reads.
DOMAIN = {  # metric -> the domain it reads
    "rhy_rhythm": "time", "harm_rhythm": "time", "timb_rhythm": "time",
    "harm_place": "place",
    "follow_mean": "time", "follow_best": "time",
    "follow_drums": "time", "follow_vocals": "time",
    # `hands` counts the notes at one event; `shuffled_attrs` permutes (x, y, dir)
    # and leaves the times alone, so a double stays a double -> time domain.
    "hands_x_strength": "time", "hands_x_coincid": "time",
    "hands_x_downbeat": "metre",
    "travel_x_strength": "place", "turn_x_strength": "place",
    # `arrange` is dominated by the per-bar note COUNT channel, which shuffling
    # (x, y, dir) leaves untouched -> time domain.
    "arrange": "time",
    # hand/colour is an attribute: `shuffled_attrs` permutes it, so it IS a test here
    "hand_stem": "place", "hand_stem_p90": "place",
}
CONTROL_DOMAIN = {  # control -> the domains it perturbs
    "metronome": {"time", "place", "metre"},
    "random_times": {"time", "metre"},
    "jitter_60ms": {"time", "metre"},
    "shuffled_attrs": {"place"},
    # ⚠️A whole-BAR rotation leaves every note on the same slot WITHIN its bar, so
    # it cannot perturb a metrical-position metric. Found by `hands_x_downbeat`
    # scoring exactly 1.000x human under it — a tie that precise is a construction,
    # not a result.
    "bar_rotated": {"time", "place"},
    "thinned_30": {"time", "place", "metre"},
    "human_wrong_song": {"time", "place", "metre"},
}

# ★TWO KINDS OF CONTROL, AND CONFLATING THEM GAVE THE WRONG VERDICT FIRST TIME.
#   DEGENERATE  — a map nobody would call good (a metronome, random times, a
#                 rotated map, another song's map). These must score FAR below the
#                 human, because a metric they can reach is a metric a lever can
#                 reach the cheap way. Pass bar: < 50 % of the human value.
#   DEGRADATION — a human map made slightly worse (60 ms of jitter, 30 % of the
#                 notes dropped). These are NOT pass/fail: a degraded human map is
#                 still a decent map and SHOULD score between ours and human. They
#                 measure how sharp the ruler is. The only failing outcome is a
#                 degradation scoring ABOVE the human, which means the metric
#                 rewards the damage.
# The first version tested both classes with one rule and marked `follow_mean`
# diagnostic-only because a 30%-thinned HUMAN map scored 0.86x of the human. That
# is the metric working, not failing: our own maps sit at 0.30x.
DEGENERATE = ("metronome", "random_times", "bar_rotated", "human_wrong_song",
              "shuffled_attrs")
DEGRADATION = ("jitter_60ms", "thinned_30")
DEGENERATE_MAX_FRACTION = 0.50


# ------------------------------------------------------------------- controls

def make_controls(notes: list[tuple], B: ss.Bars,
                  rng: np.random.Generator) -> dict[str, list[tuple]]:
    """`notes` = the HUMAN map as [(t, x, y, d, c)]. All controls derive from it so
    note count is held fixed unless the control is explicitly about note count."""
    t = np.array([n[0] for n in notes])
    attrs = [n[1:] for n in notes]
    lo, hi = float(t.min()), float(t.max())
    out: dict[str, list[tuple]] = {}

    step = (hi - lo) / max(1, len(notes) - 1)
    out["metronome"] = [(lo + i * step, 1, 0, 1, i % 2) for i in range(len(notes))]

    rt = np.sort(rng.uniform(lo, hi, len(notes)))
    out["random_times"] = [(float(rt[i]), *attrs[i]) for i in range(len(notes))]

    jt = np.sort(t + rng.uniform(-0.06, 0.06, len(t)))
    out["jitter_60ms"] = [(float(jt[i]), *attrs[i]) for i in range(len(t))]

    perm = rng.permutation(len(attrs))
    out["shuffled_attrs"] = [(float(t[i]), *attrs[perm[i]]) for i in range(len(t))]

    # bar rotation: move every note by a whole number of bars, wrapping
    span = B.edges[-1] - B.edges[0]
    shift = B.dur * max(3, B.n // 3)
    rot = []
    for (ti, *rest) in notes:
        nt = B.edges[0] + ((ti - B.edges[0] + shift) % span)
        rot.append((float(nt), *rest))
    out["bar_rotated"] = sorted(rot)

    keep = rng.random(len(notes)) > 0.30
    out["thinned_30"] = [n for n, k in zip(notes, keep) if k]
    return out


def load_human(song: str) -> list[tuple] | None:
    H = load_expert_only(REPO / "data" / "raw" / f"{song}.zip")
    if not H:
        return None
    return m1.notes_xydc(H[0], float(H[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    files = sorted(glob.glob(str(REPO / f"outputs/eval_sweep_cache/{a.arm}#s0__*.zip")))
    per_song = []
    human_pool: dict[str, list[tuple]] = {}

    for f in files:
        song = pathlib.Path(f).stem.split("__")[-1]
        hn = load_human(song)
        if hn is None:
            continue
        human_pool[song] = hn
    songs = list(human_pool)[: a.n]
    print(f"battery over {len(songs)} songs with a human Expert map\n")

    for song in songs:
        f = REPO / f"outputs/eval_sweep_cache/{a.arm}#s0__{song}.zip"
        L = scorecard._load_any(f)
        if not L:
            continue
        bm, bpm = L[0], float(L[1])
        ours = m1.notes_xydc(bm, bpm)
        human = human_pool[song]
        end = ss.song_end(song, max(n[0] for n in human + ours))
        B = ss.bars(song, bpm, end)
        if B is None or B.n < 24:
            continue
        A = ss.bar_audio_matrix(song, B)
        stems = m2.stem_onsets(song)
        if A is None or len(stems) < 3:
            continue
        nov = m4.novelty(A)
        bnds = m4.boundaries(nov) if nov is not None else []

        cands = {"human": human, "ours": ours}
        cands.update(make_controls(human, B, rng))
        other = next((s for s in songs if s != song), None)
        if other:
            cands["human_wrong_song"] = human_pool[other]

        row = {"song": song, "bars": B.n}
        for name, notes in cands.items():
            if len(notes) < 60:
                continue
            s1 = m1.song_scores(notes, B, A) or {}
            times = np.sort(np.array([n[0] for n in notes]))
            s2 = m2.score_map(times, B, stems) or {}
            s3 = m3.score_map(notes, song, B) or {}
            s4 = (m4.score_map(notes, B, bnds) or {}) if len(bnds) >= 4 else {}
            s5 = m5.score_map(notes, stems) or {}
            row[name] = ({k: s1.get(k) for k in M1_KEYS}
                         | {k: s2.get(k) for k in M2_KEYS}
                         | {k: s3.get(k) for k in M3_KEYS}
                         | {k: s4.get(k) for k in M4_KEYS}
                         | {k: s5.get(k) for k in M5_KEYS})
        per_song.append(row)
        print(f"  scored {song}")

    if not per_song:
        print("nothing scored")
        return

    names = ["human", "ours", "metronome", "random_times", "jitter_60ms",
             "shuffled_attrs", "bar_rotated", "thinned_30", "human_wrong_song"]
    print(f"\n{'='*100}\nCONTROL BATTERY — median over {len(per_song)} songs\n{'='*100}")
    # ⚠️COMMON SUBSET PER METRIC. The first version took each control's median over
    # whichever songs that control happened to score, and the battery then reported
    # human `hands_x_downbeat` = 0.2994 while eval_accent.py reported 0.1817 for the
    # same cohort — two medians over different song sets. That is the population
    # error this project keeps repeating, committed inside the tool built to catch
    # errors. Every row below is the median over the songs where EVERY control
    # produced a value.
    common = {k: [r for r in per_song
                  if all(n in r and r[n].get(k) is not None for n in names)]
              for k in ALL_KEYS}
    print("n per metric: " + ", ".join(f"{k} {len(common[k])}" for k in ALL_KEYS) + "\n")
    header = f"{'control':<18}" + "".join(f"{k:>15}" for k in ALL_KEYS)
    print(header)
    table = {}
    for name in names:
        vals = {}
        for k in ALL_KEYS:
            v = [r[name][k] for r in common[k]]
            vals[k] = round(st.median(v), 4) if len(v) >= 3 else None
        table[name] = vals
        print(f"{name:<18}" + "".join(
            (f"{vals[k]:>+15.4f}" if vals[k] is not None else f"{'-':>15}")
            for k in ALL_KEYS))

    print(f"\n{'='*100}\nVERDICT")
    print(f"  a DEGENERATE control must stay under {DEGENERATE_MAX_FRACTION:.0%} of the human value")
    print("  a DEGRADATION probe (jittered / thinned human) must not EXCEED the human")
    print("  controls that cannot perturb the domain a metric reads are excluded, named")
    print(f"{'='*100}")
    verdicts = {}
    for k in ALL_KEYS:
        hv = table["human"].get(k)
        if hv is None:
            continue
        dom = DOMAIN.get(k, "time")
        def rel(group):
            return [n for n in group
                    if dom in CONTROL_DOMAIN.get(n, {"time"})
                    and table.get(n, {}).get(k) is not None]
        deg, dgr = rel(DEGENERATE), rel(DEGRADATION)
        blind = [n for n in DEGENERATE + DEGRADATION
                 if n not in deg + dgr and table.get(n, {}).get(k) is not None]

        frac = {n: (table[n][k] / hv if hv else None) for n in deg + dgr}
        # A negative human value means the axis has no signal to protect; treat any
        # control at or above it as a failure rather than dividing by a negative.
        if hv <= 0:
            failed_deg = deg
        else:
            failed_deg = [n for n in deg if frac[n] > DEGENERATE_MAX_FRACTION]
        failed_dgr = [n for n in dgr if hv > 0 and frac[n] > 1.0]
        ok = not failed_deg and not failed_dgr

        worst_deg = max((table[n][k] for n in deg), default=None)
        verdicts[k] = {"human": hv, "domain": dom, "worst_degenerate": worst_deg,
                       "failed_degenerate": failed_deg,
                       "failed_degradation": failed_dgr,
                       "blind_by_construction": blind,
                       "retained_fraction": {n: (round(v, 3) if v is not None else None)
                                             for n, v in frac.items()},
                       "may_steer": ok}
        mark = "MAY STEER" if ok else "DIAGNOSTIC ONLY"
        why = ""
        if failed_deg:
            why = f"   (degenerate reaches it: {', '.join(failed_deg)})"
        elif failed_dgr:
            why = f"   (rewards degradation: {', '.join(failed_dgr)})"
        print(f"  {k:<17} [{dom:5s}] human {hv:+.4f}  worst degenerate "
              f"{(worst_deg if worst_deg is not None else float('nan')):+.4f}  {mark}{why}")
        print(f"{'':<20}retained: " + ", ".join(
            f"{n} {frac[n]:.2f}" for n in deg + dgr if frac[n] is not None)
            + (f" | blind: {', '.join(blind)}" if blind else ""))

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"arm": a.arm, "n_songs": len(per_song), "table": table,
             "verdicts": verdicts, "per_song": per_song}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
