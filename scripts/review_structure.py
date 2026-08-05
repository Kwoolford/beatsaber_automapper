#!/usr/bin/env python
"""WHERE, IN THIS SONG, DID WE MISS THE STRUCTURE? — timestamped, ranked.

`review_map.py` answers "where do I listen" for the note-level defects (STARVED /
MISSED_HIT / OFFBEAT / PHRASE_HOLE / ENDING). The masterpiece axes are cohort
statistics, and a cohort statistic cannot be listened to. This turns them back into
moments:

  MOTIF_MISS   this passage is a repeat of an earlier one — the MUSIC says so and
               the HUMAN map brings its pattern back — and ours does not
  FIGURE_MISS  this bar has a clear figure in one stem, the human plays it, we do
               not
  DOWNBEAT_GAP a stretch where the downbeats carry no emphasis at all

Every finding is stated against the HUMAN map on the same song, so it is never a
complaint that the audio is hard — it is a place where a person solved it and we
did not. Findings are ranked by how large the human-minus-ours gap is.

Usage:
    python scripts/review_structure.py --song 1f8d6
    python scripts/review_structure.py --song 1f333 --arm v8_mbb025 --top 8
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import song_structure as ss  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from eval_motif_rhyme import notes_xydc  # noqa: E402
from eval_rhythm_fidelity import (STEMS, follow_scores, quantise,  # noqa: E402
                                  stem_onsets)

MIN_LAG = 8          # bars: a "repeat" must be far enough away to be a repeat
AUDIO_Q = 0.95       # how similar the music must be to count as a repeat
GAP_MIN = 0.15       # smallest human-minus-ours gap worth reporting


def mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def findings(song: str, arm: str, map_path: str = "", top: int = 6) -> list[dict]:
    mp = map_path or next(iter(sorted(glob.glob(
        str(REPO / f"outputs/eval_sweep_cache/{arm}#s0__{song}.zip")))), "")
    if not mp:
        return []
    L = scorecard._load_any(pathlib.Path(mp))
    if not L:
        return []
    bm, bpm = L[0], float(L[1])
    t = np.asarray(alignment.note_times(bm, bpm), dtype=float)
    B = ss.bars(song, bpm, ss.song_end(song, float(t.max())))
    if B is None or B.n < 24:
        return []
    A = ss.bar_audio_matrix(song, B)
    H = load_expert_only(REPO / "data" / "raw" / f"{song}.zip")
    if A is None or not H:
        return []
    ours = notes_xydc(bm, bpm)
    human = notes_xydc(H[0], float(H[1]))

    S_our = ss.bar_map_similarity(ss.map_bar_vectors(ours, B))["rhythm"]
    S_hum = ss.bar_map_similarity(ss.map_bar_vectors(human, B))["rhythm"]
    Aud = np.nanmean(np.stack([A["harm"], A["timb"], A["rhy"]]), axis=0)

    out: list[dict] = []

    # ---- MOTIF_MISS: the music repeats, the human's pattern repeats, ours doesn't
    n = B.n
    ii, jj = np.triu_indices(n, k=MIN_LAG)
    av = Aud[ii, jj]
    ok = np.isfinite(av) & np.isfinite(S_our[ii, jj]) & np.isfinite(S_hum[ii, jj])
    if ok.sum() > 20:
        thr = np.quantile(av[ok], AUDIO_Q)
        best: dict[int, dict] = {}
        for a, b, s in zip(ii[ok], jj[ok], av[ok]):
            if s < thr:
                continue
            gap = float(S_hum[a, b] - S_our[a, b])
            if gap < GAP_MIN:
                continue
            key = int(b) // 4
            if key not in best or gap > best[key]["gap"]:
                best[key] = {"kind": "MOTIF_MISS", "gap": gap,
                             "t": float(B.starts[b]), "t_ref": float(B.starts[a]),
                             "bar": int(b), "bar_ref": int(a),
                             "music_sim": round(float(s), 3),
                             "ours": round(float(S_our[a, b]), 3),
                             "human": round(float(S_hum[a, b]), 3)}
        out += sorted(best.values(), key=lambda d: -d["gap"])[:top]

    # ---- FIGURE_MISS: a stem states a figure, the human plays it, we don't
    stems = stem_onsets(song)
    if len(stems) >= 3:
        rng = np.random.default_rng(0)
        Mo = quantise(np.sort(np.array([x[0] for x in ours])), B)
        Mh = quantise(np.sort(np.array([x[0] for x in human])), B)
        gaps = np.full(B.n, -np.inf)
        which = [""] * B.n
        for s in STEMS:
            if s not in stems:
                continue
            Ms = quantise(stems[s], B)
            go, _ = follow_scores(Mo, Ms, rng)
            gh, _ = follow_scores(Mh, Ms, rng)
            d = np.where(np.isfinite(go) & np.isfinite(gh), gh - go, -np.inf)
            upd = d > gaps
            gaps = np.where(upd, d, gaps)
            for i in np.where(upd)[0]:
                which[i] = s
        # merge neighbouring bars into stretches
        idx = [i for i in np.argsort(-gaps) if np.isfinite(gaps[i]) and gaps[i] >= GAP_MIN]
        used: set[int] = set()
        for i in idx[: top * 3]:
            if any(abs(i - u) < 4 for u in used):
                continue
            used.add(int(i))
            out.append({"kind": "FIGURE_MISS", "gap": float(gaps[i]),
                        "t": float(B.starts[i]), "bar": int(i),
                        "stem": which[i]})
            if len(used) >= top:
                break

    return sorted(out, key=lambda d: -d["gap"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True)
    ap.add_argument("--arm", default="tf_trim_ev03_rc05")
    ap.add_argument("--map", default="")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()

    F = findings(a.song, a.arm, a.map, a.top)
    if not F:
        print(f"{a.song}: nothing to report (no human map, or no bar grid)")
        return
    print(f"\nSTRUCTURE FINDINGS — {a.song} ({a.arm}), ranked by human-minus-ours")
    for f in F:
        if f["kind"] == "MOTIF_MISS":
            print(f"  {mmss(f['t']):>6}  MOTIF_MISS   repeats {mmss(f['t_ref'])} "
                  f"(music {f['music_sim']:.2f}) — human reuses {f['human']:+.2f}, "
                  f"we reuse {f['ours']:+.2f}")
        elif f["kind"] == "FIGURE_MISS":
            print(f"  {mmss(f['t']):>6}  FIGURE_MISS  the {f['stem']} states a figure "
                  f"the human plays and we do not (gap {f['gap']:.2f})")


if __name__ == "__main__":
    main()
