#!/usr/bin/env python
"""M1 — DOES THE MAP RHYME WHEN THE MUSIC RHYMES?

Kyle, 2026-08-04: *"We created a model to create a playable map but now need a
model to start producing masterpieces… significantly more intelligent and
intentional placements of notes."*

Every axis in the suite before tonight scores a note against the audio **at its own
instant**: is it on an onset, on the main beat, in a busy window. A map can pass all
of them note-by-note and still have no *composition*, because composition is not a
property of an instant — it is a property of a RELATION. When a chorus comes back,
a good mapper brings the pattern back. That is the difference between a map that is
correct and a map that was *written*.

**What this measures.** Split the song into bars. For every pair of bars compute how
similar the MUSIC is (harmony / timbre / groove) and how similar the MAP is (its
rhythm, its placement). Then:

    rhyme = mean(map similarity | the music is similar)
          - mean(map similarity | the music is different)

A mapper who reuses patterns on repeats scores positive. A mapper who ignores the
song's structure scores 0 — **and so does a metronome, and so does a uniform-random
map**, because a constant map is equally similar to itself everywhere. That is the
point: this is a **contrast**, and contrasts are degenerate-proof by construction,
unlike every regularity metric this project has built (`halfbeat_rate`,
`share_over_1s`) which a metronome beat a human on.

**Three controls are in the estimator itself, not bolted on:**
1. **Proximity** — nearby bars are both more audio-similar and more map-similar for
   trivial reasons. The contrast is computed WITHIN bar-distance strata and then
   pooled (`stratified_contrast`); the unstratified value is printed beside it so
   the size of that confound stays visible.
2. **Density** — our note count tracks loudness, so bars that sound alike hold a
   similar number of notes and overlap more by chance. Map-rhythm similarity is
   therefore **Cohen's kappa**, which subtracts exactly that chance term. (With
   cosine, we scored *above* the humans on Hunger — that was the confound talking.)
3. **A per-song null** — the same map with its bars circularly rotated, so the
   music/map correspondence is destroyed and nothing else is. A real effect must
   exceed its own null on the same song.

Usage:
    python scripts/eval_motif_rhyme.py --arm tf_trim_ev03_rc05
    python scripts/eval_motif_rhyme.py --arm v8_mbb025 --json outputs/motif_v8.json
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

import song_structure as ss  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402

AUDIO_VIEWS = ("harm", "timb", "rhy")
MAP_VIEWS = ("rhythm", "place")


def notes_xydc(bm, bpm: float) -> list[tuple]:
    spb = 60.0 / bpm
    return [(n.beat * spb, n.x, n.y, n.direction, n.color) for n in bm.color_notes]


def rotate(V: dict, k: int) -> dict:
    return {"rhythm": np.roll(V["rhythm"], k, axis=0),
            "place": np.roll(V["place"], k, axis=0),
            "count": np.roll(V["count"], k, axis=0)}


def song_scores(notes: list[tuple], B: ss.Bars, A: dict) -> dict | None:
    """`notes` = [(time_s, x, y, direction, color)] — taken as a plain list so the
    control battery can feed synthetic maps through the identical estimator."""
    V = ss.map_bar_vectors(notes, B)
    if V["count"].sum() < 60:
        return None
    S = ss.bar_map_similarity(V)
    Sn = ss.bar_map_similarity(rotate(V, max(3, B.n // 3)))
    out = {"notes_per_bar": round(float(V["count"].mean()), 2)}
    for av in AUDIO_VIEWS:
        for mv in MAP_VIEWS:
            c = ss.stratified_contrast(A[av], S[mv])
            cn = ss.stratified_contrast(A[av], Sn[mv])
            if not c:
                continue
            out[f"{av}_{mv}"] = round(c["contrast"], 4)
            out[f"{av}_{mv}_raw"] = round(c["contrast_raw"], 4) if c["contrast_raw"] is not None else None
            out[f"{av}_{mv}_null"] = round(cn["contrast"], 4) if cn else None
    return out


def paired(rows, key) -> dict:
    """Delegates to the shared estimator (mean + median delta, se, resolvability)."""
    return ss.paired_delta(rows, key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="tf_trim_ev03_rc05")
    ap.add_argument("--seed", default="s0")
    ap.add_argument("--songs", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    pat = str(REPO / f"outputs/eval_sweep_cache/{a.arm}#{a.seed}__*.zip")
    files = sorted(glob.glob(pat)) or sorted(glob.glob(
        str(REPO / f"outputs/eval_sweep_cache/{a.arm}__*.zip")))
    if a.songs:
        want = set(a.songs.split(","))
        files = [f for f in files if pathlib.Path(f).stem.split("__")[-1] in want]
    if not files:
        print(f"no cached maps for arm {a.arm}")
        return

    rows = []
    for f in files:
        song = pathlib.Path(f).stem.split("__")[-1]
        L = scorecard._load_any(pathlib.Path(f))
        if not L:
            continue
        bm, bpm = L[0], float(L[1])
        t = np.asarray(alignment.note_times(bm, bpm), dtype=float)
        if len(t) < 100:
            continue
        B = ss.bars(song, bpm, ss.song_end(song, float(t.max())))
        if B is None or B.n < 24:
            continue
        A = ss.bar_audio_matrix(song, B)
        if A is None:
            continue
        ours = song_scores(notes_xydc(bm, bpm), B, A)
        # ⚠️HUMAN SIDE MUST USE load_expert_only — scorecard._load_any silently
        # prefers ExpertPlus and contaminated three references before.
        H = load_expert_only(REPO / "data" / "raw" / f"{song}.zip")
        human = song_scores(notes_xydc(H[0], float(H[1])), B, A) if H else None
        if ours is None:
            continue
        rows.append({"song": song, "bars": B.n, "bar_s": round(B.dur, 2),
                     "grid": B.confidence.split(" ")[0], "ours": ours, "human": human})
        print(f"  {song:6s} bars {B.n:4d} ({B.dur:.2f}s)  "
              f"harm×place ours {ours.get('harm_place', float('nan')):+.4f} "
              f"human {human.get('harm_place', float('nan')):+.4f}" if human else
              f"  {song:6s} bars {B.n:4d} ({B.dur:.2f}s)  (no human map)")

    if not rows:
        print("nothing scored")
        return

    print(f"\n{'='*90}\nM1 MOTIF RHYME — arm {a.arm}, {len(rows)} songs")
    print(f"{'='*90}")
    print("all columns are over the PAIRED subset (songs with a human Expert map)")
    print(f"{'metric':<18} {'n':>3} {'ours':>9} {'human':>9} {'ours null':>10} "
          f"{'human null':>11} {'paired Δ':>10} {'resolvable':>11}")
    summary = {}
    for av in AUDIO_VIEWS:
        for mv in MAP_VIEWS:
            k = f"{av}_{mv}"
            # ⚠️PAIRED SUBSET ONLY. Only ~13 of the 24 songs ship a human Expert
            # map, so "our median over 24" beside "the human median over 13" is a
            # comparison across two populations — this project's single most
            # repeated mistake (hit 3× on 2026-08-04 alone). Every column below is
            # computed over the songs where BOTH sides scored.
            both = [r for r in rows
                    if r["ours"].get(k) is not None
                    and r.get("human") and r["human"].get(k) is not None]
            o = [r["ours"][k] for r in both]
            h = [r["human"][k] for r in both]
            on = [r["ours"][k + "_null"] for r in both
                  if r["ours"].get(k + "_null") is not None]
            hn = [r["human"][k + "_null"] for r in both
                  if r["human"].get(k + "_null") is not None]
            all_ours = [r["ours"][k] for r in rows if r["ours"].get(k) is not None]
            if not o:
                continue
            p = paired(rows, k)
            summary[k] = {"ours_median": round(st.median(o), 4),
                          "human_median": round(st.median(h), 4) if h else None,
                          "ours_median_all_songs": round(st.median(all_ours), 4),
                          "n_paired": len(both), "n_all": len(all_ours),
                          "ours_null_median": round(st.median(on), 4) if on else None,
                          "human_null_median": round(st.median(hn), 4) if hn else None,
                          "paired": p}
            print(f"{k:<18} {len(both):>3d} {st.median(o):>+9.4f} "
                  f"{(st.median(h) if h else float('nan')):>+9.4f} "
                  f"{(st.median(on) if on else float('nan')):>+10.4f} "
                  f"{(st.median(hn) if hn else float('nan')):>+11.4f} "
                  f"{p.get('delta', float('nan')):>+10.4f} "
                  f"{('YES' if p.get('resolvable') else 'no'):>11}")

    print("\nHOW TO READ:")
    print("  * a positive value = the map brings a pattern back when the music does")
    print("  * the NULL columns are the same map with its bars rotated; a real")
    print("    effect must clear its own null, not just zero")
    print("  * paired Δ = ours − human on the SAME song and the SAME bar grid;")
    print("    'resolvable' = |Δ| > 2 standard errors")

    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(
            {"arm": a.arm, "rows": rows, "summary": summary}, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
