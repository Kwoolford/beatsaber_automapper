#!/usr/bin/env python
"""★THE PICTURE FOR THE MASTERPIECE AXES — the song's structure beside the map's.

Kyle's P0 framing: *"Create a way for you to see the song and map in a way that
gives you my vision."* `view_main_beat.py` shows one stretch of one song in detail.
This shows the WHOLE song at once, as structure:

    panel 1   the MUSIC's self-similarity (chroma + MFCC over bars) — the bright
              off-diagonal stripes are the song repeating itself
    panel 2   OUR map's self-similarity (Cohen's kappa on the bar rhythm)
    panel 3   the HUMAN map's self-similarity, same estimator, same grid
    panel 4   per-bar rhythm fidelity: which stem each map is tracking

The diagnosis is visual and immediate. Panel 1 says *this song repeats here, here
and here*. If panels 2 and 3 differ in whether those same stripes appear, the map
that shows them is the one that was written rather than generated. Numbers for the
same comparison are in M1 (`eval_motif_rhyme.py`) and M2 (`eval_rhythm_fidelity.py`);
the picture is what makes a number believable.

🔑PNG is the primary artifact — an agent can only look at an image by rendering it
to a file and reading it back.

Usage:
    python scripts/view_structure.py --song 1f8d6
    python scripts/view_structure.py --song 1f333 --arm v8_mbb025 --out outputs/x.png
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import eval_arrangement as m4  # noqa: E402
import song_structure as ss  # noqa: E402
from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from eval_motif_rhyme import notes_xydc  # noqa: E402
from eval_rhythm_fidelity import (STEMS, follow_scores, quantise,  # noqa: E402
                                  stem_onsets)

SCOL = {"bass": "#7b4fd1", "drums": "#d1794f", "other": "#4f9bd1",
        "vocals": "#4fd18a"}


def ssm_of(notes, B):
    V = ss.map_bar_vectors(notes, B)
    return ss.bar_map_similarity(V)["rhythm"]


def lead_strip(times, B, stems, rng):
    """Per-bar (lead stem index, gain) for one map."""
    M = quantise(np.sort(np.asarray(times, dtype=float)), B)
    keys, G = [], []
    for s in STEMS:
        if s not in stems:
            continue
        g, _ = follow_scores(M, quantise(stems[s], B), rng)
        keys.append(s)
        G.append(g)
    if not G:
        return [], None
    return keys, np.vstack(G)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True)
    ap.add_argument("--arm", default="tf_trim_ev03_rc05")
    ap.add_argument("--map", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    mp = a.map or next(iter(sorted(glob.glob(
        str(REPO / f"outputs/eval_sweep_cache/{a.arm}#s0__{a.song}.zip")))), "")
    if not mp:
        print(f"no map for {a.song} / {a.arm}")
        return
    L = scorecard._load_any(pathlib.Path(mp))
    bm, bpm = L[0], float(L[1])
    t = np.asarray(alignment.note_times(bm, bpm), dtype=float)
    B = ss.bars(a.song, bpm, ss.song_end(a.song, float(t.max())))
    if B is None:
        print("no bar grid")
        return
    A = ss.bar_audio_matrix(a.song, B)
    stems = stem_onsets(a.song)
    ours = notes_xydc(bm, bpm)
    H = load_expert_only(REPO / "data" / "raw" / f"{a.song}.zip")
    human = notes_xydc(H[0], float(H[1])) if H else None

    S_aud = np.nan_to_num(np.nanmean(np.stack([A["harm"], A["timb"]]), axis=0))
    S_our = ssm_of(ours, B)
    S_hum = ssm_of(human, B) if human else None
    nov = m4.novelty(A)
    bnds = m4.boundaries(nov) if nov is not None else []

    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.15], hspace=0.34, wspace=0.18)
    panels = [("the MUSIC repeats here", S_aud, "magma"),
              (f"OUR map ({a.arm})", S_our, "viridis"),
              ("the HUMAN map", S_hum, "viridis")]
    for i, (title, M, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        if M is None:
            ax.text(0.5, 0.5, "no human Expert map", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        finite = M[np.isfinite(M)]
        vmax = float(np.quantile(finite, 0.97)) if len(finite) else 1.0
        vmin = float(np.quantile(finite, 0.10)) if len(finite) else 0.0
        ax.imshow(np.nan_to_num(M, nan=vmin), cmap=cmap, vmin=vmin, vmax=vmax,
                  origin="lower", interpolation="nearest")
        for b in bnds:
            ax.axvline(b, color="#ffffff", lw=0.6, alpha=0.35)
            ax.axhline(b, color="#ffffff", lw=0.6, alpha=0.35)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("bar")
        if i == 0:
            ax.set_ylabel("bar")

    # panel 4: which stem each map tracks, bar by bar
    rng = np.random.default_rng(0)
    ax = fig.add_subplot(gs[1, :])
    for row, (label, notes) in enumerate((("ours", ours), ("human", human))):
        if notes is None:
            continue
        keys, G = lead_strip([n[0] for n in notes], B, stems, rng)
        if G is None:
            continue
        # ⚠️First version required EVERY stem to be scoreable in a bar and drew
        # only positive gains; on a 1.74 s bar that left ~20 of 145 bars drawn and
        # the strip read as "the map follows nothing", which is a rendering
        # artefact, not a finding. Two stems is enough to name a lead, and bars
        # where the map tracks nothing are drawn faint rather than omitted.
        n_ok = np.isfinite(G).sum(axis=0)
        Gf = np.where(np.isfinite(G), G, -np.inf)
        lead = np.argmax(Gf, axis=0)
        best = np.max(Gf, axis=0)
        for bi in range(B.n):
            if n_ok[bi] < 2 or not np.isfinite(best[bi]):
                continue
            if best[bi] <= 0:
                ax.add_patch(plt.Rectangle((bi, row * 1.1), 1.0, 0.10,
                                           color="#c8c8c8", lw=0))
                continue
            ax.add_patch(plt.Rectangle((bi, row * 1.1), 1.0,
                                       float(min(1.0, 0.15 + 3.0 * best[bi])),
                                       color=SCOL.get(keys[lead[bi]], "#888"),
                                       lw=0))
    ax.set_xlim(0, B.n)
    ax.set_ylim(0, 2.3)
    ax.set_yticks([0.5, 1.6])
    ax.set_yticklabels(["ours", "human"], fontsize=9)
    ax.set_xlabel("bar — colour = the stem this bar's rhythm follows, "
                  "height = how strongly")
    for b in bnds:
        ax.axvline(b, color="#333", lw=0.6, alpha=0.5)
    handles = [plt.Line2D([], [], color=SCOL[s], lw=6, label=s)
               for s in STEMS if s in stems]
    ax.legend(handles=handles, ncol=4, fontsize=8, loc="upper right",
              frameon=False)

    fig.suptitle(f"{a.song} — structure: the song, our map, the human map "
                 f"({B.n} bars of {B.dur:.2f}s, main beat {B.ratio:g}× the fitted beat, "
                 f"grid {B.confidence.split(' ')[0]})", fontsize=11)
    out = a.out or str(REPO / f"outputs/structure_{a.song}_{a.arm}.png")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
