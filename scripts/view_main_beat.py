#!/usr/bin/env python
"""★ THE VIEW — the song's instruments and the map's notes, against the MAIN BEAT.

Kyle, 2026-08-04: *"if you had a view of all notes of the instruments and vocals
the way I did, the iteration speed would increase dramatically… Create a way for
you to see the song and map in a way that gives you my vision."*

His defect, in his words: *"every couple main beat notes were mapped instead of
most of the main beats… it hits the main flow partially."* Measured: we cover
**~0.49** of the main beat against the human's **~0.70**.

**What this draws, top to bottom, on one time axis:**

    vocals / other / drums / bass   per-stem onsets from the seeded Demucs cache
    MAIN BEAT                       the pulse the MUSIC is built on (main_beat.py
                                    picks the metrical level; ½×, 1× or 2× the
                                    fitted beat). Each beat is a tick:
                                      ● filled  = we played it
                                      ○ hollow red = WE MISSED IT   <- his defect
    OURS                            our notes, red/blue by hand, and any note NOT
                                    on a main beat is drawn hollow — so "partial
                                    main line + lots of other notes" is one glance
    HUMAN                           the human map, drawn the same way

**Design constraint**: the primary output is PNG, because an agent can only look
at an image by rendering it and reading it back. Panels default to 12s so the
ticks stay distinguishable; `--secs` trades detail for span.

Usage:
    python scripts/view_main_beat.py --song 1f8d6 --map path/to.zip --start 200 --end 248
    python scripts/view_main_beat.py --song 1f333 --map a.zip --worst   # auto-pick the worst stretch
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from main_beat import _near, _tol, coverage, find_main_beat  # noqa: E402

STEMS = ("vocals", "other", "drums", "bass")
SCOL = {"bass": "#7b4fd1", "drums": "#d1794f", "other": "#4f9bd1", "vocals": "#4fd18a"}
PANEL = 12.0


def hand_times(bm, bpm):
    spb = 60.0 / bpm
    out = [[], []]
    for n in bm.color_notes:
        if n.color in (0, 1):
            out[n.color].append(n.beat * spb)
    return np.sort(np.asarray(out[0])), np.sort(np.asarray(out[1]))


def worst_window(ours, mb, span, end):
    """The `span`-second stretch where we miss the most main beats — start here."""
    tol = _tol(mb.period)
    beats = mb.runs if len(mb.runs) >= 20 else mb.grid
    miss = np.array([t for t in beats if not _near(ours, t, tol)])
    if len(miss) == 0:
        return 0.0
    best, bs = 0.0, -1
    for t0 in np.arange(0, max(end - span, 1.0), span / 2):
        c = int(((miss >= t0) & (miss < t0 + span)).sum())
        if c > bs:
            bs, best = c, float(t0)
    return best


def load_probs(song: str, probs_dir: str | None):
    """Stage-1's raw probability, if a BEAT_PROBS_DUMP exists for this song.

    ★Added 2026-08-04 because the suite proved the defect lives HERE, not in the
    decode: in our worst windows the probability is INVERTED — 0.590 one slot off
    the main beat against 0.320 on it, where healthy windows read 0.725 on and
    0.301 off. Drawing it turns the picture from "we missed these beats" into
    "the model was pointing between them", which is the difference between a
    symptom and a cause.
    """
    for d in ([probs_dir] if probs_dir else
              ["outputs/probs_phase_2026-08-03", "outputs/probs_phase_instr_2026-08-03"]):
        f = REPO / d / f"{song}.npz"
        if f.exists():
            z = np.load(f)
            P = z["beat_probs"].max(axis=1)
            slot = 60.0 / float(z["bpm"]) / int(z["beat_subdiv"])
            return np.arange(len(P)) * slot, P, d
    return None


def draw(song, ours, ours_h, human, human_h, mb, bpm, t_start, t_end, out, label,
         probs=None):
    d = np.load(REPO / "outputs" / "stem_onset_cache" / f"{song}.npz", allow_pickle=True)
    stems = {s: np.sort(d[f"onsets_{s}"]) for s in STEMS if f"onsets_{s}" in d.files}
    tol = _tol(mb.period)
    beats = mb.runs if len(mb.runs) >= 20 else mb.grid

    lanes = (["Stage-1 p", ""] if probs is not None else []) + list(STEMS) \
        + ["MAIN BEAT", "OURS"] + (["HUMAN"] if human is not None else [])
    ypos = {n: len(lanes) - 1 - i for i, n in enumerate(lanes)}
    npanel = int(np.ceil((t_end - t_start) / PANEL))
    fig, axes = plt.subplots(npanel, 1, figsize=(20, 4.4 * npanel), squeeze=False)
    axes = axes[:, 0]

    for pi, ax in enumerate(axes):
        a, b = t_start + pi * PANEL, min(t_start + (pi + 1) * PANEL, t_end)
        if probs is not None:
            # Two lanes tall and filled: at one lane height the curve was a thin
            # unreadable band. The point of this lane is to SEE whether the peaks
            # sit on the beat markers below, so it must be legible.
            pt, pv, _ = probs
            base = ypos[""] - 0.45
            m = (pt >= a - 0.2) & (pt <= b + 0.2)
            ax.fill_between(pt[m], base, base + 1.85 * pv[m], color="#666",
                            alpha=0.30, lw=0, zorder=2)
            ax.plot(pt[m], base + 1.85 * pv[m], color="#333", lw=0.9, zorder=3)
            ax.axhline(base, color="0.8", lw=0.6, zorder=0)
            # drop a guide at each main beat so alignment is judged, not guessed
            for t in beats[(beats >= a) & (beats <= b)]:
                ax.plot([t, t], [base, base + 1.85], color="#c9002b", lw=0.5,
                        alpha=0.35, zorder=1)
        for s in STEMS:
            if s not in stems:
                continue
            v = stems[s]
            v = v[(v >= a) & (v <= b)]
            ax.vlines(v, ypos[s] - 0.3, ypos[s] + 0.3, color=SCOL[s], lw=1.5)

        # MAIN BEAT — filled if we played it, hollow red if we missed it
        # ★A MISS HAS TWO FLAVOURS AND THEY MEAN DIFFERENT THINGS:
        #   ○ red    nothing within half a period — we simply SKIPPED the beat
        #   ◔ orange we DID play near it but off the grid — we played AROUND it
        # Kyle's two complaints are exactly these: "every couple main beat notes
        # were mapped instead of most" (skipped) and "the map still maps a lot of
        # non main beat notes" (played around). Colouring them apart turns the
        # picture from descriptive into diagnostic.
        bw = beats[(beats >= a) & (beats <= b)]
        for t in bw:
            got = _near(ours, t, tol)
            if got:
                ax.plot(t, ypos["MAIN BEAT"], marker="o", ms=7, mfc="#222",
                        mec="#222", mew=1.0, zorder=4)
            else:
                nearby = _near(ours, t, mb.period * 0.5)
                ax.plot(t, ypos["MAIN BEAT"], marker="o", ms=8, mfc="none",
                        mec="#e08214" if nearby else "#c9002b", mew=2.2, zorder=4)
        ax.axhline(ypos["MAIN BEAT"], color="0.85", lw=0.6, zorder=0)

        for lane, hands in (("OURS", ours_h), ("HUMAN", human_h)):
            if lane not in ypos or hands is None:
                continue
            for col, sign, ts in (("#d1414a", -1, hands[0]), ("#3b7dd8", +1, hands[1])):
                v = ts[(ts >= a) & (ts <= b)]
                on = np.array([_near(beats, t, tol) for t in v], dtype=bool) if len(v) else v
                for t, is_on in zip(v, on):
                    ax.vlines(t, ypos[lane], ypos[lane] + sign * 0.38, color=col,
                              lw=2.0 if is_on else 1.0,
                              alpha=1.0 if is_on else 0.45)
            ax.axhline(ypos[lane], color="0.85", lw=0.6, zorder=0)

        # bar lines every 4 main beats
        for i, t in enumerate(beats[(beats >= a - mb.period) & (beats <= b)]):
            if i % 4 == 0 and a <= t <= b:
                ax.axvline(t, color="0.88", lw=1.0, zorder=0)

        if len(bw):
            _got = sum(1 for t in bw if _near(ours, t, tol))
            ax.text(0.995, 0.97, f"{_got}/{len(bw)} main beats  ({_got/len(bw):.0%})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    color="#c9002b" if _got / len(bw) < 0.7 else "#2a9d5c")
        ax.set_xlim(a, b)
        ax.set_ylim(-0.8, len(lanes) - 0.2)
        ax.set_yticks([ypos[n] for n in lanes])
        ax.set_yticklabels(lanes, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("seconds", fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    cov = coverage(ours, mb)
    covh = coverage(human, mb) if human is not None else {}
    axes[0].set_title(
        f"{song} — {label}   |   MAIN BEAT = {mb.ratio:g}× the fitted beat "
        f"(period {mb.period*1000:.0f}ms, bpm {bpm:g}, fit {mb.confidence})\n"
        f"we cover {cov.get('main_covered', float('nan')):.1%} of the main beat"
        + (f"  ·  human {covh.get('main_covered', float('nan')):.1%}" if covh else "")
        + "\n● played   ○ red = SKIPPED (nothing near)   ○ orange = played AROUND it "
        "(note nearby but off-grid)   ·   faint notes = not on the main beat",
        fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--human", default=None)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--secs", type=float, default=PANEL)
    ap.add_argument("--worst", action="store_true",
                    help="jump to the stretch where we miss the most main beats")
    ap.add_argument("--span", type=float, default=36.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--probs", default=None,
                    help="dir of BEAT_PROBS_DUMP npz; auto-detected if omitted")
    ap.add_argument("--no-probs", action="store_true")
    a = ap.parse_args()
    globals()["PANEL"] = a.secs

    L = scorecard._load_any(pathlib.Path(a.map))
    if not L:
        sys.exit(f"could not load {a.map}")
    bpm = float(L[1])
    ours = np.sort(np.asarray(alignment.note_times(L[0], bpm), dtype=float))
    ours_h = hand_times(L[0], bpm)

    hp = pathlib.Path(a.human) if a.human else REPO / "data" / "raw" / f"{a.song}.zip"
    human = human_h = None
    if hp.exists():
        H = load_expert_only(hp)
        if H:
            human = np.sort(np.asarray(alignment.note_times(H[0], float(H[1])), dtype=float))
            human_h = hand_times(H[0], float(H[1]))

    end_all = float(max(ours.max(), human.max() if human is not None else 0))
    mb = find_main_beat(a.song, bpm, end_all)
    if mb is None:
        sys.exit("could not identify a main beat (no stem cache, or too few onsets)")

    if a.worst:
        s = worst_window(ours, mb, a.span, end_all)
        t0, t1 = s, s + a.span
    else:
        t0 = a.start if a.start is not None else 0.0
        t1 = a.end if a.end is not None else min(t0 + a.span, end_all)

    out = pathlib.Path(a.out) if a.out else \
        REPO / "outputs" / f"mainbeat_{a.song}_{int(t0)}_{int(t1)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pr = None if a.no_probs else load_probs(a.song, a.probs)
    if pr is not None:
        print(f"probability lane from {pr[2]}")
    draw(a.song, ours, ours_h, human, human_h, mb, bpm, t0, t1, out,
         pathlib.Path(a.map).stem, probs=pr)


if __name__ == "__main__":
    main()
