#!/usr/bin/env python
"""Visual EDA — the SONG against the MAP, on one time axis.

Kyle, 2026-08-03: *"we were in the process of building out an eval suite that
mapped song notes to beatsaber notes you could visually see and do eda on so that
the time between me needing to give input between iterations would be slowed
down."*

**The gap this fills.** `render_map.py` draws the map (lattice, density, swing
paths) and `map_view.py` prints it as a score, but nothing has ever drawn **what
the music is doing** beside **what we played**. The offbeat defect found on
2026-08-03 -- we sit half a beat off multi-instrument hits 2.6x more often than
humans -- took three numeric scripts to find and is *obvious* in this view.

Lanes, top to bottom:

    bass / drums / other / vocals   per-stem onsets from the seeded stem cache
    k                               coincidence order per event (bar height 1-4);
                                    k>=3 events are the ones humans map 72-85% of
                                    the time
    OURS                            our notes, red/blue by hand
    HUMAN                           the human map's notes, when the song has one

Two annotations carry the diagnosis:
  * a k>=3 event with NO note within +-50 ms is ringed  -> a collision we ignored
  * one of our notes sitting within +-25 ms of half a beat from a k>=3 event is
    marked  -> we played the offbeat instead of the hit

Usage:
    python scripts/view_song_vs_map.py --song 1f333 --map outputs/.../1f333.zip
    python scripts/view_song_vs_map.py --song "SO TIRED ROCK - NUEKI" \
        --map outputs/manual/sotiredrock.zip --start 10 --end 60
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
from eval_coincidence import events_for  # noqa: E402

STEMS = ("bass", "drums", "other", "vocals")
STEM_COLOR = {"bass": "#7b4fd1", "drums": "#d1794f", "other": "#4f9bd1", "vocals": "#4fd18a"}
TOL = 0.050
PANEL_SECS = 20.0


def stem_onsets(song_id: str) -> dict[str, np.ndarray]:
    f = REPO / "outputs" / "stem_onset_cache" / f"{song_id}.npz"
    if not f.exists():
        sys.exit(f"no stem cache for {song_id!r} — build it with scripts/build_stem_onset_cache.py")
    d = np.load(f, allow_pickle=True)
    return {s: d[f"onsets_{s}"] for s in STEMS if f"onsets_{s}" in d.files}


def _nearest(ts: np.ndarray, t: float) -> float:
    if len(ts) == 0:
        return np.inf
    i = int(np.searchsorted(ts, t))
    c = [ts[j] for j in (i - 1, i) if 0 <= j < len(ts)]
    return min(abs(t - x) for x in c) if c else np.inf


def hand_times(beatmap, bpm: float) -> tuple[np.ndarray, np.ndarray]:
    """(left times, right times). Beat Saber colour 0 = red/left, 1 = blue/right."""
    spb = 60.0 / bpm
    out: list[list[float]] = [[], []]
    for n in beatmap.color_notes:
        if n.color in (0, 1):
            out[n.color].append(n.beat * spb)
    return np.sort(np.asarray(out[0])), np.sort(np.asarray(out[1]))


def draw(song: str, ours: np.ndarray, human: np.ndarray | None, bpm: float,
         start: float, end: float, out: pathlib.Path, ours_label: str,
         ours_hands=None, human_hands=None, panel_secs: float = PANEL_SECS) -> None:
    stems = stem_onsets(song)
    times, ks = events_for(song, 0.030)
    beat = 60.0 / bpm
    PANEL_SECS_L = panel_secs

    panels = int(np.ceil((end - start) / PANEL_SECS_L))
    fig, axes = plt.subplots(panels, 1, figsize=(19, 2.9 * panels), squeeze=False)
    axes = axes[:, 0]

    lanes = list(STEMS) + ["k", "OURS", "HUMAN"]
    ypos = {name: len(lanes) - 1 - i for i, name in enumerate(lanes)}

    for pi, ax in enumerate(axes):
        t0 = start + pi * PANEL_SECS_L
        t1 = min(t0 + PANEL_SECS_L, end)

        # beat grid, with downbeats emphasised every 4 beats
        b0 = int(np.floor(t0 / beat))
        for bi in range(b0, int(np.ceil(t1 / beat)) + 1):
            t = bi * beat
            if t0 <= t <= t1:
                ax.axvline(t, color="0.90" if bi % 4 else "0.72",
                           lw=1.4 if bi % 4 == 0 else 0.7, zorder=0)

        for s in STEMS:
            if s not in stems:
                continue
            v = stems[s]
            v = v[(v >= t0) & (v <= t1)]
            ax.vlines(v, ypos[s] - 0.32, ypos[s] + 0.32, color=STEM_COLOR[s], lw=1.3)

        m = (times >= t0) & (times <= t1)
        for t, k in zip(times[m], ks[m]):
            ax.vlines(t, ypos["k"] - 0.40, ypos["k"] - 0.40 + 0.20 * k,
                      color="#222", lw=1.6 if k >= 3 else 0.8,
                      alpha=1.0 if k >= 3 else 0.35)

        # Notes split by hand: left below the lane midline, right above, so hand
        # alternation is readable at a glance and a DOUBLE shows as a full-height
        # bar. Colours are Beat Saber's own (red = left, blue = right).
        for lane, hands in (("OURS", ours_hands), ("HUMAN", human_hands)):
            if hands is None:
                continue
            for color, sign, ts in (("#d1414a", -1, hands[0]), ("#3b7dd8", +1, hands[1])):
                v = ts[(ts >= t0) & (ts <= t1)]
                ax.vlines(v, ypos[lane], ypos[lane] + sign * 0.40, color=color, lw=1.6)
            ax.axhline(ypos[lane], xmin=0, xmax=1, color="0.85", lw=0.6, zorder=0)

        # THE TWO DIAGNOSES
        for t, k in zip(times[m], ks[m]):
            if k < 3:
                continue
            d = _nearest(ours, t)
            if d > TOL:                                   # collision we ignored
                ax.plot(t, ypos["k"] + 0.55, marker="o", ms=7, mfc="none",
                        mec="#c9002b", mew=1.8, zorder=5)
            # did we play the offbeat instead?
            off = min(_nearest(ours, t + beat / 2), _nearest(ours, t - beat / 2))
            if d > TOL and off <= 0.025:
                ax.plot(t, ypos["OURS"] + 0.62, marker="v", ms=8, color="#c9002b", zorder=5)

        ax.set_xlim(t0, t1)
        ax.set_ylim(-0.8, len(lanes) - 0.2)
        ax.set_yticks([ypos[n] for n in lanes])
        ax.set_yticklabels(lanes, fontsize=9)
        ax.set_xlabel("seconds", fontsize=8)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[0].set_title(
        f"{song}   —   {ours_label} vs the music   (bpm {bpm:g}, beat {beat*1000:.0f} ms)\n"
        "○ = multi-instrument hit we ignored     ▼ = we played the OFFBEAT instead",
        fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True, help="stem-cache song id, e.g. 1f333")
    ap.add_argument("--map", required=True, help="our .zip")
    ap.add_argument("--human", default=None, help="human .zip (default data/raw/<song>.zip)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--panel-secs", type=float, default=10.0,
                    help="seconds per stacked panel; lower = more readable")
    a = ap.parse_args()

    L = scorecard._load_any(pathlib.Path(a.map))
    if not L:
        sys.exit(f"could not load {a.map}")
    ours = np.sort(np.asarray(alignment.note_times(L[0], L[1]), dtype=float))
    bpm = float(L[1])
    ours_hands = hand_times(L[0], bpm)

    hp = pathlib.Path(a.human) if a.human else REPO / "data" / "raw" / f"{a.song}.zip"
    human = None
    human_hands = None
    if hp.exists():
        H = load_expert_only(hp)          # never _load_any for humans: prefers ExpertPlus
        if H:
            human = np.sort(np.asarray(alignment.note_times(H[0], H[1]), dtype=float))
            human_hands = hand_times(H[0], float(H[1]))
            print(f"human map: {len(human)} note times")
    if human is None:
        print("no human map for this song — HUMAN lane will be empty")

    end = a.end if a.end is not None else float(ours.max()) + 2.0
    out = pathlib.Path(a.out) if a.out else REPO / "outputs" / f"view_{a.song}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(a.song, ours, human, bpm, a.start, end, out, pathlib.Path(a.map).stem,
         ours_hands=ours_hands, human_hands=human_hands, panel_secs=a.panel_secs)


if __name__ == "__main__":
    main()
