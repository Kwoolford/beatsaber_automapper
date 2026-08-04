#!/usr/bin/env python
"""Visual EDA (2) — the WHOLE-SONG strip: where the map does and doesn't answer the music.

`view_song_vs_map.py` zooms in on seconds. This is the other end: one page per
song showing, on a shared time axis, whether the map's effort is going where the
music's is. It is aimed at the objections that are about *allocation over the
song* rather than about individual notes:

  W2  "It just feels really empty for no reason" (Fallen Kingdom)
  W3  "Some parts get really intense to play, even though they are not the main
       beat where you would expect the peak difficulty to be"
  W1a  we play the offbeat at multi-instrument events

Four rows, all against the same seconds axis:

  1. NPS          ours vs the human map, rolling. Gaps and over-peaks are W2/W3.
  2. INTENSITY    audio RMS as a share of the song's own peak -- Kyle's own
                  framing: *"maybe a sound compared to rest of song to easily
                  draw intensity."* Peaks here are where the peaks should be.
  3. RESPONSE     rolling share of k>=3 multi-instrument events that got a note.
                  Humans sit at 0.72-0.85; a trough is a passage we ignored.
  4. OFFBEAT      rolling halfbeat_rate -- of the events we DID answer nearby,
                  how many did we answer half a beat off. Human median 0.095.

Usage:
    python scripts/view_song_strip.py --song 1f8d6 --map outputs/.../1f8d6.zip
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

WIN = 8.0          # rolling window, seconds
HOP = 2.0
TOL = 0.050
HUMAN_RESPONSE = 0.724     # human median at k>=3 (eval_coincidence, n=263)
HUMAN_OFFBEAT = 0.095      # human median halfbeat_rate (eval_beat_phase, n=188)


def rolling(centers, values_at, win):
    return np.array([values_at(c - win / 2, c + win / 2) for c in centers])


def nearest(ts, t):
    if len(ts) == 0:
        return np.inf
    i = int(np.searchsorted(ts, t))
    c = [ts[j] for j in (i - 1, i) if 0 <= j < len(ts)]
    return min(abs(t - x) for x in c) if c else np.inf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--human", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    L = scorecard._load_any(pathlib.Path(a.map))
    if not L:
        sys.exit(f"could not load {a.map}")
    bpm = float(L[1])
    beat = 60.0 / bpm
    ours = np.sort(np.asarray(alignment.note_times(L[0], bpm), dtype=float))

    hp = pathlib.Path(a.human) if a.human else REPO / "data" / "raw" / f"{a.song}.zip"
    human = None
    if hp.exists():
        H = load_expert_only(hp)
        if H:
            human = np.sort(np.asarray(alignment.note_times(H[0], float(H[1])), dtype=float))

    times, ks = events_for(a.song, 0.030)
    ev3 = times[ks >= 3]

    end = float(max(ours.max(), times.max()))
    centers = np.arange(WIN / 2, end - WIN / 2, HOP)

    nps_o = rolling(centers, lambda t0, t1: ((ours >= t0) & (ours < t1)).sum() / WIN, WIN)
    nps_h = (rolling(centers, lambda t0, t1: ((human >= t0) & (human < t1)).sum() / WIN, WIN)
             if human is not None else None)

    def response(t0, t1):
        e = ev3[(ev3 >= t0) & (ev3 < t1)]
        if len(e) == 0:
            return np.nan
        return float(np.mean([nearest(ours, t) <= TOL for t in e]))

    def offbeat(t0, t1):
        e = ev3[(ev3 >= t0) & (ev3 < t1)]
        if len(e) < 3:
            return np.nan
        ph = []
        for t in e:
            i = int(np.searchsorted(ours, t))
            c = [ours[j] for j in (i - 1, i) if 0 <= j < len(ours)]
            if not c:
                continue
            d = min(c, key=lambda x: abs(t - x)) - t
            ph.append(abs((d + beat / 2) % beat - beat / 2))
        return float(np.mean(np.array(ph) >= 0.35 * beat)) if ph else np.nan

    resp = rolling(centers, response, WIN)
    offb = rolling(centers, offbeat, WIN)

    # intensity: RMS relative to the song's own peak (Kyle's framing)
    inten = None
    ap_ = pathlib.Path(a.audio) if a.audio else REPO / "data" / "eval_songset" / f"{a.song}.ogg"
    if ap_.exists():
        import librosa
        y, sr = librosa.load(str(ap_), sr=22050, mono=True)
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        rt = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
        inten = np.interp(centers, rt, rms / max(rms.max(), 1e-9))

    fig, axes = plt.subplots(4, 1, figsize=(17, 10), sharex=True)

    axes[0].plot(centers, nps_o, color="#1f77b4", lw=1.8, label="ours")
    if nps_h is not None:
        axes[0].plot(centers, nps_h, color="#888", lw=1.4, ls="--", label="human")
    axes[0].axhline(3.91, color="#c9002b", lw=0.9, ls=":", label="human corpus median 3.91")
    axes[0].set_ylabel("NPS")
    axes[0].legend(fontsize=8, ncol=3, loc="upper right")

    if inten is not None:
        axes[1].fill_between(centers, inten, color="#7b4fd1", alpha=0.35)
    axes[1].set_ylabel("intensity\n(RMS / peak)")
    axes[1].set_ylim(0, 1.05)

    axes[2].plot(centers, resp, color="#2a9d5c", lw=1.8)
    axes[2].axhline(HUMAN_RESPONSE, color="#c9002b", lw=1.0, ls=":",
                    label=f"human {HUMAN_RESPONSE:.2f}")
    axes[2].set_ylabel("response to\nk≥3 events")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=8, loc="lower right")

    axes[3].plot(centers, offb, color="#d1794f", lw=1.8)
    axes[3].axhline(HUMAN_OFFBEAT, color="#c9002b", lw=1.0, ls=":",
                    label=f"human {HUMAN_OFFBEAT:.3f}")
    axes[3].set_ylabel("offbeat rate\n(half-beat off)")
    axes[3].set_ylim(0, 1.05)
    axes[3].set_xlabel("seconds")
    axes[3].legend(fontsize=8, loc="upper right")

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="0.93", lw=0.7)

    axes[0].set_title(
        f"{a.song} — {pathlib.Path(a.map).stem}   (bpm {bpm:g})\n"
        "row 3 low = a passage we ignored (W2) · row 1 peak away from a row 2 peak = "
        "misallocated intensity (W3) · row 4 high = we answered the offbeat (W1a)",
        fontsize=11, loc="left")
    fig.tight_layout()
    out = pathlib.Path(a.out) if a.out else REPO / "outputs" / f"strip_{a.song}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
