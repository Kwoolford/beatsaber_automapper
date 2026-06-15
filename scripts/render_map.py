#!/usr/bin/env python
"""Mapper's-eye renderer for Beat Saber maps (Phase 1, TASK P1-2).

Turns a map into PNGs Claude (or a human) can read directly — the agent-side
ArcViewer from the fresh-eyes plan §4. Three views, all matplotlib/CPU:

  (a) LATTICE panels: time on x, the 4x3 grid unrolled on y (12 cells), a
      cut-direction arrow per note, colored by hand, with beat lines. Each panel
      is a time-unrolled window (~8 beats, like sheet music), so a handful of
      panels covers a song. Dots (any-direction) are drawn as hollow circles.
  (b) DENSITY strip (whole song): note-density curve over audio RMS, with the
      swing-simulator's violation beats marked — instantly shows dead drops,
      flat/monotone density, and parity blowups.
  (c) SWING-PATH trace: per-hand parity (forehand below / backhand above the
      midline) over time, resets and violations flagged — flow made visible.

Reads parity/violations from evaluation.swing_sim (P1-1). Read-only.

Usage:
    python scripts/render_map.py data/raw/1f1e1.zip --difficulty Expert \
        --out outputs/render/1f1e1.png
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from beatsaber_automapper.evaluation import swing_sim as ss  # noqa: E402

# cut direction -> unit (dx, dy) the saber travels; 8 = dot (no direction)
_DIR_VEC = {
    0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
    4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1),
}
_RED, _BLUE = "#e23b3b", "#3b7fe2"  # hand 0 / hand 1


def _hand_color(color: int) -> str:
    return _RED if color == 0 else _BLUE


def _cell_y(x: int, y: int) -> float:
    """Unroll the 4x3 grid to a single y coordinate (row-major, row 0 at bottom)."""
    return y * 4 + x


def _load_audio_rms(map_path: pathlib.Path, bpm: float, max_beat: float):
    """Return (beats, rms_norm) sampled over the song, or (None, None)."""
    import shutil
    import tempfile
    import zipfile

    try:
        import librosa

        from beatsaber_automapper.data.audio import load_audio
    except Exception:  # noqa: BLE001
        return None, None

    if map_path.suffix != ".zip":
        return None, None
    tmp = tempfile.mkdtemp(prefix="render_rms_")
    try:
        with zipfile.ZipFile(map_path) as zf:
            zf.extractall(tmp)
        songs = [
            p for p in pathlib.Path(tmp).iterdir()
            if p.suffix.lower() in (".egg", ".ogg", ".wav", ".mp3", ".flac")
        ]
        if not songs:
            return None, None
        wav, sr = load_audio(songs[0], target_sr=22050)
        y = wav.squeeze(0).numpy()
        hop = 2048
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        times = np.arange(len(rms)) * hop / sr
        beats = times * bpm / 60.0
        rms_norm = rms / (rms.max() + 1e-9)
        mask = beats <= max_beat + 4
        return beats[mask], rms_norm[mask]
    except Exception:  # noqa: BLE001
        return None, None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _draw_lattice(ax, notes, scorecard, b0: float, b1: float, bpm: float) -> None:
    win = [n for n in notes if b0 <= n.beat < b1]
    viol = {round(b, 3) for b in scorecard.violation_beats if b0 <= b < b1}

    for n in win:
        cy = _cell_y(n.x, n.y)
        col = _hand_color(n.color)
        if n.direction == 8 or n.direction not in _DIR_VEC:
            ax.scatter([n.beat], [cy], s=70, facecolors="none",
                       edgecolors=col, linewidths=1.8, zorder=3)
        else:
            dx, dy = _DIR_VEC[n.direction]
            norm = (dx**2 + dy**2) ** 0.5
            ln = 0.30  # arrow half-length in beat/cell units
            ax.add_patch(FancyArrowPatch(
                (n.beat - dx / norm * ln * 0.5, cy - dy / norm * ln),
                (n.beat + dx / norm * ln * 0.5, cy + dy / norm * ln),
                arrowstyle="-|>", mutation_scale=11, color=col, lw=2.0, zorder=3))

    for b in range(int(np.ceil(b0)), int(np.floor(b1)) + 1):
        ax.axvline(b, color="0.85", lw=0.8, zorder=0)
    for vb in viol:
        ax.axvspan(vb - 0.06, vb + 0.06, color="#ffcf33", alpha=0.55, zorder=1)
    for ry in (3.5, 7.5):  # row separators
        ax.axhline(ry, color="0.92", lw=0.6, zorder=0)

    ax.set_xlim(b0, b1)
    ax.set_ylim(-0.8, 11.8)
    ax.set_yticks([1.5, 5.5, 9.5])
    ax.set_yticklabels(["row0\n(bottom)", "row1", "row2\n(top)"], fontsize=7)
    t0, t1 = b0 * 60 / bpm, b1 * 60 / bpm
    ax.set_title(f"beats {b0:.0f}-{b1:.0f}  ({t0:.1f}-{t1:.1f}s)   "
                 f"{len(win)} notes, {len(viol)} parity-viol",
                 fontsize=8, loc="left")
    ax.set_xlabel("beat", fontsize=7)
    ax.tick_params(labelsize=7)


def _draw_density(ax, notes, scorecard, bpm: float, max_beat: float,
                  rms_beats, rms_vals) -> None:
    win = 2.0  # beats per density bin
    edges = np.arange(0, max_beat + win, win)
    counts, _ = np.histogram([n.beat for n in notes], bins=edges)
    nps = counts / (win * 60 / bpm)  # notes per second
    centers = edges[:-1] + win / 2

    if rms_beats is not None:
        ax.fill_between(rms_beats, 0, rms_vals * nps.max() * 1.05,
                        color="0.85", zorder=0, label="audio RMS")
    ax.plot(centers, nps, color="#2a2a2a", lw=1.3, zorder=2, label="note density (NPS)")
    for vb in scorecard.violation_beats:
        ax.axvline(vb, color="#ffb000", lw=0.4, alpha=0.5, zorder=1)
    ax.set_xlim(0, max_beat)
    ax.set_ylim(0, max(nps.max() * 1.15, 1))
    ax.set_title(f"whole-song density vs RMS   ({len(notes)} notes, "
                 f"{scorecard.violations} parity-violations, "
                 f"reset-rate {scorecard.resets / max(scorecard.n_swings, 1):.2f})",
                 fontsize=8, loc="left")
    ax.set_xlabel("beat", fontsize=7)
    ax.set_ylabel("NPS", fontsize=7)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=7)


def _draw_swing_trace(ax, scorecard, b0: float, b1: float) -> None:
    for color, hand in scorecard.per_hand.items():
        sw = [s for s in hand.swings if b0 <= s.beat < b1]
        if not sw:
            continue
        xs = [s.beat for s in sw]
        ys = [1 if s.parity.name == "BACKHAND" else -1 for s in sw]
        c = _hand_color(color)
        ax.plot(xs, ys, "-", color=c, lw=1.0, alpha=0.7, zorder=2)
        for s in sw:
            yy = 1 if s.parity.name == "BACKHAND" else -1
            if s.reset_kind == "violation":
                ax.scatter([s.beat], [yy], s=40, marker="x", color="#d11", zorder=4)
            elif s.is_reset:
                ax.scatter([s.beat], [yy], s=18, marker="o",
                           facecolors="none", edgecolors=c, zorder=3)
            else:
                ax.scatter([s.beat], [yy], s=10, color=c, zorder=3)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlim(b0, b1)
    ax.set_ylim(-1.8, 1.8)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["fore", "back"], fontsize=7)
    ax.set_xlabel("beat", fontsize=7)
    ax.tick_params(labelsize=7)


def _pick_windows(notes, max_beat: float, panel_beats: float, n: int) -> list:
    """Evenly spaced windows across the song, snapped to note-dense starts."""
    if not notes:
        return [(0, panel_beats)]
    starts = np.linspace(0, max(max_beat - panel_beats, 0), n)
    return [(float(s), float(s + panel_beats)) for s in starts]


def render_map(map_path: pathlib.Path, difficulty: str, out_png: pathlib.Path,
               n_panels: int = 4, panel_beats: float = 8.0,
               title: str | None = None, with_audio: bool = True) -> None:
    bm, bpm = ss._load_difficulty(map_path, difficulty)
    notes = sorted(bm.color_notes, key=lambda n: n.beat)
    card = ss.simulate(bm, bpm=bpm)
    max_beat = max((n.beat for n in notes), default=panel_beats)

    rms_beats, rms_vals = (None, None)
    if with_audio:
        rms_beats, rms_vals = _load_audio_rms(map_path, bpm, max_beat)

    windows = _pick_windows(notes, max_beat, panel_beats, n_panels)

    fig = plt.figure(figsize=(13, 2.0 + 2.6 * n_panels))
    gs = fig.add_gridspec(1 + 2 * n_panels, 1,
                          height_ratios=[1.6] + [2.0, 0.7] * n_panels, hspace=0.55)

    _draw_density(fig.add_subplot(gs[0]), notes, card, bpm, max_beat, rms_beats, rms_vals)
    for i, (b0, b1) in enumerate(windows):
        _draw_lattice(fig.add_subplot(gs[1 + 2 * i]), notes, card, b0, b1, bpm)
        _draw_swing_trace(fig.add_subplot(gs[2 + 2 * i]), card, b0, b1)

    fig.suptitle(title or f"{map_path.name} [{difficulty}]  bpm={bpm:.0f}",
                 fontsize=10, y=0.997)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mapper's-eye map renderer.")
    ap.add_argument("map", type=pathlib.Path)
    ap.add_argument("--difficulty", default="Expert")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--panels", type=int, default=4)
    ap.add_argument("--panel-beats", type=float, default=8.0)
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    render_map(args.map, args.difficulty, args.out, n_panels=args.panels,
               panel_beats=args.panel_beats, title=args.title,
               with_audio=not args.no_audio)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
