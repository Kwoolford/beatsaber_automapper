#!/usr/bin/env python
"""THE SONG BRIEF — a whole song as a text score an agent can read end to end.

This is the perception half of `agent_mapper/`. The generator sees a 16-note window;
an agent can see the entire song at once, and this is the artefact that lets it:
structure, per-stem rhythm bar by bar, energy, the grid to place notes on, and
**timestamped lyrics**.

★**Read the section table first, then expand only the bars you are about to map.**
The overview is ~1 row per section; `--bars` is ~1 row per bar. A 4-minute song is
~2 000 sixteenths and will not fit usefully in context as raw events, which is the
whole reason this summarises.

★**THE LYRIC REPEAT MAP IS THE MOST VALUABLE THING HERE.** When the same line is sung
twice, a human mapper maps it the same way — that is `harm_rhythm`/`rhy_rhythm`, and
`BEAT_STRUCTURE_REUSE` exists to infer it from an audio self-similarity matrix. With
lyrics you do not infer it, you read it: Hunger's chorus at 145.1 s and 247.2 s is
*literally the same words*. Map it once, reuse it deliberately, vary it on purpose.

⚠️**Two different vocal signals, and they answer different questions.**
`V` in the stem grid is a detected vocal ONSET (accurate to ~10 ms — place notes on
these). The lyric text is Whisper's word alignment (accurate to ~a syllable — use it
to know *what* is happening and where a phrase starts and ends). Do not place notes
off the lyric timestamps when a vocal onset is available.

⚠️Times are given in **seconds and bars both**, always. 30 % of our maps are at the
wrong tempo, so a bar index is not comparable between two maps of the same song;
seconds are the only ground truth.

Usage:
    python agent_mapper/brief.py <audio.ogg>                  # overview
    python agent_mapper/brief.py <audio.ogg> --bars 17-32     # the detail
    python agent_mapper/brief.py <audio.ogg> --secs 60-75
    python agent_mapper/brief.py <audio.ogg> --json out.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

CACHE = REPO / "outputs" / "brief_cache"
STEMS = ("drums", "bass", "other", "vocals")
SUBDIV = 4          # sixteenths per beat, matching the generator's grid
BEATS_PER_BAR = 4


def _mmss(t: float) -> str:
    """m:ss.ss, correct for NEGATIVE times.

    The fitted downbeat can sit slightly before t=0 (Hunger's is -21 ms), and naive
    modulo formatting turned that into "-1:59.98" — a two-minute error on the very
    first row of the brief, from a sign nobody had thought to test.
    """
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60)}:{t % 60:05.2f}"


def analyse(audio: pathlib.Path, force: bool = False) -> dict:
    """Stems, onsets, tempo and energy for one song. Cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{audio.stem}.npz"
    if f.exists() and not force:
        d = np.load(f, allow_pickle=False)
        return {"onsets": {s: d[f"on_{s}"] for s in STEMS},
                "energy": d["energy"], "energy_hz": float(d["energy_hz"]),
                "bpm": float(d["bpm"]), "phase": float(d["phase"]),
                "r": float(d["r"]), "dur": float(d["dur"])}

    import librosa
    import torch
    from beatsaber_automapper.data.audio import load_audio
    from beatsaber_automapper.data.stem_separator import separate, DEMUCS_SR
    from beatsaber_automapper.data.tempo import estimate_tempo

    y, sr = load_audio(str(audio))
    t = torch.as_tensor(y, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    stems = separate(t, sr)

    onsets = {}
    for s in STEMS:
        arr = stems[s].detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
        onsets[s] = librosa.onset.onset_detect(y=arr.astype("float32"), sr=DEMUCS_SR,
                                               units="time", backtrack=True)
    mono = np.asarray(y, dtype="float32")
    if mono.ndim > 1:
        mono = mono.mean(axis=0)
    dur = len(mono) / sr
    hop = 2048
    energy = librosa.feature.rms(y=mono, hop_length=hop)[0]
    energy_hz = sr / hop
    union = np.sort(np.unique(np.concatenate([onsets[s] for s in STEMS])))
    fit = estimate_tempo(mono, sr, onsets=union)

    np.savez(f, energy=energy, energy_hz=energy_hz, bpm=fit.bpm,
             phase=fit.phase_s, r=fit.r, dur=dur,
             **{f"on_{s}": onsets[s] for s in STEMS})
    return {"onsets": onsets, "energy": energy, "energy_hz": energy_hz,
            "bpm": fit.bpm, "phase": fit.phase_s, "r": fit.r, "dur": dur}


def grid(a: dict) -> dict:
    """The beat grid the map will be written on."""
    bpm = a["bpm"]
    spb = 60.0 / bpm
    return {"bpm": bpm, "spb": spb, "slot": spb / SUBDIV,
            "bar_s": spb * BEATS_PER_BAR, "phase": a["phase"],
            "n_bars": int(a["dur"] / (spb * BEATS_PER_BAR)) + 1}


def bar_time(g: dict, bar: int) -> float:
    """Start time of a 1-indexed bar. Anchored on the FITTED phase, not t=0."""
    return g["phase"] + (bar - 1) * g["bar_s"]


def stem_row(times: np.ndarray, t0: float, t1: float, slot: float) -> str:
    """A bar of one stem as 16 sixteenth-note cells: 'x' = onset, '.' = nothing."""
    n = BEATS_PER_BAR * SUBDIV
    cells = ["."] * n
    for t in times[(times >= t0) & (times < t1)]:
        i = int(round((t - t0) / slot))
        if 0 <= i < n:
            cells[i] = "x"
    return "".join(cells)


def energy_at(a: dict, t0: float, t1: float) -> float:
    e, hz = a["energy"], a["energy_hz"]
    i0, i1 = int(t0 * hz), max(int(t1 * hz), int(t0 * hz) + 1)
    seg = e[i0:i1]
    return float(seg.mean()) if len(seg) else 0.0


def lyric_lines(song: str) -> list[dict]:
    import lyrics as _ly
    d = _ly.load(song)
    return d["lines"] if d else []


def lyric_words(song: str) -> list[dict]:
    import lyrics as _ly
    d = _ly.load(song)
    return d["words"] if d else []


def repeat_map(lines: list[dict]) -> dict[str, list[float]]:
    """Which lyric lines recur, and when. The structure, read rather than inferred."""
    seen: dict[str, list[float]] = {}
    for ln in lines:
        key = "".join(ch for ch in ln["text"].lower() if ch.isalnum() or ch == " ").strip()
        if len(key) < 6:
            continue
        seen.setdefault(key, []).append(ln["t"])
    return {k: v for k, v in seen.items() if len(v) > 1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--bars", default=None, help="detail for bars N-M")
    ap.add_argument("--secs", default=None, help="detail for seconds A-B")
    ap.add_argument("--force", action="store_true", help="recompute the analysis cache")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a_ = ap.parse_args()
    if not a_.audio.exists():
        print(f"no such audio: {a_.audio}", file=sys.stderr)
        return 2

    a = analyse(a_.audio, a_.force)
    g = grid(a)
    song = a_.audio.stem
    lines, words = lyric_lines(song), lyric_words(song)

    print(f"SONG {song}   {_mmss(a['dur'])}   {g['bpm']:.2f} bpm "
          f"(fit r={a['r']:.3f}{'' if a['r'] >= 0.35 else ' ⚠️weak'})")
    print(f"GRID bar = {g['bar_s']:.3f}s, beat = {g['spb']:.3f}s, "
          f"1/16 = {g['slot']*1000:.1f}ms, downbeat at {g['phase']*1000:+.0f}ms, "
          f"{g['n_bars']} bars")
    tot = {s: len(a["onsets"][s]) for s in STEMS}
    print("ONSETS " + "  ".join(f"{s} {tot[s]}" for s in STEMS))
    if not lines:
        print("LYRICS ⚠️none cached — run: "
              f"python agent_mapper/lyrics.py {a_.audio}  (instrumental songs "
              "legitimately have none)")
    else:
        print(f"LYRICS {len(words)} words, {len(lines)} lines")
        # 🔴The cascade this closes: Whisper invents text on non-speech audio, and the
        # LYRIC REPEATS block below then presents that invention as the song's structural
        # backbone -- which WORKFLOW tells the agent to map against. Surface the doubt at
        # the point of use, not only in `lyrics.py`, because this is where it is believed.
        # ⚠️`a` is the ANALYSIS dict here; the audio path is `a_.audio`. Using `a.audio`
        # raised AttributeError, and a bare `except` swallowed it so the guard silently
        # never fired -- the project's own "never wrap in a bare except" landmine.
        import lyrics as _L
        _d = _L.transcribe(a_.audio, "large-v3", False, None)   # cached, so cheap
        _suspect, _why = _L.hallucination_risk(_d)
        if _suspect:
            print(f"  🔴 DO NOT TRUST THE LYRICS — {_why}. Whisper invents text on "
                  f"non-speech audio.\n     Treat as INSTRUMENTAL: ignore the lyric "
                  f"column and the LYRIC REPEATS block below.")

    if a_.bars or a_.secs:
        if a_.secs:
            s0, s1 = (float(x) for x in a_.secs.split("-"))
            b0 = max(1, int((s0 - g["phase"]) / g["bar_s"]) + 1)
            b1 = int((s1 - g["phase"]) / g["bar_s"]) + 1
        else:
            b0, b1 = (int(x) for x in a_.bars.split("-"))
        print(f"\nBARS {b0}-{b1}   (D=drums B=bass O=other V=vocal onsets; "
              f"one cell = 1/16 = {g['slot']*1000:.0f}ms)")
        print(f"{'bar':>4} {'time':>8}  {'|1e+a2e+a3e+a4e+a':<17} stem   lyric")
        for bar in range(b0, min(b1, g["n_bars"]) + 1):
            t0 = bar_time(g, bar)
            t1 = t0 + g["bar_s"]
            said = " ".join(w["word"] for w in words if t0 <= w["t"] < t1)
            for k, s in zip("DBOV", STEMS):
                row = stem_row(np.asarray(a["onsets"][s]), t0, t1, g["slot"])
                if row.strip(".") == "" and k != "D":
                    continue
                head = f"{bar:>4} {_mmss(t0):>8}" if k == "D" else " " * 13
                tail = f"   {said}" if (k == "D" and said) else ""
                print(f"{head}  |{row}| {k}{tail}")
    else:
        # Overview: one row per 8 bars, which is the phrase length human mappers use.
        print(f"\nTIMELINE (8-bar phrases; density = onsets/s per stem)")
        print(f"{'bars':>9} {'time':>8} {'energy':>7}  {'D':>5}{'B':>5}{'O':>5}{'V':>5}"
              "   lyric")
        step = 8
        emax = max(1e-9, float(np.max(a["energy"])))
        for b0 in range(1, g["n_bars"] + 1, step):
            t0 = bar_time(g, b0)
            t1 = min(t0 + g["bar_s"] * step, a["dur"])
            if t0 >= a["dur"]:
                break
            dens = {s: sum(1 for t in a["onsets"][s] if t0 <= t < t1) / max(t1 - t0, 1e-9)
                    for s in STEMS}
            e = energy_at(a, t0, t1) / emax
            said = " ".join(ln["text"] for ln in lines if t0 <= ln["t"] < t1)[:46]
            print(f"{b0:>4}-{b0+step-1:<4} {_mmss(t0):>8} {'#' * int(e * 7):<7}"
                  f"{dens['drums']:>5.1f}{dens['bass']:>5.1f}{dens['other']:>5.1f}"
                  f"{dens['vocals']:>5.1f}   {said}")
        rep = repeat_map(lines)
        if rep:
            print(f"\n★ LYRIC REPEATS — the same words sung more than once. Map a repeat")
            print("  the way you mapped its first occurrence, then vary it deliberately.")
            for k, ts in sorted(rep.items(), key=lambda kv: -len(kv[1]))[:10]:
                bars = [int((t - g["phase"]) / g["bar_s"]) + 1 for t in ts]
                print(f"    {len(ts)}× bars {bars}  "
                      f"({', '.join(_mmss(t) for t in ts)})  \"{k[:44]}\"")

    if a_.json:
        out = {"song": song, "grid": g, "duration": a["dur"],
               "onsets": {s: [round(float(t), 4) for t in a["onsets"][s]] for s in STEMS},
               "lyrics": {"lines": lines, "words": words}}
        a_.json.resolve().write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"\nwrote {a_.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
