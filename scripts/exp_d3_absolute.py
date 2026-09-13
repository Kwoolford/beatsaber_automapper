#!/usr/bin/env python
"""Does D3's ABSOLUTE branch fire on human maps?

★**Why this and not the panel.** `scripts/exp_human_panel.py` calibrated the map-only codes
against two mappers of the same song, but D3 and D4 are **song-relative** and the panel cannot
carry them: measured 2026-09-13af, two mappers' uploads of one song share a recording only **10 %**
of the time within 0.1 s — the median pair differs by **1.5 s** and a tenth by **45 s or more**
(different cuts of the track). There is no shared time base to align their energy on.

So test the branch that needs no pairing at all. `q_drops` has two modes, and the one nobody has
ever checked is the **no-human** fallback used whenever a build has no reference map:

    bad = step < 1.2 or f is None or f > lag_beats

That is an **absolute** rule — a fixed density step and a fixed one-beat answer at every energy
jump the song makes. This project has retired **five** absolute norms already (a D3 step floor
fired on a human 1f913, "odd-16th = shifted" on 1f335, the wall-coverage line on a third of human
pairs…), so an untested sixth is worth an hour.

Method: energy exactly as `agent_mapper/score.py::_energy` computes it (RMS on a 20 ms hop, scaled
so the 98th percentile is 1.0), averaged per bar on the map's own beat grid; E-jumps and the step
and lag read exactly as `queries.q_drops` reads them.

Run:
    python scripts/exp_d3_absolute.py --n 120
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import tempfile
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
CACHE = REPO / "outputs" / "d3_abs_cache"

JUMP = 0.25        # queries.q_drops
N_BARS = 2
LAG_BEATS = 1.0
STEP_MIN = 1.2     # the absolute rule's density step
BEATS_PER_BAR = 4


def load_map(zp: pathlib.Path):
    """(bpm, note beats, audio bytes) for the Expert difficulty, or None."""
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        info = next((n for n in names if n.split("/")[-1].lower() == "info.dat"), None)
        std = [n for n in names if n.lower().split("/")[-1].endswith("standard.dat")]
        diff = next((n for n in std if n.lower().split("/")[-1].startswith("expert")
                     and "plus" not in n.lower().split("/")[-1]), None)
        aud = next((n for n in names if n.lower().endswith((".egg", ".ogg"))), None)
        if not (info and diff and aud):
            return None
        meta = json.loads(zf.read(info).decode("utf-8-sig"))
        d = json.loads(zf.read(diff).decode("utf-8-sig"))
        raw = zf.read(aud)
    bpm = 0.0
    for k in ("_beatsPerMinute", "beatsPerMinute"):
        if k in meta:
            bpm = float(meta[k])
            break
    else:
        bpm = float((meta.get("audio") or {}).get("bpm") or 0.0)
    notes = d.get("colorNotes") or d.get("_notes") or []
    if bpm <= 0 or len(notes) < 150:
        return None
    beats = sorted(float(n.get("b", n.get("_time", 0.0))) for n in notes)
    return bpm, beats, raw


def energy(stem: str, raw: bytes):
    """`score.py::_energy`, cached per map."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{stem}.npz"
    if f.exists():
        z = np.load(f)
        return z["t"], z["rms"]
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".ogg") as t:
        t.write(raw)
        t.flush()
        y, sr = librosa.load(t.name, sr=22050, mono=True)
    hop = 441
    r = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    ts = librosa.frames_to_time(np.arange(len(r)), sr=sr, hop_length=hop)
    r = np.clip(r / max(float(np.percentile(r, 98)), 1e-9), 0, 1.2)
    np.savez(f, t=ts, rms=r)
    return ts, r


def verdict(bpm, beats, t, rms) -> tuple[int, int, list]:
    """(E-jumps read, how many the map FAILS the absolute rule on, the failing steps)."""
    spb = 60.0 / bpm
    nb = int(max(beats) // BEATS_PER_BAR) + 1
    bar_t = np.arange(nb + 1) * BEATS_PER_BAR * spb
    idx = np.searchsorted(t, bar_t)
    em = np.array([rms[idx[i]:max(idx[i + 1], idx[i] + 1)].mean() if idx[i] < len(rms) else 0.0
                   for i in range(nb)])
    b = np.array(beats)
    per_bar = np.array([int(((b >= i * BEATS_PER_BAR) & (b < (i + 1) * BEATS_PER_BAR)).sum())
                        for i in range(nb)])
    seen = fail = 0
    bad = []
    for i in range(3, nb - 1):
        if em[i - 1] - em[i - 2] < JUMP:
            continue
        before = per_bar[i - N_BARS:i].sum() / N_BARS
        after = per_bar[i:i + N_BARS].sum() / N_BARS
        if after < 2:
            continue                      # EMPTY's job, as in q_drops
        seen += 1
        step = after / max(before, 0.5)
        inbar = b[(b >= i * BEATS_PER_BAR) & (b < (i + 1) * BEATS_PER_BAR)]
        f = (inbar[0] - i * BEATS_PER_BAR) if len(inbar) else None
        if step < STEP_MIN or f is None or f > LAG_BEATS:
            fail += 1
            bad.append((step, None if f is None else float(f)))
    return seen, fail, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths = sorted(RAW.glob("*.zip"))
    random.Random(a.seed).shuffle(paths)
    rows = []
    for zp in paths:
        if len(rows) >= a.n:
            break
        try:
            m = load_map(zp)
            if m is None:
                continue
            bpm, beats, raw = m
            t, rms = energy(zp.stem, raw)
            seen, fail, bad = verdict(bpm, beats, t, rms)
        except Exception:  # noqa: BLE001
            continue
        if seen >= 3:
            rows.append((zp.stem, seen, fail, bad))

    tot_seen = sum(r[1] for r in rows)
    tot_fail = sum(r[2] for r in rows)
    share = np.array([r[2] / r[1] for r in rows])
    print(f"{len(rows)} human maps with >= 3 readable E-jumps, {tot_seen} jumps in total\n")
    print(f"E-jumps where a HUMAN fails D3's absolute rule: {tot_fail}/{tot_seen} = "
          f"{tot_fail / max(tot_seen, 1):.1%}")
    print(f"per map, share of its own jumps that fail: median {np.median(share):.1%}  "
          f"p25 {np.percentile(share, 25):.1%}  p75 {np.percentile(share, 75):.1%}")
    print(f"maps failing NONE of their jumps: {(share == 0).mean():.1%}")
    print(f"maps the page would call RED (>= 10 % of jumps): {(share >= 0.10).mean():.1%}")
    steps = np.array([s for r in rows for s, _ in r[3]])
    if len(steps):
        print(f"\nthe density step humans actually play at a jump they 'fail' "
              f"(rule wants >= {STEP_MIN}):")
        for q in (10, 25, 50, 75, 90):
            print(f"  p{q:<3d} {np.percentile(steps, q):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
