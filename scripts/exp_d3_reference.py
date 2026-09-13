#!/usr/bin/env python
"""Does D3's HUMAN-REFERENCE branch fire when a mapper is read against his own other difficulty?

★**Why this control works where the panel does not.** `q_drops` compares us to a reference human
at the song's energy boundaries, so it needs both maps on one time base. Two mappers' uploads of
the same song are different cuts (2026-09-13af: they match within 0.1 s only 10 % of the time), but
**one zip's Expert and ExpertPlus share the audio file exactly** — 404 of the first 800 corpus zips
carry both. That is a perfect time base and a real second opinion about the same music.

It is the branch the songset's remaining D3 reds come from, and it has never been controlled:
`bench.py` has four `humanplus-*`/`humanexp-*` rows, and 404 zips were sitting unused.

What is tested, matching `queries.q_drops` under `cross_difficulty` (which is what this is):
- **the LAG clause** — the map's first note after an E-jump, against the reference's
- **the E-drop branch** — the reference comes down and the map does not
(the level clause is skipped across difficulties, by the query's own rule, so it is not tested here
and still has no control.)

Run:
    python scripts/exp_d3_reference.py --n 150
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

JUMP = 0.25
N_BARS = 2
LAG_BEATS = 1.0
BEATS_PER_BAR = 4


def load_both(zp: pathlib.Path):
    """(bpm, expert beats, expertplus beats, audio bytes) when the zip has both."""
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        info = next((n for n in names if n.split("/")[-1].lower() == "info.dat"), None)
        exp = plus = None
        for n in names:
            b = n.lower().split("/")[-1]
            if not b.endswith("standard.dat"):
                continue
            if b.startswith("expertplus"):
                plus = n
            elif b.startswith("expert"):
                exp = n
        aud = next((n for n in names if n.lower().endswith((".egg", ".ogg"))), None)
        if not (info and exp and plus and aud):
            return None
        meta = json.loads(zf.read(info).decode("utf-8-sig"))
        de = json.loads(zf.read(exp).decode("utf-8-sig"))
        dp = json.loads(zf.read(plus).decode("utf-8-sig"))
        raw = zf.read(aud)
    bpm = 0.0
    for k in ("_beatsPerMinute", "beatsPerMinute"):
        if k in meta:
            bpm = float(meta[k])
            break
    else:
        bpm = float((meta.get("audio") or {}).get("bpm") or 0.0)

    def beats(d):
        ns = d.get("colorNotes") or d.get("_notes") or []
        return sorted(float(n.get("b", n.get("_time", 0.0))) for n in ns)
    be, bp = beats(de), beats(dp)
    if bpm <= 0 or len(be) < 150 or len(bp) < 150:
        return None
    return bpm, be, bp, raw


def energy(stem: str, raw: bytes):
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


def per_bar(beats, nb):
    b = np.array(beats)
    return np.array([int(((b >= i * BEATS_PER_BAR) & (b < (i + 1) * BEATS_PER_BAR)).sum())
                     for i in range(nb)]), b


def run(bpm, mine, his, t, rms) -> tuple[int, int, int, int]:
    """(jumps read, lag fires, drops read, E-drop fires) for `mine` against `his`."""
    spb = 60.0 / bpm
    nb = int(max(max(mine), max(his)) // BEATS_PER_BAR) + 1
    bar_t = np.arange(nb + 1) * BEATS_PER_BAR * spb
    idx = np.searchsorted(t, bar_t)
    em = np.array([rms[idx[i]:max(idx[i + 1], idx[i] + 1)].mean() if idx[i] < len(rms) else 0.0
                   for i in range(nb)])
    pm, bm = per_bar(mine, nb)
    ph, bh = per_bar(his, nb)
    jumps = lag_fire = drops = drop_fire = 0
    for i in range(3, nb - 1):
        d = em[i - 1] - em[i - 2]
        before, after = pm[i - N_BARS:i].sum() / N_BARS, pm[i:i + N_BARS].sum() / N_BARS
        if d >= JUMP:
            if after < 2:
                continue
            inm = bm[(bm >= i * BEATS_PER_BAR) & (bm < (i + 1) * BEATS_PER_BAR)]
            inh = bh[(bh >= i * BEATS_PER_BAR) & (bh < (i + 1) * BEATS_PER_BAR)]
            if not len(inh):
                continue
            hf = float(inh[0] - i * BEATS_PER_BAR)
            f = float(inm[0] - i * BEATS_PER_BAR) if len(inm) else None
            jumps += 1
            if f is None or f > hf + LAG_BEATS:
                lag_fire += 1
        elif d <= -JUMP:
            hb = ph[i - N_BARS:i].sum() / N_BARS
            ha = ph[i:i + N_BARS].sum() / N_BARS
            if hb >= 2 and ha <= 0.6 * hb and before >= 2:
                drops += 1
                if after >= 0.9 * before:
                    drop_fire += 1
    return jumps, lag_fire, drops, drop_fire


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths = sorted(RAW.glob("*.zip"))
    random.Random(a.seed).shuffle(paths)
    rows = []
    for zp in paths:
        if len(rows) >= a.n:
            break
        try:
            m = load_both(zp)
            if m is None:
                continue
            bpm, be, bp, raw = m
            t, rms = energy(zp.stem, raw)
            fwd = run(bpm, be, bp, t, rms)      # Expert read against his ExpertPlus
            rev = run(bpm, bp, be, t, rms)      # and the other direction
        except Exception:  # noqa: BLE001
            continue
        if fwd[0] + fwd[2] >= 3:
            rows.append((zp.stem, fwd, rev))

    def tally(i):
        j = sum(r[i][0] for r in rows); lf = sum(r[i][1] for r in rows)
        d = sum(r[i][2] for r in rows); df = sum(r[i][3] for r in rows)
        red = np.mean([(r[i][1] + r[i][3]) > 0 for r in rows])
        return j, lf, d, df, red

    print(f"{len(rows)} zips with both difficulties and a readable energy track\n")
    print(f"{'direction':<28s} {'lag fires':>18s} {'E-drop fires':>18s} {'MAPS RED':>9s}")
    for i, lbl in ((1, "Expert vs his ExpertPlus"), (2, "ExpertPlus vs his Expert")):
        j, lf, d, df, red = tally(i)
        print(f"{lbl:<28s} {f'{lf}/{j} = {lf/max(j,1):.1%}':>18s} "
              f"{f'{df}/{d} = {df/max(d,1):.1%}':>18s} {red:9.1%}")
    print("\n(D3 is ALWAYS_RED, so one fire in either clause reds the map.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
