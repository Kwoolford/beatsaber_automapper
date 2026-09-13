#!/usr/bin/env python
"""ANSWER — land a note when the song's energy rises, because humans answer on the beat.

★**The evidence, measured before this existed** (`PROGRESS.md 2026-09-13ag/ah). Over **774 energy
jumps in 120 human maps**, the human's first note after the jump lands at a median of **0.00 beats**
past the bar line, p75 0.25, p90 0.75. Answering promptly is one of the few things human mappers
agree on — the same measurement found they agree on *nothing* about how much to step the density
(p10 0.80, median 1.19, p90 2.00), which is why that half of D3's absolute rule was dropped.

And `q_drops`' lag clause, controlled against 150 mappers' own two difficulties, fires on
**0.0-0.3 %** of their jumps. So when it fires on us it is real: `1f333` bar 170 answers an energy
jump **2.50 beats** late, and that is one of the three reds left on that map.

## What it does

For each bar where mean energy rises by `JUMP` over the previous bar, if the map's first note in
that bar lands later than `LATE_BEATS`, **move that note back** to the earliest onset at or after
the bar line (or to the bar line itself when no onset is cached).

⚠️**It MOVES a note, it does not add one.** The note count, colours and cut directions are
untouched, so density stays put and `EMPTY`/`D6` cannot move with it — the same invariant
`_taper_rows` keeps, and for the same reason: a mechanism that also changes the density is not
isolating anything.

⚠️**Guarded by the per-hand floor.** A moved note must still leave `MIN_GAP_SEC` between it and the
previous note of the same hand, or it is left alone. The 150 ms floor is the one thing in this
project that is non-negotiable.

🔴**MUST RUN AFTER `idiomize` and `repeat.py`** — those rewrite cells and figures at fixed times;
this is the only pass that moves a time, so it goes last, before walls.

Usage:
    python agent_mapper/answer.py in.zip --out out.zip --song 1f333
    python agent_mapper/answer.py in.zip --out out.zip --song 1f333 --report
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

JUMP = 0.25          # queries.q_drops' E-jump, so the pass answers what the gate reads
LATE_BEATS = 0.75    # the human p90; later than this is a late answer
MIN_GAP_SEC = 0.150  # the per-hand floor, non-negotiable
BEATS_PER_BAR = 4


def _energy_per_bar(sid: str, bpm: float, nb: int):
    """Mean RMS per bar from `score.py`'s cache, or None when the song has not been scored."""
    from agent_mapper import score as S
    t, rms = S._energy(sid, None)
    if t is None:
        return None
    spb = 60.0 / bpm
    edges = np.arange(nb + 1) * BEATS_PER_BAR * spb
    idx = np.searchsorted(t, edges)
    return np.array([rms[idx[i]:max(idx[i + 1], idx[i] + 1)].mean() if idx[i] < len(rms) else 0.0
                     for i in range(nb)])


def _onsets(sid: str) -> np.ndarray | None:
    f = (pathlib.Path(__file__).resolve().parent.parent / "outputs" / "onset_cache" / f"{sid}.npz")
    if not f.exists():
        return None
    try:
        return np.load(f)["onsets"]
    except Exception:  # noqa: BLE001
        return None


def answer(notes: list[dict], bpm: float, sid: str, report: bool = False) -> tuple[list[dict], int]:
    """Move the first note of each late-answered energy rise onto the rise. Returns (notes, moved)."""
    if not notes:
        return notes, 0
    spb = 60.0 / bpm
    beats = [float(n.get("b", 0.0)) for n in notes]
    nb = int(max(beats) // BEATS_PER_BAR) + 1
    em = _energy_per_bar(sid, bpm, nb)
    if em is None:
        if report:
            print(f"  no energy for {sid} — nothing to answer (run score.py on it first)")
        return notes, 0
    ons = _onsets(sid)
    out = [dict(n) for n in notes]
    moved = 0
    for b in range(1, nb):
        if em[b] - em[b - 1] < JUMP:
            continue
        lo, hi = b * BEATS_PER_BAR, (b + 1) * BEATS_PER_BAR
        inbar = [i for i, t in enumerate(beats) if lo <= t < hi]
        if not inbar:
            continue
        first = min(inbar, key=lambda i: float(out[i]["b"]))
        lag = float(out[first]["b"]) - lo
        if lag <= LATE_BEATS:
            continue
        # where the song actually offers an attack
        target = float(lo)
        if ons is not None:
            cand = [float(o) / spb for o in ons if lo * spb <= float(o) < (lo + LATE_BEATS) * spb]
            if cand:
                target = min(cand)
        # the per-hand floor decides whether the move is legal
        colour = int(out[first].get("c", 0))
        prev = [float(o["b"]) for o in out
                if int(o.get("c", 0)) == colour and float(o["b"]) < float(out[first]["b"])]
        if prev and (target - max(prev)) * spb < MIN_GAP_SEC:
            if report:
                print(f"  bar {b}: {lag:.2f} beats late, but moving it would break the "
                      f"{MIN_GAP_SEC * 1000:.0f} ms floor — left alone")
            continue
        if report:
            print(f"  bar {b}: energy {em[b-1]:.2f}→{em[b]:.2f}, first note {lag:.2f} beats late "
                  f"→ moved to {target - lo:+.2f}")
        out[first]["b"] = target
        moved += 1
    return out, moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip_in", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--song", required=True)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="answer_"))
    try:
        with zipfile.ZipFile(a.zip_in) as zf:
            zf.extractall(tmp)
            names = zf.namelist()
        info = next(n for n in names if n.split("/")[-1].lower() == "info.dat")
        dat = next(n for n in names if n.lower().split("/")[-1].endswith("standard.dat"))
        meta = json.loads((tmp / info).read_text(encoding="utf-8-sig"))
        bpm = 0.0
        for k in ("_beatsPerMinute", "beatsPerMinute"):
            if k in meta:
                bpm = float(meta[k])
                break
        else:
            bpm = float((meta.get("audio") or {}).get("bpm") or 0.0)
        d = json.loads((tmp / dat).read_text(encoding="utf-8-sig"))
        notes = d.get("colorNotes") or []
        new, moved = answer(notes, bpm, a.song, report=a.report)
        assert len(new) == len(notes)
        for o, q in zip(notes, new):
            assert o.get("c") == q.get("c") and o.get("d") == q.get("d")
        d["colorNotes"] = new
        (tmp / dat).write_text(json.dumps(d), encoding="utf-8")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as zo:
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    zo.write(p, p.relative_to(tmp).as_posix())
        print(f"{a.zip_in.name}: answered {moved} energy rise(s) → {a.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
