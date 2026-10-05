#!/usr/bin/env python
"""SONGPRINT SWEEP — does a map answer the song the way a human map of the SAME song does?

    python scripts/songprint_sweep.py control              # two humans, one song: how far apart?
    python scripts/songprint_sweep.py map ours.zip --song 1f333 [--vs data/raw/1f333.zip]

Each feature is `ours − human` on one song, placed against the distribution of `human_b − human_a`
over the panel (two mappers, one song, each read against its own zip's audio). A flag says *two
humans almost never differ this much on a song* — e.g. the bass that humans answer and we ignore.
Plus the lane-family FOLLOW mix (which instrument each 8-bar section follows) as shares.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_mapper.score import load_map  # noqa: E402
from agent_mapper.songprint import songprint  # noqa: E402

EVC = ROOT / "outputs" / "event_cache"
CTL = ROOT / "outputs" / "fingerprint" / "songprint_control.json"
READABLE = 0.8   # a human map with fewer of its notes on ANY detected event is not read in time
FAM = {"drums": "drums", "bass": "bass", "vocals": "vocals", "guitar": "melodic", "piano": "melodic",
       "other": "melodic"}


def family_mix(follow: list) -> dict[str, float]:
    got = [x for x in follow if x]
    out = {f"follow:{k}": 0.0 for k in ("drums", "bass", "melodic", "vocals")}
    for x in got:
        out["follow:" + FAM[x.split(":")[0]]] += 1 / len(got)
    out["follow:silent"] = 1 - len(got) / max(len(follow), 1)
    return out


def sp(zp: pathlib.Path, sid: str) -> dict | None:
    ev = EVC / f"{sid}.6s.json"
    if not ev.exists():
        return None
    f = songprint(load_map(zp), json.loads(ev.read_text()))
    if not f:
        return None
    f.update(family_mix(f.pop("_follow")))
    return f


def control():
    P = json.loads((ROOT / "outputs/human_panel_2026-09-13.json").read_text())
    seen, deltas = set(), {}
    n = dropped = 0
    for r in P:
        k = tuple(sorted((r["a"], r["b"])))
        if k in seen:
            continue
        seen.add(k)
        fa = sp(ROOT / "data/raw" / f"{k[0]}.zip", k[0])
        fb = sp(ROOT / "data/raw" / f"{k[1]}.zip", k[1])
        if not fa or not fb:
            continue
        if min(fa["on_any"], fb["on_any"]) < READABLE:
            dropped += 1   # tempo changes (ignored by the beat->s map) or unclear timing
            continue
        n += 1
        for nm in set(fa) & set(fb):
            d = fb[nm] - fa[nm]
            deltas.setdefault(nm, []).extend([d, -d])   # both directions
    res = {nm: sorted(v) for nm, v in deltas.items()}
    CTL.write_text(json.dumps(dict(pairs=n, deltas=res)))
    print(f"{n} panel pairs (both maps cached); {dropped} dropped, a map below on_any {READABLE}")
    for nm in sorted(res, key=lambda s: (s.split(":")[0], s)):
        v = np.abs(res[nm])
        print(f"  {nm:22} n={len(v)//2:3}  |human_b - human_a|  p50 {np.median(v):.3f}  p95 {np.percentile(v,95):.3f}"
              f"  max {v.max():.3f}")


def report(ours: pathlib.Path, sid: str, vs: pathlib.Path, tail: float = 0.025):
    ctl = json.loads(CTL.read_text())
    fo, fh = sp(ours, sid), sp(vs, sid)
    if fo is None or fh is None:
        print(f"{ours.name}: no event cache for {sid} or empty map")
        return
    if fh["on_any"] < READABLE:
        print(f"\n== {ours.name}: ⚪ the human map is not readable in time (on_any {fh['on_any']:.2f};"
              " tempo changes?) — no same-song comparison")
        return
    print(f"\n== {ours.name} vs {vs.name}  (song {sid}; control = {ctl['pairs']} two-mapper songs)")
    rows = []
    for nm in sorted(set(fo) & set(fh)):
        ref = np.array(ctl["deltas"].get(nm, []))
        if len(ref) < 20:
            continue
        d = fo[nm] - fh[nm]
        rar = min(np.mean(ref <= d), np.mean(ref >= d))
        rows.append((rar, nm, fo[nm], fh[nm], d, np.percentile(ref, 2.5), np.percentile(ref, 97.5)))
    rows.sort()
    for rar, nm, a, b, d, lo, hi in rows:
        mark = "🔴" if rar <= 0.005 else "🟡" if rar <= tail else "  "
        if rar <= tail or nm.startswith("follow:"):
            print(f"  {mark} {nm:22} ours {a:6.3f}  human {b:6.3f}  diff {d:+.3f}"
                  f"   two humans differ {lo:+.3f}..{hi:+.3f} (95 %)  [{rar:.1%} of human pairs this far]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["control", "map"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--song")
    ap.add_argument("--vs")
    a = ap.parse_args()
    if a.cmd == "control":
        control()
    else:
        for p in a.paths:
            p = pathlib.Path(p)
            sid = a.song or p.stem.split("__")[-1].split("_")[0]
            vs = pathlib.Path(a.vs) if a.vs else ROOT / "data/raw" / f"{sid}.zip"
            report(p, sid, vs)


if __name__ == "__main__":
    main()
