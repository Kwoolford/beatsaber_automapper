#!/usr/bin/env python
"""OUTLIER SWEEP — where does a map sit outside every human map of its density? (unnamed defects)

Not a defect list. `agent_mapper/fingerprint.py` computes ~100 plain quantities; this places each
against the human maps NEAREST IN DENSITY (nps) and prints the ones in the far tails, with how many
human maps are that extreme. It answers the question the named queries cannot: *what about this map
would a player notice that no check was written for?*

    python scripts/outlier_sweep.py build                 # corpus table (once, ~2 min, 16 procs)
    python scripts/outlier_sweep.py control               # how many flags does a HUMAN map draw?
    python scripts/outlier_sweep.py map a.zip [b.zip ...] # the report

★Calibration matters more than the list: with ~100 correlated features a human map draws some flags
by chance, so the report prints this map's count against the held-out human distribution of counts.
A flag is a place to LOOK (read the bars), never a verdict.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TABLE = ROOT / "outputs" / "fingerprint" / "human_table.json"
K = 400          # reference = the K human maps nearest in log-nps
TAIL = 0.005     # two-sided: at most this share of the reference as extreme


def _one(p):
    from agent_mapper.score import load_map
    from agent_mapper.fingerprint import fingerprint
    try:
        m = load_map(pathlib.Path(p))
    except Exception:
        return None
    ns = m.notes
    if len(ns) < 100 or any(not (0 <= n.x <= 3 and 0 <= n.y <= 2 and 0 <= n.direction <= 8) for n in ns):
        return None
    try:
        return dict(id=pathlib.Path(p).stem, diff=m.difficulty, f=fingerprint(m))
    except Exception as e:  # noqa: BLE001
        return dict(id=pathlib.Path(p).stem, err=repr(e))


def build():
    zips = sorted((ROOT / "data" / "raw").glob("*.zip"))
    with ProcessPoolExecutor(16) as ex:
        rows = [r for r in ex.map(_one, map(str, zips), chunksize=8) if r]
    errs = [r for r in rows if "err" in r]
    rows = [r for r in rows if "f" in r]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(rows))
    print(f"{len(rows)} human maps -> {TABLE}  ({len(errs)} errors)")
    for e in errs[:5]:
        print("  err", e)


def load_table():
    rows = json.loads(TABLE.read_text())
    names = sorted({k for r in rows for k in r["f"]})
    X = np.array([[r["f"].get(k, np.nan) for k in names] for r in rows], float)
    ids = [r["id"] for r in rows]
    return names, X, ids


def place(f: dict, names, X, ids, exclude: str | None = None, k: int = K):
    """-> list of (name, value, rarity, side, p5, p50, p95, n_more_extreme, n_ref)"""
    nps = X[:, names.index("nps")]
    ok = np.array([i != exclude for i in ids])
    d = np.abs(np.log(nps) - np.log(f["nps"]))
    d[~ok] = np.inf
    ref = np.argsort(d)[:k]
    out = []
    for j, nm in enumerate(names):
        v = f.get(nm, np.nan)
        col = X[ref, j]
        col = col[np.isfinite(col)]
        if not np.isfinite(v) or len(col) < 50:
            continue
        lo = np.mean(col <= v)
        hi = np.mean(col >= v)
        # ties at a hard floor/ceiling (e.g. 0 collisions) are not rare
        rar, side = (lo, "low") if lo < hi else (hi, "high")
        out.append((nm, v, rar, side, *np.percentile(col, [5, 50, 95]),
                    int(round(rar * len(col))), len(col)))
    return out


def flags(placed, tail=TAIL):
    return [r for r in placed if r[2] <= tail]


def control(n: int = 600, seed: int = 0):
    names, X, ids = load_table()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    counts, per_feat = [], {}
    for i in pick:
        f = {nm: X[i, j] for j, nm in enumerate(names)}
        fl = flags(place(f, names, X, ids, exclude=ids[i]))
        counts.append(len(fl))
        for r in fl:
            per_feat[r[0]] = per_feat.get(r[0], 0) + 1
    counts = np.array(counts)
    res = dict(n=len(counts), tail=TAIL, k=K,
               pct={str(q): float(np.percentile(counts, q)) for q in (50, 75, 90, 95, 99)},
               mean=float(counts.mean()), zero=float(np.mean(counts == 0)),
               per_feature_fire_rate={k: v / len(counts) for k, v in
                                      sorted(per_feat.items(), key=lambda kv: -kv[1])})
    (TABLE.parent / "control.json").write_text(json.dumps(res, indent=1))
    print(f"held-out human maps {res['n']}: flags per map p50 {res['pct']['50']:.0f} "
          f"p90 {res['pct']['90']:.0f} p95 {res['pct']['95']:.0f} p99 {res['pct']['99']:.0f} "
          f"(mean {res['mean']:.2f}, zero-flag {res['zero']:.0%})")
    print("features firing most on humans:",
          ", ".join(f"{k} {v:.1%}" for k, v in list(res["per_feature_fire_rate"].items())[:12]))
    return res


def report(paths, tail=TAIL, show_near=0):
    from agent_mapper.score import load_map
    from agent_mapper.fingerprint import fingerprint
    names, X, ids = load_table()
    ctl = json.loads((TABLE.parent / "control.json").read_text()) if (TABLE.parent / "control.json").exists() else None
    for p in paths:
        p = pathlib.Path(p)
        m = load_map(p)
        f = fingerprint(m)
        pl = place(f, names, X, ids, exclude=p.stem)
        fl = sorted(flags(pl, tail), key=lambda r: r[2])
        cal = ""
        if ctl:
            c95, c99 = ctl["pct"]["95"], ctl["pct"]["99"]
            mark = "🔴" if len(fl) > c99 else "🟡" if len(fl) > c95 else "⚪"
            cal = f"  {mark} human maps draw p95 {c95:.0f} / p99 {c99:.0f}"
        print(f"\n== {p.name}  ({m.difficulty}, {f['nps']:.2f} nps, ref = {K} human maps nearest in nps)")
        n0 = sum(r[7] == 0 for r in pl)
        print(f"   {len(fl)} features beyond the {tail:.1%} tails{cal}; {n0} beyond EVERY human"
              f" (humans: 82 % draw 0, p95 2)")
        for nm, v, rar, side, p5, p50, p95, nmore, nref in fl:
            fire = ctl["per_feature_fire_rate"].get(nm, 0) if ctl else 0
            print(f"   {nm:22} {v:9.3f}  {side:4}  human p5/p50/p95 {p5:7.3f} {p50:7.3f} {p95:7.3f}"
                  f"   {nmore}/{nref} humans this far{'  (fires on '+format(fire,'.0%')+' of humans)' if fire > 0.02 else ''}")
        if show_near:
            near = sorted([r for r in pl if tail < r[2] <= show_near], key=lambda r: r[2])
            for nm, v, rar, side, p5, p50, p95, nmore, nref in near:
                print(f"   ·{nm:21} {v:9.3f}  {side:4}  human p5/p50/p95 {p5:7.3f} {p50:7.3f} {p95:7.3f}"
                      f"   {nmore}/{nref}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "control", "map"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--tail", type=float, default=TAIL)
    ap.add_argument("--near", type=float, default=0, help="also list features rarer than this")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "control":
        control()
    else:
        report(a.paths, a.tail, a.near)


if __name__ == "__main__":
    main()
