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
# Unplayable whatever the corpus says (36 / 5 244 human maps carry one — the corpus has bugs too):
# any count above zero flags, with rarity 0. verdict.py's PLAYABILITY reds the same thing.
HARD = {"two_colour_cells"}


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
        if nm in HARD and v > 0:
            rar, side = 0.0, "HARD"
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




# ============================================================================ LOCAL (8-bar windows)
# The map-wide sweep says WHAT; Kyle's complaints arrive as WHERE ("bars 33-36"). Same features,
# per 8-bar window, against the human windows nearest in LOCAL density. Map-wide-only features
# (density shape, elements per minute) are dropped: a window cannot have them.
WIN = 32                 # beats
LTABLE = TABLE.parent / "human_windows.json"
LK, LTAIL = 1000, 0.001
LOCAL_SKIP = ("win_", "last_fifth", "first_fifth", "bar_rhythm", "longest_gap", "gap_long",
              "_pm", "wall_", "nps", "events_ps", "swing_ps", "_shapes", "gap_med_beats")


def windows(m, min_notes: int = 24):
    import copy
    from agent_mapper.fingerprint import fingerprint
    if not m.notes:
        return []
    last = max(n.beat for n in m.notes)
    out = []
    for k in range(int(last // WIN) + 1):
        a, b = k * WIN, (k + 1) * WIN
        ns = [n for n in m.notes if a <= n.beat < b]
        if len(ns) < min_notes:
            continue
        w = copy.copy(m)
        w.notes, w.bombs, w.walls, w.arcs, w.chains = ns, [], [], [], []
        f = fingerprint(w)
        f["lnps"] = len(ns) / (WIN * 60 / m.bpm)
        out.append((k * WIN // 4 + 1, f))       # first bar (4/4) of the window
    return out


def _lone(p):
    from agent_mapper.score import load_map
    try:
        m = load_map(pathlib.Path(p))
    except Exception:
        return None
    if len(m.notes) < 100 or any(not (0 <= n.x <= 3 and 0 <= n.y <= 2 and 0 <= n.direction <= 8)
                                 for n in m.notes):
        return None
    sid = pathlib.Path(p).stem
    return [dict(id=sid, bar=b, f=f) for b, f in windows(m)]


def lbuild(n: int = 1500, seed: int = 0):
    zips = sorted((ROOT / "data" / "raw").glob("*.zip"))
    rng = np.random.default_rng(seed)
    zips = [zips[i] for i in rng.choice(len(zips), size=min(n, len(zips)), replace=False)]
    with ProcessPoolExecutor(16) as ex:
        rows = [w for r in ex.map(_lone, map(str, zips), chunksize=8) if r for w in r]
    LTABLE.write_text(json.dumps(rows))
    print(f"{len(rows)} human windows from {len({r['id'] for r in rows})} maps -> {LTABLE}")


def lload():
    rows = json.loads(LTABLE.read_text())
    names = sorted({k for r in rows for k in r["f"] if not any(s in k for s in LOCAL_SKIP)})
    X = np.array([[r["f"].get(k, np.nan) for k in names] for r in rows], float)
    lnps = np.array([r["f"]["lnps"] for r in rows])
    ids = np.array([r["id"] for r in rows])
    return names, X, lnps, ids


def lplace(f, names, X, lnps, ids, exclude=None, tail=LTAIL):
    d = np.abs(np.log(lnps) - np.log(f["lnps"]))
    if exclude is not None:
        d[ids == exclude] = np.inf
    ref = np.argsort(d)[:LK]
    out = []
    for j, nm in enumerate(names):
        v = f.get(nm, np.nan)
        col = X[ref, j]
        col = col[np.isfinite(col)]
        if not np.isfinite(v) or len(col) < 200:
            continue
        lo, hi = np.mean(col <= v), np.mean(col >= v)
        rar, side = (lo, "low") if lo < hi else (hi, "high")
        if nm in HARD and v > 0:
            rar, side = 0.0, "HARD"
        if rar <= tail:
            out.append((nm, v, rar, side, *np.percentile(col, [5, 50, 95]), int(round(rar * len(col))), len(col)))
    return out


def lcontrol(n: int = 300, seed: int = 1):
    """Held-out: windows of human maps NOT in the table, so no window is read against itself."""
    names, X, lnps, ids = lload()
    used = set(ids.tolist())
    zips = [z for z in sorted((ROOT / "data" / "raw").glob("*.zip")) if z.stem not in used]
    rng = np.random.default_rng(seed)
    zips = [zips[i] for i in rng.choice(len(zips), size=min(n, len(zips)), replace=False)]
    with ProcessPoolExecutor(16) as ex:
        maps = [r for r in ex.map(_lone, map(str, zips), chunksize=8) if r]
    win_flag, map_wins, per_feat, nwin = [], [], {}, 0
    for r in maps:
        k = 0
        for w in r:
            fl = lplace(w["f"], names, X, lnps, ids)
            nwin += 1
            win_flag.append(len(fl))
            k += bool(fl)
            for x in fl:
                per_feat[x[0]] = per_feat.get(x[0], 0) + 1
        map_wins.append(k)
    win_flag, map_wins = np.array(win_flag), np.array(map_wins)
    res = dict(maps=len(maps), windows=nwin, window_flag_rate=float(np.mean(win_flag > 0)),
               flagged_windows_per_map={str(q): float(np.percentile(map_wins, q)) for q in (50, 90, 95, 99)},
               per_feature_window_rate={k: v / nwin for k, v in sorted(per_feat.items(), key=lambda kv: -kv[1])})
    (TABLE.parent / "lcontrol.json").write_text(json.dumps(res, indent=1))
    print(f"held-out human maps {res['maps']} ({nwin} windows): {res['window_flag_rate']:.1%} of windows flag;"
          f" flagged windows per map p50 {res['flagged_windows_per_map']['50']:.0f}"
          f" p90 {res['flagged_windows_per_map']['90']:.0f} p95 {res['flagged_windows_per_map']['95']:.0f}"
          f" p99 {res['flagged_windows_per_map']['99']:.0f}")
    print("most-firing:", ", ".join(f"{k} {v:.2%}" for k, v in list(res["per_feature_window_rate"].items())[:10]))


def lreport(paths):
    from agent_mapper.score import load_map
    names, X, lnps, ids = lload()
    cp = TABLE.parent / "lcontrol.json"
    ctl = json.loads(cp.read_text()) if cp.exists() else None
    for p in paths:
        p = pathlib.Path(p)
        ws = windows(load_map(p))
        hits = [(b, lplace(f, names, X, lnps, ids, exclude=p.stem), f["lnps"]) for b, f in ws]
        hits = [h for h in hits if h[1]]
        cal = ""
        if ctl:
            c = ctl["flagged_windows_per_map"]
            cal = (f"  {'🔴' if len(hits) > c['99'] else '🟡' if len(hits) > c['95'] else '⚪'}"
                   f" human maps: p95 {c['95']:.0f} / p99 {c['99']:.0f}")
        print(f"\n== {p.name}: {len(hits)} of {len(ws)} 8-bar windows outside every-but-0.1 % of human"
              f" windows at their density{cal}")
        for b, fl, ln in hits:
            desc = "; ".join(f"{nm} {v:.2f} ({side}, human p50 {p50:.2f})"
                             for nm, v, rar, side, p5, p50, p95, nm_, nr in sorted(fl, key=lambda x: x[2])[:4])
            print(f"   bars {b:3}-{b+7:<3} ({ln:.1f} nps)  {desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "control", "map", "lbuild", "lcontrol", "local"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--tail", type=float, default=TAIL)
    ap.add_argument("--near", type=float, default=0, help="also list features rarer than this")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "control":
        control()
    elif a.cmd == "lbuild":
        lbuild()
    elif a.cmd == "lcontrol":
        lcontrol()
    elif a.cmd == "local":
        lreport(a.paths)
    else:
        report(a.paths, a.tail, a.near)


if __name__ == "__main__":
    main()
