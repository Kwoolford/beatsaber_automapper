"""FINGERPRINT — a broad, UNNAMED battery of map features, each placed against human maps.

★Why (Kyle, 2026-10-04): *"In the past they have all had flaws or obvious blindspots when I loaded
them up."* Every query in `scripts/queries.py` answers a defect someone already NAMED, so the flaw
that ships is always the one nobody wrote a query for (09-17 colour stacks, 10-04 88 % both-down
doubles). This module does not name defects. It computes ~100 plain, readable quantities a player
would feel (what the hands do, where they are, how the time is spaced, what the elements do) so
`scripts/outlier_sweep.py` can say *where a map sits outside every human map of its density*.

Map-only on purpose: it runs on the whole corpus with no perception cache, so the reference is
thousands of maps, not the four with stems.

    from agent_mapper.fingerprint import fingerprint
    f = fingerprint(load_map(path))   # dict name -> float (nan when undefined)
"""
from __future__ import annotations

import collections
import math

import numpy as np

UP, DOWN = {0, 4, 5}, {1, 6, 7}
VEC = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1)}
DIRS = ["U", "D", "L", "R", "UL", "UR", "DL", "DR", "dot"]
Q = 48  # ticks per beat for exact-time grouping


def _share(n, d):
    return n / d if d else float("nan")


def _angle(a, b):
    va, vb = VEC[a], VEC[b]
    c = (va[0] * vb[0] + va[1] * vb[1]) / (math.hypot(*va) * math.hypot(*vb))
    return round(math.degrees(math.acos(max(-1.0, min(1.0, c)))) / 45) * 45


def fingerprint(m) -> dict[str, float]:
    f: dict[str, float] = {}
    notes = m.notes
    spb = 60.0 / m.bpm
    if not notes:
        return f
    # ---- events: exact instants, with what each hand does there
    at = collections.defaultdict(lambda: {0: [], 1: []})
    for n in notes:
        at[round(n.beat * Q)][n.color].append(n)
    ticks = sorted(at)
    t0, t1 = ticks[0] / Q, ticks[-1] / Q
    dur_s = max((t1 - t0) * spb, 1.0)
    f["nps"] = len(notes) / dur_s
    f["events_ps"] = len(ticks) / dur_s
    f["notes_per_event"] = len(notes) / len(ticks)

    # ---- doubles (one note each hand at one instant)
    dbl = [(at[t][0][0], at[t][1][0]) for t in ticks if len(at[t][0]) == 1 and len(at[t][1]) == 1]
    f["double_share"] = _share(len(dbl), len(ticks))
    ud = [(r, b) for r, b in dbl if r.direction != 8 and b.direction != 8
          and (r.direction in UP | DOWN) and (b.direction in UP | DOWN)]
    f["dbl_mixed"] = _share(sum((r.direction in UP) != (b.direction in UP) for r, b in ud), len(ud))
    f["dbl_both_up"] = _share(sum(r.direction in UP and b.direction in UP for r, b in ud), len(ud))
    f["dbl_both_down"] = _share(sum(r.direction in DOWN and b.direction in DOWN for r, b in ud), len(ud))
    f["dbl_same_dir"] = _share(sum(r.direction == b.direction for r, b in dbl), len(dbl))
    f["dbl_has_dot"] = _share(sum(r.direction == 8 or b.direction == 8 for r, b in dbl), len(dbl))
    f["dbl_crossed"] = _share(sum(r.x > b.x for r, b in dbl), len(dbl))
    f["dbl_dx_mean"] = float(np.mean([b.x - r.x for r, b in dbl])) if dbl else float("nan")
    f["dbl_same_row"] = _share(sum(r.y == b.y for r, b in dbl), len(dbl))
    f["dbl_on_beat"] = _share(sum(round(r.beat * Q) % Q == 0 for r, b in dbl), len(dbl))
    f["stack_share"] = _share(sum(len(at[t][c]) > 1 for t in ticks for c in (0, 1)), len(ticks))
    cells = collections.Counter((round(n.beat * Q), n.x, n.y) for n in notes)
    f["cell_collisions"] = float(sum(v > 1 for v in cells.values()))

    # ---- per hand: directions, positions, transitions
    for c, hn in ((0, "L"), (1, "R")):
        hs = [n for n in notes if n.color == c]
        if not hs:
            continue
        f[f"{hn}_share"] = len(hs) / len(notes)
        dc = collections.Counter(n.direction for n in hs)
        for i, d in enumerate(DIRS):
            f[f"{hn}_dir_{d}"] = dc.get(i, 0) / len(hs)
        f[f"{hn}_diag"] = sum(dc.get(i, 0) for i in (4, 5, 6, 7)) / len(hs)
        f[f"{hn}_horiz"] = sum(dc.get(i, 0) for i in (2, 3)) / len(hs)
        xs = collections.Counter(n.x for n in hs)
        ys = collections.Counter(n.y for n in hs)
        for x in range(4):
            f[f"{hn}_col{x}"] = xs.get(x, 0) / len(hs)
        for y in range(3):
            f[f"{hn}_row{y}"] = ys.get(y, 0) / len(hs)
        cellc = collections.Counter((n.x, n.y) for n in hs)
        p = np.array(list(cellc.values()), float) / len(hs)
        f[f"{hn}_cell_entropy"] = float(-(p * np.log2(p)).sum())
        shapes = collections.Counter((n.x, n.y, n.direction) for n in hs)
        f[f"{hn}_shapes"] = float(len(shapes))
        f[f"{hn}_top5_shape_share"] = sum(v for _, v in shapes.most_common(5)) / len(hs)
        # one swing per instant (stacks collapse to their first note)
        sw, last_t = [], None
        for n in hs:
            t = round(n.beat * Q)
            if t != last_t:
                sw.append(n)
                last_t = t
        gaps = np.diff([n.beat for n in sw])
        ang, mv, reset, fast_reset, same_cell = [], [], 0, 0, 0
        for a, b, g in zip(sw, sw[1:], gaps):
            mv.append(math.hypot(b.x - a.x, b.y - a.y))
            same_cell += (a.x, a.y) == (b.x, b.y)
            if a.direction in VEC and b.direction in VEC:
                ang.append(_angle(a.direction, b.direction))
                if (a.direction in UP and b.direction in UP) or (a.direction in DOWN and b.direction in DOWN):
                    reset += 1
                    fast_reset += g * spb < 0.5
        k = max(len(sw) - 1, 1)
        if ang:
            ac = collections.Counter(ang)
            for d in (0, 45, 90, 135, 180):
                f[f"{hn}_rot{d}"] = ac.get(d, 0) / len(ang)
        f[f"{hn}_reset"] = reset / k
        f[f"{hn}_fast_reset"] = fast_reset / k
        f[f"{hn}_move_mean"] = float(np.mean(mv)) if mv else float("nan")
        f[f"{hn}_move0"] = same_cell / k
        f[f"{hn}_move_big"] = _share(sum(v >= 2.5 for v in mv), len(mv))
        if len(gaps):
            gs = gaps * spb
            f[f"{hn}_gap_p10_s"] = float(np.percentile(gs, 10))
            f[f"{hn}_gap_med_s"] = float(np.median(gs))
            f[f"{hn}_swing_ps_max4"] = float(1 / max(np.percentile(gs, 2), 1e-3))

    # ---- hands together
    f["hand_balance"] = abs(f.get("L_share", 0) - f.get("R_share", 0))
    seq = []
    for t in ticks:
        h = at[t]
        seq.append("B" if h[0] and h[1] else "L" if h[0] else "R")
    singles = [s for s in seq if s != "B"]
    f["alternation"] = _share(sum(a != b for a, b in zip(singles, singles[1:])), len(singles) - 1)
    runs, cur = [], 1
    for a, b in zip(singles, singles[1:]):
        if a == b:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    f["same_hand_run_p95"] = float(np.percentile(runs, 95)) if runs else float("nan")
    f["same_hand_run_ge4"] = _share(sum(r for r in runs if r >= 4), len(singles))
    # crossovers: red right of blue at nearby times
    lx = [(n.beat, n.x) for n in notes if n.color == 0]
    rx = [(n.beat, n.x) for n in notes if n.color == 1]
    if lx and rx:
        rb = np.array([b for b, _ in rx]); rxx = np.array([x for _, x in rx])
        cross = 0
        for b, x in lx:
            i = np.searchsorted(rb, b)
            j = [k for k in (i - 1, i) if 0 <= k < len(rb)]
            k = min(j, key=lambda k: abs(rb[k] - b))
            if abs(rb[k] - b) <= 1.0 and x > rxx[k]:
                cross += 1
        f["crossover"] = cross / len(lx)

    # ---- time: spacing of instants
    eg = np.diff([t / Q for t in ticks])
    if len(eg):
        f["gap_med_beats"] = float(np.median(eg))
        for lbl, v in (("1_16", 0.25), ("1_8", 0.5), ("1_4", 1.0), ("1_2", 2.0)):
            f[f"gap_{lbl}"] = float(np.mean(np.isclose(eg, v, atol=1e-3)))
        f["gap_triplet"] = float(np.mean(np.isclose(eg % (1 / 3), 0, atol=1e-3)
                                         & ~np.isclose(eg % 0.25, 0, atol=1e-3)))
        f["gap_long_2bar"] = float(np.sum(eg[eg >= 8]) / max(t1 - t0, 1))
        f["longest_gap_s"] = float(eg.max() * spb)
        ph = np.array([t % Q for t in ticks])
        f["on_beat"] = float(np.mean(ph == 0))
        f["on_8th_off"] = float(np.mean(ph == Q // 2))
        f["on_16th_odd"] = float(np.mean((ph == Q // 4) | (ph == 3 * Q // 4)))
        f["off_grid"] = float(np.mean(ph % (Q // 4) != 0))
        f["gap_entropy"] = float(_entropy(np.round(eg, 3)))
        # density shape over 4-bar windows
        w = np.floor((np.array([t / Q for t in ticks]) - t0) / 16).astype(int)
        cnt = np.bincount(w).astype(float)
        if len(cnt) >= 4:
            f["win_cv"] = float(cnt.std() / max(cnt.mean(), 1e-9))
            f["win_max_over_med"] = float(cnt.max() / max(np.median(cnt), 1))
            f["win_zero"] = float(np.mean(cnt == 0))
            f["win_lag1_r"] = float(np.corrcoef(cnt[:-1], cnt[1:])[0, 1]) if cnt.std() > 0 else float("nan")
            q = np.array_split(cnt, 5)
            f["last_fifth_over_mean"] = float(q[-1].mean() / max(cnt.mean(), 1e-9))
            f["first_fifth_over_mean"] = float(q[0].mean() / max(cnt.mean(), 1e-9))
        # bar-level rhythm repetition: share of bars whose onset pattern equals the previous bar's
        bars = collections.defaultdict(set)
        for t in ticks:
            bars[t // (4 * Q)].add(t % (4 * Q))
        ks = sorted(bars)
        f["bar_rhythm_repeat"] = _share(sum(bars[a] == bars[b] for a, b in zip(ks, ks[1:]) if b == a + 1),
                                        len(ks) - 1)
        pats = collections.Counter(frozenset(bars[k]) for k in ks)
        f["bar_rhythm_vocab"] = len(pats) / len(ks)

    # ---- elements
    mins = dur_s / 60
    f["bombs_pm"] = len(m.bombs) / mins
    f["walls_pm"] = len(m.walls) / mins
    f["arcs_pm"] = len(m.arcs) / mins
    f["chains_pm"] = len(m.chains) / mins
    if m.walls:
        cov = np.zeros(int(t1 + 2) * 4 + 1, bool)
        for w in m.walls:
            a = int(max(w["b"], 0) * 4); b = int(max(w["b"] + w["d"], 0) * 4)
            cov[a:b + 1] = True
        f["wall_cover"] = float(cov.mean())
        d = np.array([w["d"] for w in m.walls])
        f["wall_dur_med"] = float(np.median(d))
        f["wall_dur_cv"] = float(d.std() / max(d.mean(), 1e-9))
    else:
        f["wall_cover"] = 0.0
    return f


def _entropy(vals) -> float:
    c = collections.Counter(vals.tolist())
    p = np.array(list(c.values()), float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())
