#!/usr/bin/env python
"""QUERIES OVER THE ARRAYS — each named defect as a few lines of numpy, with an address.

★**Why (TODO P3, 2026-09-02).** The judge measures typicality; Kyle asks "wrong, and where?".
Each query here is a literal question over `score.py --npz` arrays (the song and the map on
one 1/16 lattice) that answers with `(code, t_sec, bar, why)` — a place to open in the score,
never a percentile. References are the **same song's human map** (`human` in the arrays), the
**song itself** (E, KIT, VOX, ON, MAIN), or the request. Validated on the P2 bench:

    python scripts/bench.py score queries:q_events      # EMPTY / D1 / D6
    python scripts/bench.py score queries:q_flow        # FLOW / D2
    python scripts/bench.py score queries:q_all         # everything, one pass

Every query takes the `to_arrays()` dict and returns a list of hits. Bars in a hit are the
FIRST bar of the window that fired; consecutive fires are merged into one hit whose `why`
names the span. A query that needs the human map returns [] without it — that is stated in
its docstring, and the CLEAN bench rows (human vs itself) do NOT test such a query; only the
GOOD/PREFERRED and DEFECT rows do.

Thresholds are written next to the measurement that set them (2026-09-02, bench arrays):
    q_events   4-bar windows, human ≥ 12 events: ours < 0.6× → EMPTY. Humans trivially 1.0.
               1f8d6-empty 8/32 windows · setA-1f333 29/50 · A+ (same notes) 29/50 — A+ is
               "very slow" by this query and Kyle's A+ was relative to older builds.
    q_flow     2-bar windows sliding by 1: share of events on an ODD 16th whose previous slot
               is empty (a note that starts on the "e"/"a" with nothing leading into it)
               ≥ 0.30 while ≥ 30 % of events sit on the 8th grid → FLOW (jitter, not a
               shifted grid). Humans: max 0.14 (1f913). Hunger AGENT: 29-32, 36-37, 43-47…
               ≥ 80 % of a window's events on odd 16ths (≥ 8 events) → D2 (a shifted grid):
               A+ bars 81-88 and 153-160 (100 %, human 0 %), DOD 24e6c.
"""

from __future__ import annotations

import numpy as np

BEATS_PER_BAR = 4
CODES = {0: "", 1: "vox", 2: "kik", 3: "snr", 4: "led"}


# ----------------------------------------------------------------------------- helpers
def col(arrs: dict, name: str) -> np.ndarray:
    return arrs["song"][:, list(arrs["song_names"]).index(name)]


def mcol(arrs: dict, name: str, which: str = "map") -> np.ndarray:
    return arrs[which][:, list(arrs["map_names"]).index(name)]


def hands(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = arr[:, 0:24:2]
    return (c == 1).any(axis=1), (c == 2).any(axis=1)


def notes_per_row(arr: np.ndarray) -> np.ndarray:
    return (arr[:, 0:24:2] > 0).sum(axis=1)


def has_human(arrs: dict) -> bool:
    return "human" in arrs and np.asarray(arrs["human"]).size > 0


def _t(arrs: dict, b: int) -> float:
    sel = arrs["bar"] == b
    return float(arrs["t_sec"][sel][0]) if sel.any() else 0.0


def merge(fires: list[tuple], code: str, arrs: dict, width: int) -> list[tuple]:
    """(bar, why) fires on windows of `width` bars → one hit per run of touching windows,
    as (code, t_sec, bar, why, bar_end) — the 5th field lets bench measure coverage."""
    if not fires:
        return []
    fires.sort()
    out, start, last, whys = [], fires[0][0], fires[0][0], [fires[0][1]]
    for b, why in fires[1:]:
        if b <= last + width:
            last = b; whys.append(why)
        else:
            out.append((code, _t(arrs, start), start,
                        f"bars {start}-{last + width - 1}: {whys[0]}", last + width - 1))
            start, last, whys = b, b, [why]
    out.append((code, _t(arrs, start), start,
                f"bars {start}-{last + width - 1}: {whys[0]}", last + width - 1))
    return out


# ----------------------------------------------------------------------------- q_events
def q_events(arrs: dict, W: int = 4, low: float = 0.6, high: float = 2.0) -> list[tuple]:
    """EMPTY / D6 / D1 — player EVENTS per window vs the same song's human map.

    An event is a row with any note; a two-hand double is ONE event. This is the P2 finding
    in one number: the 08-03 maps had human note counts and half the human's events.
    EMPTY: window where the human has ≥ 12 events and ours < `low`×.
    D6 (nps wasted): once, map-wide, when ≥ 50 % of our events are two-hand doubles and that
    is ≥ 20 points above the human's share — the note count looks human, the player gets
    half the events (set A: 56-66 % doubles vs humans 7-34 %; Hunger AGENT 4 % → silent).
    Placed at the EMPTY window with the most doubles, else the first window.
    D1 (very slow): once, at the first EMPTY window, when the map's median window ratio is
    below 0.7 (set A 1f333: 0.56; 1f767 0.87 — its "very slow" is not the event rate).
    D6 the other way (over-dense): window where the human has ≥ 8 events and ours ≥ `high`×
    — nps spent where the best mapper spent none (1f9a0 NEW: 657 events vs the human's 271;
    Hunger AGENT 5/52 windows; A+ 1/52). Needs the human map; silent without it.
    """
    if not has_human(arrs):
        return []
    bar = arrs["bar"]
    L, R = hands(arrs["map"]); ev = L | R; dbl = L & R
    HL, HR = hands(arrs["human"]); hev = HL | HR; hdbl = HL & HR
    mn, hn = notes_per_row(arrs["map"]), notes_per_row(arrs["human"])
    empty, dense, ratios, dshare = [], [], [], []
    for b0 in range(1, int(bar.max()) + 1, W):
        sel = (bar >= b0) & (bar < b0 + W)
        e, h = int(ev[sel].sum()), int(hev[sel].sum())
        if h >= 8 and e >= high * h:
            dense.append((b0, f"over-dense: {e} events vs human {h} ({e / h:.1f}x), "
                              f"notes {int(mn[sel].sum())} vs {int(hn[sel].sum())}"))
        if h < 12:
            continue
        ratios.append(e / h)
        d = int(dbl[sel].sum())
        dshare.append((d / max(e, 1), b0, e, d))
        if e < low * h:
            empty.append((b0, f"{e} events vs human {h} ({e / h:.2f}x), {d}/{max(e, 1)} doubles, "
                              f"notes {int(mn[sel].sum())} vs {int(hn[sel].sum())}"))
    hits = merge(empty, "EMPTY", arrs, W) + merge(dense, "D6", arrs, W)
    ds, hds = dbl.sum() / max(ev.sum(), 1), hdbl.sum() / max(hev.sum(), 1)
    if ds >= 0.5 and ds - hds >= 0.2 and dshare:
        low_w = {b for b, _ in empty}
        cand = [x for x in dshare if x[1] in low_w] or dshare
        _, b0, e, d = max(cand)
        hits.append(("D6", _t(arrs, b0), b0,
                     f"map-wide: {ds:.0%} of events are doubles (human {hds:.0%}); here {d}/{e} "
                     f"-- note count looks human, event rate is {ev.sum() / max(hev.sum(), 1):.2f}x"))
    if empty and ratios and float(np.median(ratios)) < 0.7:
        b0 = empty[0][0]
        hits.append(("D1", _t(arrs, b0), b0,
                     f"map-wide: median window event ratio {np.median(ratios):.2f}x the human "
                     f"({len(empty)}/{len(ratios)} windows under {low}x)"))
    return hits


q_events.codes = {"EMPTY", "D6", "D1"}


# ----------------------------------------------------------------------------- q_flow
def q_flow(arrs: dict, W: int = 2, iso_min: float = 0.30, grid_min: float = 0.30,
           shifted: float = 0.8) -> list[tuple]:
    """FLOW / D2 — where the rhythm jitters, and where the whole grid is shifted.

    FLOW: events that start on an odd 16th with the previous slot EMPTY (a note on the "e"
    or "a" that nothing leads into) are ≥ `iso_min` of a 2-bar window's events while
    ≥ `grid_min` of them still sit on the 8th grid — the mix a player cannot predict.
    Hunger AGENT: `··R· ···· ···· ···R|···· R·LR ···· LRL·`. Human 16th runs start on the
    beat (`RLR·`), so the human max is 0.14.
    D2: ≥ `shifted` of a window's ≥ 8 events on odd 16ths — a consistently shifted grid
    (A+ bars 81-88; DOD 24e6c reads its lattice off the kit).
    Both are read AGAINST A REFERENCE: the human map's own odd/isolated share in the same
    window, or the song's onsets without one. 1f335 is notated at 195 bpm, so its odd 16th
    is the felt 8th and the human sits there 35 % of the time (bars 43-44: 11/12) — an
    absolute rule called 20 spans of it "shifted" (2026-09-02, first customer).
    """
    bar = arrs["bar"]; sub = int(arrs["sub"])
    pos = np.arange(len(bar)) % sub

    def reads(arr):
        L, R = hands(arr); ev = L | R
        prev_empty = np.r_[True, ~ev[:-1]]
        odd = (pos % 2 == 1) & ev
        return ev, odd, odd & prev_empty, (pos % 2 == 0) & ev
    ev, odd, iso, grid = reads(arrs["map"])
    # The human (or, without one, the song's onsets) says what "odd" means here: at 195 bpm
    # (1f335) the odd 16th IS the felt 8th and the human sits on it 35 % of the time.
    if has_human(arrs):
        hev, hodd, hiso, _ = reads(arrs["human"])
    else:
        on = col(arrs, "onset") > 0.3
        hev, hodd, hiso = on, on & (pos % 2 == 1), on & (pos % 2 == 1) & np.r_[True, ~on[:-1]]
    # A shifted BAR (≥ 80 % of ≥ 4 events on odd 16ths) is all "isolated" by construction;
    # its events are D2's, never FLOW's -- so a window straddling a shift edge (A+ bars 80-81,
    # 152-153) reads only its unshifted bars for jitter.
    nb = int(bar.max()) + 1
    shifted_bar = np.zeros(nb + W, bool)
    for b in range(1, nb):
        sel = bar == b
        n = int(ev[sel].sum())
        shifted_bar[b] = n >= 4 and odd[sel].sum() >= shifted * n
    unshifted = ~shifted_bar[bar]
    flow, shift = [], []
    for b0 in range(1, nb):
        sel = (bar >= b0) & (bar < b0 + W)
        ne = int(ev[sel].sum())
        if ne < 8:
            continue
        o = odd[sel].sum() / ne
        nh = int(hev[sel].sum())
        ho = hodd[sel].sum() / nh if nh >= 4 else 0.0
        hi = hiso[sel].sum() / nh if nh >= 4 else 0.0
        if o >= shifted:
            if ho < 0.35:
                shift.append((b0, f"{o:.0%} of {ne} events on odd 16ths (reference {ho:.0%}) "
                                  f"— the grid is shifted"))
            continue
        us = sel & unshifted
        nu = int(ev[us].sum())
        if nu < 8:
            continue
        i, g = iso[us].sum() / nu, grid[us].sum() / nu
        if i >= iso_min and g >= grid_min and i >= hi + 0.2:
            flow.append((b0, f"{i:.0%} of {nu} events start on an odd 16th from silence "
                             f"(reference {hi:.0%}), {g:.0%} on the 8th grid"))
    return merge(flow, "FLOW", arrs, W) + merge(shift, "D2", arrs, W)


q_flow.codes = {"FLOW", "D2"}


# ----------------------------------------------------------------------------- q_vocals
def q_vocals(arrs: dict, W: int = 4, gap: float = 0.25, min_slots: int = 6,
             human_min: float = 0.6) -> list[tuple]:
    """D4 — the main vocals go unanswered where the human answered them.

    A MAIN=vox slot is "answered" when a note sits on it or one slot either side. Per
    4-bar window with ≥ `min_slots` vox-main slots: fire when the human answers ≥ 60 % and
    we answer ≥ `gap` less. Humans answer 70-95 % of their own vox-main slots; set A
    60-73 % — 1f767's human is itself at 70 %, so a fixed floor would fire on him.
    Measured (W=4, gap 0.25): set A 1f767 6/30 windows, 1f913 14/33, 1f8d6 10/22, A+ 3/11,
    Hunger AGENT 2/11. Needs the human map; silent without it.
    """
    if not has_human(arrs):
        return []
    bar = arrs["bar"]
    vox = col(arrs, "main").astype(int) == 1

    def answered(arr):
        L, R = hands(arr); ev = L | R
        return ev | np.r_[ev[1:], False] | np.r_[False, ev[:-1]]
    am, ah = answered(arrs["map"]), answered(arrs["human"])
    lyric = arrs["lyric"]
    fires = []
    for b0 in range(1, int(bar.max()) + 1, W):
        sel = (bar >= b0) & (bar < b0 + W) & vox
        n = int(sel.sum())
        if n < min_slots:
            continue
        h, m = ah[sel].mean(), am[sel].mean()
        if h >= human_min and m < h - gap:
            words = [w for w in lyric[sel & ~am] if any(ch.isalnum() for ch in str(w))][:4]
            fires.append((b0, f"vox answered {m:.0%} of {n} main slots vs human {h:.0%}"
                              + (f" -- unanswered: {' '.join(words)}" if words else "")))
    return merge(fires, "D4", arrs, W)


q_vocals.codes = {"D4"}


# ----------------------------------------------------------------------------- q_drops
def q_drops(arrs: dict, jump: float = 0.25, n_bars: int = 2, lag_beats: float = 1.0) -> list[tuple]:
    """D3 — the drop lands at the wrong time (or does not land).

    Drops come from the SONG: an E-jump is a bar whose mean energy rises ≥ `jump` over the
    previous bar; an E-drop falls by that much. At a jump the map must step UP: events/bar
    over the 2 bars after vs the 2 before, ≥ 0.8× the human's step, first note no more
    than `lag_beats` after the human's (without a human: step ≥ 1.2, first note within
    `lag_beats` of the bar line). At a drop the map must come down when the human does
    (human ratio ≤ 0.6, ours ≥ 0.9 → fire). The human's own jumps are 0.6-1.1× steps on
    1f913 (11, 71, 119) -- an absolute step floor fired on him, so the human decides.
    Set A 1f767 bar 85: human 3.5→6.0 at beat 0, ours 4→2 two beats late — the drop the
    verdict was about. Humans answer their own jumps at beat 0 (1f333 90, 1f767 43/85).
    """
    bar = arrs["bar"]; sub = int(arrs["sub"]); per = sub * BEATS_PER_BAR
    E = col(arrs, "energy")
    nb = int(bar.max())
    emean = np.array([E[bar == b].mean() if (bar == b).any() else 0.0 for b in range(1, nb + 1)])
    L, R = hands(arrs["map"]); ev = L | R
    hev = None
    if has_human(arrs):
        HL, HR = hands(arrs["human"]); hev = HL | HR

    def rate(e, b0, b1):
        sel = (bar >= b0) & (bar < b1)
        return e[sel].sum() / max(b1 - b0, 1)

    def first(e, b):
        sel = np.where((bar == b) & e)[0]
        return None if not len(sel) else (sel[0] - np.where(bar == b)[0][0]) / sub
    fires = []
    for b in range(3, nb - 1):
        d = emean[b - 1] - emean[b - 2]
        before, after = rate(ev, b - n_bars, b), rate(ev, b, b + n_bars)
        if d >= jump:
            step = after / max(before, 0.5)
            f = first(ev, b)
            if after < 2:
                continue        # nothing there at all: EMPTY's job, not a timing verdict
            if hev is not None:
                # the human decides what this jump asks for: his step and his first note
                hafter = rate(hev, b, b + n_bars)
                hstep = hafter / max(rate(hev, b - n_bars, b), 0.5)
                hf = first(hev, b)
                # under-stepped only if we also land short of his density after the jump
                bad = ((step < 0.8 * hstep and after < 0.8 * hafter)
                       or (hf is not None and (f is None or f > hf + lag_beats)))
                ref = f" (human ×{hstep:.1f}, first {'none' if hf is None else f'{hf:.2f}'})"
            else:
                bad = step < 1.2 or f is None or f > lag_beats
                ref = ""
            if bad:
                fires.append((b, f"E-jump {emean[b-2]:.2f}→{emean[b-1]:.2f}: events/bar {before:.1f}→{after:.1f}"
                                 f", first note {'none' if f is None else f'{f:.2f} beats'} after the bar line" + ref))
        elif d <= -jump and hev is not None:
            hb, ha = rate(hev, b - n_bars, b), rate(hev, b, b + n_bars)
            if hb >= 2 and ha <= 0.6 * hb and before >= 2 and after >= 0.9 * before:
                fires.append((b, f"E-drop {emean[b-2]:.2f}→{emean[b-1]:.2f}: human {hb:.1f}→{ha:.1f} "
                                 f"events/bar, ours {before:.1f}→{after:.1f} -- did not come down"))
    return merge(fires, "D3", arrs, 1)


q_drops.codes = {"D3"}


# ----------------------------------------------------------------------------- q_elements
def q_elements(arrs: dict, human_min_walls: int = 5) -> list[tuple]:
    """ELEMENTS — no walls where the human built them.

    Counts wall starts (a lane going from free to walled). Fires once, at the human's
    first wall, when the map has none and the human has ≥ `human_min_walls`. Arcs and
    chains are reported in `why` but never fire: the songset humans are v2 maps with 0 of
    either, so "the human has none" says nothing. Notes inside a walled lane are NOT read
    here -- the arrays carry no wall height, and human 1f767 has 4 such notes under crouch
    walls (measured 2026-09-02). Needs the human map; silent without it.
    """
    if not has_human(arrs):
        return []
    names = list(arrs["map_names"])
    li = [names.index(f"wall_lane{i}") for i in range(4)]

    def starts(arr):
        lanes = arr[:, li] > 0
        return lanes & ~np.r_[np.zeros((1, 4), bool), lanes[:-1]]
    ms, hs = starts(arrs["map"]), starts(arrs["human"])
    nm, nh = int(ms.sum()), int(hs.sum())
    if nm > 0 or nh < human_min_walls:
        return []
    s0 = int(np.argmax(hs.any(axis=1)))
    extra = []
    for k in ("arc", "chain"):
        h = int(((arrs["human"][:, names.index(f"{k}_L")] > 0) | (arrs["human"][:, names.index(f"{k}_R")] > 0)).sum())
        m = int(((arrs["map"][:, names.index(f"{k}_L")] > 0) | (arrs["map"][:, names.index(f"{k}_R")] > 0)).sum())
        extra.append(f"{k}s {m} vs {h} slots")
    return [("ELEMENTS", float(arrs["t_sec"][s0]), int(arrs["bar"][s0]),
             f"0 walls vs the human's {nh} (first at bar {int(arrs['bar'][s0])}); " + ", ".join(extra))]


q_elements.codes = {"ELEMENTS"}


# ----------------------------------------------------------------------------- q_all
QUERIES = [q_events, q_flow, q_vocals, q_drops, q_elements]


def q_all(arrs: dict) -> list[tuple]:
    """Every query, one pass, sorted by time."""
    hits = []
    for q in QUERIES:
        hits += q(arrs)
    return sorted(hits, key=lambda h: (h[1], h[0]))


q_all.codes = set().union(*(q.codes for q in QUERIES))


# ----------------------------------------------------------------------------- cli
def main() -> int:
    import argparse
    import pathlib
    import sys
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=pathlib.Path,
                    help="a map zip (built on the lattice with the song's human map as reference) "
                         "or a score.py --npz file")
    ap.add_argument("--song", help="song id for a zip (default: the 4-6 hex id in the file name)")
    ap.add_argument("--vs", default="auto", help="reference human map: zip, corpus id, or 'auto'")
    ap.add_argument("--query", default="q_all", help=", ".join(q.__name__ for q in QUERIES) + ", q_all")
    a = ap.parse_args()
    if a.src.suffix == ".npz":
        z = np.load(a.src, allow_pickle=True)
        arrs = {k: z[k] for k in z.files}
    else:
        import re
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from agent_mapper import score as S
        sid = a.song or next(iter(re.findall(r"[0-9a-f]{4,6}", a.src.stem)), None)
        m, song, how, vsm, lat, sc, mc, hc = S.build(str(a.src), sid, 4, a.vs, False)
        arrs = S.to_arrays(m, sc, mc, lat, hc)
    hits = globals()[a.query](arrs)
    for h in hits:
        code, t, b, why = h[:4]
        print(f"{code:<8s} {int(t // 60)}:{t % 60:05.2f}  bar {b:>4d}  {why}")
    from collections import Counter
    c = Counter(h[0] for h in hits)
    print(f"-- {len(hits)} hit(s): " + (", ".join(f"{k} {v}" for k, v in sorted(c.items())) or "none")
          + ("" if has_human(arrs) else "  (no human map: EMPTY/D1/D4/D6/ELEMENTS could not be asked)"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
