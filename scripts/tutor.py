#!/usr/bin/env python
"""STUDY MODE — read the best human map of THIS song on the score before building.

★**Why (TODO P2b, 2026-09-02).** A top mapper has heard a thousand answers to "the vocal
enters here", "the drop lands on bar 33", "the chorus comes back". The agent has none; it
builds from levers. The tutor puts the human's answers on the SAME lattice the agent reads
and builds on, cut at the moments the song hands out (section changes, energy steps, vocal
/ lead / drum entries, repeats) — a **situation → pattern** table the agent can read as
vocabulary and copy with `mapedit.py`.

**Which human map is the tutor.** The corpus holds ONE map per BeatSaver id and the crawl
was rating-sorted with upvote ratio ≥ 0.8 (`src/…/data/download.py`, `min_rating=0.8`),
no per-map rating is stored ⇒ `data/raw/<sid>.zip` IS the top-rated human map we have of
the song. Threshold logged here so nobody looks for a second candidate.

    tutor.py 1f8d6                       # the situations table of the human map
    tutor.py 1f8d6 --map out/x.zip       # ours beside the tutor at every situation, and
                                         # whether we answered it the tutor's way
    tutor.py 1f8d6 --bars 33-36          # zoom: the score rows of the tutor there
    tutor.py --vocab 1f333 1f767 …       # situation → pattern counts over several tutors

Situations are found in the SONG columns only (never in either map), so the tutor and
ours are cut at identical moments. A **pattern** is what a map does in the 2 bars from the
situation: events/bar, doubles share, the per-16th hand glyphs (`L R D ·`, walls as `w`),
the bar BEFORE (`pre`, where a human leaves the breath), and the first answer — how many
beats after the moment the first note lands. "Answered the tutor's way" = same pattern
word, events/bar within ±35 %, first answer within 1 beat.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

CACHE = REPO / "outputs" / "tutor_cache"
BEATS_PER_BAR = 4
SONGSET = ["1f333", "1f767", "1f913", "1f8d6"]   # the standing review set


# ----------------------------------------------------------------------------- arrays
def arrays_for(sid: str, map_path: pathlib.Path | None = None, rebuild: bool = False) -> dict:
    """The tutor alone (cached) or `map_path` with the tutor as `human` (never cached —
    builds change)."""
    from agent_mapper import score as S
    tutor = S.resolve_vs("auto", sid)
    if not tutor.exists():
        sys.exit(f"no corpus map for {sid}: {tutor} — the song has no tutor")
    if map_path is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        f = CACHE / f"{sid}.npz"
        if f.exists() and not rebuild:
            z = np.load(f, allow_pickle=True)
            return {k: z[k] for k in z.files}
        m, song, _how, _vsm, lat, sc, mc, hc = S.build(tutor, sid, 4, None)
        arrs = S.to_arrays(m, sc, mc, lat, hc)
        arrs["tutor"] = str(tutor.relative_to(REPO))
        np.savez(f, **arrs)
        return arrs
    m, song, _how, _vsm, lat, sc, mc, hc = S.build(map_path, sid, 4, "auto")
    arrs = S.to_arrays(m, sc, mc, lat, hc)
    arrs["tutor"] = str(tutor.relative_to(REPO))
    return arrs


def _col(arrs: dict, name: str) -> np.ndarray:
    return arrs["song"][:, list(arrs["song_names"]).index(name)]


def hands(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(left present, right present) per slot from a map/human array (colour 1 = L, 2 = R)."""
    col = arr[:, 0:24:2]
    return (col == 1).any(axis=1), (col == 2).any(axis=1)


# ----------------------------------------------------------------------------- situations
def find_situations(arrs: dict) -> list[dict]:
    """Moments the SONG hands out, snapped to the bar they fall in. Several kinds in one
    bar are merged into one situation."""
    sub = int(arrs["sub"]); per = sub * BEATS_PER_BAR
    bar = arrs["bar"]; n_bars = int(bar.max())
    section = arrs["section"]; lyric = arrs["lyric"]
    E = _col(arrs, "energy")
    vox = _col(arrs, "vox_midi") > 0
    lead = _col(arrs, "lead_midi") > 0
    bass = _col(arrs, "bass_midi") > 0
    kit = (_col(arrs, "kit_kick") > 0) | (_col(arrs, "kit_snare") > 0)
    by_bar: dict[int, list[str]] = collections.defaultdict(list)
    seen_sections: dict[str, int] = {str(section[0]): 1} if len(section) and section[0] else {}

    def bar_mean(x, b):
        sl = bar == b
        return float(x[sl].mean()) if sl.any() else 0.0

    for b in range(1, n_bars + 1):
        s0 = int(np.argmax(bar == b))
        if b > 1 and section[s0] != section[s0 - 1] and section[s0]:
            lab = str(section[s0])
            if lab in seen_sections:
                by_bar[b].append(f"section {lab} again (first bar {seen_sections[lab]})")
            else:
                by_bar[b].append(f"section {section[s0 - 1] or '-'}→{lab}")
                seen_sections[lab] = b
        if b > 1:
            e0, e1 = bar_mean(E, b - 1), bar_mean(E, b)
            if e1 - e0 >= 0.25:
                by_bar[b].append(f"E jump {e0 * 9:.0f}→{e1 * 9:.0f}")
            elif e0 - e1 >= 0.25:
                by_bar[b].append(f"E drop {e0 * 9:.0f}→{e1 * 9:.0f}")
    quiet = 2 * per   # an entry needs ≥ 2 silent bars before it …
    stay = 0.15       # … and the stem must then be present in ≥ 15 % of the next 2 bars
    for name, pres in (("vox", vox), ("lead", lead), ("bass", bass), ("drums", kit)):
        on = np.nonzero(pres)[0]
        for s in on:
            if s >= quiet and not pres[s - quiet:s].any() and pres[s:s + quiet].mean() >= stay:
                b = int(bar[s])
                beat_in = (s - int(np.argmax(bar == b))) / sub + 1
                if beat_in >= 4.5:          # a pickup anticipates the NEXT bar line
                    b += 1
                word = ""
                if name == "vox":
                    nxt = [w for w in lyric[s:s + per] if w and any(ch.isalnum() for ch in w)]
                    word = f" '{nxt[0]}'" if nxt else ""
                by_bar[b].append(f"{name} enters{word} @beat {beat_in:g}")
        # leaves: last slot before ≥ 2 quiet bars, after ≥ 2 bars of real presence
        for s in on:
            if (s + 1 + quiet <= len(pres) and not pres[s + 1:s + 1 + quiet].any()
                    and s >= quiet and pres[s + 1 - quiet:s + 1].mean() >= stay):
                by_bar[int(bar[s]) + 1].append(f"{name} leaves")
    out = []
    for b in sorted(by_bar):
        if b > n_bars:
            continue
        kinds = by_bar[b]
        out.append({"bar": b, "kinds": kinds, "kind": _kind(kinds),
                    "song": _song_desc(arrs, b)})
    return out


def _kind(kinds: list[str]) -> str:
    """The situation's class for the vocabulary table."""
    for k in kinds:
        if k.startswith("section") and "again" in k:
            return "repeat"
    for k in kinds:
        if k.startswith("section"):
            return "section"
    for k in kinds:
        if "enters" in k:
            return k.split()[0] + "-in"
    for k in kinds:
        if k.startswith("E jump"):
            return "E-jump"
    for k in kinds:
        if k.startswith("E drop"):
            return "E-drop"
    return kinds[0].split()[0] + "-out"


def _song_desc(arrs: dict, b: int) -> str:
    """What the song has in the 2 bars from b: E, and which stems are present."""
    bar = arrs["bar"]
    sl = (bar == b) | (bar == b + 1)
    if not sl.any():
        return ""
    E = int(round(float(_col(arrs, "energy")[sl].mean()) * 9))
    stems = []
    for name, col in (("kit", None), ("vox", "vox_midi"), ("lead", "lead_midi"), ("bass", "bass_midi")):
        if col is None:
            pres = ((_col(arrs, "kit_kick") > 0) | (_col(arrs, "kit_snare") > 0))[sl]
        else:
            pres = (_col(arrs, col) > 0)[sl]
        if pres.mean() >= 0.1:
            stems.append(name)
    main = _col(arrs, "main")[sl]
    codes = {0: "", 1: "vox", 2: "kik", 3: "snr", 4: "led"}
    m = collections.Counter(codes[int(x)] for x in main if x)
    main_s = m.most_common(1)[0][0] if m else "-"
    return f"E{E} {'+'.join(stems) or 'quiet'} main={main_s}"


# ----------------------------------------------------------------------------- patterns
def glyphs(arr: np.ndarray, s0: int, s1: int, sub: int) -> str:
    """Per-slot hand glyphs, a space per beat, `|` per bar. Walls active = lower-case w
    on an empty slot."""
    L, R = hands(arr)
    walls = arr[:, 25:29].sum(axis=1) > 0
    out = []
    for s in range(s0, s1):
        if s > s0 and (s - s0) % (sub * BEATS_PER_BAR) == 0:
            out.append("|")
        elif s > s0 and (s - s0) % sub == 0:
            out.append(" ")
        if s >= len(arr):
            out.append(" "); continue
        g = "D" if L[s] and R[s] else "L" if L[s] else "R" if R[s] else "·"
        if g == "·" and walls[s]:
            g = "w"
        out.append(g)
    return "".join(out)


def pattern(arr: np.ndarray, bar: np.ndarray, b: int, sub: int, n_bars: int = 2) -> dict:
    """What a map does in the `n_bars` bars from bar b (and the bar before)."""
    per = sub * BEATS_PER_BAR
    s0 = int(np.argmax(bar == b)) if (bar == b).any() else len(arr)
    s1 = min(s0 + n_bars * per, len(arr))
    L, R = hands(arr)
    ev = np.nonzero(L[s0:s1] | R[s0:s1])[0]
    n_ev = len(ev)
    dbl = int((L[s0:s1] & R[s0:s1]).sum())
    ev_bar = n_ev / max((s1 - s0) / per, 1e-9)
    dbl_pct = 100.0 * dbl / n_ev if n_ev else 0.0
    gaps = np.diff(ev) if n_ev > 1 else np.array([])
    gap = float(np.median(gaps)) if len(gaps) else 0.0
    # hand alternation among single-hand events
    seq = ["D" if L[s0 + s] and R[s0 + s] else "L" if L[s0 + s] else "R" for s in ev]
    singles = [h for h in seq if h != "D"]
    alt = (sum(1 for a, c in zip(singles, singles[1:]) if a != c) / (len(singles) - 1)
           if len(singles) > 1 else 0.0)
    first = float(ev[0]) / sub if n_ev else None   # beats after the bar line
    walls = float((arr[s0:s1, 25:29].sum(axis=1) > 0).mean()) if s1 > s0 else 0.0
    arcs = int((arr[s0:s1, 29:31].sum(axis=1) > 0).sum())
    chains = int((arr[s0:s1, 31:33].sum(axis=1) > 0).sum())
    if n_ev == 0:
        word = "rest"
    elif ev_bar < 2.5:
        word = "sparse"
    elif dbl_pct >= 50:
        word = "doubles"
    elif gap <= 1.0 and n_ev >= 6 and alt >= 0.7:
        word = "stream"
    elif gap <= 1.0 and n_ev >= 6:
        word = "burst"
    elif gap <= 2.0 and alt >= 0.7:
        word = "alt-8ths"
    elif gap <= 2.0:
        word = "8ths"
    elif gap <= 4.0 and alt >= 0.7:
        word = "alt-4ths"
    elif gap <= 4.0:
        word = "4ths"
    else:
        word = "mixed"
    pre_s0 = max(s0 - per, 0)
    pre_ev = int((L[pre_s0:s0] | R[pre_s0:s0]).sum())
    pre = "rest" if pre_ev == 0 else "sparse" if pre_ev < 3 else f"{pre_ev}ev"
    return dict(word=word, ev_bar=ev_bar, dbl_pct=dbl_pct, gap=gap, alt=alt, first=first,
                walls=walls, arcs=arcs, chains=chains, pre=pre,
                glyphs=glyphs(arr, s0, s1, sub))


def same_way(t: dict, o: dict) -> bool:
    """Did `o` answer the moment the way the tutor `t` did?"""
    if t["word"] != o["word"]:
        return False
    if t["ev_bar"] > 0 and abs(o["ev_bar"] - t["ev_bar"]) > 0.35 * t["ev_bar"]:
        return False
    if (t["first"] is None) != (o["first"] is None):
        return False
    if t["first"] is not None and abs(t["first"] - o["first"]) > 1.0:
        return False
    return True


def _fmt(p: dict) -> str:
    first = "—" if p["first"] is None else f"+{p['first']:.2g}b"
    extra = []
    if p["walls"] >= 0.15:
        extra.append(f"W{p['walls']:.0%}")
    if p["arcs"]:
        extra.append(f"A{p['arcs']}")
    if p["chains"]:
        extra.append(f"C{p['chains']}")
    return (f"{p['word']:<9s} {p['ev_bar']:4.1f}ev/bar {p['dbl_pct']:3.0f}%dbl first {first:<6s} "
            f"pre {p['pre']:<6s} {' '.join(extra):<8s} {p['glyphs']}")


# ----------------------------------------------------------------------------- cli
def cmd_song(a) -> int:
    from agent_mapper import score as S
    arrs = arrays_for(a.song, a.map, a.rebuild)
    sub = int(arrs["sub"]); bar = arrs["bar"]
    tutor_arr = arrs["human"] if a.map else arrs["map"]
    n_bars = int(bar.max())
    print(f"# TUTOR {arrs['tutor']}  — song {a.song} · {n_bars} bars · 1/{sub * 4} lattice · "
          f"the corpus's one human map of this song (crawl: rating-sorted, upvote ratio ≥ 0.8)")
    if a.map:
        print(f"# OURS  {a.map}")
    if a.copy or a.thin or a.fill:
        if not a.map:
            sys.exit("--copy/--thin/--fill need --map: they emit mapedit ops against OUR map")
        spec = a.copy or a.thin or a.fill
        b0, _, b1 = spec.partition("-")
        b0, b1 = int(b0), int(b1 or b0)              # "66" = one bar
        fn, word = ((emit_copy, "copy the tutor") if a.copy else
                    (emit_thin, "thin to the tutor") if a.thin else
                    (emit_fill, "fill to the tutor's slots"))
        ops = fn(arrs, b0, b1)
        print(f"# mapedit ops: {word} at bars "
              f"{b0}-{b1}  ->  save, then  mapedit.py {a.map} from ops.txt ; mapedit.py {a.map} resets")
        print("\n".join(ops))
        return 0
    if a.bars:
        m, song, how, vsm, lat, sc, mc, hc = (S.build(a.map, a.song, sub, "auto") if a.map
                                              else S.build(S.resolve_vs("auto", a.song), a.song, sub, None))
        print("\n".join(S.header_lines(m, song, sc, mc, lat, how, vsm)))
        s0, s1 = S._bar_range(a.bars, lat)
        print()
        print("\n".join(S.render_rows(m, sc, mc, lat, s0, s1, hc)))
        return 0
    sits = find_situations(arrs)
    print(f"# {len(sits)} situations (from the SONG columns). Pattern = the 2 bars from the bar; "
          f"glyphs per 1/{sub * 4}: L R D(both) · w(wall)")
    print(f"# {'bar':>4s} {'kind':<8s} {'song':<26s} {'who':<5s} pattern")
    n_same = 0
    for s in sits:
        t = pattern(tutor_arr, bar, s["bar"], sub)
        print(f"{s['bar']:>6d} {s['kind']:<8s} {s['song']:<26s} {'tutor':<5s} {_fmt(t)}")
        for k in s["kinds"]:
            print(f"{'':>6s} {'':<8s} · {k}")
        if a.map:
            o = pattern(arrs["map"], bar, s["bar"], sub)
            ok = same_way(t, o)
            n_same += ok
            print(f"{'':>6s} {'':<8s} {'':<26s} {'ours':<5s} {_fmt(o)}   {'SAME' if ok else 'differs'}")
    if a.map:
        print(f"\n# answered the tutor's way: {n_same}/{len(sits)} situations "
              f"(same pattern word, ev/bar within ±35 %, first answer within 1 beat)")
    else:
        print(f"\n# copy a pattern: tutor.py {a.song} --bars b-b+1 shows the rows; mapedit.py "
              f"places them. Whole-song view: score.py {arrs['tutor']} --song {a.song}")
    return 0


_ARROW = "↑↓←→↖↗↙↘•"


def _cells(row: np.ndarray) -> list[tuple[str, int, int, str]]:
    """(hand, x, y, arrow) for every note in one lattice row of a map array."""
    c = row[:24].reshape(12, 2)
    return [("LR"[int(c[k, 0]) - 1], k % 4, k // 4, _ARROW[int(c[k, 1]) - 1])
            for k in range(12) if c[k, 0]]


def _hands(row: np.ndarray) -> str:
    col = row[:24].reshape(12, 2)[:, 0]
    return "".join(h for h, v in (("L", 1), ("R", 2)) if (col == v).any())


def _addr(arrs: dict, i: int) -> str:
    bar, sub = arrs["bar"], int(arrs["sub"])
    s = i - int(np.where(bar == bar[i])[0][0])
    return f"{int(bar[i])}.{s // sub + 1}.{s % sub}"


def emit_copy(arrs: dict, b0: int, b1: int) -> list[str]:
    """mapedit ops that replace OUR bars b0-b1 with the tutor's cells, verbatim.

    ★The 1f767 loop (2026-09-02) did this at 9-12 (his bass-entry doubles) and 51-52
    (his vocal run) by hand; the ops are the same every time. Parity at the seams is
    the caller's problem: `mapedit.py from` refuses a violation, and `mapedit.py
    resets` names a same-parity repeat left at the boundary."""
    bar, m, h = arrs["bar"], arrs["map"], arrs["human"]
    names = list(arrs["map_names"])
    wall_col = [names.index(f"wall_lane{x}") for x in range(4)]
    out = []
    for i in range(len(bar)):
        if not (b0 <= bar[i] <= b1):
            continue
        a = _addr(arrs, i)
        if _hands(m[i]):
            out.append(f"delete {a}")
        done: set[str] = set()
        for hd, x, y, d in _cells(h[i]):
            # his same-hand towers (1f913 27.4.0) become one note; a cell under OUR wall
            # is skipped — mapedit refuses both, and refuses the whole batch
            if hd in done or m[i, wall_col[x]] > 0:
                continue
            done.add(hd)
            out.append(f"place {a} {hd} {x},{y} {d}")
    return out


def emit_thin(arrs: dict, b0: int, b1: int) -> list[str]:
    """mapedit deletes that thin OUR bars b0-b1 to the tutor's density: a note of ours
    survives only where he has a note within one lattice slot.

    ★Two lessons from 1f767 bars 13-24 (D6 2.1×): (1) a survivor on an ODD 16th whose
    neighbour is his on-beat note is now a note "from silence" — FLOW fires on it, so
    `move` it onto his slot afterwards (the ops below say which); (2) deleting the
    alternating notes leaves same-parity repeats — read `mapedit.py resets` next and
    reconcile with ONE added/removed note per hand, not a chain of flips."""
    bar, m, h, sub = arrs["bar"], arrs["map"], arrs["human"], int(arrs["sub"])
    out = []
    claimed: set[tuple[int, str]] = set()          # (slot, hand) a move already filled
    for i in range(len(bar)):
        if not (b0 <= bar[i] <= b1) or not _hands(m[i]):
            continue
        near = [j for j in (i - 1, i, i + 1) if 0 <= j < len(bar) and _hands(h[j])]
        a = _addr(arrs, i)
        if not near:
            out.append(f"delete {a}")
            continue
        s = i - int(np.where(bar == bar[i])[0][0])
        if s % 2 == 1 and i not in near:
            j = near[0]
            for hd in _hands(m[i]):
                if hd in _hands(m[j]) or (j, hd) in claimed:
                    out.append(f"delete {a} {hd}")
                    continue
                claimed.add((j, hd))
                # land in HIS cell for that hand, so the move never collides with our
                # other hand's note already in the slot (17.3.0 on 1f767)
                his = [c for c in _cells(h[j]) if c[0] == hd]
                if his:
                    x, y = his[0][1], his[0][2]
                else:
                    taken = {(c[1], c[2]) for c in _cells(m[j])}
                    side = [(2, 0), (2, 1), (3, 0), (3, 1), (1, 0), (1, 1)]
                    if hd == "L":
                        side = [(3 - x, y) for x, y in side]
                    x, y = next(c for c in side if c not in taken)
                out.append(f"move {a} {hd} {_addr(arrs, j)} {x},{y}")
    return out


_DOWN, _UP = set("↓↙↘"), set("↑↖↗")


def emit_fill(arrs: dict, b0: int, b1: int) -> list[str]:
    """mapedit places that FILL our bars b0-b1 up to the tutor's slots: wherever he has a
    note and we have nothing within one lattice slot, place one — HIS hand and cell (the
    hand alternation is the flow answer he already found), OUR direction: the vertical
    opposite of that hand's last directional swing, so the parity guard is satisfied by
    construction; a dot where his note is a dot or the hand has no history.

    ★Why not `--copy`: copy replaces our bars; fill keeps every note we placed and adds
    only the answers we skipped (EMPTY "12 events vs human 22", D4 "vox answered 65 %").
    The rhythm it adds is his; the notes around it are still ours (1f913, 2026-09-02)."""
    bar, m, h = arrs["bar"], arrs["map"], arrs["human"]
    names = list(arrs["map_names"])
    wall_col = [names.index(f"wall_lane{x}") for x in range(4)]
    # pass 1: which of his slots we fill (hand, cell, his arrow)
    fills: list[tuple[int, str, int, int, str]] = []
    for i in range(len(bar)):
        if bar[i] > b1:
            break
        if bar[i] < b0 or not _hands(h[i]):
            continue
        if any(_hands(m[j]) for j in (i - 1, i, i + 1) if 0 <= j < len(bar)):
            continue
        taken = {(c[1], c[2]) for c in _cells(m[i])}
        hands_done: set[str] = set()                 # his same-hand towers: one note per hand
        for hd, x, y, d in _cells(h[i]):
            if (x, y) in taken or hd in hands_done:
                continue
            if m[i, wall_col[x]] > 0:                # our wall owns that lane right now
                continue
            fills.append((i, hd, x, y, d))
            taken.add((x, y)); hands_done.add(hd)
    # pass 2: arrows. Alternate from the hand's last swing, but the NEXT note we already
    # placed is fixed — if the run of fills would land on its side of the swing, the first
    # of the run becomes a dot (a dot never resets), so the run re-phases to meet it.
    # (1f913 bars 1-8: fill ↑ at 5.4.0 then our 6.1.2 ↑ = a reset the fill itself made.)
    ours_dir: dict[str, list[tuple[int, str]]] = {"L": [], "R": []}
    for i in range(len(bar)):
        for hd, x, y, d in _cells(m[i]):
            if d != "•":
                ours_dir[hd].append((i, d))
    out = []
    last: dict[str, str] = {}
    ptr = {"L": 0, "R": 0}                           # next of OUR directional notes per hand
    for n, (i, hd, x, y, d) in enumerate(fills):
        while ptr[hd] < len(ours_dir[hd]) and ours_dir[hd][ptr[hd]][0] < i:
            last[hd] = ours_dir[hd][ptr[hd]][1]
            ptr[hd] += 1
        prev = last.get(hd)
        if d == "•" or prev is None:
            arrow = "•"
        else:
            arrow = "↑" if prev in _DOWN else "↓"
            if ptr[hd] < len(ours_dir[hd]):
                nj, nxt = ours_dir[hd][ptr[hd]]
                run = 1 + sum(1 for (i2, hd2, _x, _y, d2) in fills[n + 1:]
                              if hd2 == hd and d2 != "•" and i2 < nj)
                end = arrow if run % 2 == 1 else prev
                if (end in _DOWN) == (nxt in _DOWN):
                    arrow = "•"
        out.append(f"place {_addr(arrs, i)} {hd} {x},{y} {arrow}")
        if arrow != "•":
            last[hd] = arrow
    return out


def cmd_vocab(a) -> int:
    ids = a.ids or SONGSET
    table: dict[str, list[dict]] = collections.defaultdict(list)
    for sid in ids:
        try:
            arrs = arrays_for(sid, None, a.rebuild)
        except SystemExit as e:
            print(f"# skip {sid}: {e}")
            continue
        sub = int(arrs["sub"]); bar = arrs["bar"]
        for s in find_situations(arrs):
            p = pattern(arrs["map"], bar, s["bar"], sub)
            p["sid"] = sid; p["bar"] = s["bar"]
            table[s["kind"]].append(p)
    print(f"# situation → pattern over {len(ids)} tutors ({' '.join(ids)}); counts, not rates")
    print(f"# {'kind':<8s} {'n':>3s}  {'pattern words':<40s} {'ev/bar':>7s} {'dbl%':>5s} "
          f"{'first':>6s} {'pre=rest':>8s}")
    for kind, ps in sorted(table.items(), key=lambda kv: -len(kv[1])):
        words = collections.Counter(p["word"] for p in ps)
        ws = " ".join(f"{w}×{n}" for w, n in words.most_common())
        ev = statistics.median(p["ev_bar"] for p in ps)
        dbl = statistics.median(p["dbl_pct"] for p in ps)
        firsts = [p["first"] for p in ps if p["first"] is not None]
        first = f"+{statistics.median(firsts):.2g}b" if firsts else "—"
        rest = sum(1 for p in ps if p["pre"] == "rest")
        print(f"  {kind:<8s} {len(ps):>3d}  {ws:<40s} {ev:7.1f} {dbl:5.0f} {first:>6s} {rest:>5d}/{len(ps)}")
    if a.verbose:
        print()
        for kind, ps in table.items():
            for p in ps:
                print(f"  {kind:<8s} {p['sid']} bar {p['bar']:>3d}  {_fmt(p)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("song", nargs="?", help="song id (1f8d6); its corpus map is the tutor")
    ap.add_argument("--map", type=pathlib.Path, help="our build, compared at every situation")
    ap.add_argument("--bars", help="print the score rows of the tutor (or --map vs tutor) there")
    ap.add_argument("--copy", metavar="a-b",
                    help="emit mapedit ops replacing OUR bars a-b (--map) with the tutor's cells")
    ap.add_argument("--thin", metavar="a-b",
                    help="emit mapedit ops thinning OUR bars a-b to the tutor's density "
                         "(keep ours within a slot of his; move odd-16th survivors onto his slot)")
    ap.add_argument("--fill", metavar="a-b",
                    help="emit mapedit places filling OUR bars a-b up to the tutor's slots "
                         "(his hand and cell where we answer nothing; our direction)")
    ap.add_argument("--vocab", nargs="*", dest="ids", metavar="ID",
                    help="situation → pattern counts over these tutors (default: the review songset)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.ids is not None:
        return cmd_vocab(a)
    if not a.song:
        ap.error("a song id, or --vocab")
    return cmd_song(a)


if __name__ == "__main__":
    sys.exit(main())
