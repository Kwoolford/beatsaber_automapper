"""SONGPRINT — what in the SONG does a map answer? (the song-relative half of the blindspot sweep)

★Why: Kyle, 2026-09-17, on a map whose verdict page was clean: *"theres a booming bass in the
background on tempo that is so obviously ignored."* `fingerprint.py` reads the map alone and cannot
see that. This reads the map against the song's typed events (`events.py`, htdemucs_6s stems with
per-song timbre classes) and asks, per instrument, how much of it the map plays.

Every number here is read SAME-SONG: ours against the human map of the song, and that difference
against how much TWO HUMANS differ on one song (`scripts/songprint_sweep.py control`). Absolute
recall is song-dependent (a busy hi-hat makes any map's hat recall low), so it is never judged alone.

    from agent_mapper.songprint import songprint
    f = songprint(map_data, events_json)   # dict name -> float
"""
from __future__ import annotations

import collections

import numpy as np

TOL = 0.05        # s: a note answers an event within this
ACCENT_DB = 4.0   # an event this far above its stem's median loudness is an accent


def lanes(ev: dict) -> dict[str, list[dict]]:
    """Events grouped into instrument lanes. Drum classes are split when the name says what they are."""
    out = collections.defaultdict(list)
    for e in ev["events"]:
        s, c = e["stem"], e.get("cls", "")
        if s == "drums":
            k = next((w for w in ("kick", "snare", "hat", "tom", "crash", "cymbal") if w in c), "kit")
            out["drums:" + k].append(e)
            out["drums"].append(e)
        else:
            out[s].append(e)
    return out


def note_times(m) -> np.ndarray:
    spb = 60.0 / m.bpm
    return np.unique(np.round(np.array([n.beat for n in m.notes]) * spb + m.offset, 4))


def _answered(ts: np.ndarray, T: np.ndarray) -> np.ndarray:
    if not len(T) or not len(ts):
        return np.zeros(len(ts), bool)
    i = np.clip(np.searchsorted(T, ts), 1, len(T) - 1)
    d = np.minimum(np.abs(T[i] - ts), np.abs(T[i - 1] - ts))
    return d <= TOL


def songprint(m, ev: dict) -> dict[str, float]:
    T = note_times(m)
    f: dict[str, float] = {}
    if len(T) < 20:
        return f
    t0, t1 = T[0] - 1.0, T[-1] + 1.0
    L = lanes(ev)
    owned = np.zeros(len(T), bool)
    for name, es in L.items():
        es = [e for e in es if t0 <= e["t"] <= t1]
        if len(es) < 16:
            continue
        ts = np.array([e["t"] for e in es])
        loud = np.array([e.get("loud", 0.0) for e in es])
        f[f"rec:{name}"] = float(_answered(ts, T).mean())
        acc = loud >= ACCENT_DB
        if acc.sum() >= 8:
            f[f"acc:{name}"] = float(_answered(ts[acc], T).mean())
        # share of the map's instants that sit on this lane
        hit = _answered(T, np.sort(ts))
        f[f"on:{name}"] = float(hit.mean())
        if ":" not in name:
            owned |= hit
    f["on_any"] = float(owned.mean())   # notes on SOME detected event
    # the accent channel, stem-agnostic: the loudest 10 % of all events
    allv = [e for e in ev["events"] if t0 <= e["t"] <= t1]
    if allv:
        lv = np.array([e.get("loud", 0.0) for e in allv])
        ts = np.array([e["t"] for e in allv])
        top = lv >= np.percentile(lv, 90)
        f["acc:top10"] = float(_answered(ts[top], T).mean())
    # which lane each 8-bar section FOLLOWS (highest recall among lanes with >= 8 events there)
    f["_follow"] = follow(T, L, ev)
    return f


def follow(T, L, ev, bars: int = 8) -> list:
    """Per section: the lane with the highest answered share (None when the map is silent there)."""
    sec = bars * ev["bar_s"]
    out = []
    n = int(np.ceil((ev["phase"] + ev["n_bars"] * ev["bar_s"]) / sec))
    for k in range(n):
        a, b = ev["phase"] + k * sec, ev["phase"] + (k + 1) * sec
        Tk = T[(T >= a - TOL) & (T < b + TOL)]
        best, bv = None, 0.0
        if len(Tk) >= 4:
            for name, es in L.items():
                if name == "drums":
                    continue
                ts = np.array([e["t"] for e in es if a <= e["t"] < b])
                if len(ts) < 8:
                    continue
                r = _answered(ts, Tk).mean()
                if r > bv:
                    best, bv = name, float(r)
        out.append(best)
    return out
