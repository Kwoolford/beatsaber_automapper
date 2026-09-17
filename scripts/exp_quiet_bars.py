#!/usr/bin/env python
"""Do humans REST in the quietest few percent of their song's bars?

2026-09-13aj refuted "thin the quiet bars" on the 0-10 % energy band (humans still play a median 5
notes/bar there and leave only 7.7 % of bars empty) but left one question open: `1f333`'s seven-bar
rest (163-169), the BREATHING red, sits at energy percentile **0.01-0.06** of its own song, below
the resolution of that band. This splits the bottom of the distribution.

If humans leave a large share of the bottom-2 % bars empty, the song announces a rest and a narrow
builder rule is possible. If not, a rest is a style choice the audio does not announce and
BREATHING cannot be fixed from the song alone.

Two populations, reported side by side:
  * `span`  — bars between the human's first and last note, percentile within that span. This is
    where `q_breathing` reads, and the one the verdict hangs on.
  * `whole` — every bar of the song. The control arm: its 0-10 % row must reproduce the recorded
    5.0 notes/bar and 7.7 % empty before the finer rows mean anything.

A bar is IN A REST when it belongs to a run of >= 2 consecutive empty bars (`q_breathing`'s
`min_rest_bars`). Energy and loading are `exp_d3_absolute.py`'s (`score.py::_energy`).

Run:
    python scripts/exp_quiet_bars.py --n 140
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exp_d3_absolute import BEATS_PER_BAR, RAW, energy, load_map  # noqa: E402

BANDS = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.10), (0.00, 0.10), (0.10, 0.25),
         (0.25, 0.50), (0.50, 1.00)]
CUTS = (0.02, 0.05, 0.10)
MIN_REST = 2


def bars(bpm, beats, t, rms):
    """(per-bar energy, per-bar note count) on the map's own grid."""
    spb = 60.0 / bpm
    nb = int(max(beats) // BEATS_PER_BAR) + 1
    bar_t = np.arange(nb + 1) * BEATS_PER_BAR * spb
    idx = np.searchsorted(t, bar_t)
    em = np.array([rms[idx[i]:max(idx[i + 1], idx[i] + 1)].mean() if idx[i] < len(rms) else np.nan
                   for i in range(nb)])
    cnt = np.bincount((np.array(beats) // BEATS_PER_BAR).astype(int), minlength=nb)[:nb]
    return em, cnt


def in_rest(cnt):
    """True for bars inside a run of >= MIN_REST empty bars."""
    out = np.zeros(len(cnt), bool)
    i = 0
    while i < len(cnt):
        if cnt[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(cnt) and cnt[j + 1] == 0:
            j += 1
        if j - i + 1 >= MIN_REST:
            out[i:j + 1] = True
        i = j + 1
    return out


def pct(x):
    """Percentile rank of each value within its own array, in [0, 1)."""
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / len(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--song", default="1f333", help="print this song's rest percentiles too")
    a = ap.parse_args()

    paths = sorted(RAW.glob("*.zip"))
    random.Random(a.seed).shuffle(paths)
    pops = {"whole": [], "span": []}       # rows of (map, pct, count, in_rest)
    used = 0
    for zp in paths:
        if used >= a.n:
            break
        try:
            m = load_map(zp)
            if m is None:
                continue
            bpm, beats, raw = m
            t, rms = energy(zp.stem, raw)
            em, cnt = bars(bpm, beats, t, rms)
        except Exception:  # noqa: BLE001
            continue
        ok = ~np.isnan(em)
        if ok.sum() < 30:
            continue
        used += 1
        rest = in_rest(cnt)
        w = np.where(ok)[0]
        pops["whole"].append((used, pct(em[w]), cnt[w], rest[w]))
        nz = np.nonzero(cnt)[0]
        s = np.arange(nz[0], nz[-1] + 1)
        s = s[ok[s]]
        pops["span"].append((used, pct(em[s]), cnt[s], rest[s]))

    print(f"{used} human Expert maps (seed {a.seed})\n")
    for name, rows in pops.items():
        P = np.concatenate([r[1] for r in rows])
        C = np.concatenate([r[2] for r in rows])
        R = np.concatenate([r[3] for r in rows])
        M = np.concatenate([np.full(len(r[1]), r[0]) for r in rows])
        print(f"== {name}: {len(P)} bars, {R.sum()} in a {MIN_REST}+ bar rest "
              f"({R.mean():.1%}), rests in {len(set(M[R]))} of {used} maps")
        print(f"  {'band':<11}{'bars':>7}{'median n':>10}{'empty':>8}{'in rest':>9}"
              f"{'maps w/ rest here':>19}")
        for lo, hi in BANDS:
            k = (P >= lo) & (P < hi)
            mr = len(set(M[k & R]))
            print(f"  {lo:.2f}-{hi:.2f}{k.sum():>7}{np.median(C[k]):>10.1f}"
                  f"{(C[k] == 0).mean():>8.1%}{R[k].mean():>9.1%}{mr:>12d} / {used}")
        print(f"  rule 'bar below p is a rest':  precision = share of flagged bars he rests;"
              f"  recall = share of his rest bars flagged")
        for c in CUTS:
            k = P < c
            print(f"    p < {c:.2f}: precision {R[k].mean():6.1%}   recall "
                  f"{(k & R).sum() / max(R.sum(), 1):6.1%}   (base rate {R.mean():.1%})")
        print()

    # A rest is a RUN, so a bar-level rule is the wrong unit. Read maximal runs of >= L consecutive
    # quiet bars (percentile or absolute E), span population only, and ask how much of each run the
    # human leaves empty. Pre-registered: a narrow rule exists iff some row rests >= 50 % of runs.
    print("== span, RUNS of consecutive quiet bars: share of runs the human rests in "
          "(>= half the run inside a rest)")
    print(f"  {'quiet =':<14}{'L':>3}{'runs':>6}{'maps':>6}{'rested':>8}{'bars rested':>13}")
    raw_rows = []
    for zp_ in paths:
        if len(raw_rows) >= used:
            break
        try:
            m_ = load_map(zp_)
            if m_ is None:
                continue
            em_, cnt_ = bars(m_[0], m_[1], *energy(zp_.stem, m_[2]))
        except Exception:  # noqa: BLE001
            continue
        ok_ = ~np.isnan(em_)
        if ok_.sum() < 30:
            continue
        nz_ = np.nonzero(cnt_)[0]
        s_ = np.arange(nz_[0], nz_[-1] + 1)
        s_ = s_[ok_[s_]]
        raw_rows.append((em_[s_], pct(em_[s_]), cnt_[s_], in_rest(cnt_)[s_]))
    tests = [(f"pct<{c:.2f}", "p", c) for c in (0.02, 0.05)] + \
            [(f"E<{e:.2f}", "e", e) for e in (0.20, 0.30, 0.40)]
    for label, kind, thr in tests:
        for L in (2, 3, 4):
            runs = rested = brest = btot = 0
            maps = set()
            for mi, (E, P, C, R) in enumerate(raw_rows):
                q = (P < thr) if kind == "p" else (E < thr)
                i = 0
                while i < len(q):
                    if not q[i]:
                        i += 1
                        continue
                    j = i
                    while j + 1 < len(q) and q[j + 1]:
                        j += 1
                    if j - i + 1 >= L:
                        runs += 1
                        maps.add(mi)
                        r = R[i:j + 1]
                        rested += r.mean() >= 0.5
                        brest += r.sum()
                        btot += len(r)
                    i = j + 1
            if runs:
                print(f"  {label:<14}{L:>3}{runs:>6}{len(maps):>6}{rested / runs:>8.1%}"
                      f"{brest / max(btot, 1):>13.1%}")
    print()

    zp = RAW / f"{a.song}.zip"
    m = load_map(zp) if zp.exists() else None
    if m is not None:
        bpm, beats, raw = m
        t, rms = energy(zp.stem, raw)
        em, cnt = bars(bpm, beats, t, rms)
        nz = np.nonzero(cnt)[0]
        s = np.arange(nz[0], nz[-1] + 1)
        p = pct(em[s])
        rest = in_rest(cnt)[s]
        print(f"== {a.song} (its own human map), span bars {s[0] + 1}-{s[-1] + 1}")
        for i in np.where(rest)[0]:
            print(f"  bar {s[i] + 1:>4}  E {em[s[i]]:.3f}  pct {p[i]:.3f}")
        k = p < 0.10
        print(f"  its bottom-10 % bars: {k.sum()}, of which in a rest {rest[k].sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
