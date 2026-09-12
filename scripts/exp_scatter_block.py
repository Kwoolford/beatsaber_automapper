"""Per-block SCATTER candidate: flag only blocks where the SONG came back and the map did not.

The map-wide read ships already. The naive per-block version was rejected 2026-09-10 because it
could not separate our 1f8d6 from a human's own harder difficulty — but it asked EVERY block.
This one asks only the blocks the song repeats, which is a different question.

Run: python scatter_block.py
"""
import sys, pathlib, numpy as np, warnings
from collections import Counter
warnings.filterwarnings("ignore")
ROOT = pathlib.Path("/home/kyle/repos/beatsaber_automapper")
sys.path.insert(0, str(ROOT / "agent_mapper")); sys.path.insert(0, str(ROOT / "scripts"))
import score as S, queries as Q

B = 4


def song_echo(arrs, B=B):
    """_echo's question asked of the SONG: has this block's music been heard before?"""
    bar, song, names = arrs["bar"], arrs["song"], list(arrs["song_names"])
    ki = [names.index(f"kit_{k}") for k in ("kick", "snare", "hat", "crash")]
    oi = names.index("onset")
    nb = int(bar.max()); blocks = {}
    for b0 in range(1, nb + 1, B):
        idx = np.where((bar >= b0) & (bar <= b0 + B - 1))[0]
        if len(idx) < 8:
            continue
        f = [(j, tuple(int(song[s, k] > 0) for k in ki), int(song[s, oi] > 0))
             for j, s in enumerate(idx)
             if any(song[s, k] > 0 for k in ki) or song[s, oi] > 0]
        if len(f) >= 6:
            blocks[b0] = Counter(f)
    ks = sorted(blocks); out = {}
    for i, b in enumerate(ks[1:], 1):
        A = blocks[b]
        out[b] = max(sum((A & blocks[c]).values())
                     / max(sum(A.values()), sum(blocks[c].values())) for c in ks[:i])
    return out


def read(map_path, sid, vs):
    m, song, how, vsm, lat, sc, mc, hc = S.build(pathlib.Path(map_path), sid, 16, vs)
    arrs = S.to_arrays(m, sc, mc, lat, hc)
    if "human" not in arrs:
        return None
    se = song_echo(arrs)
    me = Q._echo(arrs["map"], arrs["bar"], B, 6)
    he = Q._echo(arrs["human"], arrs["bar"], B, 6)
    ks = sorted(set(se) & set(me) & set(he))
    return (np.array([se[b] for b in ks]), np.array([me[b] for b in ks]),
            np.array([he[b] for b in ks]), ks)


def fires(se, me, he, quantile, margin):
    """A block fires when the song repeats there (top `quantile` of THIS song) and the map
    echoes `margin` less than the human does at that same block."""
    if len(se) < 8:
        return 0, 0
    thr = np.quantile(se, quantile)
    sel = se >= thr
    hit = sel & ((he - me) >= margin)
    return int(hit.sum()), int(sel.sum())


ROWS = [
    ("OURS 1f767", "outputs/walls_2026-09-12/W__1f767.zip", "1f767", "auto", "want silent"),
    ("OURS 1f8d6", "outputs/walls_2026-09-12/W__1f8d6.zip", "1f8d6", "auto", "want silent"),
    ("OURS 1f333", "outputs/walls_2026-09-12/W__1f333.zip", "1f333", "auto", "WANT FIRE"),
    ("OURS 1f913", "outputs/walls_2026-09-12/W__1f913.zip", "1f913", "auto", "WANT FIRE"),
    ("CTL humanplus-1f333", "outputs/bench_fixtures/HUMANPLUS__1f333.zip", "1f333", "auto", "must be silent"),
    ("CTL humanplus-1f8d6", "outputs/bench_fixtures/HUMANPLUS__1f8d6.zip", "1f8d6", "auto", "must be silent"),
    ("CTL humanexp-1f333", "outputs/bench_fixtures/HUMANEXP__1f333.zip", "1f333",
     "outputs/bench_fixtures/HUMANPLUS__1f333.zip", "must be silent"),
    ("CTL humanexp-1f8d6", "outputs/bench_fixtures/HUMANEXP__1f8d6.zip", "1f8d6",
     "outputs/bench_fixtures/HUMANPLUS__1f8d6.zip", "must be silent"),
]

if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    data = {}
    for lab, mp, sid, vs, want in ROWS:
        try:
            r = read(mp, sid, vs)
        except Exception as e:
            print(f"{lab:22s} ERROR {e}"); continue
        if r is None:
            print(f"{lab:22s} no human reference"); continue
        data[lab] = (r, want)
        se, me, he, ks = r
        print(f"{lab:22s} blocks={len(ks):3d}  song echo {se.mean():.3f}  "
              f"map {me.mean():.3f}  human {he.mean():.3f}  map-wide gap {he.mean()-me.mean():+.3f}   [{want}]")

    print("\n--- per-block rule: song echo in the top Q of this song AND human-map echo >= margin ---")
    for q in (0.5, 0.6, 0.75):
        for margin in (0.15, 0.20, 0.25, 0.30):
            line = []
            ok = True
            for lab, ((se, me, he, ks), want) in data.items():
                n, sel = fires(se, me, he, q, margin)
                line.append(f"{lab.split()[-1]}:{n}/{sel}")
                if want.endswith("silent") and n > 0:
                    ok = False
                if want == "WANT FIRE" and n == 0:
                    ok = False
            print(f"q={q:.2f} margin={margin:.2f}  {'PASS' if ok else '    '}  " + "  ".join(line))
