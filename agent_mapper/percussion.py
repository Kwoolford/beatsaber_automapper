#!/usr/bin/env python
"""THE KIT — which drum is hitting, not just that a drum hit.

`brief.py` reports "drums 9.2 onsets/s". A human mapper hears something completely
different: *kick on 1 and 3, snare on 2 and 4, hats filling, crash on the downbeat of
the chorus*. Those are not the same information, and the difference is exactly where
our two open gaps live:

- **`double_share` 0.034 vs a human 0.146.** The current accent model is "a strong beat
  with ≥2 stems agreeing", which caps out at ~3 doubles in 24 bars. A human does not
  place doubles on strong beats — they place them on **crashes and snare accents**.
  You cannot select those without knowing which drum it was.
- **section boundaries.** A crash cymbal *is* a section marker: it is the single most
  reliable "something new starts here" signal in recorded music, and it needs no
  segmentation algorithm at all — just a classifier and a look at where they land.

## How, and why not a trained classifier
There are no drum-transcription labels in this project, so this uses physically
motivated band ratios and decay, which need none. Every kit piece is separated by
where its energy sits and how long it lasts:

| piece | energy | decay |
|---|---|---|
| kick | almost all below ~120 Hz | short |
| snare | broadband — a 150-400 Hz body **plus** 2-8 kHz noise | short |
| tom | 100-400 Hz, little top | medium, pitched |
| hat | top-heavy above 6 kHz | very short |
| **crash/ride** | top-heavy above 4 kHz | ★**long** — this is what separates it from a hat |

★**It is validated by the backbeat.** In 4/4 the snare lands on beats 2 and 4 and the
kick on 1 and 3 — a fact about music, not about this code. `--validate` measures that
concentration, so the classifier is checked against something it was not fitted to.

Usage:
    python agent_mapper/percussion.py <audio.ogg>              # kit summary + timeline
    python agent_mapper/percussion.py <audio.ogg> --bars 57-72 # the groove, bar by bar
    python agent_mapper/percussion.py <audio.ogg> --validate   # the backbeat control
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

REPO = pathlib.Path(__file__).resolve().parents[1]
CACHE = REPO / "outputs" / "percussion_cache"

# (name, symbol) — the symbol is what shows up in the per-bar groove row.
PIECES = (("kick", "K"), ("snare", "S"), ("hat", "h"), ("crash", "C"))
SYM = dict(PIECES)

BANDS = ((20, 120), (120, 400), (400, 2000), (2000, 6000), (6000, 11000))


def _features(y: np.ndarray, sr: int, onsets: np.ndarray) -> np.ndarray:
    """Per-onset: 5 band energies (fraction), plus a HIGH-FREQUENCY DECAY ratio.

    The decay column is the one that earns its place. A crash and a hi-hat have almost
    identical spectra at the moment of the strike; what separates them is that 250 ms
    later the crash is still ringing and the hat is gone.

    ⚠️**The decay window must stop at the NEXT onset.** A first version measured a
    fixed 220-320 ms tail and got a median ratio of **1.5-2.0** — energy apparently
    *rising* after the strike — because at 160 bpm an eighth note is 187 ms, so the
    window was full of the following hit. It was measuring the groove, not the decay.
    Where the next onset leaves no room the value is NaN, and `classify` treats an
    unmeasurable decay as "cannot be a crash" rather than guessing.
    """
    import librosa

    n_fft, hop = 1024, 128
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bidx = [np.where((freqs >= lo) & (freqs < hi))[0] for lo, hi in BANDS]
    frame = lambda t: int(np.clip(round(t * sr / hop), 0, S.shape[1] - 1))  # noqa: E731
    hf = np.r_[bidx[3], bidx[4]]

    out = np.zeros((len(onsets), len(BANDS) + 1), dtype="float32")
    for i, t in enumerate(onsets):
        a, b = frame(t), frame(t + 0.050)
        seg = S[:, a:max(b, a + 1)].sum(axis=1)
        tot = seg.sum() + 1e-12
        out[i, :len(BANDS)] = [seg[ix].sum() / tot for ix in bidx]

        nxt = onsets[i + 1] if i + 1 < len(onsets) else t + 1.0
        tail0, tail1 = t + 0.120, min(t + 0.320, nxt - 0.020)
        if tail1 - tail0 < 0.040:
            out[i, len(BANDS)] = np.nan          # no room to hear a decay
            continue
        e0 = S[hf, a:frame(t + 0.030) + 1].sum() + 1e-12
        e1 = S[hf, frame(tail0):frame(tail1) + 1].sum()
        # per-frame means, so a longer tail window is not automatically a louder one
        n0 = max(frame(t + 0.030) - a + 1, 1)
        n1 = max(frame(tail1) - frame(tail0) + 1, 1)
        out[i, len(BANDS)] = (e1 / n1) / (e0 / n0)
    return out


def classify(f: np.ndarray, seed: int = 0) -> list[str]:
    """Band ratios -> a kit piece, by clustering the song against ITSELF.

    ⚠️**Absolute thresholds were tried first and do not transfer between mixes.** Two
    successive hand-tuned rule sets each broke somewhere different: one labelled 47 %
    of a song's hits "tom" (no kit does that), and its replacement labelled another
    song **96 % snare / 4 % kick** while dropping the backbeat control from 0.65 to
    0.57. The reason is simple — how much of a hit's energy sits below 120 Hz depends
    on the mix, and a cutoff calibrated on one record is wrong on the next.

    ★**Within one song the mix is fixed**, so cluster the hits against each other and
    then name the clusters by physics, which is the part that *is* universal:
    the lowest-centroid cluster is the kick, the longest-ringing top-heavy one is the
    crash, the shortest top-heavy one is the hat, the rest is snare/body. Nothing here
    is fitted to the backbeat, so the backbeat stays an independent control.
    """
    from sklearn.cluster import KMeans

    lo, lomid, mid, hi, vhi, decay = (f[:, i] for i in range(6))
    top, body = hi + vhi, lomid + mid
    # A brightness score: where the energy sits, on one axis. Logs because these are
    # ratios spanning orders of magnitude and a linear KMeans would see only the loud.
    L = lambda x: np.log10(x + 1e-4)                                    # noqa: E731
    d = np.nan_to_num(decay, nan=0.0, posinf=3.0)
    X = np.column_stack([L(lo), L(body), L(top), np.clip(d, 0, 3)])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    k = min(4, len(f))
    if k < 4:
        return ["snare"] * len(f)
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)

    # Name each cluster by its centroid's physics, hardest-to-mistake first.
    stats = {}
    for c in range(k):
        m = lab == c
        stats[c] = {"lo": lo[m].mean(), "top": top[m].mean(), "body": body[m].mean(),
                    "decay": float(np.nanmean(np.where(np.isfinite(decay[m]),
                                                       decay[m], np.nan))
                                   if np.isfinite(decay[m]).any() else 0.0)}
    name: dict[int, str] = {}
    kick_c = min(stats, key=lambda c: stats[c]["top"] / (stats[c]["lo"] + 1e-6))
    name[kick_c] = "kick"
    rest = [c for c in stats if c not in name]
    bright = sorted(rest, key=lambda c: -stats[c]["top"])
    if bright:
        # Of the two brightest clusters, the one that rings longer is the crash.
        cand = bright[:2]
        crash_c = max(cand, key=lambda c: stats[c]["decay"])
        hat_c = [c for c in cand if c != crash_c]
        name[crash_c] = "crash"
        if hat_c:
            name[hat_c[0]] = "hat"
    for c in stats:
        name.setdefault(c, "snare")
    return [name[c] for c in lab]


def analyse(audio: pathlib.Path, force: bool = False) -> dict:
    """Every drum onset, labelled with a kit piece. Cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{audio.stem}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())

    import brief as _brief
    from stemcache import stems, SR

    y = stems(audio)["drums"]
    on = np.asarray(_brief.analyse(audio)["onsets"]["drums"], dtype=float)
    feats = _features(y, SR, on)
    lab = classify(feats)
    # Strength: total energy at the strike, normalised — a human hears accents, and a
    # binary "an onset happened" cannot express that some hits are twice as loud.
    e = feats[:, :len(BANDS)].sum(axis=1) * 0 + 1.0        # bands are normalised
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=128)
    fi = np.clip((on * SR / 128).astype(int), 0, len(env) - 1)
    e = env[fi]
    e = e / (np.percentile(e, 95) + 1e-9)
    hits = [{"t": round(float(t), 4), "piece": p, "vel": round(float(min(v, 1.5)), 2)}
            for t, p, v in zip(on, lab, e)]
    out = {"hits": hits, "counts": {k: lab.count(k) for k, _ in PIECES}}
    f.write_text(json.dumps(out))
    return out


def groove_row(hits: list[dict], t0: float, t1: float, slot: float, n: int = 16) -> str:
    """One bar as 16 cells of kit symbols — reads like a drum tab."""
    cells = ["."] * n
    order = {p: i for i, (p, _) in enumerate(PIECES)}       # kick beats hat in a tie
    best: dict[int, str] = {}
    for h in hits:
        if t0 <= h["t"] < t1:
            i = int(round((h["t"] - t0) / slot))
            if 0 <= i < n and (i not in best or order[h["piece"]] < order[best[i]]):
                best[i] = h["piece"]
    for i, p in best.items():
        cells[i] = SYM[p]
    return "".join(cells)


def _label_grid(hits: list[dict], g: dict, n_bars: int, sub: int = 16) -> np.ndarray:
    """[n_bars, 16] of small ints: 0 = empty, 1..4 = the kit piece in that slot."""
    code = {p: i + 1 for i, (p, _) in enumerate(PIECES)}
    grid = np.zeros((n_bars, sub), dtype=np.int8)
    slot = g["bar_s"] / sub
    for h in hits:
        b = int((h["t"] - g["phase"]) // g["bar_s"])
        if 0 <= b < n_bars:
            i = int(round((h["t"] - g["phase"] - b * g["bar_s"]) / slot)) % sub
            if grid[b, i] == 0:
                grid[b, i] = code[h["piece"]]
    return grid


def groove_repetition(hits: list[dict], g: dict, n_bars: int,
                      seed: int = 0, n_null: int = 20) -> dict:
    """★The control: does the LABELLED groove repeat from bar to bar?

    ⚠️**This replaced a backbeat control that turned out to be invalid.** The first
    version tested "snare on 2 and 4", which felt like a fact about music — but
    measured over **363 human maps**, note placement by beat-of-bar is
    0.254/0.249/0.251/0.246, i.e. flat, and only 29 % of maps even peak on beat 1
    against a 25 % chance. Drum onsets in our own stems are equally flat. A control
    the *ground truth* fails is not a control.

    This asks something the data can actually answer: a real drum groove repeats, so
    if the labels are real the bar-to-bar label agreement must beat a null in which
    the same labels are **shuffled between hits with every hit time left alone**.
    That null preserves timing, label counts and density exactly, so anything left is
    the labelling itself.
    """
    rng = np.random.default_rng(seed)
    grid = _label_grid(hits, g, n_bars)
    occupied = grid != 0

    def agree(gr: np.ndarray) -> float:
        a, b = gr[:-1], gr[1:]
        m = (a != 0) | (b != 0)
        return float((a[m] == b[m]).mean()) if m.any() else float("nan")

    real = agree(grid)
    labels = [h["piece"] for h in hits]
    nulls = []
    for _ in range(n_null):
        sh = list(labels)
        rng.shuffle(sh)
        nulls.append(agree(_label_grid(
            [{**h, "piece": p} for h, p in zip(hits, sh)], g, n_bars)))
    nulls = np.array(nulls, dtype=float)
    # The ceiling: how often do two adjacent bars agree on WHERE a hit is at all?
    # Labels can never beat this, so it says how much headroom the number had.
    a, b = occupied[:-1], occupied[1:]
    ceiling = float((a == b).mean())
    return {"real": real, "null": float(np.nanmean(nulls)),
            "null_sd": float(np.nanstd(nulls)), "ceiling": ceiling,
            "z": float((real - np.nanmean(nulls)) / (np.nanstd(nulls) + 1e-9))}


def _mmss(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 60)}:{t % 60:05.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--bars", default=None)
    ap.add_argument("--validate", action="store_true", help="the backbeat control")
    ap.add_argument("--force", action="store_true")
    a_ = ap.parse_args()
    if not a_.audio.exists():
        print(f"no such audio: {a_.audio}", file=sys.stderr)
        return 2

    import brief as _brief

    res = analyse(a_.audio, a_.force)
    hits, counts = res["hits"], res["counts"]
    a = _brief.analyse(a_.audio)
    g = _brief.grid(a)
    tot = max(sum(counts.values()), 1)

    print(f"SONG {a_.audio.stem}   {g['bpm']:.2f} bpm   {tot} drum hits")
    print("KIT  " + "   ".join(f"{SYM[p]}={p} {counts[p]} ({counts[p]/tot:.0%})"
                               for p, _ in PIECES))

    if a_.validate:
        n_bars = int((a["dur"] - g["phase"]) / g["bar_s"])
        r = groove_repetition(hits, g, n_bars)
        print("\nGROOVE-REPETITION CONTROL — do the labels repeat bar to bar?")
        print(f"  bar-to-bar label agreement : {r['real']:.3f}")
        print(f"  label-shuffled null        : {r['null']:.3f} ± {r['null_sd']:.3f}")
        print(f"  where-a-hit-is ceiling     : {r['ceiling']:.3f}")
        print(f"  z = {r['z']:+.1f}")
        print("\n  " + ("✅ the labels carry real repeating structure"
                        if r["z"] > 3 else
                        "⚠️ NOT distinguishable from shuffled labels — do not trust the "
                        "kit labels on this song"))
        return 0

    if a_.bars:
        b0, b1 = (int(x) for x in a_.bars.split("-"))
        print(f"\nGROOVE bars {b0}-{b1}   K=kick S=snare/body h=hat C=crash")
        print(f"{'bar':>4} {'time':>8}  {'|1e+a2e+a3e+a4e+a':<17}")
        for bar in range(b0, b1 + 1):
            t0 = _brief.bar_time(g, bar)
            row = groove_row(hits, t0, t0 + g["bar_s"], g["slot"])
            print(f"{bar:>4} {_mmss(t0):>8}  |{row}|")
        return 0

    print(f"\nTIMELINE (8-bar phrases; hits per second by piece)")
    print(f"{'bars':>9} {'time':>8}  " + "".join(f"{SYM[p]:>6}" for p, _ in PIECES)
          + "   crash times")
    step = 8
    for b0 in range(1, g["n_bars"] + 1, step):
        t0 = _brief.bar_time(g, b0)
        t1 = min(t0 + g["bar_s"] * step, a["dur"])
        if t0 >= a["dur"]:
            break
        seg = [h for h in hits if t0 <= h["t"] < t1]
        d = {p: sum(1 for h in seg if h["piece"] == p) / max(t1 - t0, 1e-9)
             for p, _ in PIECES}
        cr = [f"{_mmss(h['t'])}" for h in seg if h["piece"] == "crash"][:3]
        print(f"{b0:>4}-{b0+step-1:<4} {_mmss(t0):>8}  "
              + "".join(f"{d[p]:>6.1f}" for p, _ in PIECES)
              + ("   " + " ".join(cr) if cr else ""))
    print("\n  ★Crashes are section markers — a crash is where a human puts a big note.")
    print("  ★Snare hits are the accents worth spending a DOUBLE on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
