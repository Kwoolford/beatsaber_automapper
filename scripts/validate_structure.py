#!/usr/bin/env python
"""Control battery for `agent_mapper/structure.py` — are the section LETTERS real?

The boundaries are checkable by eye; the letters are the claim that needs a control,
because "map section B once and reuse it" is only sound advice if the two Bs really are
the same music.

**The ground truth is the lyrics.** Whisper's transcript knows nothing about chroma,
MFCCs, self-similarity or clustering, so if a line sung three times lands under the same
letter all three times, that agreement is independent evidence. The null re-draws the
section letters while holding the boundaries and the letter frequencies fixed.

⚠️**Pooled across songs, deliberately.** Run per song, this test is underpowered — a
song with four repeated lines cannot produce a permutation p below ~0.14 no matter how
perfect the labelling, so seven weak per-song tests answer nothing. Pooling every
repeated-line pair in the cohort into ONE permutation asks the question the cohort can
actually answer.

⚠️**1f333 is the tuning song** (its labelling threshold was set against its own lyric
repeats) and is excluded from the headline number by default.

    python scripts/validate_structure.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent_mapper"))
sys.path.insert(0, str(REPO / "src"))

import brief as _brief          # noqa: E402
import structure as _st         # noqa: E402

TUNED_ON = {"1f333"}


def pairs_for(song: str) -> tuple[list, list]:
    """(section letters in order, list of repeated-line time-pairs) for one song."""
    audio = REPO / "data" / "eval_songset" / f"{song}.ogg"
    secs = _st.analyse(audio)["sections"]
    lines = _brief.lyric_lines(song)
    groups: dict[str, list[float]] = {}
    for ln in lines:
        key = "".join(c for c in ln["text"].lower() if c.isalnum() or c == " ").strip()
        if len(key) >= 6:
            groups.setdefault(key, []).append(ln["t"])

    def idx_at(t: float) -> int | None:
        for i, s in enumerate(secs):
            if s["t0"] <= t < s["t1"]:
                return i
        return None

    prs = []
    for ts in groups.values():
        if len(ts) < 2:
            continue
        ids = [idx_at(t) for t in ts]
        ids = [i for i in ids if i is not None]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                prs.append((ids[i], ids[j]))
    return [s["label"] for s in secs], prs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-tuned", action="store_true")
    ap.add_argument("--n-null", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    cache = REPO / "outputs" / "lyrics_cache"
    songs = sorted(p.stem for p in cache.glob("*.json"))

    per, kept = [], []
    print(f"{'song':>8} {'secs':>5} {'distinct':>9} {'pairs':>6} {'agree':>6} {'null':>6}")
    for s in songs:
        try:
            labels, prs = pairs_for(s)
        except Exception as e:                                   # noqa: BLE001
            print(f"{s:>8}   FAILED {type(e).__name__}: {e}")
            continue
        if not prs:
            print(f"{s:>8} {len(labels):>5} {len(set(labels)):>9} {0:>6}"
                  "    — no repeated lyric lines, cannot test")
            continue
        real = float(np.mean([labels[i] == labels[j] for i, j in prs]))
        nulls = []
        for _ in range(200):
            sh = list(labels)
            rng.shuffle(sh)
            nulls.append(np.mean([sh[i] == sh[j] for i, j in prs]))
        tag = "  (TUNED)" if s in TUNED_ON else ""
        print(f"{s:>8} {len(labels):>5} {len(set(labels)):>9} {len(prs):>6} "
              f"{real:>6.3f} {float(np.mean(nulls)):>6.3f}{tag}")
        if s in TUNED_ON and not a.include_tuned:
            continue
        per.append((labels, prs))
        kept.append(s)

    if not per:
        print("\nno songs to test")
        return 1

    # --- ONE pooled permutation over every repeated-line pair in the cohort ---
    def pooled(orders: list[list]) -> float:
        hit = tot = 0
        for (labels, prs), lb in zip(per, orders):
            for i, j in prs:
                tot += 1
                hit += lb[i] == lb[j]
        return hit / max(tot, 1)

    real = pooled([lb for lb, _ in per])
    nulls = np.empty(a.n_null)
    for k in range(a.n_null):
        orders = []
        for labels, _ in per:
            sh = list(labels)
            rng.shuffle(sh)
            orders.append(sh)
        nulls[k] = pooled(orders)
    p = float((nulls >= real).mean())
    n_pairs = sum(len(prs) for _, prs in per)

    print(f"\n--- POOLED over {len(kept)} held-out songs ({', '.join(kept)}) ---")
    print(f"  repeated-line pairs      : {n_pairs}")
    print(f"  same-letter rate         : {real:.3f}")
    print(f"  label-shuffled null      : {nulls.mean():.3f} ± {nulls.std():.3f}")
    print(f"  p(null >= real)          : {p:.4f}   (n_null={a.n_null})")
    print("\n  " + ("✅ SECTION LABELS CONFIRMED — repeated words land under the same "
                    "letter far more than chance, on songs the threshold never saw."
                    if p < 0.05 else
                    "⚠️ NOT CONFIRMED on held-out songs — the letters may not be "
                    "trustworthy; treat section reuse as a hint, not a fact."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
