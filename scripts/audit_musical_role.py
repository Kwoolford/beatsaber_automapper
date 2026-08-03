#!/usr/bin/env python
"""Control battery for A4 (musical role) — can it see what it claims to see?

A4 returned a null on K5: our maps are *more* committed to a single stem than
human maps, the opposite of Kyle's *"trying to do the average of all of them"*.
That null is worth exactly as much as the metric behind it, and A4 has already
been rebuilt once tonight after its first version turned out to be blind (68% of
notes matched more than one stem, so entropy went near-uniform for everyone).

So before the null is quoted anywhere, four synthetic maps whose answer is known
in advance:

  follow_lead   notes placed ON the lead stem's onsets in every section.
                MUST score role_follow ~1 and high commitment. If it does not,
                the metric cannot detect perfect instrument-following and its
                null means nothing.

  follow_union  notes placed on the UNION of all stems -- deliberately "the
                average of all of them", i.e. Kyle's complaint made literal.
                MUST score LOW commitment. This is the control that decides
                whether A4 can detect the specific thing he described.

  follow_drums  notes always on drums regardless of who leads. High commitment
                but role_follow no better than chance -- the two metrics must
                come apart here, or they are measuring one thing twice.

  random_times  notes at uniform random times over the song. Both metrics must
                collapse; anything else means A4 rewards noise.

Usage:
    python scripts/audit_musical_role.py
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.data.beatmap import ColorNote  # noqa: E402

from eval_musical_role import SECTION_SEC, role_metrics, stems_for  # noqa: E402


class _BM:
    def __init__(self, notes):
        self.color_notes = notes
        self.bomb_notes: list = []


def _notes_from_times(times, bpm: float) -> list[ColorNote]:
    spb = 60.0 / bpm
    return [ColorNote(beat=t / spb, x=i % 4, y=(i // 4) % 3, color=i % 2,
                      direction=i % 8) for i, t in enumerate(sorted(times))]


def build(kind: str, stems: dict, bpm: float, n_target: int,
          rng: random.Random) -> list[ColorNote] | None:
    names = sorted(stems)
    dur = max((s.max() for s in stems.values() if len(s)), default=0.0)
    if dur <= 0:
        return None
    edges = np.arange(0.0, dur + SECTION_SEC, SECTION_SEC)
    counts = {n: np.histogram(stems[n], bins=edges)[0].astype(float) for n in names}
    base = {n: (counts[n].mean() or np.nan) for n in names}

    times: list[float] = []
    per_sec = max(3, n_target // max(1, len(edges) - 1))
    for si in range(len(edges) - 1):
        lo, hi = edges[si], edges[si + 1]
        if kind == "random_times":
            times.extend(rng.uniform(lo, hi) for _ in range(per_sec))
            continue
        if kind == "follow_union":
            pool = np.concatenate([stems[n] for n in names]) if names else np.array([])
        elif kind == "follow_drums":
            pool = stems.get("drums", np.array([]))
        else:  # follow_lead
            rel = {n: (counts[n][si] / base[n]) if base[n] == base[n] and base[n] > 0
                   else 0.0 for n in names}
            pool = stems[max(names, key=lambda n: rel[n])]
        sel = pool[(pool >= lo) & (pool < hi)]
        if len(sel) == 0:
            continue
        k = min(per_sec, len(sel))
        times.extend(float(x) for x in rng.sample(list(sel), k))
    return _notes_from_times(times, bpm) if len(times) >= 100 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12, help="songs to build controls on")
    a = ap.parse_args()

    rng = random.Random(0)
    songs = sorted(p.stem for p in (REPO / "outputs" / "stem_onset_cache").glob("*.npz"))
    kinds = ["follow_lead", "follow_drums", "follow_union", "random_times"]
    acc: dict[str, list[dict]] = {k: [] for k in kinds}
    used = 0
    for sid in songs:
        if used >= a.n:
            break
        stems = stems_for(sid)
        if not stems:
            continue
        bpm = 120.0
        ok = False
        for k in kinds:
            notes = build(k, stems, bpm, 600, rng)
            if not notes:
                continue
            r = role_metrics(_BM(notes), bpm, stems)
            if r:
                acc[k].append(r)
                ok = True
        used += int(ok)

    print(f"=== A4 CONTROL BATTERY (n={used} songs) ===\n")
    print(f"{'control':16s}{'role_follow':>14s}{'role_commitment':>18s}")
    print("-" * 48)
    for k in kinds:
        if not acc[k]:
            print(f"{k:16s}{'--':>14s}{'--':>18s}")
            continue
        f = float(np.median([r["role_follow"] for r in acc[k]]))
        c = float(np.median([r["role_commitment"] for r in acc[k]]))
        print(f"{k:16s}{f:>14.4f}{c:>18.4f}")
    print("\nmeasured maps for reference:  ours 0.2778 / 0.2325   human 0.3067 / 0.1877")
    print("\n--- READ ---")
    print("follow_lead   must score HIGH on both, or A4 cannot see instrument-following")
    print("              and its K5 null is worthless.")
    print("follow_union  is Kyle's complaint made literal ('the average of all of")
    print("              them'). Its commitment must be clearly BELOW follow_lead,")
    print("              or A4 cannot detect the thing he described.")
    print("follow_drums  must show HIGH commitment but chance-level follow -- if the")
    print("              two move together, they are one metric wearing two hats.")
    print("random_times  must collapse both.")


if __name__ == "__main__":
    main()
