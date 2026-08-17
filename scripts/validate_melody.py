#!/usr/bin/env python
"""Control battery for `agent_mapper/melody.py` — is the pitch real or is it noise?

The project rule is that a new perception axis passes a control before it steers
anything. Melody has no ground truth here (no MIDI for these songs), so this asks
three questions that noise cannot pass by accident:

1. **Mean step size.** A sung melody moves by 1-3 semitones. A tracker locking onto
   chords, bleed or partials moves much further. (Use the MEAN — the median of an
   integer-semitone interval is 1.0 for signal and noise alike.)
2. **★Cross-stem key agreement.** `vocals` (pYIN) and `other` (CQT salience) are two
   different algorithms on two different audio sources. If both independently report
   the same key, the pitches are real — a shared error mode would have to be a
   coincidence at the level of the whole song.
3. **The shuffle null.** Re-run the pitch assignment with the onset times SHUFFLED
   within the song. Real melodies get much worse (a note's pitch belongs to its own
   time); if the null scores the same, the tool is reading the stem's average and not
   its melody.

    python scripts/validate_melody.py data/eval_songset/*.ogg
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent_mapper"))
sys.path.insert(0, str(REPO / "src"))

import brief as _brief                      # noqa: E402
import melody as _mel                       # noqa: E402
from stemcache import stems, SR             # noqa: E402


def _stepstat(midis: list[int]) -> float:
    """Mean |interval|, NOT the median.

    ⚠️The first version of this control used the median and could not discriminate at
    all: intervals are integer semitones, so the median came out 1.0 for a real melody
    *and* 1.0 for the shuffled null — the statistic was quantised flat before it ever
    met the data. `step < null` scored 7 % of songs, which looked like a refutation and
    was really a blunt ruler. The mean keeps the tail, which is exactly where a
    shuffled melody gives itself away: in the big wrong leaps.
    """
    d = np.abs(np.diff(np.asarray(midis, dtype=float)))
    return float(d.mean()) if len(d) else float("nan")


def one(audio: pathlib.Path, rng: np.random.Generator) -> dict | None:
    res = _mel.analyse(audio)
    on_all = _brief.analyse(audio)["onsets"]
    s_ = stems(audio)
    row: dict = {"song": audio.stem}
    keys = {}
    for name in _mel.MELODIC:
        ev, meta = res["stems"][name], res["meta"][name]
        if len(ev) < 20:
            row[name] = None
            continue
        k, r = _mel.key_of(ev)
        keys[name] = k
        step = _stepstat([e["midi"] for e in ev])

        # --- the null: same audio, same tracker, onsets shuffled in time ---
        t, midi, voiced = (_mel._track_vocals if name == "vocals"
                           else _mel._track_salience)(s_[name], SR)
        on = np.asarray(on_all[name], dtype=float)
        lo, hi = float(on.min()), float(on.max())
        fake = np.sort(rng.uniform(lo, hi, size=len(on)))
        nev = _mel.pitch_at_onsets(fake, t, midi, voiced)
        _mel._fix_octaves(nev)
        nstep = _stepstat([e["midi"] for e in nev]) if len(nev) > 2 else float("nan")
        row[name] = {"cov": meta["coverage"], "step": step, "null_step": nstep,
                     "key": k, "r": r}
    row["key_agree"] = (len(keys) == 2 and keys.get("vocals") == keys.get("other"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+", type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    print(f"{'song':>8} | {'VOCALS cov':>10} {'step':>5} {'null':>5} {'key':>9} "
          f"| {'OTHER cov':>10} {'step':>5} {'null':>5} {'key':>9} | agree")
    rows = []
    for p in a.audio:
        try:
            r = one(p, rng)
        except Exception as e:                       # noqa: BLE001
            print(f"{p.stem:>8} | FAILED {type(e).__name__}: {e}")
            continue
        rows.append(r)
        cells = []
        for name in _mel.MELODIC:
            d = r.get(name)
            cells.append("      —          —     —         —" if not d else
                         f"{d['cov']:>10.2f} {d['step']:>5.2f} {d['null_step']:>5.2f} "
                         f"{d['key']:>9}")
        print(f"{r['song']:>8} | " + " | ".join(cells) + f" | {'YES' if r['key_agree'] else '.'}")

    print("\n--- VERDICT ---")
    for name in _mel.MELODIC:
        ds = [r[name] for r in rows if r.get(name)]
        if not ds:
            continue
        step = np.array([d["step"] for d in ds])
        null = np.array([d["null_step"] for d in ds])
        cov = np.array([d["cov"] for d in ds])
        ok = (step < null).mean()
        print(f"{name:>7}: coverage {cov.mean():.2f} (min {cov.min():.2f})   "
              f"step {np.median(step):.2f} vs null {np.median(null):.2f}   "
              f"step<null on {ok*100:.0f}% of songs")
    agree = np.mean([r["key_agree"] for r in rows if r.get("vocals") and r.get("other")])
    n_ag = sum(1 for r in rows if r.get("vocals") and r.get("other"))
    print(f"cross-stem key agreement: {agree*100:.0f}% of {n_ag} songs "
          f"(chance = 1/24 = 4%)")
    print("\nDoD: step < null on >=80% of songs AND key agreement >> 4%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
