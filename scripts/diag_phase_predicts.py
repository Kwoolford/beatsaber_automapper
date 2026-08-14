#!/usr/bin/env python
"""DOES THE PHASE WE ALREADY ESTIMATE PREDICT THE SHIFT THE MAP ACTUALLY WANTS?

`diag_grid_phase.py` established that 20 of the 39 alignment-failing songs are
rescued by a global time shift that their HUMAN map does not want — our grid is
genuinely misplaced. That measured an **oracle** shift (a per-song argmax against
the onsets we score on). It does not follow that we can *produce* that shift.

`generate.py` already runs `estimate_tempo`, which returns `TempoFit.phase_s`, and
then **logs it and throws it away** — the grid is anchored at t=0. The obvious fix
is to wire that number through. This script is the cheap test of whether that would
work, and it must run BEFORE any of it is built:

    if the fitted phase does not predict the oracle shift, wiring it through buys
    nothing, and the 20 songs need a different fix.

★The comparison has to be made **modulo one slot**. A grid offset of exactly one
slot is the same grid; only the remainder is a real displacement. Both quantities
are wrapped into +-half a slot before they are compared.

⚠️This re-fits tempo from CACHED onsets rather than from Demucs stems, so it is not
byte-identical to what `generate.py` computes at generation time (which fits
against freshly separated stems). It is the same estimator on the same songs and is
the right instrument for "does this signal carry the information", but a positive
result still has to survive the real path.

Usage:
    python scripts/diag_phase_predicts.py --json outputs/phase_predicts_2026-08-13.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.data.tempo import estimate_tempo  # noqa: E402
from beatsaber_automapper.evaluation import scorecard  # noqa: E402

COHORT = REPO / "outputs" / "wide_cohort"
AUDIO = COHORT / "audio"
BEAT_SUBDIV = 4


def _wrap(ms: float, slot_ms: float) -> float:
    """Wrap a displacement into +-half a slot: a whole slot is the same grid."""
    return (ms + slot_ms / 2.0) % slot_ms - slot_ms / 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-json", type=pathlib.Path,
                    default=REPO / "outputs" / "grid_phase_2026-08-13.json")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    oracle = {r["song"]: r for r in json.loads(a.phase_json.read_text())}
    stems = sorted(oracle)
    if a.limit:
        stems = stems[:a.limit]

    rows = []
    for i, song in enumerate(stems, 1):
        wav = next((p for p in AUDIO.glob(f"{song}.*")
                    if p.suffix.lower() in (".ogg", ".wav", ".mp3", ".egg")), None)
        onsets = scorecard.onsets_for(COHORT / f"{song}.zip")
        if wav is None or onsets is None or len(onsets) == 0:
            continue
        try:
            import librosa
            y, sr = librosa.load(str(wav), sr=None, mono=True)
            fit = estimate_tempo(y.astype("float32"), sr,
                                 onsets=np.asarray(onsets, dtype=np.float64))
        except Exception as exc:  # noqa: BLE001
            print(f"  {song}: fit failed ({exc})")
            continue

        slot_ms = 60.0 / fit.bpm / BEAT_SUBDIV * 1000.0
        # The grid sits at `phase_s`; anchoring at 0 displaces the map by -phase.
        fitted_ms = _wrap(-fit.phase_s * 1000.0, slot_ms)
        want_ms = _wrap(oracle[song]["ours_shift_ms"], slot_ms)
        rows.append({
            "song": song, "bpm_fit": fit.bpm, "r": fit.r, "trusted": bool(fit.trusted),
            "slot_ms": slot_ms, "fitted_shift_ms": fitted_ms, "oracle_shift_ms": want_ms,
            "err_ms": _wrap(fitted_ms - want_ms, slot_ms),
            "recovered": oracle[song]["recovered"],
            "bad": oracle[song]["bad"],
        })
        if i % 20 == 0:
            print(f"  ... {i}/{len(stems)} ({len(rows)} fitted)", flush=True)

    if not rows:
        print("nothing fitted")
        return 2

    tr = [r for r in rows if r["trusted"]]
    print(f"\n=== FITTED PHASE vs THE SHIFT THE MAP WANTS — n={len(rows)} "
          f"({len(tr)} trusted fits) ===\n")

    def report(name: str, grp: list[dict]) -> None:
        if len(grp) < 3:
            return
        err = [abs(r["err_ms"]) for r in grp]
        # The null: if the fitted phase carried no information, |err| would be the
        # mean |difference| of two independent draws on +-half a slot, = slot/3.
        chance = st.median([r["slot_ms"] for r in grp]) / 3.0
        f = [r["fitted_shift_ms"] for r in grp]
        o = [r["oracle_shift_ms"] for r in grp]
        c = (float(np.corrcoef(f, o)[0, 1]) if len(grp) > 2 and st.pstdev(f) > 0
             and st.pstdev(o) > 0 else float("nan"))
        print(f"  {name:<26} n={len(grp):>3}  median|err| {st.median(err):>6.1f} ms  "
              f"(chance {chance:>5.1f})  corr {c:>+.3f}")

    report("all", rows)
    report("trusted fits", tr)
    report("the failing songs", [r for r in rows if r["bad"]])
    report("phase-fixable (rec>0.035)", [r for r in rows if r["recovered"] > 0.035])

    big = sorted((r for r in rows if r["recovered"] > 0.10),
                 key=lambda r: -r["recovered"])[:12]
    if big:
        print("\n  the songs a shift rescues most — does the fit find it?")
        print(f"    {'song':<8}{'slot':>7}{'wants':>8}{'fitted':>8}{'err':>8}"
              f"{'R':>7}{'trust':>7}")
        for r in big:
            print(f"    {r['song']:<8}{r['slot_ms']:>7.0f}{r['oracle_shift_ms']:>8.1f}"
                  f"{r['fitted_shift_ms']:>8.1f}{r['err_ms']:>+8.1f}"
                  f"{r['r']:>7.3f}{str(r['trusted']):>7}")

    if a.json:
        out = a.json.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
