#!/usr/bin/env python
"""Sweep the `--pulse` knobs against the four pulse metrics AND the human values.

The first `--pulse` build overshot: `pulse_stability` 0.848 at the **98th** human
percentile against a human 0.514, and `ioi_switch_rate` 1.33 at the **0.5th** (human
14.8). It went from too loose to too rigid without passing through human.

★**All four metrics are read two-sided and together.** `pulse_stability` alone is
trivially Goodharted -- a metronome maxes it -- so an arm only counts as better if it
moves toward the human on the sequence metrics *and* keeps its note count. That is
the P0.5 DoD, and it is why `n_notes` is in the table.

Usage:
    python scripts/sweep_pulse.py --songs 1f767 1f8d6 --phrase-bars 1 2 4
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AM))
sys.path.insert(0, str(REPO / "scripts"))

KEYS = ("pulse_stability", "dominant_share", "ioi_switch_rate",
        "ioi_cond_entropy", "shuffle_lift")


def build(audio: pathlib.Path, name: str, out: pathlib.Path, extra: list[str]) -> bool:
    cmd = [sys.executable, str(AM / "autobuild.py"), str(audio), "--name", name,
           "--no-idiomize", "--out", str(out), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    return out.exists()


def metrics(zp: pathlib.Path) -> dict | None:
    from audit_eval_suite import _load_generated, _load_human
    from beatsaber_automapper.evaluation import mapjudge as mj
    from beatsaber_automapper.evaluation import rhythm
    notes = None
    for loader in (_load_human, _load_generated):
        try:
            got = loader(zp)
        except Exception:  # noqa: BLE001
            continue
        if got and got[0]:
            notes = got[0]
            break
    if not notes:
        return None
    m = dict(rhythm.rhythm_metrics(mj._BM(notes)).metrics)
    m["n_notes"] = float(len(notes))
    # Lift over the map's OWN shuffled IOI sequence: the ordering term of the P0.5
    # decomposition. A shuffle keeps the histogram exactly, so this cannot be moved
    # by changing which intervals are used -- only by holding them.
    import random
    import statistics as _st
    beats = sorted({round(n.beat, 4) for n in notes})
    d = [round(b - a, 3) for a, b in zip(beats, beats[1:])]
    d = [x for x in d if 0 < x <= 4.0]
    if len(d) >= 10:
        def _p(seq):
            return _st.fmean([1.0 if abs(x - y) < 1e-9 else 0.0
                              for x, y in zip(seq, seq[1:])])
        rng = random.Random(0)
        work = list(d)
        nulls = []
        for _ in range(40):
            rng.shuffle(work)
            nulls.append(_p(work))
        m["shuffle_lift"] = _p(d) - _st.fmean(nulls)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="+", required=True)
    ap.add_argument("--phrase-bars", nargs="+", type=int, default=[1, 2, 4])
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()
    dists = ref.get("distributions", {})

    arms: dict[str, list[str]] = {"CONTROL": []}
    for pb in a.phrase_bars:
        arms[f"pulse pb={pb}"] = ["--pulse", "--phrase-bars", str(pb)]

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pulsesweep_"))
    rows: dict[str, list[dict]] = {k: [] for k in arms}
    human: list[dict] = []
    for sid in a.songs:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        for arm, extra in arms.items():
            out = tmp / f"{arm.replace(' ', '').replace('=', '')}__{sid}.zip"
            if build(audio, f"sw_{sid}_{abs(hash(arm)) % 9999}", out, extra):
                m = metrics(out)
                if m:
                    rows[arm].append(m)
        hz = REPO / "data" / "raw" / f"{sid}.zip"
        if hz.exists():
            m = metrics(hz)
            if m:
                human.append(m)

    def pct(name: str, val: float) -> str:
        vals = dists.get(name)
        if not vals:
            return "  --"
        return f"{100 * mj.percentile_of(val, vals):4.1f}%"

    print(f"\n{'arm':<14}" + "".join(f"{k[:13]:>16}" for k in KEYS) + f"{'n_notes':>9}")
    print("-" * 90)
    for arm in (*arms, "HUMAN"):
        rs = human if arm == "HUMAN" else rows[arm]
        if not rs:
            continue
        cells = []
        for k in KEYS:
            v = statistics.median([r[k] for r in rs if r.get(k) == r.get(k)])
            cells.append(f"{v:8.3f}{pct(k, v):>8}" if arm != "HUMAN"
                         else f"{v:8.3f}{'':>8}")
        n = statistics.median([r["n_notes"] for r in rs])
        print(f"{arm:<14}" + "".join(f"{c:>16}" for c in cells) + f"{n:>9.0f}")
    print("\npercentiles are against the 1100-map human reference; 50% is the human "
          "median.\n★Read every column two-sided — 98% is as wrong as 2%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
