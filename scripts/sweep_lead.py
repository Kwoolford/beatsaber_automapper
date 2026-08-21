#!/usr/bin/env python
"""Price the per-phrase lead hand against `role_asymmetry` -- and its guard.

P0.6: with the pulse fix in, every remaining tail is hand-role. `role_asymmetry`
(mean |L-R|/(L+R) in a 2-bar window) is **1.1st human percentile on 21 of 23 maps**
-- humans are globally balanced but LOCALLY lopsided, and we split every window
evenly.

★**`role_swap_rate` is the guard and it must be read at the same time.** A map where
one hand always leads is lopsided but not human, and `handrole.py` says so in its own
docstring. So this sweep prints both, plus `ebpm_burst` -- the cost the 2026-08-14
session measured when it pinned `--runs` to 1 -- and the note count.

Deliberately does NOT call `mapjudge`: this runs while the judge's reference is being
recalibrated, and scoring against a file that is being rewritten mid-sweep is the
same class of error as editing generate.py during a run.

Usage:
    python scripts/sweep_lead.py --songs 1f767 1f8d6 --bias 0 0.25 0.4 0.6
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
sys.path.insert(0, str(REPO / "scripts"))

# Human values quoted in evaluation/handrole.py and the 2026-08-14 ebpm work.
HUMAN = {"role_asymmetry": 0.11, "role_swap_rate": 0.41, "ebpm_burst": 376.0}
KEYS = ("role_asymmetry", "role_swap_rate", "role_run_len", "ebpm_burst")


def metrics(zp: pathlib.Path) -> dict | None:
    from audit_eval_suite import _load_generated, _load_human
    from beatsaber_automapper.evaluation import flow, handrole, mapjudge as mj
    notes = None
    for loader in (_load_human, _load_generated):
        try:
            got = loader(zp)
        except Exception:  # noqa: BLE001
            continue
        if got and got[0]:
            notes, bpm = got
            break
    if not notes:
        return None
    bm = mj._BM(notes)
    m = dict(handrole.handrole_metrics(bm).metrics)
    try:
        m.update(flow.flow_metrics(bm, bpm=bpm).metrics)
    except Exception:  # noqa: BLE001
        pass
    m["n_notes"] = float(len(notes))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="+", required=True)
    ap.add_argument("--bias", nargs="+", type=float, default=[0.0, 0.25, 0.4, 0.6])
    ap.add_argument("--lead-phrase-bars", type=int, default=4)
    a = ap.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="leadsweep_"))
    rows: dict[float, list[dict]] = {b: [] for b in a.bias}
    human: list[dict] = []
    for sid in a.songs:
        audio = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        for b in a.bias:
            out = tmp / f"lead{b}__{sid}.zip"
            cmd = [sys.executable, str(AM / "autobuild.py"), str(audio),
                   "--name", f"ld_{sid}_{str(b).replace('.', '')}", "--pulse",
                   "--no-idiomize", "--out", str(out),
                   "--lead-bias", str(b),
                   "--lead-phrase-bars", str(a.lead_phrase_bars)]
            subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
            if out.exists():
                m = metrics(out)
                if m:
                    rows[b].append(m)
        hz = REPO / "data" / "raw" / f"{sid}.zip"
        if hz.exists():
            m = metrics(hz)
            if m:
                human.append(m)

    print(f"\n{'lead-bias':<12}" + "".join(f"{k[:15]:>17}" for k in KEYS) + f"{'n':>8}")
    print("-" * 90)
    for b in a.bias:
        rs = rows[b]
        if not rs:
            continue
        cells = [statistics.median([r[k] for r in rs if r.get(k) == r.get(k)])
                 if any(r.get(k) == r.get(k) for r in rs) else float("nan")
                 for k in KEYS]
        n = statistics.median([r["n_notes"] for r in rs])
        print(f"{b:<12.2f}" + "".join(f"{c:>17.3f}" for c in cells) + f"{n:>8.0f}")
    if human:
        cells = [statistics.median([r[k] for r in human if r.get(k) == r.get(k)])
                 if any(r.get(k) == r.get(k) for r in human) else float("nan")
                 for k in KEYS]
        n = statistics.median([r["n_notes"] for r in human])
        print(f"{'HUMAN':<12}" + "".join(f"{c:>17.3f}" for c in cells) + f"{n:>8.0f}")
    print("\n★Read role_asymmetry WITH role_swap_rate: a map where one hand always "
          "leads is lopsided but not human.\n⚠️ebpm_burst is the cost --runs was "
          "pinned to 1 for; it must stay inside its human band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
