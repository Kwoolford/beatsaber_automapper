#!/usr/bin/env python
"""Audit the EVALUATION SUITE itself, using degenerate control maps.

Motivation (2026-07-26, Kyle's redirect): the goal is an evaluation suite good
enough to replace the human judge — and good enough to *specify* a mapper, so an
agent could build one without ML and we could audit the architecture. Before
adding metrics, we have to know where the current suite is BLIND.

Method (standard practice for any classifier: test it on cases it must fail):
score real human maps, our production maps, and a battery of DELIBERATELY BAD
control maps with the current scorecard. A metric earns its place only if it
ranks human maps above the degenerate controls.

The controls are constructed to isolate one failure each:

  random        uniform-random (x, y, dir) at human note times. Maximal variety,
                zero musical intent. NOTE: our current diversity metrics
                (grid_coverage, dir_entropy, low row_conc/pattern_repeat) are all
                *maximized* by this map — it is the adversarial case for the
                "more diverse = more human" assumption baked into the suite.
  shuffled      human map with its (x, y, dir) triples randomly PERMUTED. This
                has byte-identical marginal distributions to the human map — same
                row/col/dir histograms, same nps — but every sequential relation
                (flow, pattern, parity) is destroyed. Any metric that scores this
                equal to human is blind to SEQUENCING, which is most of mapping.
  metronome     one single (x, y, dir) repeated at a constant interval. The
                "for-sport" degenerate: perfectly consistent, unplayably dull.
  zigzag        the classic V7 failure mode we already fixed — bottom row, two
                columns, up/down alternating. Included as a regression control:
                the suite MUST still rank this below human.

Usage:
  python scripts/audit_eval_suite.py --n 12
  python scripts/audit_eval_suite.py --n 12 --json outputs/eval_audit.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import random
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from beatsaber_automapper.data.beatmap import ColorNote  # noqa: E402
from beatsaber_automapper.evaluation import flow as fl  # noqa: E402
from beatsaber_automapper.evaluation import handrole as hro  # noqa: E402
from beatsaber_automapper.evaluation import idiom as idm  # noqa: E402
from beatsaber_automapper.evaluation import rhythm as rh  # noqa: E402
from beatsaber_automapper.evaluation import swing_sim as ss  # noqa: E402
from map_metrics import HUMAN_TARGET, map_metrics_from_seq  # noqa: E402

RAW = REPO / "data" / "raw"
CACHE = REPO / "outputs" / "eval_sweep_cache"

# Metrics the current scorecard reports, and how it interprets them.
# "human" = closer to the human value is better; high/low = monotone preference
# (which is exactly the assumption the `random` control is designed to break).
AXES = ["row_conc", "col_conc", "grid_coverage", "dir_entropy", "monotony",
        "pattern_repeat", "nps"]

# v2 axis A1 (evaluation/flow.py) — sequence-aware, so unlike the AXES above these
# are NOT invariant to shuffling the notes. `flow_dist` is the composite.
FLOW_AXES = ["angle_change", "angle_harsh_frac", "travel", "crossover",
             "handedness", "ebpm_burst", "flow_dist"]

# v2 axis A2 (evaluation/rhythm.py) — computed over note TIMES, so the only
# controls that can move these are the timing_* ones.
RHYTHM_AXES = ["pulse_stability", "ioi_cond_entropy", "ioi_switch_rate",
               "dominant_share", "ioi_entropy", "offgrid_frac"]

# v2 axis A3 (evaluation/idiom.py) — per-hand transitions drawn from the mined
# human vocabulary. Sequence-aware, so the attribute controls attack it.
IDIOM_AXES = ["idiom_coverage", "idiom_top50", "idiom_jsd", "idiom_entropy"]

# v2 axis A6 (evaluation/handrole.py) — do the two hands take different musical
# roles within a passage? Found by READING maps, not by any statistic.
HANDROLE_AXES = ["role_asymmetry", "role_swap_rate", "role_run_len"]

# The five metrics eval_sweep.py actually composites into `human_dist` — the
# scalar the sweep leaderboard ranks arms by. Reproduced here so the audit can
# report what the ranking metric itself would say about each control.
H_DIST_KEYS = ["row_conc", "col_conc", "grid_coverage", "dir_entropy", "monotony"]


def human_dist(rec_means: dict) -> float:
    """eval_sweep's composite: mean |arm - human| / human. Lower = 'more human'."""
    d = [abs(rec_means[k] - HUMAN_TARGET[k]) / abs(HUMAN_TARGET[k])
         for k in H_DIST_KEYS
         if rec_means.get(k) is not None and HUMAN_TARGET.get(k)]
    return float(np.mean(d)) if d else float("nan")


# --------------------------------------------------------------------------
# note-list <-> feature-seq plumbing
# --------------------------------------------------------------------------
def _seq_from_notes(notes: list[ColorNote], bpm: float) -> np.ndarray:
    """[L,12] feel-disc feature seq: dt(s, capped 2), x/3, y/2, dir one-hot(9)."""
    if not notes:
        return np.zeros((0, 12), dtype=np.float32)
    notes = sorted(notes, key=lambda n: n.beat)
    spb = 60.0 / bpm if bpm > 0 else 0.5
    t = np.array([n.beat * spb for n in notes], dtype=np.float64)
    dt = np.diff(t, prepend=t[0])
    seq = np.zeros((len(notes), 12), dtype=np.float32)
    seq[:, 0] = np.clip(dt, 0.0, 2.0)
    seq[:, 1] = np.array([n.x for n in notes], dtype=np.float32) / 3.0
    seq[:, 2] = np.array([n.y for n in notes], dtype=np.float32) / 2.0
    for i, n in enumerate(notes):
        seq[i, 3 + int(np.clip(n.direction, 0, 8))] = 1.0
    return seq


def _load_human(zip_path: pathlib.Path) -> tuple[list[ColorNote], float] | None:
    """Human Standard map notes + bpm, preferring Expert then ExpertPlus."""
    import shutil
    import tempfile
    import zipfile

    from beatsaber_automapper.data.beatmap import parse_difficulty_dat, parse_info_dat

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="audit_suite_"))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            # EXACT basename: "BPMInfo.dat" also ends with "info.dat", and 73 of 300
            # corpus zips list it FIRST -- picking it makes parse_info_dat find no
            # bpm and silently fall back to 120, which stretches every note time.
            info = next((n for n in names
                         if n.split("/")[-1].lower() == "info.dat"), None)
            std = [n for n in names if n.lower().split("/")[-1].endswith("standard.dat")]
            diff = None
            for cand in ("expert", "expertplus"):
                for n in std:
                    b = n.lower().split("/")[-1]
                    if b.startswith(cand) and "plus" not in b.replace(cand, "", 1):
                        diff = n
                        break
                if diff:
                    break
            if info is None or diff is None:
                return None
            for n in (info, diff):
                (tmp / pathlib.Path(n).name).write_bytes(zf.read(n))
        meta = parse_info_dat(tmp / pathlib.Path(info).name)
        bm = parse_difficulty_dat(tmp / pathlib.Path(diff).name)
        if meta is None or bm is None or not bm.color_notes:
            return None
        return list(bm.color_notes), float(meta.bpm)
    except Exception:  # noqa: BLE001
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _load_generated(zip_path: pathlib.Path) -> tuple[list[ColorNote], float] | None:
    """Notes + bpm from one of OUR generated map zips."""
    sys.path.insert(0, str(REPO / "scripts"))
    from eval_contour_follow import _load_notes_with_direction  # noqa: PLC0415
    from feel_disc_poc import _zip_bpm  # noqa: PLC0415
    try:
        recs = _load_notes_with_direction(zip_path, "Expert")
    except Exception:  # noqa: BLE001
        return None
    if not recs:
        return None
    notes = [ColorNote(beat=b, x=int(x), y=int(y), color=int(c), direction=int(d))
             for (b, x, y, c, d) in recs]
    return notes, float(_zip_bpm(str(zip_path)) or 120.0)


# --------------------------------------------------------------------------
# degenerate controls — each keeps the human note TIMES, varies what's placed
# --------------------------------------------------------------------------
def make_random(notes: list[ColorNote], rng: random.Random) -> list[ColorNote]:
    return [ColorNote(beat=n.beat, x=rng.randrange(4), y=rng.randrange(3),
                      color=rng.randrange(2), direction=rng.randrange(9))
            for n in notes]


def make_shuffled(notes: list[ColorNote], rng: random.Random) -> list[ColorNote]:
    """Identical marginals to the human map; all sequencing destroyed."""
    attrs = [(n.x, n.y, n.color, n.direction) for n in notes]
    rng.shuffle(attrs)
    return [ColorNote(beat=n.beat, x=a[0], y=a[1], color=a[2], direction=a[3])
            for n, a in zip(notes, attrs)]


def make_metronome(notes: list[ColorNote], rng: random.Random) -> list[ColorNote]:
    """One cell, one direction, constant interval."""
    if not notes:
        return []
    step = float(np.median(np.diff([n.beat for n in notes]))) if len(notes) > 1 else 0.5
    step = max(step, 1e-3)
    b0 = notes[0].beat
    return [ColorNote(beat=b0 + i * step, x=1, y=0, color=0, direction=1)
            for i in range(len(notes))]


def make_zigzag(notes: list[ColorNote], rng: random.Random) -> list[ColorNote]:
    """The original V7 failure mode: bottom row, 2 columns, up/down alternating."""
    out = []
    for i, n in enumerate(notes):
        red = i % 2 == 0
        out.append(ColorNote(beat=n.beat, x=0 if red else 2, y=0,
                             color=0 if red else 1, direction=0 if red else 1))
    return out


def make_timing_random(notes: list[ColorNote], rng: random.Random) -> list[ColorNote]:
    """Human patterns, note TIMES randomised over the same span.

    Every control above preserves the human note times, so no rhythm metric can
    possibly catch them — rhythm is invisible to attribute-shuffling. Axis A2
    needs a control that destroys timing and nothing else.
    """
    if not notes:
        return []
    lo, hi = notes[0].beat, notes[-1].beat
    beats = sorted(rng.uniform(lo, hi) for _ in notes)
    # keep the 1/16 grid so this tests RHYTHM, not off-grid placement
    beats = [round(b * 16.0) / 16.0 for b in beats]
    return [ColorNote(beat=b, x=n.x, y=n.y, color=n.color, direction=n.direction)
            for b, n in zip(beats, notes)]


def make_timing_jitter(notes: list[ColorNote], rng: random.Random) -> list[ColorNote]:
    """Human map nudged OFF the beat grid — tests the offgrid guard."""
    return [ColorNote(beat=n.beat + rng.uniform(-0.04, 0.04), x=n.x, y=n.y,
                      color=n.color, direction=n.direction) for n in notes]


CONTROLS = {
    "random": make_random,
    "shuffled": make_shuffled,
    "metronome": make_metronome,
    "zigzag": make_zigzag,
    "timing_random": make_timing_random,
    "timing_jitter": make_timing_jitter,
}


class _BM:
    """Minimal DifficultyBeatmap shim for swing_sim.simulate."""

    def __init__(self, notes):
        self.color_notes = notes
        self.bomb_notes = []


def score(notes: list[ColorNote], bpm: float) -> dict:
    rec = map_metrics_from_seq(_seq_from_notes(notes, bpm))
    bm = _BM(sorted(notes, key=lambda n: n.beat))
    try:
        rec["viol"] = int(ss.simulate(bm, bpm=bpm).violations)
    except Exception:  # noqa: BLE001
        rec["viol"] = None
    try:
        frep = fl.flow_metrics(bm, bpm=bpm)
        rec.update(frep.metrics)
        rec["flow_dist"] = frep.flow_dist
    except Exception:  # noqa: BLE001
        pass
    try:
        rec.update(rh.rhythm_metrics(bm).metrics)
    except Exception:  # noqa: BLE001
        pass
    try:
        rec.update(idm.idiom_metrics(bm).metrics)
    except Exception:  # noqa: BLE001
        pass
    try:
        rec.update(hro.handrole_metrics(bm).metrics)
    except Exception:  # noqa: BLE001
        pass
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12, help="number of human maps to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write full results here")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    raws = sorted(RAW.glob("*.zip"))
    rng.shuffle(raws)

    cohorts: dict[str, list[dict]] = {k: [] for k in ("human", *CONTROLS)}
    used = 0
    for zp in raws:
        if used >= a.n:
            break
        loaded = _load_human(zp)
        if loaded is None:
            continue
        notes, bpm = loaded
        if len(notes) < 100:
            continue
        cohorts["human"].append(score(notes, bpm))
        for name, fn in CONTROLS.items():
            cohorts[name].append(score(fn(notes, rng), bpm))
        used += 1

    # our production maps, for reference — scored through the same `score()` path
    # as every other cohort so the comparison is apples-to-apples
    prod = []
    for m in sorted(glob.glob(str(CACHE / "prod__*.zip")))[: a.n]:
        loaded = _load_generated(pathlib.Path(m))
        if loaded is None:
            continue
        notes, bpm = loaded
        prod.append(score(notes, bpm))
    if prod:
        cohorts["prod(ours)"] = prod

    def mean(rows, k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    order = ["human", "prod(ours)", "random", "shuffled", "metronome", "zigzag",
             "timing_random", "timing_jitter"]
    order = [o for o in order if cohorts.get(o)]

    print(f"=== EVAL-SUITE AUDIT — {used} human maps + degenerate controls ===\n")
    hdr = f"{'cohort':14s}" + "".join(f"{k:>15s}" for k in AXES) + f"{'viol':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name in order:
        rows = cohorts[name]
        line = f"{name:14s}" + "".join(f"{mean(rows, k):15.3f}" for k in AXES)
        line += f"{mean(rows, 'viol'):8.1f}"
        print(line)
    print(f"\n{'(human target)':14s}" +
          "".join(f"{HUMAN_TARGET.get(k) or float('nan'):15.3f}" for k in AXES))

    # ---- v2 axis A1: flow / ergonomics (sequence-aware) ----
    if any(cohorts[n] and cohorts[n][0].get("flow_dist") is not None for n in order):
        print("\n\n=== v2 axis A1 — FLOW / ERGONOMICS (sequence-aware) ===")
        print("flow_dist = mean |robust z| vs the human corpus; LOWER = more human\n")
        fhdr = f"{'cohort':14s}" + "".join(f"{k:>18s}" for k in FLOW_AXES)
        print(fhdr)
        print("-" * len(fhdr))
        for name in order:
            rows = cohorts[name]
            print(f"{name:14s}" + "".join(f"{mean(rows, k):18.3f}" for k in FLOW_AXES))

    # ---- cohort-level distribution comparison (the mode-collapse-proof view) ----
    if cohorts.get("prod(ours)"):
        print("\n\n=== A1 cohort comparison vs the human DISTRIBUTION ===")
        print("shift  = (cohort median - human median) / human MAD  (0 = human-like)")
        print("spread = cohort MAD / human MAD  (<1 = under-dispersed / mode-collapsed)\n")
        chdr = (f"{'cohort':14s}" +
                "".join(f"{k:>22s}" for k in fl.SEQUENCE_KEYS) +
                f"{'flow_gap':>11s}{'min_spread':>12s}")
        print(chdr)
        print("-" * len(chdr))
        for name in order:
            cc = fl.cohort_comparison(cohorts[name])
            if "_summary" not in cc:
                continue
            cells = "".join(
                f"{cc[k]['shift']:+10.2f}/{cc[k]['spread']:<11.2f}" if k in cc
                else f"{'--':>22s}" for k in fl.SEQUENCE_KEYS)
            s = cc["_summary"]
            print(f"{name:14s}{cells}{s['flow_gap']:11.2f}{s['min_spread']:12.2f}")

    # ---- v2 axis A2: rhythm (only the timing_* controls can move these) ----
    print("\n\n=== A2 RHYTHM cohort comparison vs the human DISTRIBUTION ===")
    rhdr = (f"{'cohort':14s}" + "".join(f"{k:>22s}" for k in rh.SEQUENCE_KEYS)
            + f"{'rhythm_gap':>13s}{'min_spread':>12s}")
    print(rhdr)
    print("-" * len(rhdr))
    for name in order:
        cc = rh.cohort_comparison(cohorts[name])
        if "_summary" not in cc:
            continue
        cells = "".join(
            f"{cc[k]['shift']:+10.2f}/{cc[k]['spread']:<11.2f}" if k in cc
            else f"{'--':>22s}" for k in rh.SEQUENCE_KEYS)
        s = cc["_summary"]
        print(f"{name:14s}{cells}{s['rhythm_gap']:13.2f}{s['min_spread']:12.2f}")

    # ---- v2 axis A3: idiom vocabulary ----
    print("\n\n=== A3 IDIOM cohort comparison vs the human DISTRIBUTION ===")
    ihdr = (f"{'cohort':14s}" + "".join(f"{k:>22s}" for k in idm.SEQUENCE_KEYS)
            + f"{'idiom_gap':>12s}{'min_spread':>12s}")
    print(ihdr); print("-" * len(ihdr))
    for name in order:
        cc = idm.cohort_comparison(cohorts[name])
        if "_summary" not in cc:
            continue
        cells = "".join(
            f"{cc[k]['shift']:+10.2f}/{cc[k]['spread']:<11.2f}" if k in cc
            else f"{'--':>22s}" for k in idm.SEQUENCE_KEYS)
        s2 = cc["_summary"]
        print(f"{name:14s}{cells}{s2['idiom_gap']:12.2f}{s2['min_spread']:12.2f}")

    # ---- v2 axis A6: hand-role division ----
    print("\n\n=== A6 HAND-ROLE cohort comparison vs the human DISTRIBUTION ===")
    hhdr = (f"{'cohort':14s}" + "".join(f"{k:>22s}" for k in hro.SEQUENCE_KEYS)
            + f"{'handrole_gap':>14s}{'min_spread':>12s}")
    print(hhdr); print("-" * len(hhdr))
    for name in order:
        cc = hro.cohort_comparison(cohorts[name])
        if "_summary" not in cc:
            continue
        cells = "".join(
            f"{cc[k]['shift']:+10.2f}/{cc[k]['spread']:<11.2f}" if k in cc
            else f"{'--':>22s}" for k in hro.SEQUENCE_KEYS)
        s3 = cc["_summary"]
        print(f"{name:14s}{cells}{s3['handrole_gap']:14.2f}{s3['min_spread']:12.2f}")

    # ---- what does the sweep's OWN ranking scalar say about each control? ----
    print("\n\n=== eval_sweep's ranking metric (h_dist, lower = 'more human') ===")
    print("this is the number the sweep leaderboard picks winning arms by\n")
    hd = {name: human_dist({k: mean(cohorts[name], k) for k in H_DIST_KEYS})
          for name in order}
    for name, v in sorted(hd.items(), key=lambda kv: kv[1]):
        flag = ""
        if name not in ("human", "prod(ours)") and v <= hd.get("prod(ours)", float("inf")):
            flag = "  <== ranks BETTER than our production maps"
        print(f"  {name:14s} h_dist {v:6.3f}{flag}")

    # ---- the actual audit: does each metric rank human above each control? ----
    print("\n\n=== BLIND SPOTS — metrics that FAIL to rank human above a control ===")
    print("(a metric earns its place if human beats the controls it is RESPONSIBLE for)")
    print("Responsibility is per axis, because each control destroys one thing:")
    print("  attribute controls (random/shuffled/metronome/zigzag) -> layout + flow axes")
    print("  timing controls    (timing_random/timing_jitter)      -> rhythm axis (A2)")
    print("A rhythm metric scoring `shuffled` == human is CORRECT, not blind: that")
    print("control preserves every note time. Only the mismatches below are real.\n")
    RESPONSIBLE = {
        **{k: {"random", "shuffled", "metronome", "zigzag"} for k in AXES + ["viol"] + FLOW_AXES},
        **{k: {"timing_random", "timing_jitter", "metronome"} for k in RHYTHM_AXES},
        **{k: {"random", "shuffled", "metronome", "zigzag"} for k in IDIOM_AXES},
        **{k: {"shuffled", "metronome", "zigzag"} for k in HANDROLE_AXES},
    }
    hum = cohorts["human"]
    blind: dict[str, list[str]] = {}
    for k in AXES + ["viol"] + FLOW_AXES + RHYTHM_AXES + IDIOM_AXES + HANDROLE_AXES:
        h = mean(hum, k)
        for name in order:
            if name in ("human", "prod(ours)"):
                continue
            if name not in RESPONSIBLE.get(k, set()):
                continue  # this control does not attack what this metric measures
            c = mean(cohorts[name], k)
            if np.isnan(h) or np.isnan(c):
                continue
            # "closer to the human value is better" is the suite's own rule, so a
            # control is only CAUGHT if it sits further from the human value than
            # a real human map's own spread. Controls that land ON the human value
            # (or beyond it in the "good" direction) are undetected.
            spread = float(np.std([r[k] for r in hum if r.get(k) is not None]) or 1e-9)
            if abs(c - h) < spread:
                blind.setdefault(k, []).append(name)
    if not blind:
        print("  none — every metric separates human from every control.")
    for k, names in sorted(blind.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:16s} cannot distinguish human from: {', '.join(names)}")

    caught_by = {name: [] for name in order if name not in ("human", "prod(ours)")}
    for name in caught_by:
        for k in AXES + ["viol"] + FLOW_AXES + RHYTHM_AXES + IDIOM_AXES + HANDROLE_AXES:
            if name in RESPONSIBLE.get(k, set()) and name not in blind.get(k, []):
                caught_by[name].append(k)
    print("\n=== per-control: which metrics catch it? ===")
    for name, ks in caught_by.items():
        status = "CAUGHT" if ks else "*** UNDETECTED BY THE ENTIRE SUITE ***"
        print(f"  {name:12s} {status}")
        if ks:
            print(f"               by: {', '.join(ks)}")

    if a.json:
        out = pathlib.Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"n_human": used,
             "cohorts": {k: v for k, v in cohorts.items()},
             "blind_spots": blind}, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
