#!/usr/bin/env python
"""Control battery for the per-map judge — the only real evidence it works.

`calibrate_mapjudge.py` guarantees the held-out human accept rate lands near
1-alpha; that is arithmetic, not a finding. What has to be MEASURED is whether the
judge **rejects a map that is obviously broken, one map at a time** -- because that
is the claim being made: that an agent can decide a map is good without Kyle playing
it.

So every human map in the held-out slice is turned into the six degenerate controls
from `audit_eval_suite.py` (attribute-destroying: random / shuffled / metronome /
zigzag; timing-destroying: timing_random / timing_jitter) and scored **at n=1**, the
way a freshly authored map would be.

DoD, fixed before running (docs/eval_suite_v2.md principle 1):
    human accept   >= 0.85   (it is ~0.90 by construction; below 0.85 means broken splits)
    every control  <= 0.10   accepted

★2026-09-02 (P0.2): the verdict now also carries the UNDILUTED ALIGNMENT FLOOR
(`mapjudge.ALIGN_FLOOR_METRIC`), which by design fails the worst-aligned ~10 % of
humans on top of the pooled gate's ~10 %. So under `--audio` the human bar is read
on the `no-floor` column (the pooled gate alone, which is what the conformal
guarantee covers) and the ~0.79 in `accept` is the priced cost, not a broken split.
The `offbeat` control must be <= 0.10 in `accept` -- that is what the floor is for.

It also reports **per-metric discrimination** -- the AUC of each metric's
nonconformity, human vs control. A metric that cannot separate a human map from a
metronome is dead weight in the aggregate and dilutes the metrics that can; those
get named here rather than left in silently.

★An axis-aware caveat inherited from `audit_eval_suite.py`: `random`, `shuffled` and
`zigzag` preserve the human note TIMES, so timing metrics are blind to them BY
CONSTRUCTION, and `timing_*` preserve the attributes. Per-metric AUC is therefore
reported against the controls that actually attack what each metric measures.

Usage:
    python scripts/audit_mapjudge.py --n 300
"""
from __future__ import annotations

import argparse
import json
import pathlib
import math
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from audit_eval_suite import CONTROLS, _load_human  # noqa: E402
from beatsaber_automapper.evaluation import mapjudge as mj  # noqa: E402
from calibrate_mapjudge import CORPUS_OFFSET, corpus  # noqa: E402

# Which controls attack which kind of metric. A metric is only credited (or
# blamed) on the controls that can possibly move it.
ATTRIBUTE_CONTROLS = ["random", "shuffled", "zigzag", "metronome"]
TIMING_CONTROLS = ["timing_random", "timing_jitter", "metronome"]
DENSITY_CONTROLS = ["too_dense", "too_sparse"]


def make_too_dense(notes, rng):
    """Every gap filled with an extra note -- attacks nps / peak_nps.

    ★**Why this control had to be added.** The first run reported `nps` and
    `peak_nps` as DEAD at AUC 0.507/0.467, and that reading was wrong: every
    pre-existing control preserves the note COUNT and the song span, so no density
    metric can possibly move on them. They were blind by construction, not dead --
    the same axis-aware mistake `audit_eval_suite.py` records for the rhythm axis.
    A metric with no control that attacks it is untested, and calling it dead would
    have deleted the only thing in the judge that measures how busy a map is.
    """
    out = []
    for i, n in enumerate(notes):
        out.append(n)
        if i + 1 < len(notes):
            mid = (n.beat + notes[i + 1].beat) / 2.0
            if mid > n.beat + 1e-6:
                out.append(type(n)(beat=mid, x=n.x, y=n.y,
                                   color=1 - n.color, direction=n.direction))
    return out


def make_too_sparse(notes, rng):
    """Three quarters of the notes removed -- degenerate emptiness.

    ★**The first version dropped every THIRD note and the battery failed on it at
    0.692 accepted.** That was the control being wrong, not the judge: an Expert map
    with a third of its notes gone is about a *Hard* map, and Hard maps are real
    human artifacts, so the ground truth does not fail this control. The rule is
    already on the books -- **a control the ground truth fails is not a control**
    (`docs/perception_scorecard.md`, where the backbeat control was discarded for
    the same reason). `--density-sweep` measures the whole curve instead; this
    control is set at a keep fraction the sweep shows is genuinely degenerate
    (0.25 keep = 1.02 nps, accepted 0.000).
    """
    return [n for i, n in enumerate(notes) if i % 4 == 0]


def make_offbeat(notes, rng):
    """Every note shifted a QUARTER BEAT later -- attacks the ALIGNMENT axis alone.

    ★**Why this control had to be added (2026-08-21).** The audio axis passed the
    battery without adding anything: on `timing_random` the beat-domain metrics beat
    it (`ioi_cond_entropy` 0.996 vs `onset_precision` 0.885) and on `timing_jitter` it
    is weak (0.694 vs `pulse_stability` 0.997). Not because the axis is useless, but
    because **no control here produces the failure mode it exists for** -- notes with
    perfectly human rhythm that sit off the music. Every existing control damages the
    IOI sequence, which the beat-domain metrics catch first. A metric with no control
    that isolates it is untested, exactly as `make_too_dense` records for density.

    ★**The shift is a quarter beat for a reason**: it is a whole number of 1/16 grid
    slots, so `offgrid_frac` cannot move (a shift off the grid would be caught
    trivially and by construction -- the `[PHASE]` finding), and it preserves every
    interval, so `pulse_stability`, `dominant_share`, `ioi_switch_rate` and
    `ioi_cond_entropy` are unchanged. Note count, colours, positions and directions
    are untouched, so geometry, idiom and hand-role cannot move either.
    ⇒**Only the audio axis can see this map is wrong**, which is the point.
    At 120-188 bpm a quarter beat is 80-125 ms, comfortably outside the 50 ms
    matching tolerance.

    ⚠️A human would call the result a different map, not a broken one -- it is
    rhythmically intact and merely displaced against the song. That is precisely the
    defect `onset_precision` reports on OUR maps.
    """
    import copy
    out = []
    for n in notes:
        m = copy.copy(n)
        m.beat = n.beat + 0.25
        out.append(m)
    return out


EXTRA_CONTROLS = {"too_dense": make_too_dense, "too_sparse": make_too_sparse,
                  "offbeat": make_offbeat}


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a random `pos` scores above a random `neg`), ties counted as half."""
    if not pos or not neg:
        return float("nan")
    neg_s = sorted(neg)
    import bisect
    total = 0.0
    for v in pos:
        lo = bisect.bisect_left(neg_s, v)
        hi = bisect.bisect_right(neg_s, v)
        total += lo + 0.5 * (hi - lo)
    return total / (len(pos) * len(neg))



def thin(notes, keep: float, rng):
    """Keep a `keep` fraction of notes, evenly -- a difficulty reduction."""
    if keep >= 1.0:
        return list(notes)
    step = 1.0 / keep
    idx = {int(i * step) for i in range(int(len(notes) * keep))}
    return [n for i, n in enumerate(notes) if i in idx]


def thicken(notes, mult: float, rng):
    """Insert notes into the gaps until the map is `mult` times as dense."""
    out = list(notes)
    extra = int(len(notes) * (mult - 1.0))
    if extra <= 0:
        return out
    gaps = sorted(range(len(notes) - 1),
                  key=lambda i: notes[i + 1].beat - notes[i].beat, reverse=True)
    for i in gaps[:extra]:
        n, nx = notes[i], notes[i + 1]
        mid = (n.beat + nx.beat) / 2.0
        if mid > n.beat + 1e-6:
            out.append(type(n)(beat=mid, x=n.x, y=n.y,
                               color=1 - n.color, direction=n.direction))
    return sorted(out, key=lambda n: n.beat)


def density_sweep(ref, a) -> int:
    """How far from human density must a map be before the judge notices?

    ★**Why this replaces a pass/fail on a `too_sparse` control.** The battery's
    first run accepted a map with every third note removed at 0.692, which looked
    like a judge defect. It is not obviously one: an Expert map with a third of its
    notes removed is roughly a *Hard* map, and Hard maps are real human artifacts.
    `docs/perception_scorecard.md` already records the rule that decided this --
    **a control the ground truth fails is not a control** (the backbeat control was
    thrown out for exactly this). So the honest question is not "does it reject
    thinning" but "at what thinning does it start to", and the answer is a curve.

    This matters directly for D1 (*"very slow"*): the judge's usefulness on our own
    maps depends on whether it can see a density defect at the size ours has.
    """
    rng = random.Random(a.seed)
    raws = corpus(a.seed)[CORPUS_OFFSET:]
    span = int(1100 * 1.25) + 40
    held = raws[2 * span:3 * span]

    keeps = [1.0, 0.8, 0.67, 0.5, 0.33, 0.25, 0.15, 0.10]
    mults = [1.25, 1.5, 2.0, 3.0]
    rows: dict[str, list] = {f"keep {k:.2f}": [] for k in keeps}
    rows.update({f"x{m:.2f}": [] for m in mults})

    scored = 0
    for zp in held:
        if scored >= a.n:
            break
        loaded = _load_human(zp)
        if loaded is None:
            continue
        notes, bpm = loaded
        if len(notes) < 50 or not (30.0 < bpm < 400.0):
            continue
        scored += 1
        for k in keeps:
            v = thin(notes, k, rng)
            if len(v) >= 20:
                rows[f"keep {k:.2f}"].append((mj.judge(mj.map_record(v, bpm), ref), v))
        for m in mults:
            v = thicken(notes, m, rng)
            rows[f"x{m:.2f}"].append((mj.judge(mj.map_record(v, bpm), ref), v))

    print(f"\ndensity resolution of the judge, {scored} held-out human maps\n")
    print(f"{'variant':<12} {'nps':>6} {'accept':>7} {'no-parity':>10} "
          f"{'p med':>7}  worst metric")
    print("-" * 66)
    out = {}
    for name, entries in rows.items():
        rs = [r for r, _ in entries if r.n_scored]
        if not rs:
            continue
        acc = sum(1 for r in rs if r.verdict(a.alpha) == "PASS") / len(rs)
        accs = sum(1 for r in rs
                   if not math.isnan(r.p_value) and r.p_value >= a.alpha) / len(rs)
        pmed = sorted(r.p_value for r in rs)[len(rs) // 2]
        npss = [next((m.value for m in r.metrics if m.name == "nps"), float("nan"))
                for r in rs]
        npss = [v for v in npss if v == v]
        npsmed = sorted(npss)[len(npss) // 2] if npss else float("nan")
        tally: dict[str, int] = {}
        for r in rs:
            w = r.worst(1)
            if w:
                tally[w[0].name] = tally.get(w[0].name, 0) + 1
        worst = max(tally, key=tally.get) if tally else "-"
        out[name] = {"accept": acc, "accept_stats_only": accs, "p_median": pmed,
                     "nps_median": npsmed, "worst_metric": worst, "n": len(rs)}
        print(f"{name:<12} {npsmed:>6.2f} {acc:>7.3f} {accs:>10.3f} {pmed:>7.3f}  {worst}")

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.with_name("mapjudge_density_sweep.json").write_text(
        json.dumps({"n_maps": scored, "alpha": a.alpha, "rows": out}, indent=1) + "\n")
    print(f"\nwrote {a.json.with_name('mapjudge_density_sweep.json')}")
    return 0


def _onsets_for(zp):
    """The judge's cached onsets for this corpus map, or None."""
    import numpy as _np
    f = REPO / "outputs" / "onset_cache" / f"{zp.stem}.npz"
    if not f.exists():
        return None
    z = _np.load(f)
    k = list(z.keys())
    return _np.asarray(z[k[0]], dtype=float) if k else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="held-out human maps to audit")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density-sweep", action="store_true",
                    help="instead of the battery, measure the judge's density "
                         "RESOLUTION: accept rate vs how much a human map is "
                         "thinned or thickened")
    ap.add_argument("--json", type=pathlib.Path,
                    default=REPO / "outputs" / "mapjudge_audit.json")
    ap.add_argument("--audio", action="store_true",
                    help="score every variant against the song's cached onsets so the "
                         "ALIGNMENT axis is audited too; without this the battery runs "
                         "on 21 metrics and alignment never discriminates anything")
    a = ap.parse_args()

    ref = mj.load_reference()
    rng = random.Random(a.seed)
    if a.density_sweep:
        return density_sweep(ref, a)

    # The held-out slice is the THIRD span in the calibrator's layout. Re-deriving
    # it here (rather than reusing the cached records) is deliberate: the controls
    # need the note lists, not the metrics.
    raws = corpus(a.seed)[CORPUS_OFFSET:]
    span = int(1100 * 1.25) + 40
    held = raws[2 * span:3 * span]

    all_controls = dict(CONTROLS)
    all_controls.update(EXTRA_CONTROLS)
    cohorts: dict[str, list] = {"human": []}
    for c in all_controls:
        cohorts[c] = []
    # per-metric nonconformity, per cohort
    umetrics: dict[str, dict[str, list[float]]] = {k: {} for k in cohorts}

    scored = 0
    n_no_onsets = 0
    for zp in held:
        if scored >= a.n:
            break
        loaded = _load_human(zp)
        if loaded is None:
            continue
        notes, bpm = loaded
        if len(notes) < 50 or not (30.0 < bpm < 400.0):
            continue
        song_onsets = _onsets_for(zp) if a.audio else None
        if a.audio and song_onsets is None:
            # A map without cached onsets would be scored on 21 metrics and pooled
            # with 23-metric ones -- the exact mixing the dual calibration sets exist
            # to prevent. Skip it rather than dilute the cohort.
            n_no_onsets += 1
            continue
        scored += 1

        variants = {"human": notes}
        for cname, fn in all_controls.items():
            variants[cname] = fn(list(notes), rng)

        for vname, vnotes in variants.items():
            # ★Every variant is scored against the SAME song's onsets. That is the
            # point: the two timing controls move note TIMES while the music stays
            # put, so the alignment axis is the one instrument that should catch them
            # directly rather than through a beat-domain proxy.
            rec = mj.map_record(vnotes, bpm, onsets=song_onsets)
            res = mj.judge(rec, ref, label=f"{zp.name}:{vname}")
            cohorts[vname].append(res)
            for m in res.metrics:
                umetrics[vname].setdefault(m.name, []).append(m.u)

        if scored % 50 == 0:
            print(f"  ... {scored} maps")

    print(f"\nscored {scored} held-out human maps x {1+len(all_controls)} variants "
          f"= {scored*(1+len(all_controls))} judgements, each at n=1"
          + (f"   [AUDIO: {n_no_onsets} skipped, no cached onsets]"
             if a.audio else "   [no audio axis - 21 metrics]") + "\n")

    # ---------------- accept rates ----------------
    print(f"{'cohort':<15} {'accept':>7} {'no-floor':>9} {'no-parity':>10} {'p median':>9} "
          f"{'s_mean':>8} {'viol>0':>7}  verdict")
    print("-" * 82)
    summary = {}
    order = ["human"] + list(all_controls)
    for name in order:
        rs = [r for r in cohorts[name] if r.n_scored]
        if not rs:
            continue
        acc = sum(1 for r in rs if r.verdict(a.alpha) == "PASS") / len(rs)
        # The same verdict with the parity hard gate removed. Without this the
        # audit cannot tell whether the judge is doing statistical work or just
        # re-running swing_sim: 88-98% of the attribute controls are parity-illegal.
        acc_stats = sum(1 for r in rs
                        if not math.isnan(r.p_value) and r.p_value >= a.alpha) / len(rs)
        # The verdict with only the P0.2 alignment floor removed: the pooled gate
        # plus parity, i.e. the judge as it was before 2026-09-02.
        acc_nofloor = sum(1 for r in rs if (r.viol or 0) == 0 and not r.nps_fail
                          and not math.isnan(r.p_value) and r.p_value >= a.alpha) / len(rs)
        pmed = sorted(r.p_value for r in rs)[len(rs) // 2]
        smean = sum(r.s_mean for r in rs) / len(rs)
        smax = sum(r.s_topk for r in rs) / len(rs)
        vfrac = sum(1 for r in rs if (r.viol or 0) > 0) / len(rs)
        if name == "human":
            ok = "PASS" if acc_nofloor >= 0.85 else "FAIL (splits broken?)"
            if acc_nofloor >= 0.85 and acc < 0.85:
                ok += f"  (floor costs {acc_nofloor - acc:.3f}, priced in P0.2)"
        else:
            ok = "PASS" if acc <= 0.10 else "FAIL"
        summary[name] = {"accept": acc, "accept_no_floor": acc_nofloor,
                         "accept_stats_only": acc_stats,
                         "p_median": pmed, "s_mean": smean,
                         "s_topk": smax, "viol_frac": vfrac, "n": len(rs)}
        print(f"{name:<15} {acc:>7.3f} {acc_nofloor:>9.3f} {acc_stats:>10.3f} {pmed:>9.3f} "
              f"{smean:>8.3f} {vfrac:>7.3f}  {ok}")

    # ---------------- per-metric discrimination ----------------
    # Per CONTROL, not pooled. Pooling hid the truth on `dominant_share`: the
    # metronome drives it to 1.0 (extreme high) while timing_random drives it DOWN,
    # where a one-sided "high" metric correctly scores 0 -- so the pooled AUC read
    # 0.522 and called a working metric dead. A metric earns its place if ANY
    # control that attacks it is separated.
    ctrl_names = list(all_controls)
    print(f"\nper-metric discrimination: AUC of nonconformity, control vs human "
          f"(0.50 = blind to that control)")
    hdr = f"{'metric':<18}" + "".join(f"{c[:9]:>10}" for c in ctrl_names) + f"{'BEST':>8}"
    print(hdr)
    print("-" * len(hdr))
    metric_rows = {}
    for name, axis, _tail, _note in mj.CANDIDATES:
        hu = umetrics["human"].get(name)
        if not hu:
            continue
        per_ctrl = {}
        for c in ctrl_names:
            per_ctrl[c] = auc(umetrics[c].get(name, []), hu)
        vals = [v for v in per_ctrl.values() if v == v]
        best = max(vals) if vals else float("nan")
        metric_rows[name] = {"per_control": per_ctrl, "relevant_auc": best,
                             "human_u": sum(hu) / len(hu)}
        row = f"{name:<18}" + "".join(
            f"{per_ctrl[c]:>10.3f}" if per_ctrl[c] == per_ctrl[c] else f"{'-':>10}"
            for c in ctrl_names) + f"{best:>8.3f}"
        if best == best and best < 0.60:
            row += "  <- DEAD"
        print(row)

    dead = [m for m, r in metric_rows.items()
            if r["relevant_auc"] == r["relevant_auc"] and r["relevant_auc"] < 0.60]
    if dead:
        print(f"\n⚠️ {len(dead)} metric(s) below 0.60 AUC against the controls that "
              f"attack them:\n   {', '.join(dead)}")
        print("   These dilute the mean aggregate. Consider dropping them "
              "(mapjudge.DROPPED) and re-calibrating.")

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(
        {"n_maps": scored, "alpha": a.alpha, "cohorts": summary,
         "metrics": metric_rows, "dead": dead}, indent=1) + "\n")
    print(f"\nwrote {a.json}")

    all_ok = (summary.get("human", {}).get("accept_no_floor", 0) >= 0.85 and
              all(summary[c]["accept"] <= 0.10
                  for c in all_controls if c in summary))
    print(f"\nDoD: {'MET' if all_ok else 'NOT MET'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
