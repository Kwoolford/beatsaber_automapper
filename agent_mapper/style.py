#!/usr/bin/env python
"""Map in a chosen STYLE — as a target position on continuous axes, not a preset genre.

**Kyle:** *"…and can map to whatever style you want."*

★**The obvious build was tested first and REFUTED.** The natural design is a set of
named style clusters mined from the human corpus ("this mapper is a *flow* mapper,
that one is *tech*"), so a style is a cluster you aim at. Measured over **1 098 human
maps × 21 metrics**, robust-scaled: k-means silhouette peaks at **0.153** at k = 2 and
falls monotonically with k, against a null of **0.105** where every metric is shuffled
independently and no joint style can survive.

    k        2      3      4      5      6      7      8
    sil    0.153  0.143  0.130  0.125  0.112  0.108  0.109
    null   0.105  0.092  0.097

**Human mapping style is a CONTINUUM.** A silhouette of 0.15 is not a clustered space,
and naming clusters in it would invent a taxonomy the data does not contain — the same
mistake the downbeat detector and the backbeat control would have been, both avoided
by measuring the corpus first.

**So a style here is a set of TARGET PERCENTILES** in the human distribution:
*"denser than 80 % of human maps, crossing over more than 70 % of them, smoother than
75 %."* That construction has three things going for it:

1. It matches the shape the data actually has.
2. It is **checkable** — `mapjudge` already reports every metric as a human percentile,
   so "did I hit the style?" is answered on the same ruler as "is this defective?".
3. It is **user-facing**, which is the standing requirement: these are meant to become
   knobs a player can move, so they have to be quantities with an intelligible meaning.
   "Denser than 80 % of human maps" is one; "cluster 3" is not.

⚠️**The presets below are CHOICES, not discoveries.** The corpus supplies the axes and
the percentiles; which corner of that space to call "flowing" is a naming convention
and nothing more. They are starting points for his ear to move.

⚠️**A style target is not a quality target.** The judge still gates separately: a map
can hit its style exactly and be defective, and `p` is the number that says so.

Usage:
    python agent_mapper/style.py --list
    python agent_mapper/style.py --show dense
    python agent_mapper/style.py --check outputs/autobuild_ab767.zip --style flowing
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Each preset is {metric: target percentile in the human distribution}. Only the
# metrics a style actually means to move are listed; everything unlisted is left to
# land wherever the music puts it, which is the point -- a style is a few deliberate
# choices, not a full specification of a map.
PRESETS: dict[str, dict[str, float]] = {
    # The floor: sit where a typical human map sits on everything the style layer
    # can steer. This is what autobuild targets by default.
    "human": {"nps": 0.50, "crossover": 0.50, "angle_change": 0.50,
              "travel": 0.50, "idiom_top50": 0.50},
    # Busier, without going near the density Kyle called unplayable. ⚠️The judge
    # rejects at 6.10 nps (accept 0.125) and he called 6.18 unplayable, so `nps`
    # deliberately stops at p80 rather than p95.
    "dense": {"nps": 0.80, "peak_nps": 0.70, "crossover": 0.55,
              "angle_change": 0.45, "travel": 0.60},
    # Smooth and continuous: small wrist rotations, common idioms, moderate travel.
    "flowing": {"angle_change": 0.25, "idiom_top50": 0.75, "idiom_coverage": 0.65,
                "travel": 0.40, "crossover": 0.45, "nps": 0.45},
    # Awkward on purpose: bigger rotations, rarer moves, more crossing over.
    "technical": {"angle_change": 0.75, "idiom_coverage": 0.35, "crossover": 0.75,
                  "travel": 0.75, "ioi_switch_rate": 0.70, "nps": 0.60},
    # Room to breathe. Kyle named the pacing he liked: "when there is a slow spot we
    # let the player breathe."
    "calm": {"nps": 0.30, "peak_nps": 0.25, "ebpm_burst": 0.25,
             "angle_change": 0.35, "travel": 0.35},
}

# Which build knob moves which metric. Only these are STEERABLE; a style may name a
# metric that no knob controls, and `plan_for` reports that honestly instead of
# pretending the target was actionable.
STEERABLE = {
    "nps": "autobuild --nps",
    "crossover": "idiomize --crossover",
    "idiom_coverage": "idiomize --top-k",
    "angle_change": "(follows from the vocabulary; not directly steerable)",
    "travel": "(follows from the vocabulary; not directly steerable)",
}


def value_at(metric: str, pct: float, reference: dict) -> float | None:
    """The human value at a given percentile, from the judge's own distributions."""
    d = reference["distributions"].get(metric)
    if not d:
        return None
    i = int(round(max(0.0, min(1.0, pct)) * (len(d) - 1)))
    return float(d[i])


def resolve(style: str | dict, reference: dict) -> dict[str, dict]:
    """Turn a style into concrete target VALUES, with the knob that moves each."""
    spec = PRESETS[style] if isinstance(style, str) else style
    out = {}
    for metric, pct in spec.items():
        out[metric] = {
            "target_pct": pct,
            "target_value": value_at(metric, pct, reference),
            "knob": STEERABLE.get(metric),
        }
    return out


def build_args(style: str | dict, reference: dict) -> dict:
    """The concrete build parameters a style implies, for `autobuild`."""
    r = resolve(style, reference)
    args = {}
    if "nps" in r and r["nps"]["target_value"] is not None:
        args["nps"] = round(r["nps"]["target_value"], 3)
    if "crossover" in r and r["crossover"]["target_value"] is not None:
        args["crossover"] = round(r["crossover"]["target_value"], 3)
    if "idiom_coverage" in r:
        # Vocabulary depth trades directly against coverage: depth 500 forces
        # coverage to ~1.0 because the top 500 IS ~90 % of human usage; 1000
        # reproduces the human 0.909; deeper drifts to ~0.885. Measured sweep in
        # idiomize.VOCAB_DEPTH.
        p = r["idiom_coverage"]["target_pct"]
        args["top_k"] = 500 if p >= 0.85 else (1000 if p >= 0.5 else 2000)
    return args


def check(res, style: str | dict, tol: float = 0.20) -> list[dict]:
    """Did the map land where the style asked? Per metric, on the judge's own ruler.

    `tol` is in PERCENTILE units: a target of p80 is met anywhere in p60-p100. The
    band is wide on purpose -- these axes carry real seed variance, and a tight band
    would turn a style into something to over-fit.
    """
    spec = PRESETS[style] if isinstance(style, str) else style
    got = {m.name: m.pct for m in res.metrics}
    rows = []
    for metric, want in spec.items():
        have = got.get(metric)
        rows.append({
            "metric": metric, "want_pct": want, "got_pct": have,
            "hit": None if have is None else abs(have - want) <= tol,
            "knob": STEERABLE.get(metric),
        })
    return rows


def report(rows: list[dict], style_name: str) -> str:
    hit = sum(1 for r in rows if r["hit"])
    n = sum(1 for r in rows if r["hit"] is not None)
    out = [f"style `{style_name}`: {hit}/{n} targets met",
           f"  {'metric':<18} {'want':>6} {'got':>6}  knob"]
    for r in rows:
        g = f"{r['got_pct']*100:5.1f}%" if r["got_pct"] is not None else "    - "
        mark = "ok " if r["hit"] else ("?? " if r["hit"] is None else "MISS")
        out.append(f"  {mark} {r['metric']:<14} {r['want_pct']*100:5.0f}% {g}  "
                   f"{r['knob'] or ''}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show")
    ap.add_argument("--check", type=pathlib.Path)
    ap.add_argument("--style", default="human")
    ap.add_argument("--tol", type=float, default=0.20)
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()

    if a.list:
        print("styles (target percentiles in the human distribution):\n")
        for name, spec in PRESETS.items():
            s = "  ".join(f"{k} p{int(v*100)}" for k, v in spec.items())
            print(f"  {name:<10} {s}")
        print("\n⚠️These are naming conventions over a CONTINUUM, not clusters found "
              "in the corpus -- see the module docstring for the measurement.")
        return 0

    if a.show:
        r = resolve(a.show, ref)
        print(f"style `{a.show}`\n  {'metric':<18} {'pct':>5} {'value':>10}  knob")
        for m, d in r.items():
            v = f"{d['target_value']:.3f}" if d["target_value"] is not None else "-"
            print(f"  {m:<18} {d['target_pct']*100:>4.0f}% {v:>10}  {d['knob'] or ''}")
        print(f"\n  build args: {build_args(a.show, ref)}")
        return 0

    if a.check:
        res = mj.judge_zip(a.check, reference=ref)
        rows = check(res, a.style, a.tol)
        print(f"{a.check.name}: judge says {res.verdict()} (p={res.p_value:.3f})\n")
        print(report(rows, a.style))
        print("\n⚠️Hitting a style is not the same as being good -- the judge's "
              "verdict above is the separate question.")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
