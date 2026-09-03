#!/usr/bin/env python
"""Build a complete map of ANY song, end to end, and judge it — no human in the loop.

**Kyle, 2026-08-20:** *"build the agent framework so visibly that you can confidently
validate and create any map from any song."* This is that loop, run as one command:

    SEE      events.py      the song as 14-20 typed note classes, with a trust
                            verdict per stem
             structure.py   where the sections are, which repeat, and their energy
    PLAN     here           per section: who carries it, and how dense it should be
    BUILD    mapctl auto    hand assignment, the 150 ms per-hand floor, parity
    DRESS    idiomize       note cells redrawn from the mined human vocabulary
    JUDGE    mapjudge       PASS/FAIL at n=1 against 1 100 human maps

★**Density is hit by keeping the LOUDEST events, not by thinning arbitrarily.**
`--every 3` drops every third onset regardless of what it is; this picks an accent
percentile per section so the surviving events are the ones the song emphasises.
When a budget forces notes to be dropped, dropping the quiet ones is the musical
choice, and `events.py` measures loudness relative to each stem's own median so the
same percentile means the same thing on every stem and every song.

★**Section energy sets the budget, so the map breathes.** Kyle named this himself as
something to protect: *"when there is a slow spot we let the player breathe."* An
intro at energy 0.41 and a drop at 1.00 get different note budgets from the same
target, rather than the flat ~8 nps the ML pipeline used to emit everywhere.

⚠️**A PASS here means NOT DEFECTIVE, not good.** The judge gates against the human
corpus median and his standing target is the best mappers, so this is a floor. It
also has no audio axis yet, so it cannot tell you the notes are on the beat.

Usage:
    python agent_mapper/autobuild.py data/eval_songset/1f767.ogg --name auto767
    python agent_mapper/autobuild.py <audio> --name X --nps 4.5 --no-idiomize
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AM))

# Human Expert median, from the mapjudge reference over 1100 maps. A TARGET, not a
# ceiling -- Kyle called 6.18 nps unplayable and the judge independently rejects at
# 6.10, so the usable band is roughly 2.7-5.1.
HUMAN_NPS = 4.17
# Melodic stems, in the order they are preferred as a section's carrier. Drums are
# the backbone and are handled separately.
MELODIC = ("vocals", "other", "guitar", "piano", "bass")


def run(args: list[str], quiet: bool = True) -> str:
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True,
                       cwd=REPO)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"{' '.join(args[-4:])}\n{r.stdout}\n{r.stderr}")
    if not quiet:
        print(r.stdout.rstrip())
    return r.stdout


def plan(audio: pathlib.Path, nps: float) -> list[dict]:
    """Per section: the carrier class, the backbone, and each one's accent budget."""
    import brief as B
    import events as E
    import structure as S

    a = B.analyse(audio)
    d = E.analyse(audio)
    sec = S.analyse(audio)
    S.roles(sec["sections"], a)
    trust = d.get("trust") or {}

    # Only offer classes from a stem whose labelling survived its control. An
    # untrusted stem is still usable -- as ONE lane, which is what `stem` alone means.
    def classes_in(stem: str, b0: int, b1: int) -> dict[str, int]:
        if trust.get(stem) is False:
            return {}
        out: dict[str, int] = {}
        for e in d["events"]:
            if e["stem"] == stem and b0 <= e["bar"] <= b1:
                out[e["cls"]] = out.get(e["cls"], 0) + 1
        return out

    # ★★NORMALISE THE ENERGY CURVE, do not just apply it. `0.55 + 0.60*energy` spans a
    # breathing 55 % to a full 115 % of target, but its DURATION-WEIGHTED MEAN is only
    # 0.919 across the songset (energy averages ~0.64, not 0.5), so asking for 4.17 nps
    # silently delivers 3.83 -- an 8 % density shortfall built into the arithmetic, on
    # an axis where we already sit at the 19th human percentile. Dividing by the
    # weighted mean keeps every section's RELATIVE breathing identical (the shape Kyle
    # named as something to protect) while making the map as a whole hit the target.
    def _mult(s) -> float:
        return 0.55 + 0.60 * float(s.get("energy") or 0.5)

    _tot = sum(max(s["t1"] - s["t0"], 0.1) for s in sec["sections"]) or 1.0
    _wmean = sum(max(s["t1"] - s["t0"], 0.1) * _mult(s)
                 for s in sec["sections"]) / _tot or 1.0

    rows = []
    for s in sec["sections"]:
        b0, b1 = s["bar0"], s["bar0"] + s["bars"] - 1
        dur = max(s["t1"] - s["t0"], 0.1)
        budget = nps * dur * (_mult(s) / _wmean)

        # ★★RECRUIT AS MANY CARRIERS AS THE BUDGET NEEDS.
        # Taking only the single busiest class made the note supply, not the
        # requested density, the real ceiling: on Hunger the two streams we looked at
        # hold **1 965** candidate events while the song has **4 813** across six
        # stems, so `--nps 9` silently delivered 6.25 and every song came out at
        # roughly the same difficulty. Kyle: *"the objective is to be able to map
        # whatever difficulty we want."* Rank the classes and take them until the
        # section's melodic share can actually be paid for.
        ranked = sorted(
            ((f"{stem}/{cls}", n)
             for stem in MELODIC
             for cls, n in classes_in(stem, b0, b1).items()),
            key=lambda kv: -kv[1],
        )
        carrier, n_carrier = (ranked[0] if ranked else (None, 0))
        carriers = [c for c, _ in ranked[:1]]
        n_pool = n_carrier

        drums = sum(1 for e in d["events"]
                    if e["stem"] == "drums" and b0 <= e["bar"] <= b1)
        # Split the budget: the carrier takes the larger share in a loud section,
        # the drums carry a quiet one where a melodic line is sparse anyway.
        share = 0.55 if float(s.get("energy") or 0.5) >= 0.5 else 0.35
        # Recruit further classes only when the top one cannot cover its share. Each
        # added class is a real instrument line, so this widens the map rather than
        # doubling notes onto times already played.
        want_melodic = budget * share
        for name, n in ranked[1:]:
            if n_pool >= want_melodic:
                break
            carriers.append(name)
            n_pool += n
        rows.append({
            "bar0": b0, "bar1": b1, "role": s.get("role"), "label": s["label"],
            "energy": round(float(s.get("energy") or 0), 2), "dur": round(dur, 1),
            "budget": int(budget),
            "carrier": carrier, "carrier_n": n_carrier,
            "carriers": carriers, "pool_n": n_pool, "share": share,
            "carrier_pct": _pct(want_melodic, n_pool),
            "drums_n": drums,
            "drums_pct": _pct(budget * (1 - share), drums),
        })
    return rows


def _pct(want: float, have: int) -> float | None:
    """Accent percentile that keeps ~`want` of `have` events (None = keep all)."""
    if not have:
        return None
    return None if want >= have else round(max(want / have, 0.02), 3)


def build(audio: pathlib.Path, name: str, rows: list[dict], verbose: bool,
          pulse: bool = False, phrase_bars: int = 4, lead_bias: float = 0.0,
          lead_phrase_bars: int = 4, pulse_fill: int = 1,
          pulse_sync: float = 0.3, snap_onsets: bool = False,
          adaptive_subdiv: bool = False, seed: int = 0,
          doubles: bool = False, accent_slots: str = "0,2,4,6,8,10,12,14",
          doubles_rate: float = 0.3, phase_shift: float = 0.0,
          phase_calibrate: bool = True, density_search: bool = False) -> None:
    init = [str(AM / "mapctl.py"), "init", str(audio), "--name", name, "--fresh"]
    if phase_shift:
        init += ["--phase-shift", str(phase_shift)]
    init += ["--phase-calibrate" if phase_calibrate else "--no-phase-calibrate"]
    if adaptive_subdiv:
        init += ["--adaptive-subdiv"]
    run(init)
    for r in rows:
        bars = f"{r['bar0']}-{r['bar1']}"
        if pulse:
            # ★ONE pass over BOTH streams, then hold one interval per phrase.
            # Two passes is what P0.5 measured as the merge cost (drums alone
            # `pulse_stability` 0.387, the union 0.329, human 0.514), and a pulse
            # grid chosen twice independently is not a pulse.
            # All recruited carriers go into ONE merged pass — layering separate
            # passes is what cost the pulse (drums alone 0.387, a union of two
            # independent passes 0.329, human 0.514).
            follow = ",".join(x for x in [("drums" if r["drums_n"] else None),
                                          *(r.get("carriers") or [r["carrier"]])] if x)
            if not follow:
                continue
            cmd = [str(AM / "mapctl.py"), "auto", name, "--bars", bars,
                   "--follow", follow, "--wide", "--pulse",
                   "--phrase-bars", str(phrase_bars),
                   "--pulse-fill", str(pulse_fill),
                   "--pulse-sync", str(pulse_sync)]
            if snap_onsets:
                cmd += ["--snap-onsets"]
            if doubles:
                # ★Humans put both hands on ~16 % of note instants; we placed ZERO
                # until 2026-08-21. `mapctl` had the flag the whole time.
                cmd += ["--doubles", "--accent-slots", str(accent_slots)]
                if doubles_rate < 1.0:
                    cmd += ["--doubles-rate", str(doubles_rate)]
            if lead_bias > 0:
                # ⚠️`--seed` used to reach `idiomize` ONLY, so the lead-hand RNG ran at
                # seed 0 no matter what was asked for. Every "3 seeds" reading of a
                # hand-role metric was therefore one seed wearing three hats.
                cmd += ["--lead-bias", str(lead_bias),
                        "--lead-phrase-bars", str(lead_phrase_bars),
                        "--seed", str(seed)]
            # One budget for the merged stream: the section's whole accent budget,
            # not the drums/carrier split, which only existed to feed two passes.
            # The accent percentile is searched to hit the budget in SURVIVING
            # slots; a percentile computed from the event count undershoots by
            # whatever fraction of the two streams collides on the grid.
            cmd += ["--target-notes", str(int(r["budget"]))]
            run(cmd, quiet=not verbose)
            continue
        # Drums first: the backbone defines the pulse, and `auto` assigns hands over
        # the MERGED timeline, so a later pass cannot hand one hand two fast swings.
        # ★`--density-search` (2026-09-02j): the default path asks for a PERCENTILE of the
        # event pool, and its own comment predicted what that costs -- "a percentile computed
        # from the event count undershoots by whatever fraction of the two streams collides
        # on the grid". Measured on 1f333: a plan budgeting 1367 notes delivered 972 (-29 %),
        # and the miss grows with the request (--nps 9 asks 2465, lands 1347) because the
        # percentile saturates at 1.0 while the pool it is a fraction OF is only the single
        # top carrier -- the classes `plan` recruited are dropped on the floor here. The flag
        # spends the budget the way the pulse path already does: search the percentile for the
        # one whose SURVIVING slots hit the target, over every recruited carrier.
        if density_search:
            # ONE pass over drums + the carrier, with the section's WHOLE budget. Splitting it
            # two ways and searching each half independently still undershot (nps 5.0: 1082 of
            # 1367 budgeted), because the halves collide on the shared grid and neither search
            # can see the other's slots -- the same reason the pulse path merges. One search
            # over the union is self-correcting.
            # ★NO RECRUITMENT HERE (measured 2026-09-03). Adding the classes `plan` recruited
            # buys NOTHING on density -- the percentile search alone meets the request on the
            # same 3 of 4 songset songs -- and it costs quality: on 1f333 it pushed FLOW from
            # 18 flagged bars to 22 (🟡 8 % -> 🔴 10 %) and the tutor from 25/49 down to 22/49.
            # The recruited classes are extra instrument lines the search does not need once it
            # can reach the budget by lowering the percentile on the carrier it already has.
            follow = ",".join(x for x in [("drums" if r["drums_n"] else None),
                                          r["carrier"]] if x)
            if follow:
                run([str(AM / "mapctl.py"), "auto", name, "--bars", bars,
                     "--follow", follow, "--wide",
                     "--target-notes", str(max(int(r["budget"]), 1))], quiet=not verbose)
            continue
        if r["drums_n"]:
            cmd = [str(AM / "mapctl.py"), "auto", name, "--bars", bars,
                   "--follow", "drums", "--wide"]
            if r["drums_pct"]:
                cmd += ["--accent-pct", str(r["drums_pct"])]
            run(cmd, quiet=not verbose)
        if r["carrier"]:
            cmd = [str(AM / "mapctl.py"), "auto", name, "--bars", bars,
                   "--follow", r["carrier"], "--wide"]
            if r["carrier_pct"]:
                cmd += ["--accent-pct", str(r["carrier_pct"])]
            run(cmd, quiet=not verbose)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--style", default=None,
                    help="a style name from style.py --list. Sets the density and "
                         "vocabulary targets; --nps overrides its density.")
    ap.add_argument("--nps", type=float, default=None)
    ap.add_argument("--no-idiomize", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # ★**A CUT-DIRECTION / VARIETY LEVER** (2026-08-24). `idiomize`'s `width` keeps
    # only the N highest-COUNT candidates before drawing, and human idiom frequency is
    # vertical-dominated -- so the truncation is what strips our diagonals. Measured
    # over 23 songs x 3 seeds (arm gaps >> the ±0.005-0.021 seed spread):
    #   width  diagonal   idiom_top50   idiom_coverage   (human 0.415 / 0.404 / 0.909)
    #     3     0.237        0.557          0.990   <- default
    #     5     0.303        0.449          0.929
    #     8     0.394        0.325          0.783
    # ⚠️**The default stays 3 deliberately.** It was chosen by READING two maps side by
    # side, and a decision made by eye is not one to reverse on axis numbers alone
    # (the same rule that keeps BEAT_GRID_PHASE off). This flag exists so the
    # alternative can be PLAYED -- see LISTENING.md.
    # ★See mapctl.PHASE_BIAS_BEATS. Default ON since 2026-09-02 (P0 [PCAL]): a
    # measured estimator bias, held-out validated, corpus-independent. The rationale
    # for the flip is on the mapctl side.
    ap.add_argument("--phase-calibrate", dest="phase_calibrate", action="store_true",
                    help="correct the measured +0.053-beat bias in our phase estimator "
                         "(held-out: onset_precision 0.888->0.917, human agreement "
                         "0.642->0.709). Default ON")
    ap.add_argument("--no-phase-calibrate", dest="phase_calibrate", action="store_false",
                    help="build on the raw, uncorrected phase estimate")
    ap.set_defaults(phase_calibrate=True)
    # ★LEG 3's second travel lever: re-weights candidates by HOW FAR each move
    # travels, where --width sets how MANY are considered. See idiomize.TRAVEL_TARGET.
    ap.add_argument("--travel-target", type=float, default=None,
                    help="grid-units/sec the sampler prefers a move to cover "
                         "(default 4.167). Higher = wider reaches")
    ap.add_argument("--width", type=int, default=None,
                    help="idiom candidate-pool width (default 3). Higher = more "
                         "diagonals and a broader local vocabulary; 5 sits closest "
                         "to the human idiom_top50/idiom_coverage.")
    ap.add_argument("--verbose", action="store_true")
    # ★★TWO ENTRY POINTS, NOT ONE LEVER (P4 decide-and-log, 2026-09-02). Doubles, the
    # lead hand and `--doubles-rate` are applied ONLY inside the `--pulse` branch of
    # `build()`; the default two-pass path ships 0 % doubles while `--no-doubles`
    # claimed "ON by default". The 08-03 maps' 51-70 % doubles came from the pulse
    # path, the 09-02 tutor DoD's 0 % from this one. Measured on the songset with
    # the P3 queries (Kyle's verdicts) and the tutor: NOPULSE 43 hits / 25 of 99
    # situations his way; PULSE 53 hits (FLOW 21, D2 8: notes on the odd 16th between
    # the lead's 8ths, the Hunger AGENT pattern) / 19 of 99; doubles bolted onto the
    # drums pass 47 / 21. Only the judge (typicality, p 0.57 -> 0.66) prefers pulse,
    # so the default stays here and `--pulse` is a request. WORKFLOW.md said "on".
    ap.add_argument("--pulse", dest="pulse", action="store_true",
                    help="hold one interval per phrase, from a single merged pass over "
                         "drums+carrier (P0.5); the ONLY path with doubles and a lead "
                         "hand. Off by default: it draws FLOW/D2 jitter (2026-09-02)")
    ap.add_argument("--no-pulse", dest="pulse", action="store_false",
                    help="the default two-pass path (drums, then carrier)")
    ap.set_defaults(pulse=False)
    ap.add_argument("--phrase-bars", type=int, default=4)
    ap.add_argument("--lead-bias", type=float, default=0.0,
                    help="probability a phrase's lead hand repeats (P0.6); --pulse "
                         "only. WORKFLOW.md's operating point is 0.2 (0.3 overshoots)")
    ap.add_argument("--lead-phrase-bars", type=int, default=4)
    ap.add_argument("--pulse-fill", type=int, default=1,
                    help="lattice points held across a quiet gap (P0.7)")
    ap.add_argument("--no-doubles", dest="doubles", action="store_false",
                    help="disable both-hands accents inside --pulse (there they are on "
                         "by default since 2026-08-21: double_share 0.000 -> 0.127 vs "
                         "human 0.205, n=23). The default path has NO doubles; "
                         "add them with mapedit.py where a stem enters (tutor.py)")
    ap.set_defaults(doubles=True)
    ap.add_argument("--accent-slots", default="0,2,4,6,8,10,12,14",
                    help="slots eligible for doubles. The eighth-note set matches the "
                         "human on-beat share (0.64 vs 0.635); `0,8` pins 100%% of "
                         "doubles exactly on a beat, which is the mechanical feel")
    ap.add_argument("--phase-shift", type=float, default=0.0,
                    help="TEST ONLY: degrade the grid by N beats (causal probe)")
    # ★[FULL] IS THE DEFAULT since 2026-09-02 (P0 decide-and-log): the absence of an
    # element 96-100 % of human maps carry is a defect, and no metric can see it, so
    # waiting on an ear to turn it on was the wrong default. 0 = notes only ([V2]).
    ap.add_argument("--walls", type=int, default=89,
                    help="add N walls (0 = none). Human median is 89 and 96%% of human "
                         "maps have them; we shipped zero until 2026-08-22. ⚠️No metric "
                         "in the suite can see walls — only his ear can judge them. "
                         "Default 89 (human median) since 2026-09-02")
    ap.add_argument("--arcs", type=int, default=90,
                    help="add N arcs (v3 human median 90, present in 100%% of v3 maps). "
                         "Purely additive — notes are untouched. Default 90")
    ap.add_argument("--chains", type=int, default=16,
                    help="add N chains (v3 human median 16, present in 71%%). Extends an "
                         "existing swing, so parity is re-checked after. Default 16")
    ap.add_argument("--notes-only", action="store_true",
                    help="[V2] output: no walls, arcs or chains (same as --walls 0 "
                         "--arcs 0 --chains 0)")
    ap.add_argument("--doubles-rate", type=float, default=0.3,
                    help="fraction of eligible accent slots that become doubles. 0.3 "
                         "measured n=23: rate 0.166 vs human 0.237, 23/23 PASS. 0.5 "
                         "matches the median better but is not resolvably closer "
                         "per-song (1.40x vs 1.55x human-human spread) and costs a PASS")
    ap.add_argument("--density-search", action="store_true",
                    help="spend the section budget by SEARCHING the accent percentile "
                         "(what the pulse path already does) instead of asking for a "
                         "fixed fraction of the top carrier's events, and follow every "
                         "carrier `plan` recruited. Off by default: it changes the "
                         "default build's note count.")
    ap.add_argument("--adaptive-subdiv", action="store_true",
                    help="1/8-beat slots below 150 bpm (P1.0)")
    ap.add_argument("--snap-onsets", action="store_true",
                    help="reconcile the placing detector with the judge's (P0.7)")
    ap.add_argument("--pulse-sync", type=float, default=0.3,
                    help="syncopation-restore threshold; lower restores more real "
                         "onsets (P0.7/P0.8)")
    ap.add_argument("--json", type=pathlib.Path)
    a = ap.parse_args()

    from beatsaber_automapper.evaluation import mapjudge as mj
    ref = mj.load_reference()

    # A style sets the targets; an explicit --nps still wins, because the density is
    # the one knob Kyle has already given a verdict on (6.18 unplayable).
    sargs = {}
    if a.style:
        import style as ST
        sargs = ST.build_args(a.style, ref)
        print(f"=== STYLE `{a.style}` -> {sargs}")
    nps = a.nps if a.nps is not None else sargs.get("nps", HUMAN_NPS)
    # P0.1: a density the CALLER asked for (flag or style preset) is a request the
    # judge gates against; the corpus default is not a request.
    nps_requested = a.nps if a.nps is not None else sargs.get("nps")
    if a.notes_only:
        a.walls = a.arcs = a.chains = 0

    print(f"=== SEE: {a.audio.name}")
    rows = plan(a.audio, nps)
    print(f"{'bars':<12} {'role':<10} {'nrg':>5} {'budget':>7}  carrier "
          f"(events -> accent pct)")
    print("-" * 78)
    for r in rows:
        cp = f"{r['carrier_pct']}" if r["carrier_pct"] else "all"
        dp = f"{r['drums_pct']}" if r["drums_pct"] else "all"
        print(f"{r['bar0']:>4}-{r['bar1']:<7} {str(r['role']):<10} {r['energy']:>5.2f} "
              f"{r['budget']:>7}  {str(r['carrier']):<18} "
              f"({r['carrier_n']}->{cp})  drums({r['drums_n']}->{dp})")

    print(f"\n=== BUILD")
    build(a.audio, a.name, rows, a.verbose, pulse=a.pulse,
          phrase_bars=a.phrase_bars, lead_bias=a.lead_bias,
          lead_phrase_bars=a.lead_phrase_bars, pulse_fill=a.pulse_fill,
          pulse_sync=a.pulse_sync, snap_onsets=a.snap_onsets,
          adaptive_subdiv=a.adaptive_subdiv, seed=a.seed,
          doubles=a.doubles, accent_slots=a.accent_slots,
          doubles_rate=a.doubles_rate, phase_shift=a.phase_shift,
          phase_calibrate=a.phase_calibrate, density_search=a.density_search)
    out = a.out or (REPO / "outputs" / f"autobuild_{a.name}.zip")
    run([str(AM / "mapctl.py"), "export", a.name, "--out", str(out)], quiet=False)

    if not a.no_idiomize:
        print(f"\n=== DRESS (idiomize)")
        import idiomize as I
        kw = {}
        if "crossover" in sargs:
            kw["crossover"] = sargs["crossover"]
        if "top_k" in sargs:
            kw["top_k"] = sargs["top_k"]
        if "width" in sargs:
            kw["width"] = sargs["width"]
        if a.travel_target is not None:
            kw["travel_target"] = a.travel_target
        # An explicit --width beats the style's choice: it is the knob being A/B'd.
        if a.width is not None:
            kw["width"] = a.width
        n, nfb = I.idiomize_zip(out, out, seed=a.seed, **kw)
        print(f"  re-placed {n - nfb}/{n} note cells from the human vocabulary")

    # ★★WALLS — the element 96 % of human maps have and we shipped ZERO of.
    # Measured 2026-08-22 over the same 23 songs: human median **89 walls per map**,
    # ours **0**. `walls.py` was built on 2026-08-19 against 16 504 vanilla human walls
    # and never wired into this loop, so every map the agent has produced is missing a
    # whole physical layer. 🔴**No metric can see this** — adding 84 walls + 48 arcs +
    # 16 chains moves every axis by exactly 0.000, because the suite scores notes and
    # nothing else. That is precisely why it went unnoticed for three days.
    # 🔴🔴**MUST RUN AFTER `idiomize`.** `walls.py` places each wall in a lane no note
    # occupies, but `idiomize` REDRAWS every note's column afterwards — so walls chosen
    # against the pre-idiomize layout end up on top of notes. Measured on the first
    # attempt: **12 notes trapped inside an active wall**, i.e. unplayable, while the
    # lane/duration/width statistics all still matched the human idiom perfectly.
    # ★Only the collision check caught it; no axis in the suite can see walls at all.
    if a.walls:
        import walls as W
        n_walls = W.add_walls(out, out, per_map=a.walls, seed=a.seed)
        print(f"\n=== WALLS")
        print(f"  added {n_walls} walls (human median 89; 96 % of human maps have them)")

    # ★ARCS and CHAINS — the other two elements measured but never wired.
    # Among v3 human maps (the only format where they can exist): **arcs in 100 %,
    # median 90 per map; chains in 71 %, median 16**. ⚠️A first pass read 0 % for both
    # because 767 of 791 corpus maps are v2, where the objects do not exist — the
    # project's own landmine: *check the feature was AVAILABLE before calling it rare.*
    # ★An arc is purely additive (its own object between two positions, notes
    # untouched). A chain extends the swing an existing note starts, so it is checked
    # against the swing simulator below rather than trusted.
    if a.arcs:
        import arcs as A
        n_arcs = A.add_arcs(out, out, target=a.arcs)
        print(f"\n=== ARCS")
        print(f"  added {n_arcs} arcs (v3 human median 90; 100 % of v3 maps have them)")
    if a.chains:
        import chains as C
        n_chains = C.add_chains(out, out, target=a.chains, seed=a.seed)
        print(f"\n=== CHAINS")
        print(f"  added {n_chains} chains (v3 human median 16; 71 % of v3 maps)")

    print(f"\n=== JUDGE")
    # ★The alignment axis needs the song's cached onsets. Without them the judge
    # scores 21 metrics and the P0.2 floor (is the map ON the music?) cannot apply --
    # which is the one question this loop most needs answered. Say so loudly.
    from beatsaber_automapper.evaluation import scorecard as SC
    onsets = SC.onsets_for(a.audio)
    if onsets is None:
        print("  ⚠️NO CACHED ONSETS for this song -> no alignment axis, no P0.2 floor. "
              "Run scripts/build_onset_cache.py on it first.")
    res = mj.judge_zip(out, reference=ref, onsets=onsets, nps_request=nps_requested)
    print(mj.report(res))
    if a.style:
        import style as ST
        print()
        print(ST.report(ST.check(res, a.style), a.style))
    if a.json:
        a.json.write_text(json.dumps(
            {"song": a.audio.stem, "plan": rows, "map": str(out),
             "verdict": res.verdict(), "p": res.p_value,
             "rank": res.rank_score, "n_notes": res.n_notes,
             "worst": [{"metric": m.name, "value": m.value, "pct": m.pct}
                       for m in res.worst(8)]}, indent=1) + "\n")
    return 0 if res.verdict() == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
