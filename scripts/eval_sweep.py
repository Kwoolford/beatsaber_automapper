#!/usr/bin/env python3
"""Multi-song / multi-arm evaluation sweep harness (2026-06-30).

Purpose: test many generation theories per night and get a leaderboard, instead
of hand-running one config at a time. Each ARM is a named set of env vars + CLI
flags for scripts/generate.py; each SONG is a cached full-length audio file with
a PRECOMPUTED reference onset density (the expensive Demucs step, cached once).
For every (arm, song) it generates a map (cached on disk) and scores it with the
DoD density-corr (+ note count, CV); prints an arm×song Spearman matrix with
mean and pass-count, and writes a JSON leaderboard.

Subcommands
-----------
  build-songset --n N         extract N full-length (>=MIN_DUR s) songs from
                              data/raw into data/eval_songset/ and cache refs.
  sweep [--arms a,b,...]       run the arms (default: all defined) over the songset.
  list-arms                    print the registered arms.

Add a theory = add one entry to ARMS below (name -> env dict + extra flags).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
SONGSET = REPO / "data" / "eval_songset"
CACHE = REPO / "outputs" / "eval_sweep_cache"
MIN_DUR = 150.0          # seconds; below this, density-corr Spearman is too noisy
WIN_SEC = 2.0
SR = 44100

BEAT_CKPT = "logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LAYOUT_CKPT = "logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"
# Shelved per-instrument Stage-1 ckpt (TASK 2, 2026-06-05). Killed at the time on
# val_f1_avg_tol (0.600 vs the 0.603 baseline) -- a metric we've since established
# anti-correlates with map quality three separate times. Re-evaluating on the v2
# suite per docs/stage1_instrument_rebuild.md Phase 0, B-0 (2026-07-28).
BEAT_CKPT_V7INSTR = "logs/beat_classifier/version_7/checkpoints/beat-epoch=00-val_f1_avg_tol=0.600.ckpt"

# B-1 (2026-07-29): the honest instrument retrain. version_7 was confounded -- its
# only surviving ckpts were epochs 0/2/7 (old hardcoded save_top_k=3 + early stop on
# val_f1_avg_tol), so B-0 compared an epoch-0 model against version_4's epoch-11 and
# could not separate "instrument features hurt" from "less training". version_8 ran
# 18 full epochs with --save-top-k -1, so EVERY epoch survived. val_f1_avg_tol
# oscillates in a narrow band (0.562-0.599) with no trend -- deliberately NOT the
# selection metric here; the v2 suite selects.
B1_DIR = REPO / "logs" / "beat_classifier" / "version_8" / "checkpoints"


def _b1_ckpt(epoch: int) -> str:
    """Resolve a version_8 epoch checkpoint by number (val_f1 suffix varies)."""
    hits = sorted(B1_DIR.glob(f"beat-epoch={epoch:02d}-val_f1_avg_tol=*.ckpt"))
    if not hits:
        raise FileNotFoundError(f"no version_8 checkpoint for epoch {epoch} in {B1_DIR}")
    return str(hits[0].relative_to(REPO))


# Spaced subset -- enough to see the SHAPE of the suite-vs-epoch curve without
# paying for all 18. If the curve has a clear interior optimum, fill in around it.
B1_EPOCHS = (0, 3, 6, 9, 12, 15, 17)

# ---- ARMS: name -> (env overrides, extra generate flags). Add theories here. ----
_DS25 = {"DENSITY_SELECT": "1", "DENSITY_SELECT_GAMMA": "2.5"}
# A theory = one entry: name -> (env overrides, extra generate.py flags).
# History (2026-06-30): the density-select gamma sweep found g2.5 best (5/6 pass);
# a Stage-2 temperature sweep was a dead end (layout collapse was a decode bug, now
# fixed in generate_phrase). Keep the live comparison set lean.
# 2026-06-30 (PM-3): the hardened scorecard exposed two layout-quality gaps that
# survived the decode-bug fix — grid_coverage ~0.65 vs human 0.96 and dir_entropy
# ~0.60 vs 0.80: the model under-uses the 12 grid cells and the 9 cut directions.
# The sweep decodes layout GREEDILY (temp 0.0 → nucleus collapses to argmax), so
# those numbers are the model's *argmax* diversity. Two no-retrain levers, both on
# the production density config (dsel_g2.5): (a) stochastic decode (raise temp+top_p
# lets the tail through); (b) the env-gated frequency penalty (deterministic
# anti-repeat), now extended to the DIR role via LAYOUT_DIV_D so it can move
# dir_entropy, not just grid_coverage.
_DIV = {"LAYOUT_DIVERSITY": "1"}
# 2026-07-23: grid_cov/dir_entropy gaps CLOSED (temp 0.9/top_p 0.97 promoted to
# generate.py prod defaults; composite monotony 0.44 ≈ human 0.43). Remaining
# layout-quality residual is the HIDDEN sub-signal `pattern_repeat` (adjacent-
# identical (x,y,dir) tuples; human ≈ 0.002) now surfaced as its own column.
# The cumulative LAYOUT_DIV_* penalty over-flattens the whole distribution
# (div10 → grid 1.0 / rows 0.35, past human) so it's the wrong tool for adjacency.
# NEW lever = windowed ADJACENCY anti-repeat (LAYOUT_ANTIREPEAT window +
# LAYOUT_AR_STRENGTH): penalize only tokens seen in the last-W emissions per role,
# breaking back-to-back loops WITHOUT touching global cell/dir spread.
# 2026-07-23 PROMOTED: the sweep winner ar_w1_s2 (W=1/S=2.0) is now the baked-in
# LAYOUT_ANTIREPEAT/LAYOUT_AR_STRENGTH default in layout_model.py, so `prod` (the
# density-select-only config) now inherits anti-repeat and IS the new production
# control. `noar` (LAYOUT_ANTIREPEAT=0) preserves the pre-promotion baseline for
# regression; keep g2.5_div10 as the over-flatten failure-mode reference.
def _ar(w: str, s: str) -> dict[str, str]:
    return {**_DS25, "LAYOUT_ANTIREPEAT": w, "LAYOUT_AR_STRENGTH": s}
ARMS: dict[str, tuple[dict[str, str], list[str]]] = {
    "prod":        (_DS25, []),                                             # control = NEW PRODUCTION (W1/S2 baked default + temp 0.9/top_p 0.97)
    # Byte-identical config to `prod`. Decode is stochastic (temp 0.9/top_p 0.97)
    # and generate.py has no seed flag, so prod vs prod_rep measures the NOISE
    # FLOOR of every arm comparison in this file. Added 2026-07-27 after a
    # regeneration at near-identical BPM moved the idiom gap 2.41 -> 1.76, which
    # is larger than several differences that had been read as signal.
    "prod_rep":    (_DS25, []),
    # --- STAGE-1 IOI PRIOR (2026-07-27) — the rhythm gap is onset SELECTION ---
    # Part D ruled out tempo detection: correcting BPM moves rhythm 2.41 -> 2.37
    # and makes it WORSE on the songs that were actually mis-detected. rule_mapper
    # showed rhythm is inherited entirely from the onset layer. So this changes
    # WHICH slots Stage-1 picks: within each density-allocated window, maximise
    # model prob + lambda * human P(interval | previous interval) instead of
    # taking the top-k by probability, which just reproduces the audio's own
    # periodicity. Window allocation is untouched, so density_corr is preserved.
    # Single-song probe at lambda=1: dominant_share 0.924 -> 0.653 (human 0.509),
    # switch rate 1.2 -> 4.3 (human 13.7), viol 0, note count held -- but it
    # over-produces 1/16 and under-produces 1/4, so sweep the strength.
    # ioi05/ioi1/ioi2 below were generated by the MAXIMISING version of the
    # selector and are kept only as the record of that failure: maximising a
    # diagonal-dominant bigram produces long homogeneous runs, so rhythm got
    # WORSE than baseline (switch rate 5.38 -> 3.18 vs a human 13.65) even though
    # the interval histogram moved toward human. The argmax of a distribution is
    # not a sample from it.
    "ioi05":       ({**_DS25, "BEAT_IOI_PRIOR": "0.5"}, []),
    "ioi1":        ({**_DS25, "BEAT_IOI_PRIOR": "1.0"}, []),
    "ioi2":        ({**_DS25, "BEAT_IOI_PRIOR": "2.0"}, []),
    # --- HAND OFFSET (2026-07-27) — the unified fix for A2 rhythm + A6 hand role ---
    # Dumping beat_probs next to human note times showed our maps NEVER place a
    # note on an odd 16th (0 of 679 slots); the human map puts 248 there, and
    # those are exactly the slots we miss. Cause: hand lockstep. Human hands are
    # interleaved by a 16th 32% of the time, ours 0.2%, and the union can only
    # reach an odd 16th if the hands are offset. So A2 and A6 are ONE defect.
    # This MOVES one hand by a 16th at shared slots instead of deleting it (which
    # is what BEAT_HAND_ROLE did, costing 24% of the notes).
    # Single-song probe at 0.5: within-window IOI CV 0.102 -> 0.377 (human 0.354),
    # pulse 0.874 -> 0.588 (human 0.591), switch 1.2 -> 16.0 (human 13.5),
    # note count and parity held. 0.8 overshoots.
    "ho03":        ({**_DS25, "BEAT_HAND_OFFSET": "0.3"}, []),
    "ho05":        ({**_DS25, "BEAT_HAND_OFFSET": "0.5"}, []),
    "ho07":        ({**_DS25, "BEAT_HAND_OFFSET": "0.7"}, []),
    # spacing-aware variant: pick the neighbour that keeps the hand's own gaps
    # even, to recover the flow regression (which came from angle_change, not
    # travel, so the travel penalty cannot fix it)
    "ho03s":       ({**_DS25, "BEAT_HAND_OFFSET": "0.3",
                     "BEAT_HAND_OFFSET_SPACING": "1"}, []),
    "ho05s":       ({**_DS25, "BEAT_HAND_OFFSET": "0.5",
                     "BEAT_HAND_OFFSET_SPACING": "1"}, []),
    "ho03s_best":  ({**_DS25, "BEAT_HAND_OFFSET": "0.3",
                     "BEAT_HAND_OFFSET_SPACING": "1",
                     "LAYOUT_TRAVEL_PENALTY": "1.0", "COLOR_SEP_MODE": "extreme"}, []),
    "ho05_best":   ({**_DS25, "BEAT_HAND_OFFSET": "0.5",
                     "LAYOUT_TRAVEL_PENALTY": "1.0", "COLOR_SEP_MODE": "extreme"}, []),
    # min-gap guard: only offset when the moved note stays >=2 slots from this
    # hand's other notes. The flow regression was ebpm_burst (243 -> 360
    # swings/min vs a human 250), NOT angle_change which slightly improved.
    "ho03g":       ({**_DS25, "BEAT_HAND_OFFSET": "0.3", "BEAT_HAND_OFFSET_MINGAP": "2"}, []),
    "ho05g":       ({**_DS25, "BEAT_HAND_OFFSET": "0.5", "BEAT_HAND_OFFSET_MINGAP": "2"}, []),
    "ho05g3":      ({**_DS25, "BEAT_HAND_OFFSET": "0.5", "BEAT_HAND_OFFSET_MINGAP": "3"}, []),
    # ---- corrected: SAMPLE the prior instead of maximising it ----
    "iois1":       ({**_DS25, "BEAT_IOI_PRIOR": "1.0"}, []),
    "iois2":       ({**_DS25, "BEAT_IOI_PRIOR": "2.0"}, []),
    "iois4":       ({**_DS25, "BEAT_IOI_PRIOR": "4.0"}, []),
    "iois2_best":  ({**_DS25, "BEAT_IOI_PRIOR": "2.0",
                     "LAYOUT_TRAVEL_PENALTY": "1.0", "COLOR_SEP_MODE": "extreme"}, []),
    "noar":        ({**_DS25, "LAYOUT_ANTIREPEAT": "0"}, []),               # pre-promotion baseline (anti-repeat OFF) — regression reference
    "ar_w1_s2":    (_ar("1", "2.0"), []),                                   # promoted config, explicit (== prod default now)
    "ar_w2_s2":    (_ar("2", "2.0"), []),                                   # 2-step window, moderate
    "ar_w3_s3":    (_ar("3", "3.0"), []),                                   # 3-step window, stronger loop-break
    "g2.5_div10":  ({**_DS25, **_DIV, "LAYOUT_DIV_X": "1.0", "LAYOUT_DIV_Y": "1.0", "LAYOUT_DIV_D": "1.0"}, []),  # over-flatten reference
    # --- eval-suite v2 axis A1 (flow/ergonomics) levers, added 2026-07-27 ---
    # travel: our hands move ~50% further per second than human hands (flow
    # `travel` shift +2.48 human-MADs). Penalize long jumps in short windows.
    "tp1":         ({**_DS25, "LAYOUT_TRAVEL_PENALTY": "1.0"}, []),
    "tp2":         ({**_DS25, "LAYOUT_TRAVEL_PENALTY": "2.0"}, []),
    "tp4":         ({**_DS25, "LAYOUT_TRAVEL_PENALTY": "4.0"}, []),
    # crossover: enforce_color_separation moves EVERY wrong-side note, so we
    # measure crossover 0.000 vs a human median of 0.218. "extreme" keeps the
    # mild one-column crossovers the model chose; "off" is the ablation.
    "xsep_ext":    ({**_DS25, "COLOR_SEP_MODE": "extreme"}, []),
    "xsep_off":    ({**_DS25, "COLOR_SEP_MODE": "off"}, []),
    # do the two levers compose, or fight?
    "tp2_xsep":    ({**_DS25, "LAYOUT_TRAVEL_PENALTY": "2.0", "COLOR_SEP_MODE": "extreme"}, []),
    # --- eval-suite v2 axis A2 (rhythm) lever, added 2026-07-27 ---
    # Our hands fire simultaneously on 85.6% of beats (human 17.5%), which is what
    # makes the union rhythm metronomic. Soft-penalise the right hand on slots the
    # left hand already took. Single-song probe: il0.5 -> simultaneity 0.12 (human
    # 0.175) but little rhythm gain; il0.9 -> simultaneity 0.02 (too far) but
    # cond-entropy 0.49 and switch-rate 12.2, both near human (0.54 / 13.7). The
    # sweet spot is somewhere between, hence three strengths.
    "il5":         ({**_DS25, "BEAT_HAND_INTERLEAVE": "0.5"}, []),
    "il7":         ({**_DS25, "BEAT_HAND_INTERLEAVE": "0.7"}, []),
    "il9":         ({**_DS25, "BEAT_HAND_INTERLEAVE": "0.9"}, []),
    # --- eval-suite v2 axis A3 (idiom) lever, added 2026-07-27 ---
    # Boost cut directions that COMPLETE a known human idiom given this hand's
    # previous note. Single-song probe at strength 2.0: coverage 0.759 -> 0.946
    # (human 0.919), top50 0.337 -> 0.398 (human 0.386), viol still 0, travel
    # untouched. Slight overshoot past human suggests ~1.0-1.5 is the sweet spot.
    "ib1":         ({**_DS25, "LAYOUT_IDIOM_BONUS": "1.0"}, []),
    "ib2":         ({**_DS25, "LAYOUT_IDIOM_BONUS": "2.0"}, []),
    "ib3":         ({**_DS25, "LAYOUT_IDIOM_BONUS": "3.0"}, []),
    # rhythm lever + the best-guess flow levers, to check the axes do not fight
    "il7_tp1_xsep": ({**_DS25, "BEAT_HAND_INTERLEAVE": "0.7",
                      "LAYOUT_TRAVEL_PENALTY": "1.0", "COLOR_SEP_MODE": "extreme"}, []),
    # everything that looked good, together — the candidate next production config
    "combo":       ({**_DS25, "BEAT_HAND_INTERLEAVE": "0.7",
                     "LAYOUT_TRAVEL_PENALTY": "1.0", "COLOR_SEP_MODE": "extreme",
                     "LAYOUT_IDIOM_BONUS": "1.5"}, []),
    # --- eval-suite v2 axis A6 (HAND ROLE) lever, added 2026-07-27 ---
    # Our worst axis by far: handrole_gap 3.50 vs a human 0.34, and worse than a
    # uniformly random map. Human mappers give ONE hand the lead in a passage then
    # swap; we split every bar evenly. BEAT_HAND_ROLE reassigns which hand plays
    # each already-selected onset (times untouched), targeting the measured human
    # reference: asymmetry 0.115, swap rate 0.461, doubles 0.175.
    # Single-song probe at strength 1.0 OVERSHOOTS asymmetry (0.241), so sweep down.
    "hr05":        ({**_DS25, "BEAT_HAND_ROLE": "0.5"}, []),
    "hr075":       ({**_DS25, "BEAT_HAND_ROLE": "0.75"}, []),
    "hr10":        ({**_DS25, "BEAT_HAND_ROLE": "1.0"}, []),
    # the two PROVEN levers on their own (flow PASS + idiom PASS), no interleave,
    # no idiom bonus — this is the honest promotion candidate
    "best":        ({**_DS25, "LAYOUT_TRAVEL_PENALTY": "1.0",
                     "COLOR_SEP_MODE": "extreme"}, []),
    # proven pair + hand role = candidate to pass all four axes
    "best_hr":     ({**_DS25, "LAYOUT_TRAVEL_PENALTY": "1.0",
                     "COLOR_SEP_MODE": "extreme", "BEAT_HAND_ROLE": "0.5"}, []),
    # --- TRACK B / Phase 0 (2026-07-28): cheap re-eval of the shelved per-
    # instrument Stage-1 ckpt (version_7). Same layout/decode config as `prod`,
    # only the beat classifier + --use-instr change, so any axis delta is
    # attributable to the instrument representation, not to Track A levers.
    "v7instr":     (_DS25, ["--beat-ckpt", BEAT_CKPT_V7INSTR, "--use-instr"]),
    # --- TRACK A-1 (2026-07-28) — DIFFICULTY (axis A7 playfeel, nps sub-metric).
    # Kyle: "this is Expert, not Expert+" -- we generate 6.18 NPS against a human
    # Expert median 3.91-4.46. BEAT_DIFFICULTY_SCALE scales the total note budget
    # that DENSITY_SELECT allocates across windows, so the density-tracks-song-
    # structure shape is preserved and only its overall level drops. Single-song
    # smoke test (1f333, NOTE: a known half-tempo probe trap, verify on all 24):
    # scale 0.68 took nps 4.78 -> 3.71, diagonal_share ~unchanged (0.518->0.505,
    # as expected -- this lever only touches budget, not direction, that's A-2).
    "ds065":       ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.65"}, []),
    "ds07":        ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.70"}, []),
    "ds075":       ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.75"}, []),
    # --- TRACK A-2 (2026-07-28) -- DIRECTION IDIOM. Kyle: "obsessed with 45-
    # degree notes" -- diagonal_share 0.513 vs human 0.358, up/down inverted
    # (0.468 vs human 0.563). Traced to LAYOUT_ANTIREPEAT running on ROLE_DIR:
    # penalizing a repeated up/down cut pushes the model toward diagonals as the
    # least-recently-used escape. LAYOUT_ANTIREPEAT_ROLES=xy narrows the penalty
    # to X/Y only, leaving DIR to the model's own distribution. DoD: diagonal
    # share inside the human range without dir_entropy collapsing back to the
    # pre-2026-07-23 monotony (watch both ends).
    "ar_xy":       ({**_DS25, "LAYOUT_ANTIREPEAT_ROLES": "xy"}, []),
    "ar_xy_ds07":  ({**_DS25, "LAYOUT_ANTIREPEAT_ROLES": "xy",
                     "BEAT_DIFFICULTY_SCALE": "0.70"}, []),
    # --- 2026-07-28/29 follow-up: ds065 alone was the biggest single-lever win
    # of the whole session -- scorecard.py on the 24-map cohorts: flow 0.71->0.28
    # PASS, idiom 1.85->0.58 PASS, rhythm 2.37->0.71 (bar 0.70, essentially at the
    # line), playfeel 2.29->1.23 (bar 1.00), handrole 3.23->2.54 (bar 2.00).
    # rhythm/playfeel/handrole all improve MONOTONICALLY as the scale drops
    # (ds07 rhythm 1.11 > ds065 rhythm 0.71), and NPS at ds065 (4.66) is still
    # slightly above the human Expert ceiling (4.46) -- so push the scale lower
    # to see if rhythm/playfeel cross their bars, and pair with the previously-
    # built BEAT_HAND_ROLE lever (default OFF, strength 0.5 was the honest
    # candidate from 2026-07-27) since difficulty scaling alone plateaus
    # handrole around 2.5, still short of the 2.00 bar.
    "ds05":        ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.50"}, []),
    "ds055":       ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55"}, []),
    "ds06":        ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.60"}, []),
    "ds065_hr05":  ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.65", "BEAT_HAND_ROLE": "0.5"}, []),
    "ds06_hr05":   ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.60", "BEAT_HAND_ROLE": "0.5"}, []),
    # ds055 alone already clears 4/5 axes (2026-07-29) -- check whether the A-2
    # direction lever adds anything on top before considering this "done".
    "ar_xy_ds055": ({**_DS25, "LAYOUT_ANTIREPEAT_ROLES": "xy",
                     "BEAT_DIFFICULTY_SCALE": "0.55"}, []),

    # --- A6 / HAND LEAD (2026-08-01) ---------------------------------------
    # scripts/eval_spread_breakdown.py attributed the whole handrole failure to a
    # single sub-metric: `role_asymmetry`, human 0.115 vs ours 0.026-0.046, cohort
    # spread 0.27 against a 0.35 bar. Upstream of it is the double rate -- we put
    # both hands on the same slot 84-94% of the time against a human 23%, which
    # makes a per-window lead arithmetically impossible.
    # BEAT_HAND_LEAD biases each hand's per-window budget SHARE while keeping its
    # total fixed, so unlike BEAT_HAND_ROLE (which deleted ~24% of the notes to
    # manufacture asymmetry) no note is lost. Value = target local asymmetry.
    # All arms sit on ds055, the best-scoring density: prod density is a tier too
    # dense to judge anything at, and the lever composes with the difficulty scale.
    # GRID CENTRED BY SMOKE TEST, not by guesswork (2026-08-01, song 1f8a3):
    # BEAT_HAND_LEAD=0.30 realised role_asymmetry 0.247 against an OFF baseline of
    # 0.043, i.e. the metric lands at ~0.82x the requested share -- the lever
    # OVERSHOOTS the human 0.115 badly at the values first written here (0.2-0.5).
    # Note count was preserved (678 -> 687, +1.3%), which is the point: the old
    # BEAT_HAND_ROLE lost ~24%. So the interesting band is 0.10-0.25, centred on
    # the ~0.14 predicted to land on the human median.
    "hl010_ds055":  ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                      "BEAT_HAND_LEAD": "0.10"}, []),
    "hl014_ds055":  ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                      "BEAT_HAND_LEAD": "0.14"}, []),   # predicted human median
    "hl018_ds055":  ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                      "BEAT_HAND_LEAD": "0.18"}, []),
    "hl025_ds055":  ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                      "BEAT_HAND_LEAD": "0.25"}, []),
    # The smoke test also dropped `role_swap_rate` 0.436 -> 0.282 (human 0.461):
    # a lead block outlives handrole.py's 8-beat measuring window, so neighbouring
    # windows share a leader and the dominant hand swaps less often. handrole_gap
    # averages |shift| over asymmetry AND swap rate, so that regression can eat the
    # asymmetry win. This arm raises the swap rate to compensate.
    "hl014_sw07_ds055": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                          "BEAT_HAND_LEAD": "0.14",
                          "BEAT_HAND_LEAD_SWAP": "0.70"}, []),
    # does the lead compose with the direction lever that flipped handrole on its
    # own? ar_xy_ds055 passed handrole but collapsed idiom's spread instead.
    "hl014_ar_xy_ds055": ({**_DS25, "LAYOUT_ANTIREPEAT_ROLES": "xy",
                           "BEAT_DIFFICULTY_SCALE": "0.55",
                           "BEAT_HAND_LEAD": "0.14"}, []),

    # --- ROBUSTNESS AROUND THE hl014 OPTIMUM (2026-08-01) -------------------
    # hl014_ds055 PASSES all 5 axes, but its neighbours hl010 and hl018 both FAIL,
    # which is a sharp enough optimum to be worth distrusting. The mechanism does
    # explain it -- realised role_asymmetry is linear in the setting (0.0917 /
    # 0.1197 / 0.1538 for 0.10 / 0.14 / 0.18) and hl014 is simply the arm that
    # lands on the human 0.115 -- but "the explanation is tidy" is not evidence.
    # These fill in the gaps: if the pass survives 0.12-0.16 it is a plateau with
    # a real basin; if only 0.14 passes it is a knife edge tuned to the bars and
    # must NOT be promoted on the scorecard alone.
    "hl012_ds055":  ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                      "BEAT_HAND_LEAD": "0.12"}, []),
    "hl016_ds055":  ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                      "BEAT_HAND_LEAD": "0.16"}, []),
    # a different SEED at the winning setting: same target asymmetry, different
    # lead/swap pattern. If the pass is real it survives re-seeding; if it does not,
    # the arm was fitted to one particular arrangement of leads.
    "hl014_seed1_ds055": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                           "BEAT_HAND_LEAD": "0.14",
                           "BEAT_HAND_LEAD_SEED": "1"}, []),
    # lower density on the winner -- playfeel 0.76 has room, and fewer notes is the
    # other route toward the human double share (0.785 vs 0.231).
    "hl014_ds05":   ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.50",
                      "BEAT_HAND_LEAD": "0.14"}, []),

    # --- EMPIRICAL NOISE FLOOR (2026-08-01) --------------------------------
    # Seeds 0 and 1 of the SAME configuration scored handrole_gap 1.04 and 0.26 --
    # a spread of 0.78 against a documented floor of +-0.29. The floor is wrong by
    # ~3x on the axis we care most about, which means several of today's
    # fine-grained handrole rankings (e17 vs e15 especially) are not resolvable.
    # These three more seeds give 5 samples of an identical config, i.e. an actual
    # per-axis variance estimate rather than an assumed one. Cheapest possible way
    # to stop over-reading small differences across every future sweep.
    "hl014_seed2_ds055": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                           "BEAT_HAND_LEAD": "0.14",
                           "BEAT_HAND_LEAD_SEED": "2"}, []),
    "hl014_seed3_ds055": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                           "BEAT_HAND_LEAD": "0.14",
                           "BEAT_HAND_LEAD_SEED": "3"}, []),
    "hl014_seed4_ds055": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                           "BEAT_HAND_LEAD": "0.14",
                           "BEAT_HAND_LEAD_SEED": "4"}, []),

    # --- A8 / BEAT-GRID QUANTISATION (2026-08-02) ---------------------------
    # Axis A8 measured our notes against the AUDIO for the first time and found
    # them 75-82% on a real onset (human 96.6%) with a timing scatter of 11.7ms
    # (human 8.7ms). The offset HISTOGRAM says where that comes from: human
    # offsets are a unimodal peak on the onset, ours are FLAT across the whole
    # +-50ms window. Flat is the signature of a grid.
    #
    # `_quantize_to_beat_grid` snaps every Stage-1 onset to a 1/8 grid: spacing
    # 46ms, so displacement is uniform on +-23ms, predicted MAD 11.6ms against a
    # MEASURED 11.7ms. And the grid is built from the detected bpm, which is exact
    # on 1 of 21 eval songs (median error 0.74%, four songs at 2/3 tempo), so it
    # also slides against the music as the song goes on. Stage-1's own frames are
    # 11.6ms apart -- the model's timing is 4x finer than what this leaves of it.
    #
    # Arms halve the displacement bound in turn (23.2 -> 11.6 -> 5.8 -> 0 ms).
    # VERDICT LOGIC: if offset_mad_ms falls with the bound, the scatter was
    # quantisation and this is a decode-time fix for the defect Kyle actually
    # hears. If it does NOT fall, the scatter is the model's, this lever is dead,
    # and the alignment work moves to Stage 1 (where the fix is a retrain).
    # WATCH: q0 puts notes off the 1/16 grid entirely, and human maps are 94-99%
    # ON that grid -- expect A2 rhythm's offgrid guard to push back, which is the
    # real trade-off this sweep is here to price.
    # ☠ RETIRED BEFORE THEY RAN — `BEAT_GRID_SUBDIV` is a NO-OP on the production
    # path. It gates `_quantize_to_beat_grid`, which belongs to the older
    # frame-based decoder; the v7/v10 path emits on Stage-1's own slot grid, and a
    # q16 arm produced a map byte-identical in grid terms to the q8 control (zero
    # odd-16ths in either). Caught by checking the first generated map against the
    # control instead of trusting the flag — the sweep was killed 4 minutes in.
    #   "q16_ds055", "q32_ds055", "q0_ds055", "q16_hl014_ds055", "q0_hl014_ds055"
    # The premise was wrong too: human maps sit on the SAME 1/4-beat grid we do
    # (557 of 561 notes on 1f767). The grid is not too coarse, it is in the wrong
    # PLACE. Superseded by the oracle-bpm arms below.

    # --- A8 / IS IT THE TEMPO? (2026-08-02) ---------------------------------
    # Our detected bpm is exact on 1 of 21 eval songs: median error 0.74%, and four
    # songs land at 2/3 of the true tempo. Stage-1 places every note on a 1/4-beat
    # slot grid built from that bpm, so on nearly every song the grid slides against
    # the music as it plays -- which is exactly "the consistent beat of the song is
    # not where the notes are played" (Kyle, 2026-08-01).
    #
    # These arms hand the generator the TRUE bpm from the human map's Info.dat.
    # That is an ORACLE and cannot ship; it is here to settle attribution, because
    # no observational metric can separate "wrong tempo" from "wrong note choice".
    #
    # VERDICT LOGIC
    #   precision recovers toward the human 0.930 -> the defect IS tempo estimation.
    #       That is a solved problem outside this repo (beat trackers that return
    #       phase, or fitting the grid to the detected onsets we already compute for
    #       A8), and it becomes the top build item.
    #   precision barely moves -> the tempo is not the binding constraint. Suspects
    #       in order: grid PHASE (detect_bpm throws away librosa's beat positions
    #       and the grid is anchored at t=0), then Stage-1 slot selection itself.
    "obpm_ds055":       ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                          "BEAT_BPM_ORACLE": "outputs/true_bpm_eval_songset.json"}, []),
    "obpm_hl014_ds055": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                          "BEAT_HAND_LEAD": "0.14",
                          "BEAT_BPM_ORACLE": "outputs/true_bpm_eval_songset.json"}, []),
    "obpm_prod":        ({"BEAT_BPM_ORACLE": "outputs/true_bpm_eval_songset.json"}, []),

    # --- THE SHIPPABLE VERSION OF THE ORACLE (2026-08-02) -------------------
    # `BEAT_TEMPO_FIT=1` fits tempo AND phase to the per-stem onsets (the same
    # onsets A8 scores against) instead of trusting librosa's tempo scalar. On the
    # 23 eval songs it recovers the human-declared bpm EXACTLY on 21, where the
    # current detector manages 1. Unlike the obpm_* arms it reads no human map, so
    # whatever it earns here it keeps in production.
    #
    # Read these against the obpm_* arms, not against ds055/prod: the oracle is the
    # CEILING (a perfect tempo), and the question is how much of that ceiling a
    # real estimator reaches. Reaching it on 21 of 23 songs and losing 2 to
    # metrical-level ties is the expected shape.
    "tf_ds055":       ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                        "BEAT_TEMPO_FIT": "1"}, []),
    "tf_hl014_ds055": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55",
                        "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1"}, []),
    "tf_prod":        ({"BEAT_TEMPO_FIT": "1"}, []),

    # --- RE-TUNE DENSITY ON THE CORRECTED GRID (2026-08-02) -----------------
    # A correct tempo changes how many 1/4-beat slots exist per second, so every
    # density lever in this repo was fitted against the wrong grid. Measured on the
    # cached oracle arms rather than guessed:
    #
    #     arm                 nps   shift (human MADs)   notes/map
    #     human              3.909        --                --
    #     ds055              4.02       +0.20             800
    #     obpm_ds055         4.42       +0.88             850
    #
    # The scale that lands on the human median is 0.55 * 3.909/4.42 ~= 0.486, so
    # these bracket it. Without this, a corrected-tempo arm fails playfeel for a
    # reason that has nothing to do with alignment, and the fix would look like a
    # regression.
    "tf_ds045": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.45", "BEAT_TEMPO_FIT": "1"}, []),
    "tf_ds048": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48", "BEAT_TEMPO_FIT": "1"}, []),
    "tf_ds052": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.52", "BEAT_TEMPO_FIT": "1"}, []),
    # and the same on the hand-lead config, which is the best-known arm otherwise
    "tf_hl014_ds048": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                        "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1"}, []),

    # --- K1 TAIL TRIM (2026-08-03) ------------------------------------------
    # Kyle: "notes playing about 5 seconds after the song ends". Measured, 8/24
    # of our maps place notes past the last detected onset; 1f8d6 runs 11 notes
    # 4.43 s past it, against a human corpus that essentially never does.
    # BEAT_TRIM_TAIL is the grace in seconds allowed after the last librosa
    # onset. Single-song probe on 1f8d6: tail notes 11 -> 2, tail seconds
    # 4.43 -> 0.53, and it costs 4 notes out of 494.
    # ★ It is NOT expected to fix the drift -- the probe moved drift only
    # 0.429 -> 0.378 against a human p90 of 0.145. Tail notes and end-of-song
    # decay are two defects; this arm prices the cheap one and, more importantly,
    # checks the trim does not regress the other five axes.
    "tf_hl014_ds048_trim": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                             "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1",
                             "BEAT_TRIM_TAIL": "0.5"}, []),

    # --- K1 DECAY: ONSET-EVIDENCE WEIGHTING (2026-08-03) ---------------------
    # Measured cause: on 1f8d6's outro, windows with ZERO detected onsets carry
    # wmean 0.28-0.42 -- as high as the body of the song -- so density-select
    # hands ~35 notes to a region with ~2 real onsets. wmean IS the defect, so
    # nothing computed from it can fix this. These arms multiply the window
    # weight by an INDEPENDENT signal: audio onset density from librosa.
    #
    # Single-song probe on 1f8d6 (seed 0, on top of trim): drift 0.378 -> 0.130
    # (human p90 0.145), q5 precision 0.622 -> 0.86, overall precision
    # 0.886 -> 0.930, note count 490 -> 501. ★ ONE SONG, ONE SEED -- this is
    # exactly the single-song probe trap in the landmine list, hence this sweep.
    #
    # ⚠️ THE RISK TO WATCH IS DETECTOR-FITTING. We weight by librosa-on-mix and
    # A8 scores against a per-stem onset union; the two correlate, so some of the
    # precision gain may be fitting the grader rather than the music. Two checks:
    # (a) do the other five axes stay flat, and (b) does the gain concentrate on
    # the "ours alone" songs (1f336, 1f3d7, 1f767, 1f65d, 1f333) rather than on
    # the ones whose HUMAN map drifts too (1f8d6, 1f8ce)? Landing far ABOVE human
    # precision would be the tell that we are grading ourselves.
    "tf_hl014_ds048_trim_ev05": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                                  "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1",
                                  "BEAT_TRIM_TAIL": "0.5",
                                  "BEAT_ONSET_EVIDENCE": "0.5"}, []),
    "tf_hl014_ds048_trim_ev10": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                                  "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1",
                                  "BEAT_TRIM_TAIL": "0.5",
                                  "BEAT_ONSET_EVIDENCE": "1.0"}, []),

    # --- BUDGET ALLOCATION vs PRECISION (2026-08-02) ------------------------
    # After the tempo fix, the entire remaining alignment gap is onset_precision
    # (0.902 vs human 0.930; scatter is already better than human). Two candidate
    # causes were tested and BOTH behaved opposite to intuition:
    #
    # 1. Total density does NOT move precision. Across tf_ds045/048/052/055 the
    #    note rate goes 3.63 -> 4.42 nps and precision stays 0.895/0.902/0.904/
    #    0.902. Logged as a prediction beforehand and falsified.
    # 2. Stage-1 probability DOES know where the music is -- AUROC 0.755 against
    #    "this slot sits on a detected onset", and its top decile is 0.986 precise
    #    against a 0.687 base rate. So this is NOT purely a representation gap.
    #
    # Replaying selection policies over a BEAT_PROBS_DUMP at a fixed budget shows
    # where it actually lives -- the budget ALLOCATION, in the opposite direction
    # from the obvious guess:
    #
    #     global top-k by probability     0.948
    #     per-window gamma = 1.0          0.944
    #     per-window gamma = 2.5 (ship)   0.937
    #     per-window gamma = 4.0          0.919
    #     per-window gamma = 8.0          0.894
    #
    # A HIGH gamma concentrates the budget into loud windows, which forces more
    # notes deeper down those windows' probability ranking while starving quiet
    # windows that hold a few excellent onsets. A probability FLOOR does nothing
    # (0.937 unchanged at every quantile) because per-window top-k already avoids
    # the weak slots within a window.
    #
    # VERDICT LOGIC: gamma 1.0-1.5 should buy ~+0.01 precision, which is about the
    # +0.0067 needed to clear the alignment bar. WATCH density_corr: gamma was
    # raised to 2.5 on 2026-06-30 precisely to make density track the music
    # (+0.53, 5/6 songs). If alignment and density_corr trade off directly, that is
    # a real tension to report, not a knob to quietly pick a side on.
    "tf_g1_ds048":   ({"DENSITY_SELECT": "1", "DENSITY_SELECT_GAMMA": "1.0",
                       "BEAT_DIFFICULTY_SCALE": "0.48", "BEAT_TEMPO_FIT": "1"}, []),
    "tf_g15_ds048":  ({"DENSITY_SELECT": "1", "DENSITY_SELECT_GAMMA": "1.5",
                       "BEAT_DIFFICULTY_SCALE": "0.48", "BEAT_TEMPO_FIT": "1"}, []),
    "tf_hl014_g15_ds048": ({"DENSITY_SELECT": "1", "DENSITY_SELECT_GAMMA": "1.5",
                            "BEAT_DIFFICULTY_SCALE": "0.48", "BEAT_HAND_LEAD": "0.14",
                            "BEAT_TEMPO_FIT": "1"}, []),

    # --- RHYTHMICALLY COHERENT THINNING (2026-08-02) ------------------------
    # Re-tuning density to the human rate COSTS rhythm, and the sub-metrics say
    # exactly why (shift in human MADs):
    #
    #     arm         nps   pulse_stability  ioi_cond_entropy   gap
    #     tf_ds055   4.42        -0.06            +0.47        0.25
    #     tf_ds048   3.88        -0.66            +1.20        0.64
    #     tf_ds045   3.63        -1.11            +1.61        1.06
    #
    # Removing notes makes the map lose its PULSE and makes intervals less
    # predictable, because thinning by probability keeps confident notes wherever
    # they happen to fall and breaks the runs that make a rhythm legible. Humans at
    # 3.9 nps have a pulse; we at 3.9 nps (thinned from 4.4) do not.
    #
    # `BEAT_IOI_PRIOR` switches selection from per-window top-k to `_ioi_dp_select`,
    # which SAMPLES from softmax(log p + lam * log P(interval | previous)) using the
    # human interval bigram. It was built on 2026-07-27 for precisely this and has
    # been default-off ever since — and it could not have been judged fairly before
    # today, because the grid it samples on was wrong on 20 of 21 songs.
    #
    # VERDICT LOGIC: rhythm recovers toward tf_ds055's 0.25 while playfeel keeps the
    # re-tune's gain -> the pair is the answer and density becomes promotable.
    # Rhythm recovers but PRECISION drops -> sampling costs alignment (it is
    # deliberately not greedy), which is the same trade the gamma sweep is pricing;
    # read the two together before choosing.
    "tf_ioi05_ds048": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                        "BEAT_TEMPO_FIT": "1", "BEAT_IOI_PRIOR": "0.5"}, []),
    "tf_ioi1_ds048":  ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                        "BEAT_TEMPO_FIT": "1", "BEAT_IOI_PRIOR": "1.0"}, []),
    "tf_hl014_ioi1_ds048": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                             "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1",
                             "BEAT_IOI_PRIOR": "1.0"}, []),
    # Seeds, because a single run of this arm could not be compared to anything.
    # The 5-seed floor measured 2026-08-02 is alignment sd 0.092, flow 0.116,
    # handrole 0.317 — so a one-seed comparison against the 5-seed baseline would
    # be exactly the unresolvable difference this session spent the night
    # documenting.
    "tf_hl014_ioi1_ds048_s1": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                                "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1",
                                "BEAT_IOI_PRIOR": "1.0",
                                "BEAT_HAND_LEAD_SEED": "1"}, []),
    "tf_hl014_ioi1_ds048_s2": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                                "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1",
                                "BEAT_IOI_PRIOR": "1.0",
                                "BEAT_HAND_LEAD_SEED": "2"}, []),

    # --- SEEDS OF THE BEST ARM (2026-08-02) ---------------------------------
    # `tf_hl014_ds048` scores alignment 0.40 against a 0.39 bar -- a "fail" by
    # 0.01, and THERE IS NO MEASURED NOISE FLOOR FOR THE ALIGNMENT AXIS. The
    # 5-seed floor run (2026-08-01) predates A8 entirely, so calling 0.40 a fail
    # asserts a precision the suite has never demonstrated. That is the same
    # mistake as the assumed +-0.29 handrole floor, which turned out to be ~3x
    # understated.
    #
    # These four seeds do three jobs at once: give A8 its first measured floor,
    # test whether the 4/6 is stable or another seed lottery (2 of 5 identical
    # seeds passed 5/5 last time), and satisfy the re-seed precondition that every
    # verdict script in this repo now demands before a promotion.
    "tf_hl014_ds048_s1": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                           "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1",
                           "BEAT_HAND_LEAD_SEED": "1"}, []),
    "tf_hl014_ds048_s2": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                           "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1",
                           "BEAT_HAND_LEAD_SEED": "2"}, []),
    "tf_hl014_ds048_s3": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                           "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1",
                           "BEAT_HAND_LEAD_SEED": "3"}, []),
    "tf_hl014_ds048_s4": ({**_DS25, "BEAT_DIFFICULTY_SCALE": "0.48",
                           "BEAT_HAND_LEAD": "0.14", "BEAT_TEMPO_FIT": "1",
                           "BEAT_HAND_LEAD_SEED": "4"}, []),
}

# --- TRACK B / B-1 (2026-07-30): score the instrument retrain BY THE SUITE. ---
# Two families, added programmatically so the epoch subset lives in one place:
#   b1_e<NN>        -- version_8 epoch NN at PROD density. Directly comparable to
#                      `prod` (version_4, no instrument features) and to `v7instr`
#                      (the confounded B-0 arm). Isolates the representation.
#   b1_e<NN>_ds055  -- the same ckpt at the Track A difficulty scale, because the
#                      density lever composes with everything and prod density is
#                      a tier too dense to judge anything at (2026-07-29).
# Verdict logic: if the best-by-suite b1 epoch beats `prod` on the 5-axis scorecard,
# the instrument representation earns its retrain; if the whole epoch CURVE sits at
# or below prod, B-0's regressions were the representation after all, not the
# undertraining, and Track B needs B-2 (per-stem MERT) rather than more epochs.
for _ep in B1_EPOCHS:
    ARMS[f"b1_e{_ep:02d}"] = (
        _DS25, ["--beat-ckpt", _b1_ckpt(_ep), "--use-instr"],
    )
    ARMS[f"b1_e{_ep:02d}_ds055"] = (
        {**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55"},
        ["--beat-ckpt", _b1_ckpt(_ep), "--use-instr"],
    )
    # + HAND LEAD (2026-08-01). The B-1 interim showed the instrument model and
    # BEAT_HAND_LEAD are attacking the SAME variable from opposite ends: as
    # version_8 trains, its double share falls 0.978 -> 0.890 and role_asymmetry
    # rises 0.0138 -> 0.0420, which is exactly what the lever forces post-hoc.
    # So the interesting candidate is the two together, not either alone -- a
    # learned representation that un-locksteps the hands, plus a lever that gives
    # one of them the lead. Only worth GENERATING for the best epoch; the family
    # is defined here so no code change is needed once that epoch is known.
    ARMS[f"b1_e{_ep:02d}_ds055_hl014"] = (
        {**_DS25, "BEAT_DIFFICULTY_SCALE": "0.55", "BEAT_HAND_LEAD": "0.14"},
        ["--beat-ckpt", _b1_ckpt(_ep), "--use-instr"],
    )
    # LOWER DENSITY on top of the instrument model (2026-08-01). b1_e12_ds055
    # PASSES handrole -- the first arm ever to, at judgeable density -- and its
    # three remaining failures are all knife-edge AND all density-sensitive: flow
    # sits exactly on the 0.50 bar, playfeel misses by 0.03, idiom's spread by
    # 0.02. The instrument model already emits fewer notes than the control (656
    # vs 800), so a smaller scale is the cheapest shot at converting all three at
    # once without touching the handrole win.
    ARMS[f"b1_e{_ep:02d}_ds05"] = (
        {**_DS25, "BEAT_DIFFICULTY_SCALE": "0.50"},
        ["--beat-ckpt", _b1_ckpt(_ep), "--use-instr"],
    )
    ARMS[f"b1_e{_ep:02d}_ds045"] = (
        {**_DS25, "BEAT_DIFFICULTY_SCALE": "0.45"},
        ["--beat-ckpt", _b1_ckpt(_ep), "--use-instr"],
    )

sys.path.insert(0, str(REPO / "scripts"))
from eval_alignment import _separate_stems, _detect_onsets_librosa, _load_generated_beatmap, _beat_to_seconds  # noqa: E402
from eval_density_corr import _bin_counts, _spearman, _pearson  # noqa: E402

# Map-only quality axes (no Demucs): row/col spread, grid coverage, dir variety,
# monotony (the original complaint) + playability. Shared with the human-baseline
# command via scripts/map_metrics.py so every metric is computed identically.
try:
    from map_metrics import map_metrics, HUMAN_TARGET, BETTER  # noqa: E402
    from best_of_n_poc import swing_violations  # noqa: E402
    _HAVE_MAP = True
except Exception as _e:  # noqa: BLE001
    print(f"(map-quality axes unavailable: {_e})")
    _HAVE_MAP = False
    HUMAN_TARGET, BETTER = {}, {}


def _list_songs() -> list[pathlib.Path]:
    return sorted(p for p in SONGSET.glob("*") if p.suffix.lower() in (".ogg", ".mp3"))


def _ref_npz(song: pathlib.Path) -> pathlib.Path:
    return song.with_suffix(".ref.npz")


def _get_ref(song: pathlib.Path) -> tuple[np.ndarray, float]:
    """Reference onset times (drums∪other librosa) + duration; cached per song."""
    cache = _ref_npz(song)
    if cache.exists():
        d = np.load(cache)
        return d["ref_times"], float(d["duration"])
    import librosa
    dur = float(librosa.get_duration(path=str(song)))
    stems = _separate_stems(song, SR)
    drum_on = _detect_onsets_librosa(stems.get("drums", np.zeros(1)), SR)
    other_on = _detect_onsets_librosa(stems.get("other", np.zeros(1)), SR)
    ref_times = np.union1d(drum_on, other_on)
    np.savez(cache, ref_times=ref_times, duration=dur)
    return ref_times, dur


def build_songset(n: int) -> None:
    SONGSET.mkdir(parents=True, exist_ok=True)
    import librosa
    have = _list_songs()
    print(f"songset has {len(have)} songs; target {n}")
    raw = sorted((REPO / "data" / "raw").glob("*.zip"))
    for zp in raw:
        if len(_list_songs()) >= n:
            break
        name = zp.stem
        dst = SONGSET / f"{name}.ogg"
        if dst.exists():
            continue
        try:
            with zipfile.ZipFile(zp) as zf:
                egg = next((m for m in zf.namelist() if m.lower().endswith((".egg", ".ogg"))), None)
                if not egg:
                    continue
                data = zf.read(egg)
            dst.write_bytes(data)
            dur = float(librosa.get_duration(path=str(dst)))
            if dur < MIN_DUR:
                dst.unlink()
                continue
            print(f"  + {name}  dur={dur:.0f}s  — computing ref onsets …")
            _get_ref(dst)
        except Exception as e:
            print(f"  ! {name}: {e}")
            if dst.exists():
                dst.unlink()
    final = _list_songs()
    print(f"songset now {len(final)} songs: {[s.stem for s in final]}")


def _true_bpm(song: pathlib.Path) -> float | None:
    """BPM declared in the human map for this song, if we have it.

    Tempo detection is wrong on 30% of the eval set (7/23 songs; see
    scripts/bpm_octave_probe.py), including two at exactly half tempo, where the
    beat grid is twice as coarse in real time and the fast notes cannot be
    represented at all. Worse, the mis-tempo maps score BETTER on the beat-domain
    rhythm axis, so the confound actively distorts our measurements.

    Passing the human-declared BPM removes tempo detection as a confound from
    evaluation. This is an EVALUATION-ONLY fix — production has no human map to
    read a BPM from, and the detector itself still needs real work.
    """
    src = REPO / "data" / "raw" / f"{song.stem}.zip"
    if not src.exists():
        return None
    try:
        from feel_disc_poc import _zip_bpm
        b = _zip_bpm(str(src))
        return float(b) if b else None
    except Exception:  # noqa: BLE001
        return None


def _gen(label: str, arm: str, song: pathlib.Path, force: bool,
         true_bpm: bool = False, seed: int | None = None) -> pathlib.Path | None:
    """Generate one map. `arm` selects the config; `label` names the cache entry.

    They differ when an arm is replicated across seeds (`--seeds N`), where the
    label carries the seed so each replicate caches separately.
    """
    env_over, extra = ARMS[arm]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{label}__{song.stem}.zip"
    if out.exists() and not force:
        return out
    if true_bpm:
        b = _true_bpm(song)
        if b:
            extra = [*extra, "--bpm", str(b)]
    env = dict(os.environ)
    env.update(env_over)
    if seed is not None:
        # Read by scripts/generate.py -> generation.seeding.seed_everything.
        # Without it, decode sampling and post-processing draw from unseeded
        # RNGs and an "identical" arm scores differently every run.
        env["BSA_SEED"] = str(seed)
    cmd = [
        sys.executable, "scripts/generate.py", str(song), "--v7", "--difficulty", "Expert",
        "--beat-ckpt", BEAT_CKPT, "--layout-ckpt", LAYOUT_CKPT,
        # Decode at production generate.py defaults. Promoted 2026-07-23 to
        # temp 0.9/top_p 0.97 (closes grid_cov/dir_entropy vs human at h_dist 0.05).
        # Arms can still override via extra flags.
        "--section-gate", "loud_only", "--temperature", "0.9", "--top-p", "0.97",
        "--output", str(out), *extra,
    ]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print(f"  ! gen failed {label}/{song.stem}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'rc='+str(r.returncode)}")
        return None
    return out


def _score(zip_path: pathlib.Path, ref_times: np.ndarray, duration: float) -> dict:
    notes, bpm = _load_generated_beatmap(zip_path, "Expert")
    gen_times = np.array(sorted(_beat_to_seconds(b, bpm) for b, _x, _c in notes), dtype=np.float64)
    dur = float(max(duration, gen_times.max() if len(gen_times) else 0.0))
    gen_d = _bin_counts(gen_times, dur, WIN_SEC)
    ref_d = _bin_counts(ref_times, dur, WIN_SEC)
    n = min(len(gen_d), len(ref_d))
    gen_d, ref_d = gen_d[:n], ref_d[:n]
    # onset-alignment proxy: fraction of generated notes within 50 ms of a real
    # reference onset (are notes placed on actual musical events?).
    onset_hit = None
    if len(ref_times) and len(gen_times):
        ref_sorted = np.sort(ref_times)
        idx = np.searchsorted(ref_sorted, gen_times).clip(1, len(ref_sorted) - 1)
        dl = np.abs(gen_times - ref_sorted[idx - 1])
        dr = np.abs(gen_times - ref_sorted[idx])
        onset_hit = float((np.minimum(dl, dr) <= 0.05).mean())
    rec = {
        "spearman": _spearman(gen_d, ref_d),
        "pearson": _pearson(gen_d, ref_d),
        "gen_cv": float(gen_d.std() / gen_d.mean()) if gen_d.mean() else 0.0,
        "n_windows": int(n),
        "onset_hit": onset_hit,
        "monotony": None, "row_conc": None, "viol": None,
    }
    if _HAVE_MAP:
        try:
            rec.update(map_metrics(zip_path, "Expert"))  # row_conc, col_conc, grid_coverage, dir_entropy, monotony, pattern_repeat, nps, n_notes
        except Exception:  # noqa: BLE001
            pass
        try:
            rec["viol"] = swing_violations(zip_path, "Expert")
        except Exception:  # noqa: BLE001
            pass
    # eval-suite v2 axis A1 — flow/ergonomics (sequence-aware). Reported per map
    # here; the ARM is ranked by the COHORT statistic (flow_gap), never by a mean
    # of per-map distances — see docs/eval_suite_v2.md §A1 lesson 1.
    try:
        rec.update(_flow_metrics_for(zip_path))
    except Exception:  # noqa: BLE001
        pass
    # A3 idiom + A6 hand role. These were missing from the sweep for a while, so
    # a sweep report showed only two of the four scored axes and hand role -- our
    # largest defect -- was invisible unless you ran the scorecard separately.
    try:
        rec.update(_axis_metrics_for(zip_path))
    except Exception:  # noqa: BLE001
        pass
    return rec


def _axis_metrics_for(zip_path: pathlib.Path) -> dict:
    from beatsaber_automapper.data.beatmap import ColorNote
    from beatsaber_automapper.evaluation import handrole as _hro
    from beatsaber_automapper.evaluation import idiom as _idm
    from eval_contour_follow import _load_notes_with_direction

    recs = _load_notes_with_direction(zip_path, "Expert")
    if not recs:
        return {}
    notes = [ColorNote(beat=b, x=int(x), y=int(y), color=int(c), direction=int(d))
             for (b, x, y, c, d) in recs]

    class _BM:
        color_notes = sorted(notes, key=lambda n: n.beat)
        bomb_notes: list = []

    bm = _BM()
    return {**_idm.idiom_metrics(bm).metrics, **_hro.handrole_metrics(bm).metrics}


def _flow_metrics_for(zip_path: pathlib.Path) -> dict:
    from beatsaber_automapper.data.beatmap import ColorNote
    from beatsaber_automapper.evaluation import flow as _fl
    from eval_contour_follow import _load_notes_with_direction
    from feel_disc_poc import _zip_bpm

    recs = _load_notes_with_direction(zip_path, "Expert")
    if not recs:
        return {}
    notes = [ColorNote(beat=b, x=int(x), y=int(y), color=int(c), direction=int(d))
             for (b, x, y, c, d) in recs]

    class _BM:
        color_notes = notes
        bomb_notes: list = []

    return dict(_fl.flow_metrics(_BM(), bpm=float(_zip_bpm(str(zip_path)) or 120.0)).metrics)


def _load_human_baseline() -> None:
    """Refresh HUMAN_TARGET from a cached human-baseline run, if present."""
    f = CACHE / "human_baseline.json"
    if _HAVE_MAP and f.exists():
        try:
            for k, v in json.loads(f.read_text()).items():
                if isinstance(v, dict) and "mean" in v:
                    HUMAN_TARGET[k] = round(v["mean"], 3)
        except Exception:  # noqa: BLE001
            pass


def _acquire_cache_lock() -> pathlib.Path:
    """Refuse to run two sweeps against the same cache directory.

    Learned the hard way 2026-07-27: an overnight script was accidentally
    launched twice, both instances wrote the same `outputs/eval_sweep_cache/
    <arm>__<song>.zip` paths concurrently, and 11 map zips came out corrupt
    ("Bad CRC-32", "Bad magic number for central directory"). The scores for
    those maps were silently lost. A lock is cheaper than re-running a sweep.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = CACHE / ".sweep.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except Exception:  # noqa: BLE001
            pid = -1
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise SystemExit(
                f"another sweep is already running (pid {pid}). Concurrent sweeps "
                f"corrupt the cached maps — wait for it, or remove {lock} if that "
                f"process is definitely dead.")
        print(f"(clearing stale lock from dead pid {pid})")
    lock.write_text(str(os.getpid()))
    return lock


def sweep(arms: list[str], force: bool, true_bpm: bool = False,
          seeds: int = 0) -> None:
    lock = _acquire_cache_lock()
    try:
        _sweep_inner(arms, force, true_bpm, seeds)
    finally:
        lock.unlink(missing_ok=True)


def _expand_seeds(arms: list[str], seeds: int,
                  true_bpm: bool = False) -> list[tuple[str, str, int | None]]:
    """Expand each arm into `seeds` replicates -> [(label, arm, seed), ...].

    seeds=0 keeps the historical one-unseeded-run-per-arm behaviour. Replicates
    are labelled `<arm>#s<n>` so every downstream table treats them as ordinary
    arms and the final aggregation can group them back by base arm.

    A `--true-bpm` run gets its own `#truebpm` label too. It used to share a
    cache key with the normal run of the same arm and silently overwrite it —
    a landmine that only stayed harmless because the flag was never forwarded.
    """
    suf = "#truebpm" if true_bpm else ""
    if seeds <= 0:
        return [(f"{a}{suf}", a, None) for a in arms]
    return [(f"{a}{suf}#s{n}", a, n) for a in arms for n in range(seeds)]


def _sweep_inner(arms: list[str], force: bool, true_bpm: bool = False,
                 seeds: int = 0) -> None:
    songs = _list_songs()
    if not songs:
        print("no songs — run: eval_sweep.py build-songset --n 6")
        return
    _load_human_baseline()
    plan = _expand_seeds(arms, seeds, true_bpm)
    base_of = {label: arm for label, arm, _ in plan}
    labels = [label for label, _, _ in plan]
    if seeds > 0:
        print(f"sweep: {len(arms)} arms × {seeds} seeds × {len(songs)} songs "
              f"= {len(plan) * len(songs)} maps\n")
    else:
        print(f"sweep: {len(arms)} arms × {len(songs)} songs\n")
    refs = {s: _get_ref(s) for s in songs}
    results: dict[str, dict[str, dict]] = {}
    import time as _time
    for ai, (label, arm, seed) in enumerate(plan, 1):
        results[label] = {}
        for si, s in enumerate(songs, 1):
            t0 = _time.time()
            zp = _gen(label, arm, s, force, true_bpm, seed)
            if zp is None:
                continue
            try:
                rec = _score(zp, *refs[s])
                results[label][s.stem] = rec
                print(f"  [{ai}/{len(plan)} {label}] [{si}/{len(songs)} {s.stem[:14]}] "
                      f"row_conc={rec.get('row_conc')} spear={rec.get('spearman'):+.2f} "
                      f"viol={rec.get('viol')} ({_time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  ! score failed {label}/{s.stem}: {e}")
        done = results[label]
        sp = [v["spearman"] for v in done.values()]
        print(f"  [{label}] scored {len(done)}/{len(songs)}  mean Spearman={np.mean(sp):+.3f}" if sp else f"  [{label}] none scored")
    arms = labels

    song_names = [s.stem for s in songs]
    print("\n=== density_corr Spearman (DoD >= 0.41) ===")
    hdr = "arm".ljust(12) + "".join(s[:10].rjust(11) for s in song_names) + "      mean   #pass"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for arm in arms:
        row = results[arm]
        cells, sp, npass = [], [], 0
        for s in song_names:
            if s in row and row[s].get("spearman") is not None:
                v = row[s]["spearman"]; sp.append(v); npass += int(v >= 0.41)
                cells.append(f"{v:+.3f}".rjust(11))
            else:
                cells.append("    --     ")
        mean = float(np.mean(sp)) if sp else float("nan")
        summary[arm] = {
            "mean_spearman": mean, "n_pass": npass, "n_scored": len(sp),
            "per_song": {s: results[arm].get(s) for s in song_names},
        }
        # aggregate every numeric map/audio metric as a mean over songs
        for k in ("row_conc", "col_conc", "grid_coverage", "dir_entropy", "monotony",
                  "pattern_repeat", "onset_hit", "gen_cv", "nps", "n_notes"):
            vals = [r[k] for r in row.values() if r.get(k) is not None]
            summary[arm][f"mean_{k}"] = float(np.mean(vals)) if vals else None
        viol = [r["viol"] for r in row.values() if r.get("viol") is not None]
        summary[arm]["total_viol"] = int(np.sum(viol)) if viol else None
        # composite human-distance: mean |arm - human| / human over the map-shape
        # metrics that have a human target. Lower = more human-like layout.
        dists = []
        for k in ("row_conc", "col_conc", "grid_coverage", "dir_entropy", "monotony"):
            mv, hv = summary[arm].get(f"mean_{k}"), HUMAN_TARGET.get(k)
            if mv is not None and hv:
                dists.append(abs(mv - hv) / abs(hv))
        summary[arm]["human_dist"] = round(float(np.mean(dists)), 3) if dists else None
        print("".join([arm.ljust(12)] + cells) + f"   {mean:+.3f}    {npass}/{len(sp)}")

    # quality-vs-human table — every metric with its human target + arrow
    cols = [  # (summary key, header, human-target key in HUMAN_TARGET)
        ("mean_row_conc", "row_conc", "row_conc"), ("mean_col_conc", "col_conc", "col_conc"),
        ("mean_grid_coverage", "grid_cov", "grid_coverage"), ("mean_dir_entropy", "dir_ent", "dir_entropy"),
        ("mean_monotony", "monoton", "monotony"), ("mean_pattern_repeat", "prep", "pattern_repeat"),
        ("mean_onset_hit", "onset_hit", None),
        ("mean_gen_cv", "gen_cv", None), ("mean_nps", "nps", None), ("total_viol", "viol", None),
        ("human_dist", "h_dist↓", None),  # composite layout distance to human (lower=better)
    ]
    def arrow(htk):
        return {"low": "↓", "high": "↑"}.get(BETTER.get(htk), "") if htk else ""
    print("\n=== quality vs human (mean over songs) ===")
    print("arm".ljust(12) + "".join(f"{h}{arrow(tk)}".rjust(10) for _k, h, tk in cols))
    tgt = "HUMAN".ljust(12) + "".join(
        (f"{HUMAN_TARGET[tk]:.2f}" if tk and HUMAN_TARGET.get(tk) is not None else "·").rjust(10)
        for _k, _h, tk in cols)
    print(tgt); print("-" * (12 + 10 * len(cols)))
    for arm in arms:
        s = summary[arm]
        def _f(k, fmt="{:.3f}"):
            return (fmt.format(s[k]) if s.get(k) is not None else "--").rjust(10)
        print(arm.ljust(12) + "".join(
            _f(k, "{:.0f}" if k in ("total_viol",) else "{:.2f}") for k, _h, _tk in cols))

    # ---- v2 axis A1: flow/ergonomics, ranked by the COHORT statistic ----
    try:
        from beatsaber_automapper.evaluation import flow as _fl
        fcols = _fl.SEQUENCE_KEYS
        print("\n=== flow / ergonomics (v2 axis A1) — shift = median offset in human MADs ===")
        print("rank arms by flow_gap (mean |shift|); spread <1 = under-dispersed vs human")
        print("arm".ljust(12) + "".join(f"{k:>20s}" for k in fcols)
              + "crossover".rjust(11) + "flow_gap".rjust(10) + "min_spr".rjust(9))
        for arm in arms:
            rows = [r for r in results[arm].values() if r]
            cc = _fl.cohort_comparison(rows)
            if "_summary" not in cc:
                continue
            cells = "".join(
                f"{cc[k]['shift']:+9.2f}/{cc[k]['spread']:<10.2f}" if k in cc
                else f"{'--':>20s}" for k in fcols)
            xo = cc.get("crossover", {}).get("median")
            s = cc["_summary"]
            print(arm.ljust(12) + cells
                  + (f"{xo:11.3f}" if xo is not None else f"{'--':>11s}")
                  + f"{s['flow_gap']:10.2f}{s['min_spread']:9.2f}")
        print(f"{'HUMAN':12s}" + "".join(f"{'+0.00/1.00':>20s}" for _ in fcols)
              + f"{0.218:11.3f}{0.0:10.2f}{1.0:9.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"(flow axis unavailable: {e})")

    # ---- v2 axis A2: rhythm, also ranked by the cohort statistic ----
    try:
        from beatsaber_automapper.evaluation import rhythm as _rh
        print("\n=== rhythm (v2 axis A2) — our largest measured gap ===")
        print("arm".ljust(12) + "".join(f"{k:>20s}" for k in _rh.SEQUENCE_KEYS)
              + "dom_share".rjust(11) + "rhy_gap".rjust(9) + "min_spr".rjust(9))
        for arm in arms:
            rows = [r for r in results[arm].values() if r]
            cc = _rh.cohort_comparison(rows)
            if "_summary" not in cc:
                continue
            cells = "".join(
                f"{cc[k]['shift']:+9.2f}/{cc[k]['spread']:<10.2f}" if k in cc
                else f"{'--':>20s}" for k in _rh.SEQUENCE_KEYS)
            ds = cc.get("dominant_share", {}).get("median")
            s = cc["_summary"]
            print(arm.ljust(12) + cells
                  + (f"{ds:11.3f}" if ds is not None else f"{'--':>11s}")
                  + f"{s['rhythm_gap']:9.2f}{s['min_spread']:9.2f}")
        print(f"{'HUMAN':12s}" + "".join(f"{'+0.00/1.00':>20s}" for _ in _rh.SEQUENCE_KEYS)
              + f"{0.509:11.3f}{0.0:9.2f}{1.0:9.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"(rhythm axis unavailable: {e})")

    # ---- v2 axes A3 (idiom) + A6 (hand role), same cohort statistic ----
    for _title, _mod, _gapkey in (
        ("idiom (v2 axis A3)", "idiom", "idiom_gap"),
        ("HAND ROLE (v2 axis A6) — our worst axis", "handrole", "handrole_gap"),
    ):
        try:
            import importlib
            _m = importlib.import_module(f"beatsaber_automapper.evaluation.{_mod}")
            print(f"\n=== {_title} ===")
            print("arm".ljust(12) + "".join(f"{k:>20s}" for k in _m.SEQUENCE_KEYS)
                  + "gap".rjust(9) + "min_spr".rjust(9))
            for arm in arms:
                rows = [r for r in results[arm].values() if r]
                cc = _m.cohort_comparison(rows)
                if "_summary" not in cc:
                    continue
                cells = "".join(
                    f"{cc[k]['shift']:+9.2f}/{cc[k]['spread']:<10.2f}" if k in cc
                    else f"{'--':>20s}" for k in _m.SEQUENCE_KEYS)
                sm = cc["_summary"]
                print(arm.ljust(12) + cells
                      + f"{sm[_gapkey]:9.2f}{sm['min_spread']:9.2f}")
            print(f"{'HUMAN':12s}" + "".join(f"{'+0.00/1.00':>20s}"
                                             for _ in _m.SEQUENCE_KEYS)
                  + f"{0.0:9.2f}{1.0:9.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"({_title} unavailable: {e})")

    if seeds > 0:
        _seed_aggregate(base_of, labels)

    out = CACHE / "leaderboard.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    _write_report(summary, results, song_names, arms, cols)
    return summary, results, song_names


AXES = ("alignment", "rhythm", "flow", "idiom", "handrole", "playfeel")


def _score_label(label: str) -> dict | None:
    """Six-axis scorecard for one seed replicate, from its cached maps."""
    from beatsaber_automapper.evaluation import alignment, playfeel, scorecard
    zips = sorted(CACHE.glob(f"{label}__*.zip"))
    if not zips:
        return None
    loaded, prec, nps = [], [], []
    for p in zips:
        try:
            r = scorecard._load_any(p)
        except Exception:  # noqa: BLE001
            continue
        if not r:
            continue
        loaded.append(r)
        nps.append(playfeel.playfeel_metrics(r[0], bpm=r[1]).metrics["nps"])
        if r[2] is not None:
            prec.append(alignment.alignment_metrics(
                r[0], bpm=r[1], onsets=r[2]).metrics["onset_precision"])
    if not loaded:
        return None
    res = scorecard.score_cohort(loaded, label)
    fin = lambda v: [x for x in v if x == x]  # noqa: E731
    return {"axes": {a.name: a.gap for a in res["axes"]},
            "npass": sum(1 for a in res["axes"] if a.passed),
            "prec": float(np.median(fin(prec))) if fin(prec) else float("nan"),
            "nps": float(np.median(fin(nps))) if fin(nps) else float("nan")}


def _seed_aggregate(base_of: dict[str, str], labels: list[str]) -> None:
    """Report each arm as a mean ± sd over its seeds, and say what is resolvable.

    The point of the table. Five runs of a byte-identical config once scored
    4, 2, 1, 3 and 5 of six axes, so a single run cannot rank anything. An arm's
    score is the mean over seeds; a difference from the control smaller than
    2 sd is printed as noise rather than as a result.
    """
    bases: dict[str, list[str]] = {}
    for label in labels:
        bases.setdefault(base_of[label], []).append(label)

    # Keep the seed number, so two arms can be compared at the SAME seed.
    by_seed: dict[str, dict[int, dict]] = {}
    scored: dict[str, list[dict]] = {}
    for base, labs in bases.items():
        got = {}
        for lab in labs:
            r = _score_label(lab)
            if r:
                got[int(lab.rsplit("#s", 1)[1])] = r
        if got:
            by_seed[base] = got
            scored[base] = [got[k] for k in sorted(got)]
    if not scored:
        print("\n(seed aggregate unavailable: nothing scored)")
        return

    def ms(rows: list[dict], key: str) -> tuple[float, float, int]:
        v = [r["axes"][key] for r in rows
             if key in r["axes"] and r["axes"][key] == r["axes"][key]]
        if not v:
            return float("nan"), float("nan"), 0
        return (float(np.mean(v)),
                float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))

    order = list(scored)
    ctrl = order[0]
    print(f"\n=== SEED AGGREGATE — mean ± sd over seeds (control = {ctrl}) ===")
    print("An arm is its mean, not its luckiest run. A delta inside 2 sd is NOT a")
    print("result: the same config scored 4/2/1/3/5 of six axes before seeding.")
    print("\n" + "metric".ljust(12)
          + "".join(f"{b[:22]:>24s}" for b in order)
          + ("" if len(order) < 2 else f"{'delta':>9s}{'resolvable?':>13s}"))
    print("-" * (12 + 24 * len(order) + (0 if len(order) < 2 else 22)))

    for key in (*AXES, "npass", "prec", "nps"):
        if key in AXES:
            cells = [ms(scored[b], key) for b in order]
        else:
            cells = []
            for b in order:
                v = [r[key] for r in scored[b] if r[key] == r[key]]
                cells.append((float(np.mean(v)) if v else float("nan"),
                              float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                              len(v)))
        line = key.ljust(12) + "".join(f"{m:>16.3f} ±{s:<6.3f}" for m, s, _ in cells)
        if len(order) >= 2:
            (m0, s0, _), (m1, s1, _) = cells[0], cells[1]
            d = m1 - m0
            # 2 sd of the pooled per-seed spread. Anything smaller is the seed
            # lottery, not the lever under test.
            pooled = float(np.sqrt(s0 ** 2 + s1 ** 2))
            verdict = "yes" if abs(d) >= 2 * pooled and pooled > 0 else "NO (noise)"
            line += f"{d:>+9.3f}{verdict:>13s}"
        print(line)

    ident = [b for b in order if len(scored[b]) > 1
             and all(r["npass"] == scored[b][0]["npass"] for r in scored[b])]
    print(f"\nseeds per arm: " + ", ".join(f"{b}={len(scored[b])}" for b in order))
    if ident:
        print("reproducible pass count (all seeds agree): " + ", ".join(ident))

    # ---- paired comparison: same lever, same seed ----------------------------
    # Both arms start from the same RNG state, so their early decode draws match
    # and much of the seed effect cancels. The pairing is partial, not perfect —
    # the draw sequences diverge once the configs make different numbers of
    # decisions — so treat a narrower paired sd as an empirical result to check,
    # not an assumption. If it IS narrower, a lever can be ranked with far fewer
    # seeds than the unpaired 2 sd test needs.
    if len(order) >= 2:
        a, b = order[0], order[1]
        shared = sorted(set(by_seed[a]) & set(by_seed[b]))
        if len(shared) >= 2:
            print(f"\n=== PAIRED vs {a} (same seed both sides, n={len(shared)}) ===")
            print(f"{'axis':12s}{'paired delta':>16s}{'sd(paired)':>13s}"
                  f"{'sd(unpaired)':>15s}{'verdict':>13s}")
            for key in AXES:
                d = [by_seed[b][s]["axes"][key] - by_seed[a][s]["axes"][key]
                     for s in shared
                     if key in by_seed[a][s]["axes"] and key in by_seed[b][s]["axes"]]
                d = [x for x in d if x == x]
                if len(d) < 2:
                    continue
                md, sd = float(np.mean(d)), float(np.std(d, ddof=1))
                _, s0, _ = ms(scored[a], key)
                _, s1, _ = ms(scored[b], key)
                unp = float(np.sqrt(s0 ** 2 + s1 ** 2))
                verdict = "yes" if sd > 0 and abs(md) >= 2 * sd else "NO (noise)"
                print(f"{key:12s}{md:>+16.3f}{sd:>13.3f}{unp:>15.3f}{verdict:>13s}")
            print("\nIf sd(paired) << sd(unpaired), the seed effect is shared and")
            print("pairing is the cheaper way to rank arms. If they are similar,")
            print("the lever perturbs the decode too early for pairing to help.")


def _render(arm: str, song: str) -> str | None:
    """Render an arm's cached map for `song`; return a repo-relative png path."""
    zp = CACHE / f"{arm}__{song}.zip"
    if not zp.exists():
        return None
    rdir = CACHE / "renders"; rdir.mkdir(exist_ok=True)
    png = rdir / f"{arm}__{song}.png"
    if not png.exists():
        r = subprocess.run(
            [sys.executable, "scripts/render_map.py", str(zp), "--difficulty", "Expert",
             "--out", str(png), "--no-audio"], cwd=REPO, capture_output=True, text=True)
        if not png.exists():
            return None
    return str(png.relative_to(CACHE))


def _write_report(summary, results, song_names, arms, cols) -> None:
    """Emit a single self-contained report.md: tables vs human + embedded renders
    of the headline arm vs the control, so a sweep is judged at a glance."""
    import datetime
    lines = [f"# Eval sweep report — {datetime.datetime.now():%Y-%m-%d %H:%M}",
             f"\n{len(arms)} arms × {len(song_names)} songs. DoD: density_corr Spearman ≥ 0.41.\n"]
    # density_corr table
    lines.append("## density_corr (Spearman, DoD ≥ 0.41)\n")
    lines.append("| arm | " + " | ".join(s[:10] for s in song_names) + " | mean | #pass |")
    lines.append("|" + "---|" * (len(song_names) + 3))
    for arm in arms:
        per = summary[arm]["per_song"]
        cellvals = [(f"{per[s]['spearman']:+.2f}" if per.get(s) and per[s].get("spearman") is not None else "—") for s in song_names]
        lines.append(f"| {arm} | " + " | ".join(cellvals)
                     + f" | **{summary[arm]['mean_spearman']:+.3f}** | {summary[arm]['n_pass']}/{summary[arm]['n_scored']} |")
    # quality vs human table
    lines.append("\n## quality vs human (mean over songs)\n")
    hdr = [h for _k, h, _tk in cols]
    lines.append("| arm | " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    humanrow = [(f"{HUMAN_TARGET[tk]:.2f}" if tk and HUMAN_TARGET.get(tk) is not None else "·") for _k, _h, tk in cols]
    lines.append("| **HUMAN** | " + " | ".join(humanrow) + " |")
    for arm in arms:
        s = summary[arm]
        vals = [(f"{s[k]:.0f}" if k == "total_viol" and s.get(k) is not None
                 else f"{s[k]:.2f}" if s.get(k) is not None else "—") for k, _h, _tk in cols]
        lines.append(f"| {arm} | " + " | ".join(vals) + " |")
    # headline arm (best mean_spearman) vs first/control arm, rendered
    headline = max(arms, key=lambda a: summary[a]["mean_spearman"] if not np.isnan(summary[a]["mean_spearman"]) else -9)
    ctrl = "control" if "control" in arms else arms[0]
    song0 = song_names[0]
    lines.append(f"\n## renders — {song0}\n")
    for label, ra in (("control", ctrl), ("headline (best density_corr)", headline)):
        p = _render(ra, song0)
        lines.append(f"**{label}** (`{ra}`)\n" + (f"\n![{ra}]({p})\n" if p else "\n_(render unavailable)_\n"))
    (CACHE / "report.md").write_text("\n".join(lines))
    print(f"wrote {CACHE / 'report.md'}")


def human_baseline(n: int) -> dict:
    """Compute map-only metric distributions over n human maps from data/raw.

    Writes outputs/eval_sweep_cache/human_baseline.json and prints mean/p10/p90 so
    every metric in the leaderboard has a real human reference (not a hard-coded
    guess). Refreshes map_metrics.HUMAN_TARGET for this process.
    """
    if not _HAVE_MAP:
        print("map_metrics unavailable"); return {}
    raw = sorted((REPO / "data" / "raw").glob("*.zip"))
    rows: list[dict] = []
    for zp in raw:
        if len(rows) >= n:
            break
        for diff in ("Expert", "ExpertPlus", "Hard"):
            try:
                m = map_metrics(zp, diff)
                if m.get("n_notes", 0) > 20:
                    rows.append(m); break
            except Exception:  # noqa: BLE001
                continue
    if not rows:
        print("no human maps scored"); return {}
    keys = ("row_conc", "col_conc", "grid_coverage", "dir_entropy", "monotony",
            "pattern_repeat", "nps", "n_notes")
    base = {}
    print(f"\n=== HUMAN baseline (n={len(rows)} maps) ===")
    print("metric".ljust(16) + "mean".rjust(9) + "p10".rjust(9) + "p90".rjust(9))
    for k in keys:
        vals = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        if not len(vals):
            continue
        base[k] = {"mean": float(vals.mean()), "p10": float(np.percentile(vals, 10)),
                   "p90": float(np.percentile(vals, 90))}
        HUMAN_TARGET[k] = round(float(vals.mean()), 3)
        print(f"{k:16s}{vals.mean():9.3f}{np.percentile(vals,10):9.3f}{np.percentile(vals,90):9.3f}")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "human_baseline.json").write_text(json.dumps(base, indent=2))
    print(f"\nwrote {CACHE / 'human_baseline.json'}")
    return base


def main() -> None:
    try:  # line-buffer stdout so nohup/background sweeps show live progress
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-songset"); b.add_argument("--n", type=int, default=6)
    sw = sub.add_parser("sweep")
    sw.add_argument("--arms", default=None, help="comma list; default all")
    sw.add_argument("--force", action="store_true", help="regenerate even if cached")
    sw.add_argument("--seeds", type=int, default=0,
                    help="Run each arm N times with seeds 0..N-1 and score it as the "
                         "mean +- sd over them, flagging any difference inside 2 sd as "
                         "noise. 0 = one unseeded run per arm (the historical behaviour, "
                         "under which five identical configs scored 4/2/1/3/5 of six "
                         "axes). Use >= 3 before believing any ranking.")
    sw.add_argument("--true-bpm", action="store_true",
                    help="generate with the human map's declared BPM. Tempo detection is "
                         "wrong on 30%% of the eval set and the beat-domain rhythm axis "
                         "REWARDS the error, so this removes a real confound. Evaluation "
                         "only -- production has no human BPM to read.")
    hb = sub.add_parser("human-baseline"); hb.add_argument("--n", type=int, default=40)
    sub.add_parser("list-arms")
    a = ap.parse_args()
    if a.cmd == "build-songset":
        build_songset(a.n)
    elif a.cmd == "human-baseline":
        human_baseline(a.n)
    elif a.cmd == "list-arms":
        for k, (e, x) in ARMS.items():
            print(f"  {k:14s} env={e} flags={x}")
    elif a.cmd == "sweep":
        arms = a.arms.split(",") if a.arms else list(ARMS)
        bad = [x for x in arms if x not in ARMS]
        if bad:
            sys.exit(f"unknown arms: {bad}")
        # --true-bpm was parsed but never forwarded, so the flag silently did
        # nothing; fixed 2026-08-02 along with the seed work.
        sweep(arms, a.force, a.true_bpm, a.seeds)


if __name__ == "__main__":
    main()
