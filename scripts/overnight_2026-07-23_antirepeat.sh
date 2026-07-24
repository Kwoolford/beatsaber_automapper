#!/usr/bin/env bash
# Overnight 2026-07-23 — MONOTONY / ADJACENCY ANTI-REPEAT sweep.
#
# Context: the grid_coverage/dir_entropy gaps are CLOSED — temp 0.9/top_p 0.97 was
# promoted to generate.py production defaults (h_dist 0.19 -> 0.05) and composite
# monotony is now ~human (0.44 vs 0.43). Kyle picked "monotony / pattern_repeat"
# as the next target. A smoke test surfaced that `pattern_repeat` (adjacent-
# identical (x,y,dir) tuples) is ALREADY ~human (~0.0) in shipped maps, so the
# real question is whether a DETERMINISTIC adjacency lever can make the overall
# layout MORE human-like (lower composite h_dist) than pure stochastic decode,
# without the over-flattening the cumulative LAYOUT_DIV_* penalty causes.
#
# NEW lever = windowed adjacency anti-repeat (models/layout_model.py):
#   LAYOUT_ANTIREPEAT=W (recent-window size) + LAYOUT_AR_STRENGTH=S. Penalizes only
#   tokens emitted in the last-W steps PER ROLE (X/Y/DIR), breaking back-to-back
#   loops WITHOUT flattening the whole-phrase distribution. Smoke (1f333, W1/S2):
#   grid_cov 0.67->1.0, dir_ent 0.72->0.80(=human), monotony 0.43(=human),
#   col_conc 0.29(~human) — i.e. closes grid/dir while HOLDING human concentration.
#
# Arms (all on production density-select g2.5; control decodes at new prod 0.9/0.97):
#   prod        control = NEW PRODUCTION (temp 0.9/top_p 0.97, no anti-repeat)
#   ar_w1_s2    pure adjacency (W=1): forbid immediate per-role repeat
#   ar_w2_s2    2-step window, moderate
#   ar_w3_s3    3-step window, stronger loop-break
#   g2.5_div10  over-flatten reference (cumulative penalty; col_conc collapses)
#
# DoD (an arm "wins" if):
#     h_dist (composite layout distance to human) STRICTLY BELOW prod's h_dist
#   while HOLDING:
#     mean_spearman (density_corr) >= 0.41,   monotony <= 0.46 (not worse than base),
#     pattern_repeat <= 0.05 (human ~0.002),  col_conc >= 0.20 (NOT over-flattened),
#     row_conc <= 0.60,   total_viol == 0.
# WINNER => promote LAYOUT_ANTIREPEAT=W/LAYOUT_AR_STRENGTH=S to production layout
# config and render vs prod for Kyle. If NO arm beats prod's h_dist without
# over-flattening (col_conc<0.20) or spiking monotony, the temp-nudge prod is the
# ceiling for no-retrain layout levers -> next step is a targeted diversity-reg
# fine-tune (distinct from the superseded entropy-reg which over-diversified).
set -euo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
LOG=logs/overnight/antirepeat_2026-07-23.log
mkdir -p logs/overnight
echo "=== monotony/anti-repeat sweep @ $(date) ===" | tee "$LOG"

# --force: regenerate every arm (control included) so all share identical code.
python scripts/eval_sweep.py sweep \
    --arms prod,ar_w1_s2,ar_w2_s2,ar_w3_s3,g2.5_div10 --force 2>&1 | tee -a "$LOG"

echo "=== VERDICT @ $(date) ===" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import json
lb = json.load(open("outputs/eval_sweep_cache/leaderboard.json"))
prod = lb.get("prod", {})
prod_h = prod.get("human_dist")
print(f"prod (control) human_dist = {prod_h}")
print(f"{'arm':12s} {'h_dist':>7s} {'grid':>6s} {'dir':>6s} {'monot':>6s} {'prep':>6s} "
      f"{'col':>6s} {'row':>6s} {'dens':>6s} {'viol':>4s}  verdict")
print("-" * 92)
best = None
for arm, d in lb.items():
    h = d.get("human_dist"); gc = d.get("mean_grid_coverage", 0.0)
    de = d.get("mean_dir_entropy", 0.0); mono = d.get("mean_monotony", 1.0)
    prep = d.get("mean_pattern_repeat", 1.0); cc = d.get("mean_col_conc", 0.0)
    rc = d.get("mean_row_conc", 1.0); dens = d.get("mean_spearman", 0.0)
    viol = d.get("total_viol", 0)
    hold = (dens >= 0.41 and mono <= 0.46 and prep <= 0.05
            and cc >= 0.20 and rc <= 0.60 and viol == 0)
    beats = (arm != "prod" and h is not None and prod_h is not None and h < prod_h)
    ok = hold and beats
    if arm == "prod":
        verdict = "control"
    elif ok:
        verdict = "*** DoD MET (more human than prod)"
    elif not hold:
        verdict = "over-flatten/regress hold"
    else:
        verdict = "holds but no h_dist gain"
    hs = f"{h:.3f}" if h is not None else "  --"
    print(f"{arm:12s} {hs:>7s} {gc:6.2f} {de:6.2f} {mono:6.2f} {prep:6.2f} "
          f"{cc:6.2f} {rc:6.2f} {dens:6.2f} {viol:4d}  {verdict}")
    if ok and (best is None or h < best[1]):
        best = (arm, h)
print("-" * 92)
if best:
    print(f"WINNER: {best[0]} (h_dist {best[1]:.3f} < prod {prod_h}) while holding "
          f"density/monotony/pattern_repeat/playability -> promote its LAYOUT_ANTIREPEAT/"
          f"LAYOUT_AR_STRENGTH to production layout config; render vs prod for Kyle.")
else:
    print("NO WINNER: no anti-repeat arm is more human-like than the temp-nudge prod "
          "without over-flattening (col_conc<0.20) or spiking monotony. The temp nudge "
          "is the no-retrain ceiling -> next step is a targeted diversity-reg fine-tune.")
PY
echo "COMPLETE @ $(date)" | tee -a "$LOG"
