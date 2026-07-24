#!/usr/bin/env bash
# Overnight 2026-06-30 (PM-3) — LAYOUT DIVERSITY sweep.
#
# The hardened scorecard (PM-2) exposed two layout-quality gaps that survived the
# decode-bug fix:
#     grid_coverage ~0.65 vs human 0.96   (model under-uses the 12 grid cells)
#     dir_entropy   ~0.60 vs human 0.80   (model under-uses the 9 cut directions)
# The sweep decodes layout GREEDILY (temp 0.0 -> nucleus collapses to argmax), so
# those are the model's *argmax* diversity. Two no-retrain levers, both on the
# production density config (dsel_g2.5 = control):
#   (a) g2.5_temp   -> stochastic decode (temp 0.9, top_p 0.97): let the tail through.
#   (b) g2.5_div05/10 -> env-gated frequency penalty (deterministic anti-repeat),
#       NOW extended to the DIR role via LAYOUT_DIV_D so it can move dir_entropy,
#       not just grid_coverage (previously X/Y only).
#
# DoD (an arm "wins" if):
#     grid_coverage >= 0.80  AND  dir_entropy >= 0.72
#   while HOLDING:
#     mean_spearman (density_corr) >= 0.41,  row_conc <= 0.60,
#     col_conc >= 0.20 (not over-flattened),  total_viol == 0.
# A winner becomes the new production layout config. If a penalty over-flattens
# (col_conc < 0.20 or monotony spikes), the gentler strength is preferred.
set -euo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate
LOG=logs/overnight/layoutdiv_2026-06-30.log
mkdir -p logs/overnight
echo "=== layout-diversity sweep @ $(date) ===" | tee "$LOG"

# --force: regenerate all arms (control included) so every arm shares identical code.
python scripts/eval_sweep.py sweep \
    --arms dsel_g2.5,g2.5_temp,g2.5_div05,g2.5_div10 --force 2>&1 | tee -a "$LOG"

echo "=== VERDICT @ $(date) ===" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import json, pathlib
lb = json.load(open("outputs/eval_sweep_cache/leaderboard.json"))
H_GRID, H_DIR = 0.96, 0.80
print(f"{'arm':12s} {'grid_cov':>8s} {'dir_ent':>7s} {'col_conc':>8s} {'row_conc':>8s} "
      f"{'dens':>6s} {'#pass':>5s} {'viol':>4s}  verdict")
print("-" * 78)
print(f"{'HUMAN':12s} {H_GRID:8.2f} {H_DIR:7.2f} {0.29:8.2f} {0.49:8.2f} "
      f"{'.':>6s} {'.':>5s} {'.':>4s}")
best = None
for arm, d in lb.items():
    gc = d.get("mean_grid_coverage", 0.0); de = d.get("mean_dir_entropy", 0.0)
    cc = d.get("mean_col_conc", 0.0); rc = d.get("mean_row_conc", 1.0)
    dens = d.get("mean_spearman", 0.0); npass = d.get("n_pass", 0)
    viol = d.get("total_viol", 0)
    hold = dens >= 0.41 and rc <= 0.60 and cc >= 0.20 and viol == 0
    hits = gc >= 0.80 and de >= 0.72
    ok = hold and hits
    verdict = "*** DoD MET" if ok else ("holds, gap open" if hold else "REGRESSED hold")
    print(f"{arm:12s} {gc:8.2f} {de:7.2f} {cc:8.2f} {rc:8.2f} "
          f"{dens:6.2f} {npass:5d} {viol:4d}  {verdict}")
    if ok:
        # prefer the closest-to-human that still holds; tie-break on gentlest (lowest gc overshoot)
        score = (H_GRID - gc) ** 2 + (H_DIR - de) ** 2
        if best is None or score < best[1]:
            best = (arm, score)
print("-" * 78)
if best:
    print(f"WINNER: {best[0]} meets grid>=0.80 & dir>=0.72 while holding density/playability "
          f"-> promote to production layout config, render vs control for Kyle.")
else:
    print("NO WINNER: neither lever closes the grid/dir gap without regressing. "
          "If penalty over-flattens (col_conc<0.20) the model logits are the ceiling -> "
          "next step is a diversity-reg fine-tune (targeted, unlike the superseded entropy-reg).")
PY
echo "COMPLETE @ $(date)" | tee -a "$LOG"
