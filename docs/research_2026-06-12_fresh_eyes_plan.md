# Fresh-Eyes Research & Plan — 2026-06-12

Written in response to Kyle's strategic reset request: "find the missing piece." This synthesizes
8 architectures of internal evidence, the current (interrupted) reward-pivot state, and outside
research on rhythm-game chart generation.

## 1. Honest state of the project

- V7 two-stage (MERT → BeatClassifier WHEN → LayoutPhraseModel WHAT) is production. Plateau is
  confirmed and well-characterized: align-F1 ~0.40, density-corr ~0.15 (flat ~8 NPS), contour at
  chance, late-song/final-chorus collapse, postprocessor rewrites ~48% of swing directions.
- Every scoped-V8 lever (T0–T5) tested null. Per-slot metrics hit the same ~0.60 subjectivity
  ceiling across 3 independent feature families.
- Reward pivot is mid-flight: handcrafted reward **2a is DEAD** (AUC human-vs-V7 = 0.31 — it ranks
  our maps MORE human than humans). Learned discriminator **2b smoke = AUC 1.0 @ n=65**
  (suspiciously perfect). The 06-11 overnight scale+ablation run **never executed** (empty log,
  no JSONs — interrupted by the dual-boot switch). V7 cohort sits at 74/400 maps.
- ⚠️ 22+ commits unpushed; new uncommitted work (`feel_disc_poc.py`, `gen_v7_cohort.py`,
  `overnight_2026-06-11.sh`). `git push origin main` still needs Kyle's auth.

## 2. Diagnosis: the missing piece is the judge, not the generator

Kyle's own framing nails it: *"the evaluation loop never played a Beat Saber level."* Eight
architectures optimized proxies (per-slot F1, token accuracy) that we now have hard evidence
anti-correlate with quality (val_token_acc anti-correlates with align-F1; the 2a reward scored V7
maps MORE human than humans). The generator has been flying blind because nothing in the loop can
perceive "clunky," "monotonous," or "dead drop." Three perception channels are missing, and all
three are now buildable:

1. **A simulated player** (deterministic): swing-path simulation à la the community's
   [JoshaParity](https://github.com/Joshabi/JoshaParity) — tracks forehand/backhand state per
   hand, swing rotation, resets, angle deltas. Catches *unplayable* (parity violations, wrist
   breaks) and quantifies *flow* without any ML. The community considers parity/flow the #1
   discriminator of good vs soulless maps (BSMG wiki: flow = fluid motion + intuitive direction
   changes; "keeping parity is almost always better").
2. **A learned taste model** (statistical): the 2b discriminator — human vs our-generator, MERT
   audio-conditioned. This is the reward that ranks "feels human."
3. **Vision** (perceptual): Claude is multimodal. Render maps as mapper's-eye images (time-lattice
   with note arrows, per phrase) and the agent can *look at the map* the way Kyle does in
   ArcViewer — closing the loop that currently bottlenecks on Kyle's eyes.

On Kyle's two theories:
- **"Too much freedom / not learning"** — supported. The model never learned contour or density
  (confirmed null), so sampling freedom produces noise and the postprocessor papers over it
  (48% direction rewrites = the model has no authority over its own output). The fix is not lower
  temperature; it is *selection pressure* (generate N, keep what the judges like) plus *structure
  constraints* (below).
- **"Segment the song, generate per segment, maintain parity between common phrases, stitch"** —
  half-built already (V7 generates per 16-beat phrase; song-memory cross-attn was the implicit
  version and it collapsed on final choruses). The missing half is making structure EXPLICIT,
  which is exactly how human mappers work: identify chorus₁ ≡ chorus₂ ≡ chorus₃, map ONE chorus,
  copy with small variations, hand-craft transitions. Implicit attention failed; explicit
  copy-with-variation is the right formalization of Kyle's idea. Fluidity at seams = parity-aware
  stitching (the simulator from #1 validates seams).

## 3. Outside research (what others learned)

- **Beat Sage / DeepSaber** (Dance Dance Convolution lineage): two-stage WHEN/WHAT like ours;
  community verdict "soulless" — no structure awareness, no parity model, no quality objective.
  We have independently reproduced their plateau. Confirms: more per-slot supervised learning
  will not escape it.
- **[InfernoSaber](https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper)** (active
  2025, 4-model pipeline w/ conv autoencoder music encoding, adjustable difficulty): the most-used
  open automapper; same fundamental criticism. Difficulty-as-a-knob is a UX feature worth copying.
- **[GenéLive! (DeNA, Love Live!)](https://arxiv.org/abs/2202.12823)** — the only *shipped in
  production* chart generator. Key insights: beat-phase features ("beat guide") and multi-scale
  onset conv kernels; crucially they kept humans in the loop and the model drafts → human polishes.
- **[Beat-Aligned Spectrogram-to-Sequence](https://arxiv.org/pdf/2311.13687)** + osu! transformer
  line of work: beat-aligned tokenization is the consensus representation (validates keeping V7's
  grid — matches our V8-0 PoC finding that the grid covers 94–99% of human timing).
- **[TaikoNation](https://arxiv.org/pdf/2107.12506)**: explicitly argues *patterning* (congruent,
  human-like note groupings) is what separates human charts from DDC-style output — the academic
  version of "monotonous diagonals" feedback. Their fix: model patterns, not slots.
- **Community quality tooling**: [JoshaParity](https://github.com/Joshabi/JoshaParity) (C# swing
  simulator), [bs-parity](https://github.com/GalaxyMaster2/bs-parity) (error checker),
  [BSMG mapping wiki](https://bsmg.wiki/mapping/basic-mapping.html) (flow/parity/emphasis rules in
  prose — directly encodable), BeatLeader replays (real player swing data; future goldmine for a
  playability ground truth). Lighting: community tool *Lolighter* does rule-based energy→lights —
  validates the decorator approach.

Nobody in the field has closed the loop with (a) a learned human-vs-generated reward + (b) a swing
simulator + (c) multimodal perception. That combination is our differentiator, and (a) is already
smoke-green in this repo.

## 4. The plan

Phased, each with a hard gate. No phase starts until the previous gate is read.

### Phase 0 — Resume the interrupted reward run (tonight, zero new code)
Re-launch cohort growth to 400 (`gen_v7_cohort.py`, ~24s/map ≈ 2.2h for the remaining 326) then
`overnight_2026-06-11.sh` (feel-disc @ scale + dt/spatial/dir ablations — already written).
**Gate:** held-out AUC(human vs V7) ≥ 0.75 on the `none` arm AND no single ablation arm collapses
to ~0.5 (which would mean the reward is a one-feature fingerprint, e.g. it only reads our
metronomic timing). Smoke AUC 1.0 makes degeneracy the main risk — the ablations exist to catch it.

**OUTCOME (2026-06-12): GATE PASSED, with a saturation finding + validated fix.**
- All 4 arms (none/dt/spatial/dir) = held-out AUC 1.0000 @ 400 V7 + 400 human. No ablation
  collapse → not a one-feature fingerprint; V7's tells are in every feature group at once.
- BUT at 60 epochs the model is saturated: V7 logits all ≈ −10.23 (within-V7 sd = 0.3% of the
  human gap) → perfect detector, USELESS ranker for best-of-N.
- **Fix validated: early stopping.** At 1 epoch: AUC 0.994 (separation intact), within-V7 logit
  sd = 14% of gap with smooth percentile structure → a usable candidate ordering exists.
  Phase-2 rule: pick the reward checkpoint by max within-generator ranking spread subject to
  AUC ≥ 0.9; sharpen later with harder negatives (best-of-N survivors, light human corruptions).
- Artifacts: `outputs/feel_disc_{none,dt,spatial,dir}_2026-06-11.json`,
  `outputs/feel_disc_2026-06-12.pt` (60-ep ckpt), `outputs/feel_disc_scores{,_ep1}_2026-06-12.json`.
- Ops postmortem: the 06-11 chain originally never ran (dual-boot interruption); relaunched
  06-12 and completed. Side effect: `eval_contour_follow._load_notes_with_direction` leaked one
  15MB tempdir per zip load (1,610 dirs ≈ 24GB) and filled the root partition — FIXED (rmtree
  in finally) + `CLAUDE_CODE_TMPDIR` moved to giga_speed.

### Phase 1 — Map Perception ("play the map") (~2-4 sessions)
1. **Swing simulator** (`evaluation/swing_sim.py`): Python port of JoshaParity's core — per-hand
   parity state machine, swing angle sequence, reset/wrist-break detection, swing-EBPM. Output:
   per-map scorecard + per-seam validity. Extends existing `evaluation/playability.py`. Unit-test
   against maps with known violations (we can author tiny fixtures).
2. **Renderer** (`scripts/render_map.py`): matplotlib mapper's-eye lattice — time on x, 4×3 grid
   unrolled on y, arrows for cut direction, color per hand, beat lines, audio RMS strip underneath.
   NOT a single in-game frame: each panel is a time-unrolled window (8–16 beats ≈ several seconds
   of notes, like sheet music), so a handful of panels covers a song. Plus a per-hand swing-path
   trace (the simulator's predicted wrist trajectory) — flow made visible. Claude reads these
   directly → agent-side ArcViewer. (Kyle stays the final judge; this stops every iteration
   needing him.)
3. **Evaluation protocol for full songs (Kyle's question: songs are long, and what is "good"?).**
   Three answers, layered:
   - **Coverage by tier, not frame-by-frame.** Tier 1 (100% of timeline, cheap): swing simulator +
     2b reward + density-vs-RMS correlation run over the whole map. Tier 2 (macro, 1 image): a
     whole-song strip — note-density curve over audio RMS with the structure plan and seams
     overlaid; instantly shows dead drops / flat density / missing outro. Tier 3 (micro, ~12–20
     panels per song, MULTI-panel by design): stratified windows — one per *unique* section type,
     every seam, the drop, intro/outro, PLUS any window the Tier-1 judges flag (worst parity /
     lowest reward). Vision effort goes where the deterministic judges point, so song length
     doesn't blow up the budget.
   - **Reference for "good" = comparative, never absolute.** We hold 5,374 human maps; every
     vision-eval batch renders generated windows blind-shuffled with human windows from the SAME
     section type and similar NPS. Claude ranks; generated output "passes" when it stops being
     identifiable. This is the discriminator trick applied to the vision channel — no absolute
     aesthetic judgment is ever needed, only "which of these is the human one, and why."
   - **Calibration gate before trusting it:** render 5 human + 5 V7 maps blind; Claude's ranking
     must separate human from V7 AND its stated reasons must match Kyle's known complaints
     (diagonals, monotony, dead drops). Simulator must flag 0 violations on human Expert maps and
     >0 on raw (pre-postprocess) V7 output. If either fails, fix the perception before using it
     to steer generation. Periodic re-anchor: Kyle ArcViewers 1 map per milestone and we diff his
     verdict against the panel's.

### Phase 2 — Structure-first generation + selection pressure (the new architecture, ~1-2 weeks)
This is Kyle's segmentation idea, formalized:
1. **Structure plan**: MERT phrase-fingerprint self-similarity → segment the song into labeled
   sections (A/B/chorus/drop instances). Deterministic given the song.
2. **Per-unique-segment generation**: generate candidates only for *unique* section types;
   instances of the same type get the same base pattern + controlled variation (fill variation,
   mirrored hands on repeat 2, etc.). Kills monotony AND inconsistency simultaneously — repeats
   are recognizably similar by construction, the thing song-memory attention failed to learn.
3. **Best-of-N + judges**: for each unique segment, sample N=8–32 candidates from the existing
   LayoutPhraseModel (no retrain); hard-filter by swing simulator; rank by 2b reward; top
   candidate wins. This is where the reward earns its keep — selection, not training. Cheap,
   debuggable, and each judge's vote is inspectable.
4. **Parity-aware stitching**: seams solved as a constraint problem (entry/exit hand state per
   segment from the simulator); insert/flip transition notes only at seams. Replaces most of the
   global postprocessor — the model's choices stop being overwritten wholesale.
5. **Style seeds**: `--seed` + temperature + per-seed variation profile (e.g. vertical-heavy vs
   lateral-heavy sampling bias) → regenerate for a *different* map of the same song on demand.
6. **No arcs/chains in generation (Kyle 2026-06-12).** The grammar already carries
   ARC_HEAD/ARC_TAIL/CHAIN_HEAD/CHAIN_TAIL kinds (`swing_tokenizer.py` IDs 39–42); constrained
   sampling already masks illegal roles, so masking these kinds at inference is a few lines.
   Rationale: arcs/chains multiply the judge surface (simulator + vision must understand them)
   for zero flow benefit — they're decoration. **Arc decorator** becomes an optional postprocess
   pass later (sibling of the lighting decorator): rule-based — connect same-hand notes across
   gaps ≥ N beats when directions align, sustain-detection from stems for long-note arcs. Chains
   only ever via decorator too, if at all.
**Gate (North Star):** Kyle ArcViewers 3 songs (incl. one clear-drop EDM) and reports: drop is
mapped, choruses feel consistent, no parity complaints, "would play again." Plus: density-corr
≥0.4 (structure plan should finally crack this — density comes from the plan, not the model),
0 simulator violations, reward(generated) within human range *with* the diversity ablations clean.

### Phase 3 — Only if Phase-2 selection isn't enough: train on the reward
DPO/preference fine-tune of LayoutPhraseModel on (reward-winner ≻ reward-loser) candidate pairs
from Phase 2's own sampling, or RL-style reward-weighted regression. Do not start this while
best-of-N is still improving maps — selection is safer than optimization against a learned reward
(Goodhart risk is real given 2a's failure mode).

### Phase 4 — Lighting decorator (parallel-izable any time after Phase 1's renderer)
v1 exactly as Kyle proposed: section energy (already computed) → preset bank (strobe on drops,
slow fades on verses, off in silences), beat-locked. The structure plan from Phase 2 makes this
near-free: presets keyed by section type + energy. v2 (optional, later): train a small
lighting-event model on human maps' lighting tracks conditioned on section type + energy — same
two-stage shape as notes, only if v1 looks lifeless in-game.

## 5. Explicitly NOT doing
- No V9 full rebuild, no continuous-time backbone (V8-0 PoC refuted it), no bigger context /
  whole-song attention (the 5090 isn't the bottleneck; the objective was).
- No more per-slot-F1-targeted retrains. The ceiling is real; stop paying it tribute.
- No new per-slot input features (3 families went null).

## 6. Risks
- **2b reward degenerate** (reads metronomic-timing fingerprint, not taste) → ablations in Phase 0
  catch it; mitigation = condition on audio (MERT) + diversify negatives (corruptions of HUMAN
  maps + other automappers' output, e.g. InfernoSaber, as additional negative classes).
- **Best-of-N too slow** → candidates are per-unique-segment (a song is ~4–6 unique sections), and
  24s/map full-pipeline cost is dominated by Demucs/MERT which is shared across candidates. Layout
  sampling is the cheap part. Estimate <2 min/song for N=16.
- **Stitching feels mechanical** → seams are where human mappers spend effort too; simulator
  validates correctness, vision eval judges feel; worst case allow 1-beat overlap regeneration.
- **Vision eval drift** → always calibrate against held-out human maps in the same batch.

## 7. Immediate next actions
1. `git push origin main` (Kyle: needs your auth — 22+ commits at risk) + commit the 06-11 files.
2. Relaunch Phase 0 tonight (cohort → 400, then the existing overnight script).
3. While it runs: build Phase-1 renderer (no GPU needed).
