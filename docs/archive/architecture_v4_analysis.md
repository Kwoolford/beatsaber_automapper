# V4 Architecture Analysis — Flow & Playability

**Date:** 2026-04-13
**Triggering observations:** User + auto-EDA on `reference_20260413_v14_seq.zip` (Expert generation from v14 checkpoint).

## Observed Failure Modes

| # | Symptom | Data |
|---|---------|------|
| F1 | Model produces 50% parity violations pre-postproc | `fix_parity` corrected 686/1370 notes |
| F2 | Over-use of diagonal (45°-tilted) directions for no musical reason | ~40% of final dirs are 4/5/6/7 despite training dist ~20% |
| F3 | Physically impossible follow-through patterns | User: "(bottom-mid, up-right) → (top-left, down-right)" = 2D teleport, wrong entry angle |
| F4 | Zero arcs/chains/bombs emitted | 0 of each in output despite non-zero training weight |
| F5 | Extreme mode collapse | Top (col,row,dir,color) pattern = 9.5% of notes; only 35/216 combos used |
| F6 | Fixed 2-notes-per-beat chord output | 680/690 beats have exactly 2 notes, 10 have 1, zero singles-only phrases |

## Root Cause Map

### RC-A — No "follow-through" signal anywhere
The model's flow loss (`seq_module._compute_flow_loss`) penalizes same-parity directions vs. previous onset **but ignores grid positions**. The true physical playability constraint is:

> "The current swing direction should roughly align with the direction of movement from the previous note's position to the current note's position."

For the user's example:
- Prev = `(x=1, y=0, dir=5 up-right)`; Curr = `(x=0, y=2, dir=7 down-right)`
- Movement vector `(cx−px, cy−py) = (−1, +2)` → normalized ≈ `(−0.45, +0.89)` (going up-left)
- Curr direction vector = `(+0.7, −0.7)` (going down-right)
- Dot product ≈ **−0.94** (opposite direction — teleport)

No training loss penalizes this; it's parity-valid so `fix_parity` ignores it.

### RC-B — Flow loss alpha is too weak
`flow_loss_alpha=0.25` vs. main CE loss which averages ~1.0 on 7-token note sequences. Parity signal is ~25% of gradient magnitude — not strong enough to override learned token-frequency priors.

### RC-C — Flow loss doesn't check intra-onset parity
Within a single onset, multiple same-color notes can have same parity. `_compute_flow_loss` only compares current target to the *previous onset's* direction, never to other notes in the *same* target.

### RC-D — Postprocess `_choose_flow_direction` is aggressively diagonal
`postprocess.py:782` biases toward diagonals:
- `|dx| ≥ 2`: **always** diagonal (dirs 4/5/6/7)
- `|dx| == 1`: diagonal 50% of the time via `(curr_x + curr_y) % 2`
- Edge columns (0, 3): diagonal ~50% via `curr_y % 2`

Because `fix_parity` + `convert_dot_notes` together rewrote ~50% of the map's notes in our test, this single function is responsible for the diagonal overuse observed.

### RC-E — Undertraining on rare tokens
- Token dropout = 0.05 (was 0.10 before) → model memorizes common patterns instead of reconstructing from context
- Early-stopped at epoch 13 with patience=5 → no time to develop diverse representations
- Mode collapse on `(col=1, row=0, dir=down, red)` = 9.5% of all notes suggests tonic "default" behavior
- Arc/chain/bomb events appear in <2% of training onsets; at low temp+top-p=0.85, their logits never clear the nucleus

### RC-F — Inference EOS boost cuts sequences short
`beam_search.py:222`: `logits[EOS] += 1.0` in event_type phase once any notes are emitted. Combined with `logits[EOS] += 5.0` in between_events at note cap — strong bias toward stopping at 2 notes. Explains F6.

## Fix Plan — Training (applied for v15 run)

### T1. Add **Follow-Through Loss** `_compute_follow_through_loss` (NEW)
Differentiable penalty on misaligned swing-direction vs. movement-vector.

```python
# For each same-color pair (prev, curr) within target or prev→target:
#   movement = normalize((curr.x - prev.x, curr.y - prev.y))
#   expected_dir_probs = softmax(logits_at_curr_dir_pos)
#   for each dir d, dir_vec_d = precomputed 9-dir unit vectors
#   penalty = sum over d: expected_dir_probs[d] * (1 - movement · dir_vec_d) / 2
```

The loss: weighted sum over direction logits of `(1 - alignment)`. Zero-distance movement → neutral (skip). Uses soft probabilities so gradient flows.

Applied between:
- Last note of previous onset → first note of current onset (cross-onset)
- Each consecutive same-color note within current target (intra-onset)

Initial weight: `follow_through_alpha = 0.35` (larger than parity because this is the harder signal).

### T2. Strengthen parity signal: `flow_loss_alpha` 0.25 → 0.4

Parity is simpler than follow-through but still undersignaled. Pushing to 0.4 roughly matches CE magnitude on the `dir` token position.

### T3. Add **intra-onset parity** to `_compute_flow_loss`
Currently only checks current note vs. previous onset's last direction. Extend to: for each consecutive same-color pair **within the current target**, penalize same-parity probability mass.

### T4. Token dropout 0.05 → 0.10
Restore the previous value; forces the model to reconstruct from context instead of memorizing common patterns. Helps F5 mode collapse.

### T5. Rare-event CE reweighting
In `_build_token_weights`, raise weights for `ARC_START=3.0`, `ARC_END=3.0`, `CHAIN=3.0`, `BOMB=2.0`. These events are <2% of training onsets; without reweighting, the model rarely produces them at inference.

### T6. **Expert-only training**
User authorized. Filter `SequenceDataset` to `difficulty_idx=3` (Expert). Expected samples: ~850K train / ~100K val. Smaller, more consistent distribution → faster convergence per epoch. ExpertPlus can be retrained separately once architecture is validated.

### T7. Patience 5 → 10; `max_samples_per_epoch` 750K → 500K
Shorter epochs × more epochs = finer checkpoints, more chances for loss to decline.

## Fix Plan — Inference (applied post-training, not required for v15 run)

### I1. Diagonal penalty during dir phase
At `note_dir` grammar phase in `beam_search.apply_constraints`, add logit penalties of `−0.5` to dirs 4/5/6/7 so model prefers 0/1 unless it strongly votes for diagonal. This is *softer* than the current straight-direction +1.0 boost; still allows diagonals when the model is confident.

### I2. Trajectory-continuity mask
If previous same-color note's position + implied exit vector places the saber far from current position, penalize directions that would require "backward entry". Mirror of the training-time loss.

### I3. Remove aggressive EOS boost
Drop `logits[EOS] += 1.0` in event_type phase. Keep the `+5.0` only at `note_count >= max_notes`. This lets the model emit more events per onset (rare arcs/chains get a chance).

### I4. Rare-event temperature boost at event_type
When the model's softmax distribution over `(NOTE, BOMB, WALL, ARC_START, ARC_END, CHAIN, EOS)` is >90% NOTE, raise temperature to 1.3 on that single position. Breaks NOTE-dominance.

## Fix Plan — Post-processing (applied post-training)

### P1. Rebalance `_choose_flow_direction`
- `|dx| >= 3`: always diagonal
- `|dx| == 2`: diagonal 60% (was 100%)
- `|dx| == 1`: diagonal 20% (was 50%)
- `|dx| == 0`: straight 100% (was ~50%)

### P2. Follow-through-aware parity fix
When `fix_parity` needs to replace a direction, additionally ensure the replacement direction has positive dot product with the movement vector. Mirrors the training loss.

### P3. Cap parity-fix rate
If >40% of notes need parity fix, log a warning. Indicates model regression.

## What v15 Training Will Test

This run tests **training-side changes only** (T1–T7). Inference and postproc changes are post-training. If the training-side changes work, the model itself should produce:
- <10% parity violations pre-postproc (vs. 50% in v14)
- <15% diagonal directions pre-postproc (vs. ~25% in v14)
- Zero "impossible follow-through" patterns in sampled outputs
- Non-trivial arc/chain/bomb counts (>5 of each in a 3-minute song)
- Higher pattern diversity (>80 unique combos vs. 35)

## Training Config for v15

```yaml
# configs/model/sequence.yaml
token_dropout: 0.10       # was 0.05
flow_loss_alpha: 0.4      # was 0.25
follow_through_alpha: 0.35  # NEW
intra_onset_parity: true    # NEW flag

# configs/train.yaml  
max_epochs: 100
early_stopping_patience: 10    # was 5
data.dataset.max_samples_per_epoch: 500000  # was 750000
data.dataset.difficulty_filter: [3]  # Expert only (was [3, 4])
data.dataset.batch_size: 256
data.dataset.num_workers: 16
```

Expected per-epoch time at 500K samples: ~10 min. 15 hours ≈ 90 epochs (with early stop after ~patience=10 epochs of no improvement).

## Success Criteria

1. Val_loss lower than v14's 1.090
2. Generated Expert map shows:
   - Parity violation rate < 15% (pre-postproc)
   - Follow-through score > 0.7 (mean dot product over consecutive pairs)
   - ≥ 3 arcs AND ≥ 3 chains emitted
   - At least 80 unique `(col, row, dir, color)` combos
   - No 2D teleports: 0 consecutive pairs with movement-dir dot product < −0.5

If any of the above fail, revise the loss weights or move to inference-side fixes (I1–I4).
