# V6 Swing Event Grammar — Locked Specification

**Status:** Locked as of 2026-05-11. Do not change token IDs or vocab size without a full
re-preprocess of all cohort data.

Source of truth implementation: `src/beatsaber_automapper/data/swing_tokenizer.py`.

---

## Overview

Stage 2 generates a **single globally-ordered stream of swing events**, one token group per
physical cut or obstacle. The stream covers the entire song from beat 0 to EOS.

Walls are excluded from the swing stream and added via rule-based postprocessing (like lighting
in V5+). All other v3 object types — colorNotes, sliders (arcs), burstSliders (chains), bombNotes
— are encoded in the stream.

---

## Event Format

Each event is a fixed-length run of tokens. The length depends on KIND (token position 2):

```
SWING event (7 tokens):  [HAND] [Δt] [KIND] [X] [Y] [DIR] [FIELD_D]
CHAIN_TAIL event (6 tokens): [HAND] [Δt] [CHAIN_TAIL] [X] [Y] [SQUISH]
BOMB event (5 tokens):   [HAND=NONE] [Δt] [BOMB] [X] [Y]
```

Token positions and their semantics by KIND:

| Pos | NOTE       | ARC_HEAD   | ARC_TAIL   | CHAIN_HEAD | CHAIN_TAIL | BOMB       |
|-----|------------|------------|------------|------------|------------|------------|
| 0   | HAND       | HAND       | HAND       | HAND       | HAND       | HAND=NONE  |
| 1   | Δt         | Δt         | Δt         | Δt         | Δt         | Δt         |
| 2   | NOTE       | ARC_HEAD   | ARC_TAIL   | CHAIN_HEAD | CHAIN_TAIL | BOMB       |
| 3   | X (0-3)    | X (0-3)    | X (0-3)    | X (0-3)    | TAIL_X     | X (0-3)    |
| 4   | Y (0-2)    | Y (0-2)    | Y (0-2)    | Y (0-2)    | TAIL_Y     | Y (0-2)    |
| 5   | DIR (0-8)  | DIR (0-8)  | DIR (0-8)  | DIR (0-8)  | SQUISH_bin | —          |
| 6   | ANGLE_bin  | MU_bin     | MU_bin     | SLICE_bin  | —          | —          |

**HAND mapping:**
- `HAND_LEFT (3)` = red saber (color=0)
- `HAND_RIGHT (4)` = blue saber (color=1)
- `HAND_NONE (5)` = bombs (no saber involvement)

**FIELD_D (position 6) semantics by KIND:**
- NOTE → ANGLE_bin: quantized angle_offset (-45° to +45°, 7 bins)
- ARC_HEAD / ARC_TAIL → MU_bin: arc curvature multiplier (0.0–2.0, 9 bins)
- CHAIN_HEAD → SLICE_bin: burst slider slice count (2–32, 31 bins)

**Arc mid_anchor_mode:** always decoded as 0 (the common case). This saves a token at the cost
of losing mid_anchor=1/2 fidelity, which is acceptable because mid_anchor ≠ 0 is rare (<2% of arcs).

---

## Vocabulary

Total vocab size: **118 tokens**.

```
Token ID  Field             Values
--------  -----             ------
0         PAD               —
1         BOS               —
2         EOS               —

3         HAND_LEFT         red saber
4         HAND_RIGHT        blue saber
5         HAND_NONE         bomb

6–37      Δt_bins[0..31]    beats since previous event
          (32 bins, see Δt Quantization table below)

38        NOTE              standard directional cut
39        ARC_HEAD          arc/slider head
40        ARC_TAIL          arc/slider tail
41        CHAIN_HEAD        burst slider head
42        CHAIN_TAIL        burst slider tail
43        BOMB              bomb note

44–47     X_bins[0..3]      grid column (left→right)

48–50     Y_bins[0..2]      grid row (bottom→top)

51–59     DIR_bins[0..8]    cut direction
          0=up 1=down 2=left 3=right
          4=up-left 5=up-right 6=down-left 7=down-right 8=any

60–66     ANGLE_bins[0..6]  angle offset (-45 -30 -15 0 +15 +30 +45 degrees)

67–75     MU_bins[0..8]     arc curvature (0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)

76–106    SLICE_bins[0..30] chain slice count (2..32)

107–117   SQUISH_bins[0..10] chain squish factor (0.0 0.1 … 1.0)
```

---

## Δt Quantization

32 bins covering 0 to 64+ beats. Nearest-bin assignment (no clipping — values beyond 64 beats
map to bin 31). Absolute beats accumulate via prefix sum during decoding.

```
Bin  Value (beats)    Bin  Value (beats)
---  -------------    ---  -------------
 0    0.0000          16    1.0
 1    0.0625          17    1.5
 2    0.1250          18    2.0
 3    0.1875          19    2.5
 4    0.2500          20    3.0
 5    0.3125          21    3.5
 6    0.3750          22    4.0
 7    0.4375          23    4.5
 8    0.5000          24    5.0
 9    0.5625          25    6.0
10    0.6250          26    7.0
11    0.6875          27    8.0
12    0.7500          28   12.0
13    0.8125          29   16.0
14    0.8750          30   32.0
15    0.9375          31   64.0
```

**First event Δt:** beats from beat 0 (i.e., the absolute beat of the first event).

---

## Ordering Rules

Events at **different beats** are ordered by increasing beat value.

Events at the **same beat** are ordered by:
1. Kind priority: NOTE < ARC_HEAD < CHAIN_HEAD < ARC_TAIL < CHAIN_TAIL < BOMB
2. Within same kind: HAND_LEFT (color=0) before HAND_RIGHT (color=1)

This ordering is canonical — it must match between encoding and decoding for the round-trip to
hold and for the model to learn consistent ordering.

**Δt for events at the same beat** (i.e., chords): the second and subsequent events at the same
beat have Δt = 0 (bin 0).

---

## Arc / Chain Self-Connect Policy

ARC_HEAD and ARC_TAIL are separate events in the stream. During decoding, they are matched by
**HAND** in FIFO order: the first unmatched ARC_HEAD of HAND=H is paired with the first ARC_TAIL
of HAND=H encountered later in the stream. Same rule for CHAIN_HEAD / CHAIN_TAIL.

If an ARC_HEAD has no matching ARC_TAIL (malformed output), it is dropped from the decoded
beatmap — not converted to a regular note.

If an ARC_TAIL has no preceding ARC_HEAD of the same hand, it is dropped.

This means the model must learn to always emit ARC_TAIL after ARC_HEAD of the same hand. The
beam search grammar mask enforces a soft budget (at most 4 unmatched arc heads per hand at any
time) to discourage runaway arc-head emission.

---

## What Is Not Encoded (V6-0)

- **Obstacles/walls** — excluded from the swing stream. Added via `generate.py`'s rule-based
  wall layer (same pattern as rule-based lighting in V5).
- **basicBeatmapEvents and colorBoostBeatmapEvents** — lighting. Still handled by
  `generation/lighting_rules.py`.
- **arc mid_anchor_mode** — always decoded as 0.

---

## Stream Framing

A complete encoded beatmap is a 1-D list of integers:

```
[BOS] <event_0> <event_1> … <event_N-1> [EOS]
```

For training, the sequence is split into windows of `max_seq_len` tokens with stride `stride`.
BOS appears only at the start of the full sequence; EOS appears only at the end.

For the dataset, each training sample is a window of tokens plus the corresponding saber-state
tensor pre-computed from the full sequence (see `data/saber_state.py`).

---

## Round-Trip Tolerance

The round-trip test (`tests/test_swing_tokenizer.py`) accepts a decode as correct if:
- **Count match:** same number of colorNotes, sliders, burst_sliders, bomb_notes as input.
- **Beat tolerance:** each decoded event's beat is within ±0.1 beat of the nearest original event
  of the same type + hand.
- **Discrete fields exact:** color/hand, x, y, direction, angle (within 1 bin), mu (within 1 bin),
  slice count (within 1 bin), squish (within 1 bin).

The ±0.1 beat tolerance accounts for cumulative Δt quantization drift.
