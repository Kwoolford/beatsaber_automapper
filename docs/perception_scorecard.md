# Do we perceive a song the way a human mapper does?

**Kyle's brief, 2026-08-16:** *"keep building crazy good tools until you believe you have
the same insights into a song to map as a human does."*

This is the falsifiable version of that question. Each row is something a human mapper
has in their head while mapping. The point of writing it down is that "do we have the
same insights" is otherwise unanswerable — and the rows we can now cross off are only
worth crossing off because each one has a **control it passes**, not because a tool
exists.

**Honest headline: no, not yet — but the list of what is missing is now short and named,
and three of the four biggest gaps closed today.**

## The scorecard

| # | what a human mapper perceives | status | evidence |
|---|---|---|---|
| 1 | tempo and the beat grid | ✅ | grid phase within **±8 ms** measured at the subdivision harmonic, 6 songs |
| 2 | the subdivision a part sits on (8ths vs 16ths) | ✅ | onsets concentrate at R@½ = 0.24–0.62; human maps place 0.52 on-beat / 0.31 eighth / 0.17 sixteenth |
| 3 | which beat is beat 1 (the downbeat) | ⬜ **not needed** | over **363 human maps**, note placement by beat-of-bar is 0.254/0.249/0.251/0.246 and only 29 % peak on beat 1 vs 25 % chance. Human mappers do not weight bar position. **Do not build a downbeat detector.** |
| 4 | where the sections are, and which repeat | ✅ **CONFIRMED** | `structure.py`; repeated lyric lines land under the same letter 0.485 vs a shuffled null 0.317, **p = 0.019**, 97 pairs over 6 **held-out** songs |
| 5 | section role (intro / build / drop / breakdown) | ⚠️ built, uncontrolled | energy-shape heuristic; it does put Hunger's breakdown at bar 131 with nrg 0.11 |
| 6 | the words, and when they are sung | ✅ | `lyrics.py`, Whisper on the separated vocal, 9 songs cached |
| 7 | **the melody — what note, and which way it moves** | ✅ **PARTLY CONFIRMED** | `melody.py`; two independent trackers agree on the **key on 36 % of 14 songs against a 4 % chance** (p ≈ 0.0005). The step-size control lacks power — see below |
| 8 | **which drum is hitting** | ✅ **CONFIRMED (3 of 4 songs)** | `percussion.py`; labelled groove repeats bar-to-bar at **z = +12.7…+25.7** vs a label-shuffled null. The 4th song self-reports as untrustworthy (z = +1.5) |
| 9 | how hard a hit is (accent / dynamics) | ⚠️ partial | per-hit velocity from the onset envelope; no control yet |
| 10 | build-ups, risers, drops, breaks | ⚠️ partial | only as an energy delta between sections |
| 11 | swing vs straight feel | ❌ | not built |
| 12 | the hook — the moment the song is *about* | ❌ | not built, and probably not automatable |
| 13 | **what it feels like to play** | ❌ | fundamental; only Kyle's ear closes this |

## What each control actually established

**Structure (row 4) is the strongest result.** The threshold that decides "same section"
was set on Hunger alone, where the lyric repeat map gives the answer away (choruses at
bars 59/115/195, verses at 37/91) — and it is right there, choruses one letter, verses
another, breakdown its own. It then held up on **six songs it had never seen**. This is
the "longitudinal view" from the original ask, and it is now real rather than asserted.

**Melody (row 7) passes the control that matters and fails one that cannot resolve.**
pYIN on `vocals` and CQT salience on `other` are different algorithms on different audio;
agreeing on the song's key 9× more often than chance is not something noise does. The
step-size null (shuffle the onset times, keep the f0 track) is weak by construction —
a random time still lands on a nearby pitch, because a singer's range is narrow — so its
50 % is **"not measurable with this null"**, not a refutation. A stronger null would swap
f0 tracks *between songs*.

**Percussion (row 8) is a labelling that repeats, not a labelling that is named right.**
The control proves the classes are consistent and carry structure. It does *not* prove
the cluster called "crash" is a crash. Treat the names as a vocabulary, the separation as
evidence.

## Three things that were nearly built and should not be

1. **A downbeat detector.** Row 3. Half a day of work avoided by measuring the human
   corpus first.
2. **A backbeat control** ("snare on 2 and 4"). It is not true of the ground truth. A
   control the ground truth fails is not a control.
3. **Pitch-driven note placement to fix `travel`.** Built and **measured as refuted** —
   see below. This one had a strong intuition behind it and was wrong.

## The refutation worth keeping: travel is not a contour problem

`travel` (grid distance per second between a hand's consecutive swings) is ours **4.77**
against a human **12.53**. The obvious hypothesis was that our hands barely move because
nothing told the placer *where* to go, and pitch would fix it. Measured on 1f767, 449
notes, parity clean in every arm:

| placement | travel | crossover | rows used | columns used |
|---|---|---|---|---|
| before (`--wide`, buggy) | — | 0.000 | 0, 2 | **2 of 4** |
| `--wide` (bug fixed) | **4.770** | 0.000 | 0, 2 | 4 of 4, evenly |
| `--pitch` (level → column, within hand) | **3.556** ⬇ | 0.000 | **0, 1, 2** | 4 of 4 |
| `--pitch --pitch-span full` (interval → jump) | **4.789** | 0.523 ⬆⬆ | 0, 1, 2 | 4, bimodal |
| **human** | **12.53** | 0.183 | all | all |

★**Following the melody literally makes the hands move LESS** (4.77 → 3.56), and the
reason is musical: melodies move in small steps, so contour-following parks consecutive
notes in the same cell. Mapping the *interval* to a jump instead recovers the baseline
but overshoots crossover to **0.523 against a human 0.183**.

⇒ **`travel` is a property of the note SEQUENCE, not of any per-note rule.** A human gets
there by making consecutive same-hand notes land far apart *as a pattern* — which is the
next hypothesis, and is not a perception problem at all. All the perception in the world
does not place a note; that is a separate defect and it is now isolated.

## What was kept from that work

- **A real bug fixed**: `--wide` never widened anything. Hands strictly alternate and the
  column was chosen by a *global* note counter, so `k % 2` was perfectly correlated with
  which hand was playing — the left hand only ever saw even `k`. Measured as exactly two
  distinct columns across a 449-note map; now four, evenly.
- `--pitch` survives as a **lever, not a default**, per the standing rule that
  well-behaved levers are user-facing even when they do not fix a defect. It is the only
  thing that has ever put notes in the **middle row** (95 of 449; previously zero).
- `--pitch` refuses to run on a stem the melody tool does not trust (coverage < 0.45), so
  it declines on Hunger's screamed vocals rather than inventing a contour.

## What to do next, in order

1. **`travel` as a sequence property** — the refutation above says where to look.
2. **Wire `structure.py` into `mapctl`**: map section `D` once and reuse it at every later
   `D`. This is the confirmed result and it is not yet used by the placer.
3. **Wire `percussion.py` into doubles**: spend them on crashes and snare accents rather
   than on "a strong beat with ≥2 stems agreeing" (`double_share` 0.034 vs human 0.146).
4. Rows 9–11, cheapest first.
