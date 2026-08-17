# `agent_mapper/` — what the first build taught

## 2026-08-14 — the toolkit exists, and building one map with it found four bugs

Built `brief.py` (perceive), `lyrics.py` (timestamped words from the vocals stem) and
`mapctl.py` (a session you author into), then used them to build a complete 1 081-note
map of **Hunger** from a **19-line plan**. Every number below is on the same ruler the
ML maps are scored with — no special-casing.

| | notes | nps | precision | ebpm_burst | travel | doubles | parity |
|---|---|---|---|---|---|---|---|
| **AGENT (hand-built)** | 1081 | **4.00** | 0.983 | **376** | 4.42 | 0.000 | 0 |
| ML baseline | 1328 | 4.96 | 0.960 | 376 | 5.94 | 0.391 | 0 |
| HUMAN | 2254 | 8.35 | 0.956 | 376 | 12.53 | 0.146 | 0 |

⚠️**The precision is CIRCULAR and must not be quoted as a win.** `auto` places notes
*on* the onsets the metric then scores against, so 0.983 is guaranteed by construction
— the same circularity flagged for `BEAT_GRID_PHASE=search`. What is *not* circular:
nps lands on the human Expert median (3.91), burst rate matches the human exactly, and
parity is clean.

★**Two real gaps remain, both nameable**: travel **4.42 vs the human's 12.53** (our
hands barely move — `auto` only uses two columns and two rows per hand) and
**doubles 0.000 vs 0.146** (we never play both hands at one instant).

## The four bugs, each found by measuring rather than by reading

1. **`_mmss` turned a −21 ms downbeat into "−1:59.98"** — a two-minute error on the
   first row of every brief, from a sign nobody had tested.
2. **Layering two `auto` passes doubled the burst rate** (376 → 752 against a human
   376). Each pass tracked its own hand and parity state, so the second pass landed
   notes *between* the first's and handed one hand two fast swings.
3. ★**The per-hand floor, and the first fix that looked applied and did nothing.**
   Measured across **31 723 human gaps in 40 songs**: a human hand almost never swings
   twice inside ~**150 ms** (cohort p5 148 ms; Hunger's human map floors at 160 ms).
   Ours allowed **80 ms**. The first guard checked only the *previous* note of that
   hand — but a later pass inserts notes *between* fixed ones, so 70 violations
   survived and the metric did not move at all. Checking **both neighbours** fixed it:
   ebpm 752 → **376**, min gap 159 ms, zero violations.
4. `DifficultyBeatmap` needs an explicit `version`.

## A hypothesis of mine, refuted by its own sweep
I expected **hand runs** to look more human than strict alternation, since
`role_asymmetry` is human 0.115 against our 0.026. For burst speed it is the
**opposite**: `runs=1` gives exactly the human's 376 and `runs≥2` gives 752, because
`ebpm_burst` is a **per-hand** rate and alternating is precisely what keeps each hand
slow. `--runs` survives as a stylistic knob; its default is 1 for a measured reason.

★**And the number that reframed it**: the human's *average* per-hand rate is **3.99**,
essentially identical to our 3.96 — while their burst rate is half. **The defect was
never the average, it was the fast tail.**

## What this says about the ML track
Nothing yet — the map has not been played. That is the whole point of building it:
`docs/eval_suite_v2.md`'s axes have been measured to not track Kyle's ear (M-F ranks
the map he called *"really empty"* second-best), so a hand-built map that scores well
proves only that it scores well. **Installed as `AUTO Hunger [AGENT]` for his verdict.**

## v2 — accents, and the value of not reinventing a solved problem

| | notes | nps | precision | ebpm | travel | doubles | parity |
|---|---|---|---|---|---|---|---|
| **AGENT v2** | 1261 | 4.66 | 0.984 | **376** | 4.60 | 0.034 | **0** |
| ML baseline | 1328 | 4.96 | 0.960 | 376 | 5.94 | 0.391 | 0 |
| HUMAN | 2254 | 8.35 | 0.956 | 376 | 12.53 | 0.146 | 0 |

**Doubles**, added because humans use them to buy density *without* speeding either
hand up — `hands_x_downbeat` is human 0.182 against our 0.036, and the note there was
that *"we spend doubles on 2/3 of all events so they mark nothing"*.
★**Order turned out to matter more than the rule**: in a dense alternating pass both
hands are always busy, so a double can never be placed — bolting them onto a finished
pass gave 0.016. Running a **sparse accent pass first and filling around it** gives
0.034. ⚠️Still 4× below the human 0.146, because only strong beats (slots 0 and 8) with
≥2 stems agreeing qualify; bar downbeats alone capped it at ~3 doubles in 24 bars.

### 🔴 Two rounds of hand-rolled parity repair, and the lesson
Adding doubles introduced **13 parity violations** — caught by `check`, before export,
which is what it is for. I fixed the "insert between two existing notes" case for
parity the same way I had for the gap floor… and got 13 → **5**, at a cost of **380
notes** in skips.
★**Then I stopped and used the parity fixer that already exists.**
`postprocess.fix_parity` has flow-aware look-ahead and is the model `swing_sim`
actually scores against; running it on check/export gives **0 violations and 0 resets**
(from 381) with **no** notes lost. The hand-rolled skipping was then deleted — it was
solving a solved problem and paying for it in notes.
⇒**Keep `auto`'s alternation as a starting guess and let the validated component own
correctness.** The gap floor stays hand-rolled because it is about *timing*, which
`fix_parity` does not address.

### Still open, both measured
- **travel 4.60 vs human 12.53** — `auto` uses two columns and two rows per hand, so
  the hands barely move. The human map is far more physical.
- **doubles 0.034 vs 0.146** — needs a broader accent model than "strong beat with
  stem agreement".


## 2026-08-16 — three perception axes, and the one hypothesis they refuted

Kyle: *"mostly work on the manual mapping suite and keep building crazy good tools until
you believe you have the same insights into a song to map as a human does."* The answer,
written down falsifiably, is **[`docs/perception_scorecard.md`](../docs/perception_scorecard.md)**
— 13 rows of what a human mapper perceives, each marked with the control it passes.
**Short version: not yet, but the gap is now named, and three of the four biggest rows
closed.**

### Built, each with a control
| tool | what it adds | control | verdict |
|---|---|---|---|
| `stemcache.py` | separate once, analyse many ways | — | infrastructure |
| `melody.py` | **pitch** — what note, and which way it moves | two independent trackers agree on the key on 36 % of 14 songs vs 4 % chance | **PARTLY CONFIRMED** |
| `percussion.py` | **which drum** is hitting | labelled groove repeats bar-to-bar, z = +12.7…+25.7 vs a shuffled null | **CONFIRMED, 3 of 4 songs** |
| `structure.py` | **sections, and which repeat** | repeated lyric lines share a letter 0.485 vs null 0.317, p = 0.019, 6 **held-out** songs | **CONFIRMED** |

### ★The refutation: `travel` is not a contour problem
Our hands barely move (`travel` 4.77 vs a human 12.53) and the obvious cause was that
nothing told the placer *where* to go. Built it, measured it, wrong:

- `--pitch` (level → column) made travel **worse, 4.77 → 3.56**. Melodies move in small
  steps, so following the contour parks consecutive notes in the same cell.
- `--pitch-span full` (interval → jump size) recovered the baseline (4.789) but
  overshot crossover to **0.523 against a human 0.183**.

⇒ **`travel` is a property of the note SEQUENCE, not of any per-note rule**, and is
therefore not a perception defect at all. Isolating that is the useful part.

### A real bug: `--wide` never widened anything
Hands strictly alternate and the column came from a *global* note counter, so `k % 2` was
perfectly correlated with the hand — the left hand only ever saw even `k`. Measured as
exactly **two** distinct columns across a 449-note map; now four, evenly. Fixed.

### Two controls thrown out for being wrong, not for failing
- **The backbeat** ("snare on 2 and 4"). Over **363 human maps**, note placement by
  beat-of-bar is 0.254/0.249/0.251/0.246 and only 29 % of maps peak on beat 1 against a
  25 % chance. Human structure is entirely in the **subdivision** (0.52 on-beat, 0.31
  eighth, 0.17 sixteenth). ⇒ **a downbeat detector is not worth building**, which this
  nearly was.
- **The melody step control** used a *median* of integer semitone intervals, quantised to
  1.0 for signal and noise alike. It read as a refutation and was a blunt ruler.

### Landmines paid for
- librosa's `voiced_prob` is **0.01–0.16 even where the flag is True and the pitch is
  right** — gating on it threw away 95 % of a correctly tracked vocal line.
- Segmenting f0 into notes independently of the onsets gives **48 "notes" for a 343-word
  song** (vibrato flips the rounded semitone every ~35 ms). Anchor **one pitch per onset**.
- Absolute spectral thresholds **do not transfer between mixes** — two hand-tuned drum
  classifiers each broke somewhere different (47 % "tom"; 96 % "snare"). Cluster each
  song against itself.
- Beat-level chroma clustering finds the **chord loop**, not the sections. Bar-level
  features with delay embedding put the window at phrase scale.
- The checkerboard novelty curve is almost entirely negative (~4 % above zero), so gate
  peaks on **prominence**, never height.
- Hunger's vocals are genuinely **unpitched** (metalcore; pYIN voiced-on-loud 0.19 vs
  0.91–0.99 elsewhere). The one song we hand-mapped is the worst case for a melody tool.
