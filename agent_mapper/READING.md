# Reading a map: the process, the habits, and how I catch my own mistakes

> **Kyle, 2026-08-21:** *"you should be able to piece together enough note sheets, compare them
> side by side in a text format to the map and evaluate by yourself, not using a metric that's
> computed if the map is fun to play or not"* … *"not everything needs to be a metric… just
> document the process you find works for yourself when creating a map from scratch. What habits
> to look out for you. How you find errors or unfun parts to play."* … *"the human ground truth is
> subjective. Songs can be mapped a lot of different ways and all be fun."*

This is that document. It is deliberately not a metrics reference — `WORKFLOW.md` has the commands
and `PROGRESS.md` has the numbers. This is **what to do with your eyes**.

---

## 1. THE BAR IS NOT THE HUMAN MAP. IT IS THE SPREAD BETWEEN TWO HUMANS.

The single most useful thing measured all session: **197 pairs of different mappers on the same
song.** They do NOT agree with each other. Correlation between two humans on the same song:
legibility 0.10, recurrence 0.01, crossover −0.01, density 0.30. **Two good mappers make
genuinely different maps.**

⇒**"Different from the human map" is not a defect.** The question is always:

> is my map further from that song's human than two humans are from each other?

**Median |mapper A − mapper B| on the same song** — this is the yardstick, use it:

| property | human-human spread | typical value |
|---|---|---|
| legibility (top-5 cell share) | **0.121** | 0.598 |
| recurrence (cell returns within 8 notes) | **0.113** | 0.522 |
| crossover (hand in the other half) | **0.078** | 0.218 |
| doubles (both hands at once) | **0.058** | 0.177 |
| density (nps) | **0.609** | 4.36 |

★Anything inside 1.0× of that spread is **a valid interpretation, not an error.** Above ~1.5× is a
real gap. This is how to hold "the ground truth is subjective" and "we still have defects" at once.
⚠️Two humans agree on **where the climax goes only 43 %** of the time (n=197). **Peak placement is
taste.** Do not "fix" a map because its climax is somewhere else.

---

## 2. THE READING PASS — what I actually look at, in order

### a. `--sections` first, and read the SHAPE not the level
```bash
python scripts/map_view.py <map.zip> --sections
```
Read the `density` sparkline as a contour. Ask: **does it breathe?** A map that is `▃▃▃▃` for
thirteen sections is flat even if every section is individually fine. Compare max/min: ours was
2.63× on one song against the human's 1.83× — I was *more* dramatic, which is also a defect.
★**`nps` and `peak_nps` percentiles will both say "human median" while the shape is wrong.** The
numbers describe the level; only the sparkline shows the arc.

### b. The `lead` column is the fastest read on the biggest defect
A human map alternates its lead hand between sections (`R0.14`, `R0.06`, `L0.04`, …). A column of
`--` means both hands play everything and no passage belongs to a hand.

### c. Then read an actual passage, side by side, aligned in SECONDS
```bash
python scripts/map_view.py <ours.zip> --vs <human.zip> --secs 30-45
```
Look down each hand's column, not across the row. What I look for, in the order it jumps out:

1. **Does a cell come back?** Human hands return to the same few cells constantly — `1,0 ↓`
   nine times in one passage. If every line is a new `col,row,arrow`, the map is a scatter and
   there is nothing to lock into. **This is the single most reliable read for "unfun".**
2. **Does each hand stay home?** A human's left hand lives in columns 0–1 for whole passages. If
   the left column keeps showing `2,x` and `3,x`, the hands are tangling.
3. **Does the arrow alternate ↓ then ↑?** That is the natural swing cycle. Long runs of the same
   direction mean the wrist is resetting silently.
4. **Are the gaps even?** Read the `beat` column. Human gaps sit on 0.5 / 1.0 with deliberate
   breaks. A column of 0.25 / 0.5 / 0.25 / 0.75 with no pattern is a stumble, not a rhythm.
5. **Is there ever a double?** `L … │ R …` on one row. Humans use them on ~18 % of instants as
   accents. A map with none is missing a gesture.

### d. Jump to the concrete failures
```bash
python scripts/map_view.py <map.zip> --find violations   # unplayable, non-negotiable
python scripts/map_view.py <map.zip> --find oov          # outside the human vocabulary
python scripts/map_view.py <map.zip> --find doubles      # read how they are placed, not how many
```

---

## 3. HABITS I HAVE TO WATCH FOR IN MYSELF

These are mine, caught repeatedly, and they recur:

🔴**0. THE PAGE PROPOSES; ONLY THE COHORT DISPOSES — and my eye has now been wrong twice
running.** Reading `1f767` on 2026-08-22 I was sure the right hand *"crosses into the left half
constantly"* where the human's stays home. Measured: ours **0.193** vs human **0.174**, a difference
of **0.73× the human-human spread — inside the range**. I then guessed the crossovers must at least
*cluster* differently; that was also wrong (0.964 vs 1.050, similar). **A passage that looks
lopsided usually is not; a handful of adjacent examples is exactly what randomness produces.**
★Use the page to generate the hypothesis and the cohort to keep or kill it — every time.

🔴**1. I over-read a single song.** Every strong pattern I saw once failed to generalise:
- *"the human repeats a 4-note figure"* — the ABAB rate was identical for both. It was vocabulary
  SIZE, not repetition.
- *"bass and other peak where the human's climax is"* — true on one song, chance across 23.
- *"humans place doubles as a symmetric both-hands-up gesture"* — true twice on one song; across
  109 maps the pairing is mixed 45 % / down 31 % / up 24 %.
★**Rule: read to form the hypothesis, then measure it on ≥20 songs before believing it.**

🔴**2. I quote the metric that flatters the change.** The judge scored my map **0.878** and the
human's map of the same song **0.274**. A high `p` means *blander*, not better — our maps sit
closer to the corpus centre than real maps do. **Human `p` band is 0.28–0.57. Above 0.7 is a
warning.**

🔴**3. I accept a ceiling from too little data.** A first scan over 600 zips found **4** mapper
pairs and gave a peak-agreement ceiling of 0.25 — which made our 0.26 look *at ceiling, therefore
not a defect*. Scanning all 5 373 gave **0.431** and reversed the conclusion.

🔴**4. My filters hide the error I need to see.** `grep -E "placed|pulse"` swallowed a tool
REFUSING my request; an `&&` chain aborted because `autobuild` exits 1 on a FAIL verdict and I
compared a hash of empty input; a bare `except` ate an `AttributeError` so a guard silently never
fired. ★**When a step produces no output, that is the finding — go look.**

🔴**5. I assume a knob does something.** Four levers this session were dead or undocumented:
`idiomize --width` (accepted, threaded, never used — width 1 and 12 gave byte-identical maps),
`--doubles`, `build_onset_cache --audio-dir`, `--adaptive-subdiv`. ★**Change the knob and diff the
output before building on it.**

---

## 4. HOW I FIND THE UNFUN PARTS

In the order they have actually worked:

1. **Read a passage against the same song's human.** Nothing else has found as much. The legibility
   defect — 23/23 maps below the human — was invisible to all 23 metrics and obvious on the page.
2. **Look for what is ABSENT, not what is wrong.** Zero doubles. Zero hand-lead alternation. A
   density contour with no arc. Absence does not raise a number.
3. **Read the tool's own refusal.** `structure --validate` saying *"do not trust these letters"*,
   or `--follow bass/low-stab` saying *"the class labelling FAILED its control on this song"*, is
   the map telling you the plan is built on sand.
4. **Check the thing the pass was supposed to fix, not the aggregate.** A PASS with the named
   defect unmoved is not a fix — `1f9a0` went FAIL→PASS while its `onset_precision` moved 0.474 →
   0.499, i.e. not at all.
5. **Ask what a player DOES here.** Fifteen notes with no repeated cell is fifteen separate
   decisions. That is the difference between a passage you learn and a passage you survive.

---

## 5. WHAT IS ACTUALLY LEFT (measured against the human-human spread, shipped config)

| property | ratio to human-human spread | verdict |
|---|---|---|
| legibility | 0.72× | ✅ inside — a valid interpretation |
| recurrence | 0.81× | ✅ inside |
| crossover | 0.73× | ✅ inside |
| doubles | 1.54× | 🟡 marginal (was 3.54× before `--doubles`) |
| **density** | **1.92×** | 🔴 **the real remaining gap** |

**Density is the one honest failure.** We build every song at ~4 nps (range 3.15–4.46) while human
maps for the same songs range **1.88–7.51**. On the fast songs we sit at **0.48–0.59** of the
human. ⇒**We flatten every song to the same difficulty**, which is the exact opposite of *"map
whatever difficulty we want"*.
⚠️And audio does not predict it: the best single feature (bass onset rate) reaches r = 0.47, and a
linear fit beats the constant by 0.08 nps. **Density is a CHOICE. It has to be an explicit input,
not something inferred.**
