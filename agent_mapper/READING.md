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
| travel | **0.975** | 4.43 |

★Anything inside 1.0× of that spread is **a valid interpretation, not an error.** Above ~1.5× is a
real gap. This is how to hold "the ground truth is subjective" and "we still have defects" at once.
⚠️Two humans agree on **where the climax goes only 43 %** of the time (n=197). **Peak placement is
taste.** Do not "fix" a map because its climax is somewhere else.

---

## 2. THE READING PASS — what I actually look at, in order

### 0. ★2026-09-02 — THE SCORE first: song and map on one lattice, every slot

> **Kyle:** *"The model doesn't have the visibility that I do when evaluating a map… rows are
> possible note placements, columns are the notes, matched with the song in note-sheet format
> with lyrics and all. This granular visibility with deep timings is what the model does not
> have. This would catch the obvious errors more than a metric. This is the eval suite."*

```bash
python agent_mapper/score.py <map.zip> --song 1f333 --sections            # triage: one row per bar
python agent_mapper/score.py <map.zip> --song 1f333 --vs auto --bars 33-36 # zoom, human beside
python agent_mapper/score.py <map.zip> --song 1f333 --vs auto --npz s.npz  # arrays for queries
```

**How to read a page.** Left of `│E ON MAIN│` is the song, right is the map, and with `--vs`
the human map of the same song sits at the far right as the answer key. Every 1/16 slot is a row
— **the empty rows are the point**: "VOX is singing and L/R are blank" is only visible when the
silence is drawn.

- Read `KIT` (K kick · S snare · h hat · C crash; upper = loud) against `L`/`R`: a good map's
  notes sit on K and S. Notes with `●` in `±ms` but a blank KIT cell are on *something* (the
  guitar 8ths) — that is what "notes flow in an odd way" looked like on Hunger AGENT bar 34.
- Read `VOX pitch + lyric` against `L`/`R`: a syllable with nothing beside it is a missed vocal.
  On 1f8d6 (Kyle: *"feels really empty"*) the agent struck only on the beat while "gave",
  "every", "night", "go" fell between beats unanswered; the human answered them.
- Read `DBL` and `ALT`: a column of `D … LR` is two-hand claps on every beat — 8 notes/bar that
  feel like 4. That was the EMPTY map (59 % doubles vs the human's 7 %).
- `TRV`/`ROT` per hand: cells travelled and degrees rotated from the hand's previous note.
  `ALT R!` = the same hand twice within a beat.
- `E` (0–9) is the audio's energy; a step in E with no step in `L`/`R` density is a drop the
  map missed. `MAIN` names the line the player is following (vox > kik > snr > led-in-rests).
- `±ms` is the note's distance to the nearest reference onset: `●` ≤50 · `○` ≤120 · `✗` nothing
  there. `H±ms` is the same for the human note — if both are +20 the clock is 20 ms early, not
  the notes.

**The triage page (`--sections`)** flags per bar: `THIN-vs-human`, `DENSE-vs-human`,
`MAIN-unanswered`, `VOX-silent-map`, `EMPTY-loud`, `BURST-no-onsets`, `✗n`. Flags are where to
zoom, not verdicts — the A+ map of 1f333 carries 23 THIN flags and Kyle promoted it. Zoom with
`--bars` and read the rows; clean bars stay coarse.

**What the score does NOT see** (honest, 2026-09-02): a global signature for *flow* — the
bar-level counts of Hunger AGENT (odd flow) and A+ (preferred) barely differ (on-KIT 85 % vs
78 %); the difference is on-beat share (36 % vs 57 %) and the per-hand direction sequence, which
the page shows but no query yet names. That is TODO P3 FLOW. Caveats printed in the header
(melody coverage, lyric language probability, missing caches) are real — a blank VOX lane on a
screamed vocal at cov 0.28 is the song.

### 0b. ★2026-09-02 — STUDY the tutor before building (`scripts/tutor.py`)

The corpus holds one human map per song, crawled rating-sorted with upvote ratio ≥ 0.8 ⇒
`data/raw/<sid>.zip` is the best human map we have of it — **the tutor**. Read it BEFORE
building, on the score, cut at the moments the song hands out:

```bash
python scripts/tutor.py 1f8d6                                # situation → what the human did there
python scripts/tutor.py 1f8d6 --bars 102-103                 # the rows, to copy cells and cuts
python scripts/tutor.py 1f8d6 --map outputs/x.zip            # ours beside the tutor, SAME / differs
python scripts/tutor.py --vocab                              # the review songset's four tutors, pooled
python agent_mapper/score.py x.zip --song 1f8d6 --vs 1f8d6   # --vs takes a corpus id now
```

**Situations come from the SONG columns only** (section changes and repeats, E jumps/drops ≥
0.25, a stem entering after ≥ 2 silent bars and staying, a stem leaving), so the tutor and our
build are cut at identical bars. A **pattern** is the 2 bars from that bar: a word (`rest sparse
doubles stream burst alt-8ths 8ths alt-4ths 4ths mixed`), events/bar, doubles %, the first
answer in beats after the bar line, the bar BEFORE (`pre rest` = the breath), and the glyphs
per 1/16 (`L R D · w`). "Answered the tutor's way" = same word, events/bar within ±35 %, first
answer within 1 beat — the count is the DoD, not a rate.

**What the four tutors said (`--vocab`, 2026-09-02, 99 situations):** `vox-in` → alt-8ths
10/17 at **0 % doubles**, first answer `+0b`; `drums-in` → **doubles 4/6** and `bass-in` →
doubles 3/5 (a double is the human's ENTRY accent, not a texture); `E-drop` → **rest 5/10**;
`E-jump` → the bar before is **rest 4/5**; `repeat` → alt-8ths/alt-4ths, the same figure as
the first time (1f913 bars 43 and 83 are glyph-identical). Nearly every first answer is `+0b`:
humans hit the moment on the bar line.

**What copying looks like (1f8d6, PROGRESS 2026-09-02e):** at the breakdown (bar 102, A→C, E
5→2) the tutor keeps **one hand on the lead's own notes** (`R··· R··· R··· R·R·`, every note a
`led` MAIN row); at the drop (110) three **doubles + walls, then a wall-only bar**; at the
section entry (62) a double on the bar line then **L on the lead's off-beats, R on the kit**;
in the outro one note per bar inside full walls. Read the rows with `--bars`, place them with
`mapedit.py from edits.txt` — reverse the tutor's cut sequence when our parity enters on the
other foot, keep its cells. Then `--map` to confirm the count moved.

### 0c. ★2026-09-02 — ASK the arrays (`scripts/queries.py`) — every hit is an address
```bash
python scripts/queries.py <map.zip>                 # song id from the file stem, --vs auto
python scripts/queries.py <map.zip> --song 1f8d6 --query q_flow
python scripts/bench.py score queries:q_all         # does the reader still agree with Kyle?
```
The score shows; the queries ASK, in Kyle's codes, and answer `code mm:ss bar why` so the next
command is `score.py --bars a-b`. Six of them, bench-tested against 19 of his verdicts (0 false
fires on the four songset humans, every labelled defect hit, AGENT > BEFORE on FLOW 19 % vs 2 %):

| ask | code | the read |
|---|---|---|
| are we playing the song? | **EMPTY** / **D1** | player EVENTS (a double is ONE) per 4 bars vs the human's: < 0.6× is empty; median < 0.7 is "very slow" |
| are we wasting nps? | **D6** | ≥ 2× the human's events in a window; or ≥ 50 % doubles map-wide and 20 pts over his |
| does it flow? | **FLOW** | ≥ 30 % of events start on an odd 16th out of silence *while* ≥ 30 % sit on the 8th grid — jitter, not a grid |
| is the grid shifted? | **D2** | ≥ 80 % odd 16ths where the reference has < 35 %. **Not ±ms alignment** — humans sit off-sound as often as we do |
| are we singing? | **D4** | vox-MAIN slots with a note within ±1 slot, ≥ 25 pts under the human — it names the unanswered words |
| did the drop land? | **D3** | at an E-jump: first note > 1 beat after his, or step and density both < 0.8× his; at an E-drop: he halves, we don't |
| the other elements? | **ELEMENTS** | 0 walls where he has ≥ 5 (arcs/chains reported only — the songset humans are v2) |

★★**The reference is the same song's human map (`--vs auto`), never an absolute norm.** Two
absolute rules died in one afternoon: a D3 step floor fired on human 1f913 (its own jumps are
0.6–1.1× steps) and "odd 16th = shifted" called 20 spans of 1f335 shifted — at 195 bpm the human
sits on the odd 16th 35 % of the time, it IS the felt 8th. Without a human the queries that need
one stay silent and say so; only FLOW/D2 fall back to the song's onsets.
**Refuted reads — do not retry without a new column:** hand-role histogram distance (backwards:
A+ 0.16 < AGENT 0.54), 16th bursts followed by rest, ±ms as D2, D5 "random bursts" as
16th-runs-without-onsets or isolated fast clusters (humans do both as often) — **D5 has no
locator**; notes-under-walls (no wall height in the arrays; human 1f767 has 4 under crouch walls).
**What they said about today's defaults:** `TUTOR__1f8d6`/`NOTUTOR__1f8d6` → 0 hits (ratios
0.6–1.5, vox 84 % vs 88 %) while the tutor scores 4/15 — the queries measure *wrong*, the tutor
measures *like him*; both are needed. `NEW__1f335` → EMPTY ×16 (bars 77-88: 1 event vs 31);
`NEW__1f9a0` → D6 ×5 (2–2.6× his events in every chorus).

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

### c2. ★NEW 2026-08-24 — read the two channels that used to be invisible
```bash
python scripts/map_view.py <map.zip> --bars a-b --align --elements
```
**`--align` is how you hear the map.** Per note: **●** on a sound (≤50 ms) · **○** near-miss
(50–120 ms) · **✗** nothing there. A run of ✗ is the player's *"why am I hitting this?"*, and it is
invisible in every aggregate. The whole-map ●/○/✗ split prints underneath — ★read that too, because
a passage is a sample and rule 0 below is that the page proposes while the cohort disposes.

**`--elements` is how you see the other three.** 🔴**Until this date no reading tool could see
walls, arcs or chains at all** — `mapjudge`'s 23 metrics move by exactly **0.000** when 89 walls +
90 arcs + 16 chains are added, so a `[FULL]` map read as notes-only. Now the sheet carries a
`lanes` column (`██··` = which columns are blocked *right now* — the dodge decision), gesture marks,
and an audit: **notes trapped inside walls** (this has shipped — 12 of them), and **dodge windows in
seconds** (how long the player had to leave the lane).
Both are scored against **2 688 human maps**, so the output is a percentile, not a bare number.
⚠️Presence is tracked separately from amount: 90 % of human maps have walls but only 31 % have
chains, so "we have 16 chains" is only meaningful against the maps that use chains at all.

### d. Jump to the concrete failures
```bash
python scripts/map_view.py <map.zip> --find violations   # unplayable, non-negotiable
python scripts/map_view.py <map.zip> --find oov          # outside the human vocabulary
python scripts/map_view.py <map.zip> --find doubles      # read how they are placed, not how many
```

### 0d. ★2026-09-02 — THE GATE (`scripts/verdict.py`) and the ops that clear a red
```bash
python scripts/verdict.py <map.zip>                       # queries + tutor + judge, one page, SHIP? — exit 1 = no
python scripts/tutor.py <sid> --map <map.zip> --copy a-b  # ops: his cells replace ours in bars a-b
python scripts/tutor.py <sid> --map <map.zip> --thin a-b  # ops: ours survive only beside his slots
python scripts/tutor.py <sid> --map <map.zip> --fill a-b  # ops: his slots we skipped, his hand+cell, OUR parity-safe arrow (EMPTY/D4, not a paste)
python agent_mapper/mapedit.py <map.zip> from ops.txt     # apply, guarded (parity, 150 ms gap, walls)
python agent_mapper/mapedit.py <map.zip> resets           # same-parity repeats as addresses
```
The verdict is the only thing that decides. Every 🔴 carries bars and the tool that fixes them;
`SHIP? YES` needs no red, and a PLAYABILITY 🟡 means resets above `2×human+2`. Three rules the
1f767 loop (172 ops, 27 resets → 0, judge PASS) paid for: **the zip is the artefact once
`mapctl export` has run** (walls/arcs/chains are zip surgery — `mapctl clear` + `auto` rebuilds
the session, not the map); **reconcile a reset with ONE note per hand** (place back the kick you
thinned, delete the stray, or `flip <addr> <hand> X` — a dot absorbs the flip); and **never flip
the second note of the pair** — that error walks forward through every note after it. Reading
order for a red: `score.py --bars a-b` first, ops second, re-read the same bars third.

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
| **density** | understood | ceiling is the song's own event supply, not the code |
| travel | 1.12× | 🟡 marginal |
| `vertical_share` | 1.32× | 🟡 marginal — largest NON-CIRCULAR gap left |

**Density is the one honest failure.** We build every song at ~4 nps (range 3.15–4.46) while human
maps for the same songs range **1.88–7.51**. On the fast songs we sit at **0.48–0.59** of the
human. ⇒**We flatten every song to the same difficulty**, which is the exact opposite of *"map
whatever difficulty we want"*.
⚠️And audio does not predict it: the best single feature (bass onset rate) reaches r = 0.47, and a
linear fit beats the constant by 0.08 nps. **Density is a CHOICE. It has to be an explicit input,
not something inferred.**
