# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md) — including the full session-by-session archive from 2026-06 to
2026-08-02. Evaluation-suite design rationale is in [`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule for keeping it that way:** when an item finishes, move the *outcome and what it taught* into
PROGRESS.md and delete it from here. A completed item is history, not work. Curated 2026-08-02, when
this file had reached 4,076 lines.

---

## 📍 CURRENT STATE (2026-08-03)

**The model was PROMOTED for the first time in the project's history.** Kyle graded *Hunger* (Aether
Realm) **A+** and said ship it. `generate.py` defaults now carry the full config; a bare invocation
reproduces the exact map he played.

📖 **The baseline is documented in full at [`docs/BASELINE_2026-08-03.md`](docs/BASELINE_2026-08-03.md)**
— architecture, all eight promoted defaults with their evidence, measured position vs the human
corpus, how we got here, and every landmine. **Read that before changing anything.** It is the thing
new work is measured against.

**Song names** (Kyle asked for names, not ids): `1f333` = **Hunger** (Aether Realm) · `1f8d6` =
**Fallen Kingdom (2022 Remap)** (CaptainSparklez) · `1f913` = **Digital Life Hacker** (Wotoha) ·
`1f767` = **アリスブルー** (HoneyWorks) · plus **SO TIRED ROCK** (NUEKI), Kyle's motivation song.

**★ Two standing methodology rules:**
1. **Never calibrate the human corpus through `scorecard._load_any`** — it prefers ExpertPlus. Use
   `calibrate_playfeel.load_expert_only`. Three human references were wrong because of this.
2. **Ask "norm or aspiration?" before calibrating any axis.** Kyle's target is the **best** mappers,
   so "the human cohort passes it" is not a validity check for aspirational axes.
   ⚠️He has **declined to name exemplary mappers for now** — *"we aren't close to exemplary"* — so that
   cohort is blocked by his choice, not by oversight.

**★ And the lesson that cost the most to learn**: *a lever can pass every axis in the suite and still
carry a defect no axis measures.* `BEAT_ONSET_EVIDENCE` degraded reachability and nothing noticed
until Kyle's correction forced the metric to exist.

---

## 🔴 P0 — SEED LOTTERY: **CLOSED** (cause + fix in PROGRESS.md)

Nothing in the generation path was ever seeded. `generate.py --seed` / `BSA_SEED` +
`eval_sweep --seeds N`. Verified: same seed → byte-identical map, from a fresh process, after a whole
sweep ran in between.

**Three habits to keep**
1. **Score every arm at ≥3 seeds and quote the sd.** ⚠️n=3 *underestimates* sd (idiom's went
   0.043 → 0.107 by adding two seeds) — treat n=3 as a screen, n=5 as a verdict.
2. **`npass` is not a ranking statistic** — an identical config scored 4, 4, 2. Rank on per-axis gaps
   with error bars.
3. **Pairing helps alignment only** (sd 0.033 vs 0.143) — it rides the postprocess `random` stream;
   the other axes ride the torch decode, which diverges once configs differ.

**Open**: the spread bar (0.35) sits inside the noise, so pass/fail on it is a coin flip. The bar is
not miscalibrated (human `min_spread` is 0.923; 0.35 was set as a mode-collapse alarm) — we simply sit
on it at 0.39–0.46. **Recommendation: stop gating on spread**, report it with its sd, keep a hard
alarm near 0.15. Not done unilaterally: it changes scorecard semantics and breaks comparability.


## ✅ P1 — PROMOTED 2026-08-03 (Kyle's call, on *Hunger*)

> *"The vast majority of the 1f333 song is A+ and better than what we had before so promote it."*

Eight defaults flipped in `generate.py` / `postprocess.py`: `BEAT_TEMPO_FIT=1`,
`BEAT_DIFFICULTY_SCALE=0.48`, `DENSITY_SELECT=1`, `DENSITY_SELECT_GAMMA=2.5`, `BEAT_HAND_LEAD=0.14`,
`BEAT_TRIM_TAIL=0.5`, `BEAT_ONSET_EVIDENCE=0.3`, `BEAT_REACH=3:0.3:0.5`. All still env-overridable.

**Verified**: a bare `generate.py` with no env vars reproduces `ExpertStandard.dat` sha `a432690c…`
on Hunger at seed 0 — byte-identical to the map he played. **Baseline: `docs/BASELINE_2026-08-03.md`.**

⚠️Carried into the baseline as known costs: `BEAT_ONSET_EVIDENCE` **degrades reachability** (repaired
by `BEAT_REACH`) and pushes **peak nps 6.25 → 6.50 against a human 5.5** — which is very likely part
of what W3 is about. Its only non-circular evidence is the rhythm axis.

---

## 🎯 W1–W7 — FROM KYLE'S PLAY-THROUGH OF THE PROMOTED MODEL (2026-08-03)

He played `outputs/kyle_review_2026-08-03/*_AFTER2_reach.zip`, graded **Hunger** A+ and told us to
promote. These are his remaining objections, in his order of annoyance. **Baseline being measured
against: `docs/BASELINE_2026-08-03.md`.**

★ **His standing instruction on how to work these**: *"I'm hesitant to change much because we have a
great foundation so we really need to tread carefully, make isolated and tactical changes, and
document like crazy."* One lever at a time, ≥3 seeds, and nothing promoted without his ear.

⚠️ He also declined to name exemplary mappers *yet*: *"we aren't close to exemplary."* So the
best-mapper cohort (needed for aspirational axes) stays blocked **by his choice**, not by oversight.

---

### 🔴 W1 — THE MODEL DOES NOT FIND THE CORE TEMPO-CARRYING INSTRUMENT ★ his biggest complaint
> *"Our model still fundamentally struggles to find the core aha tempo/instrument that a mapper
> obviously adheres to."*

**Evidence, all from his ear:**
- **SO TIRED ROCK** (his motivation song, *"always sucked on our model"*): a deep dooming bass lays
  the tempo for the whole song and *"the notes are stubbornly not being placed on this tempo. They
  are being placed on all of the other little sounds."*
- **SO TIRED ROCK @ 0:14** — a big guitar drop *"I don't think we have ever generated notes for
  across every model."*
- **SO TIRED ROCK @ 0:46** — a booming guitar plays 3 notes in sync with the booming bass and we map
  **nothing**: *"this epic coordination of instruments colliding doesn't exist."*
- **Digital Life Hacker**: a deep powerful bass on an interval other notes line up to (people
  chanting *"hey, hey, hey, hey"*) *"is not even recognized… they get confused on more electric songs."*

★ **This is almost certainly a REPRESENTATION gap, not a decode one, and we already have the receipt**:
Stage-1 `version_4` has only `drum_proj` + `mix_proj` — **no instrument projection**. It *literally
cannot hear the guitar* (recorded 2026-07-27). No decode lever can fix an input the model never sees.
That makes this **Track B**, and the largest open item in the project.

**His own proposals, all worth building:**
1. *"Maybe the demucs could parse and allow us to specify?"* — an explicit lead-instrument channel.
2. *"Maybe demucs should flag specific alignments when key instruments hit the same beat consistently
   and that could be a big flag for when a note should get placed."* — a **coincidence detector**.
3. *"Also maybe a sound compared to rest of song to easily draw intensity."* — relative loudness.

### ✅ MEASURED 2026-08-03 — his hypothesis is RIGHT, but the gap is elsewhere. Full data in PROGRESS.md.

- **His coincidence idea is CONFIRMED**: humans map a 4-instrument collision **84.5 %** of the time
  (response 0.407 → 0.575 → 0.724 → 0.845 as k goes 1 → 4, n=263).
- **Control passed**: `k` is **not** a loudness proxy — conditioning on onset-strength deciles retains
  **110 %** of the lift and `corr(k, strength)` is **−0.146**. `BEAT_ONSET_EVIDENCE` does not capture it.
- 🔴**BUT WE ARE NOT COINCIDENCE-BLIND**: our lift **1.915 vs human 1.732** — we respond *more* steeply.
  We under-respond **uniformly at every k** (0.352 vs 0.504 = **0.70×**), and **0.70 ≈ C5's
  distinct-times ratio 467/626 = 0.75** ⇒ **W1's symptom and C5's root cause are plausibly the same
  defect.** ⇒ ❌**DO NOT build "weight the budget by coincidence count"** — that pushes on the one
  thing we are not failing at.

### 🔴 W1a — THE LIVE DEFECT: we play the OFFBEAT at multi-instrument events
`scripts/eval_beat_phase.py` — `halfbeat_rate` = share of k≥3 events whose nearest note sits in the
outer third of the beat:

| cohort | n | halfbeat_rate |
|---|---|---|
| ours | 72 | **0.245** |
| human | 188 | **0.095** |
| SO TIRED ROCK (ours) | 1 | **0.316** ← past our own p90 |

**2.6× the human rate**, worst on his motivation song. At SO TIRED ROCK 0:14 the phase histogram is
**bimodal** — 203 events on-beat, **111 at exactly −½ beat** (0.244 s at 123 BPM).
★**No existing axis can see this**: a note parked on a lone-stem "little sound" is still on a real
onset, so it **passes A8**. Third instance of *a lever can pass every axis and still carry a defect no
axis measures.*

**Where the defect lives — measured 2026-08-03, PARTLY CONFIRMED.** A 24-song `BEAT_PROBS_DUMP` run
compared Stage-1's probability at each k≥3 event slot against the slot half a beat away:
**`win_rate` median 0.573** — it prefers the right slot, but only just, landing in the *pre-registered*
"partial, commit nothing either way" band (≥0.60 decode / ≤0.55 Track B). ★But
**`corr(win_rate, halfbeat_rate) = −0.494` over 23 songs**: the songs where Stage-1 cannot separate the
slots are the songs where we play the offbeat (Fallen Kingdom 0.900 → halfbeat 0.056; 1f336 0.49 →
0.31). ⇒ real driver, thin (57 %) edge for any decode lever.

⚠️🔴**DO NOT try to fix this with grid phase.** The slot grid is `subdiv=4`, so **a half beat is two
whole slots** and the grid already has a slot in both places; a phase shift moves notes by at most
±half a slot (≤61 ms). This is a **selection** defect, not a grid-placement one. (C2 stays valid for
its own purpose — songs whose grid really is misplaced — it is just not this lever.)

**Tasks**
1. 🔴⚠️**`halfbeat_rate` MAY NOT STEER A LEVER — it failed the control battery**
   (`scripts/audit_phase_metrics.py`, 2026-08-03): a **metronome scores 0.036 against the human
   0.084**, i.e. better. A constant pulse covers the beat grid densely, so minimising this metric can
   be achieved by becoming metronomic — the "for-sport" degenerate. It stays valid as a *diagnostic*
   vs human maps at matched density, which is all it has been used for. Any lever here must be
   co-scored against a **metronome guard** (rhythm A2 / `pulse_stability`), not merely against
   `on_event_rate`. ⚠️It is also insensitive to small timing error (`timing_jitter` moves it 0.0843 →
   0.0859, inside noise) — it detects wrong-slot *selection*, not sloppiness.
2. ⚠️**Do not commit a GPU night to a decode lever on a 57 % edge** — that is what the pre-registered
   band says.
3. 🔴**TRACK B AS ALREADY BUILT DOES NOT FIX THIS — tested, clean negative.** B-1 (`version_8` ep 12,
   `--use-instr`) vs prod `version_4` on the same 24 songs: `win_rate` 0.5797 vs 0.5731, **paired delta
   +0.0098 ± 0.0135 se (t = 0.73), better on 12 of 23 songs.** Not resolvable. Knowing *which
   instrument* plays does not tell the model *where the downbeat is* — B-1's real win was
   un-lockstepping the hands, a different capability. **Do not spend a retrain night on this
   expecting W1a to move.**
4. ★**The live hypothesis instead: give Stage-1 an explicit METRICAL-POSITION feature** — where each
   slot sits within the beat and the bar, from the tempo fit `data/tempo.py` already computes and
   nothing consumes. ⚠️This is **not** the refuted "shift the grid by the phase" idea: phase as an
   *input to the probability* is a different mechanism from phase as an *offset to the grid*, and only
   the latter dies to the half-beat-is-two-slots arithmetic. Neither `version_4` nor `version_8`
   encodes metrical position at all — which would explain a model that finds the active region
   (2.0–2.9× a random slot) but picks within it at 57 %.
5. ⚠️Unexplained, check before building: under v8, **1f767** reports `vs_random` **64×** and **1f9a0**
   `p_on_event` **0.0079** — the instrument model's probabilities are far peakier on some songs.
6. Cheap unblocked next step: check whether `win_rate` correlates with anything about the SONG (tempo,
   genre, stem separability) — Fallen Kingdom at 0.900 vs 1f336 at 0.49 is a 2× spread and the reason
   is unknown.
7. ✅The battery has now been RUN (`scripts/audit_phase_metrics.py`) — see item 1 for its verdict.

**DoD**: `halfbeat_rate` moves from 0.245 toward the human 0.095 with `on_event_rate` held or improved,
at ≥3 seeds — then Kyle's ear on SO TIRED ROCK.

---

### 🔴 W2 — FALLEN KINGDOM IS TOO EMPTY; and he wants a "how many notes" lever
> *"Its on beat, but its also an expert song and we shouldn't be afraid to play a simple beat thats
> medium tempo. Review the first minute and youll see what I mean. It just feels really empty for no
> reason."* … *"Maybe we should introduce a toggle that is 'How many notes do you want'."*

**Evidence**: we play *"like 1 out of 2/3 notes"* of an obvious slow beat. Baseline nps on Fallen
Kingdom is **3.21** against a corpus median of 3.91 — and this is a *slow* song, so a fixed global
budget under-serves it.

★**Now visible in one picture** — `scripts/view_song_strip.py --song 1f8d6` (added 2026-08-03):
- our NPS runs **~0.5 below the human's for essentially the whole first 175 s** — the "empty" feeling
  is a persistent level offset, not a few missed moments;
- **response to k≥3 multi-instrument events falls to 0.0 across ~185–200 s** and to 0.1–0.2 at
  ~90–100 s and ~210–230 s — whole passages with the music hitting hard and no notes;
- its **offbeat rate is low** (halfbeat_rate 0.056, better than the human 0.095) ⇒ **W2 is a distinct
  defect from W1a**, and fixing one will not fix the other.

⚠️**Do not just raise `BEAT_DIFFICULTY_SCALE` globally** — Hunger is A+ at the current budget and he
separately complains that parts of it are *too* intense (W3). The defect is **per-song / per-section
allocation**, not the global total.

### ✅ DIAGNOSED 2026-08-03 — **the budget is a fixed fraction of supply; Stage-1 is innocent**
Task 2 below is **done**, and the answer is decisive. On Fallen Kingdom's first minute Stage-1 scores
the human's note slots at **0.797** against **0.0032** for slots generally (~250×) — and it scores the
**48 human notes we skipped at 0.734**, essentially as high as the ones we played. The model is
pointing right at them and the decode declines. Across 13 songs, `used/supply` (slots emitted ÷ slots
with prob > 0.5):

| cohort | median | spread (p90 − p10) |
|---|---|---|
| ours | **0.582** | **0.115** |
| human | **0.854** | **0.435** |

⇒ we under-use the supply by **~45 %**, and our fraction is **nearly constant** while the human's
varies ~4× more (0.520 on 1f9a0 → 1.294 on 1f8a3). **A global budget cannot serve songs needing
different amounts — Kyle's complaint, measured.** ⇒**W2 is fixable in the decode today**, unlike W1a.

**Tasks**
1. Build `BEAT_NOTE_BUDGET` as a **user-facing multiplier** (he wants it in the UI — see
   [[feedback-levers-are-user-facing]]): clean range, monotone behaviour, default 1.0 = baseline.
2. 🔴🔴**DO NOT CHASE HUMAN DENSITY — REFUTED 2026-08-04 BY KYLE'S OWN VERDICT.** Distinct-nps as a
   fraction of each song's *own* human map: **Hunger (graded A+) 0.650**, **Fallen Kingdom ("really
   empty") 0.781**, アリスブルー 0.867. **The song he loved is the furthest below its human; the song he
   called empty is denser relative to its human.** The ratio is backwards from his verdict, so a
   budget lever aimed at human density optimises the wrong thing. (My earlier note here said to target
   the human `used/supply` median 0.854 — withdrawn. The 0.582-vs-0.854 measurement stands; the
   inference from it does not.)
   ⚠️And **none of tonight's metrics separates the two songs**: k≥3 response is *better* on Fallen
   Kingdom (0.667) than Hunger (0.545), and >1 s phrase holes are near-identical (0.538 vs 0.500).
3. ★**THE LIVE QUESTION — sharpen "empty" before sweeping again.** His words were *"we play like 1 out
   of 2/3 notes of an obvious slow beat"*: a claim that a **simple repeating pulse gets played
   intermittently**, not a claim about totals. Build the instrument for *that* — e.g. in sections with
   a steady pulse, what share of consecutive on-pulse positions carry a note, ours vs human. **W4's
   lesson applies directly**: `tail_ratio` returned a clean 1.000 null and the defect was real; a null
   from a blunt instrument is "not yet measurable", not "refuted".
   ★**Measured 2026-08-03 — the marginal note is much worse than the average note.** Notes added by
   `nb130` on Fallen Kingdom are **31.8 % k=0** (vs our existing 9.5 %) and **0.9 % k≥3** (vs 21.3 %),
   while still landing on a real human note ~56 % of the time. So a global bump closes the **count**
   gap (497 → 607 vs human 646) and not the **quality** gap. **Likely mechanism**:
   `DENSITY_SELECT_GAMMA=2.5` concentrates budget into loud windows, so extra budget goes deeper down
   the ranking *inside windows already served* while quiet windows holding good onsets stay starved
   (documented independently in C1). ⇒**next experiment: lower γ and raise budget TOGETHER.**
4. ⚠️The corpus median (3.91) was already a suspect target; it is now doubly so — it is measured over
   ~200 random corpus maps, a **different song population** from our 24-song eval set, so comparing
   our cohort nps to it compares two different things. Use each song's own human map, and even then
   only as context, never as a target (see 2).

**DoD**: Fallen Kingdom's first minute plays the main beat; Hunger does not get denser.

---

### 🟠 W3 — INTENSITY IS MISALLOCATED (revive the shelved structure work)
> *"Some parts of the song get really intense to play, even though they are not the main beat where
> you would expect the peak difficulty to be. Maybe we should revive some of the old work we did with
> assigning intensity to each part of the song like beat drop and what not and having higher nps for
> those sections."*

**Evidence**: baseline peak nps **6.5 vs human 5.5**, and on Hunger it reaches **9.5**. So we do have
peaks — they are in the wrong places. Corroborated by the A5 **structure axis, built and shelved as a
negative result** (2026-07-27) — it should be re-opened with this sharper target.

**Tasks**
1. Detect sections and classify intensity (drop / chorus / verse / breakdown). `_detect_sections_energy`
   already exists and `section_gate="loud_only"` uses it — start there rather than rebuilding.
2. Add his relative-loudness idea: intensity as **energy relative to the rest of the song**, not absolute.
3. Allocate the note budget by section intensity, so peaks land on drops.

**DoD**: peak nps lands in the sections Kyle would name as peaks; Hunger's over-intense passages calm
down without losing its A+ character.

---

### 🟠 W4 — PHRASES ARE NOT RESPECTED (absorbs the old K4)
> *"It also still feels like the phrase level playing isn't fully fledged, like normally a mapper
> builds a sequence through different sections of the song and we still aren't generating according to
> full phrases, a few times the singer is still finishing a sentence and there's no notes."*

**Measured, still failing**: notes-per-onset against each map's own median — Hunger's 1:30–1:33
build-up sits at **0.87×** against a 0.9× DoD, and **3:20–3:28 at 0.54×**. Design against the 3:20
window; it is far worse and barely moved under any lever so far.

### ✅ CONFIRMED 2026-08-04 — `scripts/eval_phrase_abandon.py`. Full data in PROGRESS.md.

| metric | ours (n=60) | human (n=120) |
|---|---|---|
| **`share_over_1s`** — sung phrases containing a **>1 s** hole with no notes | **0.539** | **0.250** |
| `share_over_2s` | 0.074 | **0.000** |
| `med_hole` | 1.071 s | 0.698 s |

**2.2×.** More than half our sung phrases contain a second or more of silence; the human median for a
>2 s hole is **zero**. ⚠️**The first metric (`tail_ratio`, density of the final third ÷ the first two
thirds) reported NO defect — both cohorts exactly 1.000.** A ratio of densities cannot see a hole.
Both are kept in the script, the blunt one labelled, so nobody re-derives it and believes it.

**Tasks**
1. Phrase boundaries are now cheap and already implemented (`vocal_phrases()`). The lever: when a
   vocal phrase is active, do not let the note stream go silent for >1 s. ⚠️Build it as a **budget
   redistribution**, not an insertion of notes at arbitrary times — W2 showed the marginal note is
   drawn from a much worse pool (31.8 % of added notes sit near no onset at all), and filling a hole
   with filler is exactly that failure.
2. ⚠️**Run the control battery first, and expect a `metronome` failure**: a constant pulse leaves no
   holes at all and would score *better than human* here, exactly as it did on `halfbeat_rate`.
   Assume this cannot steer a lever until shown otherwise.
3. Re-open the shelved A5 structure axis with "cover the whole crescendo, not just its peak".

**DoD**: `share_over_1s` falls from 0.539 toward the human 0.250 **without** the added notes repeating
W2's k-distribution collapse (check with `scripts/view_ab_diff.py`), at ≥3 seeds. Then Kyle's ear.

---

### 🟡 W5 — DOT (ANY-DIRECTION) BLOCKS ARE USED DECORATIVELY *(he is deferring this)*
> *"During the big iconic part of the song around 1.13 the drop uses any directional slice blocks.
> Its just unnecessary and should use directional blocks."*

★ **His rule, worth recording verbatim** — a dot block has exactly two legitimate purposes:
1. **Multi-note swings**, where several notes are sliced at once and you want to make it easier.
2. **A multi-directional single swing** — e.g. a swing that goes up *and* to the right in one
   interval, on harder maps.

*"I've purposefully ignored this for the time being and figured we could incorporate this feature
later."* — so **do not build this before W1–W4**; it is recorded so it is not lost.

**Task**: restrict dot emission to those two cases. Note `convert_dot_notes` already converts most
dots; check why some survive at 1:13 in Hunger.

---

### 🟡 W6 — MULTI-NOTE SWINGS ARE A MISSING FEATURE
> *"For future reference, big drops that are grand and not note heavy like the one at 1.14 is a
> perfect candidate for multi note swings."*

We have no concept of a deliberate multi-note swing (several notes taken in one motion). This is a
**new capability**, not a fix — and it is the legitimate use of dot blocks from W5, so the two should
be built together.

---

### ✅ W7 — DIAGNOSED **AND FIXED**, awaiting Kyle's ear (lever default OFF)
> *"The final note of the song did not line up together, the map was like .5 seconds late."*

**Read it literally — "together" means the two hands.** On Hunger the map plays doubles at 271.596 /
271.755 / 271.915 and then ends on a **lone red at 272.074**; the blue hand simply stops. Cohort
metric (final event not a double while ≥3 of the previous 4 were): **ours 0.159 vs human 0.036 —
4.4×**. Seed-dependent, not song-dependent (9/24 songs on ≥1 seed, **0/24 on all three**).

✅**`BEAT_TRIM_TAIL` is EXONERATED** — our last note sits **0.47 s before** the cut point, and there is
**no general "we end late" defect**: `last_onset − last_note` is 0.723 s for us vs 0.789 s for humans
over 13 songs. **Do not re-open this as a timing bug.** Full write-up in PROGRESS.md.

✅**`BEAT_END_RESOLVE=0.75` BUILT AND VALIDATED at 3 seeds × 24 songs** (2026-08-04): orphaned ending
**0.1528 → 0.0139** (human 0.036 — it lands *below* the human rate) and it **costs nothing** — rhythm,
idiom, playfeel, precision and nps are unchanged to three decimals; alignment, flow and handrole move
≤0.014, all in the improving direction. Paired note-count check over 34 matched (seed, song) pairs:
**28 deltas of 0, 6 of −1, never positive.** It drops the orphan rather than inventing a partner.
**Only step left: Kyle plays it.** ⚠️Do not promote unilaterally.

⚠️**Sweep-table reading trap**: the `delta / resolvable?` column compares the **second** arm to the
control, not the last. Difference the columns directly or `endres` looks like it costs +0.382 playfeel
when it costs 0.000.

---

### 🛡️ Confirmed positives — protect these
- **Hand-lead alternation**: *"a giant difference maker... noticeably great impact on the flow."*
- **Density pacing**: *"when there is a slow spot we let the player breathe... we no longer have the
  monotony flood of notes."*

Any lever that regresses A6 hand-role or the density-select behaviour trades away something Kyle has
explicitly valued by ear. Check both before promoting anything.

---

## 🛠️ SPARE-TIME STANDING TASK — the visual EDA suite (Kyle, 2026-08-03)

> *"We were in the process of building out an eval suite that mapped song notes to beatsaber notes you
> could visually see and do eda on so that the time between me needing to give input between iterations
> would be slowed down. If at any point overnight you have spare time continue to build out that
> functionality."*

**Purpose: raise the number of questions answerable without his ear.** Every iteration that currently
needs him to put on a headset is a day of latency. Work this whenever a GPU job is running and there is
no CPU-bound experiment queued.

**What exists already** — extend these, do not restart:
- `scripts/render_map.py` — PNG lattice panels, density strip over audio RMS, swing-path parity trace.
- `scripts/map_view.py` — the map as a readable text score, with stem lanes.
- `scripts/cohort_eda.py` — per-cohort reference stats.

**The actual gap, named by tonight's work**: nothing plots **per-instrument onsets against our notes**.
The offbeat defect (W1a — we sit half a beat off multi-instrument hits 2.6× more than humans) was found
through three separate numeric scripts and would have been **obvious at a glance** in a view with the
bass/drums/other/vocals onset lanes drawn against our note times and the human's. The seeded
`outputs/stem_onset_cache/` (274 songs) already has everything needed.

**Build order**
1. **Song-vs-map alignment view** — stem onset lanes + coincidence order `k` per event, our notes and
   the human's on the same time axis, beat grid drawn. Mark k≥3 events we missed and notes sitting a
   half-beat off one. This is the view that makes W1a visible.
2. **Whole-song strip** — nps, `halfbeat_rate` and per-section intensity along the song, so W2/W3
   (empty songs, misallocated peaks) are readable without playing.
3. **A/B diff view** — two arms on one axis, only the notes that differ highlighted. Most sweeps change
   little; showing everything hides the change.
4. Make it a single HTML artifact per song rather than loose PNGs, so he can scroll one file.

**DoD**: Kyle can open one file per song and answer *"did this lever help, and where"* without playing
the map. It never replaces his ear as the promotion gate — it replaces his ear as the **triage** gate.

---

## 🔵 P2 — CARRIED FORWARD

### C1 — Precision sits at the greedy optimum; gains need better probabilities, not better picking
Three decode levers were tried and **none moved onset precision off ~0.90**: density (0.895–0.904
across 3.63–4.42 nps), γ allocation (0.902 / 0.907 / 0.898, non-monotone), and a probability floor
(no effect at any quantile). Two predictions were logged in advance and both falsified. The IOI prior
*did* move it — to **0.769**, downward — proving selection controls precision but only from the
greedy optimum downward. **Stop hunting decode knobs.** Remaining leads in order: the threshold/NMS
stage (in replay, a min-distance of 2–3 slots alone costs 0.948 → 0.923), then Stage-1 itself (AUROC
0.755 against "this slot is on a real onset" — informative but not sharp).

### C2 — Grid PHASE, but only where the human control says the fault is ours
Decomposition: 0.756 →(tempo)→ 0.887 →(optimal global shift)→ 0.906 → human 0.930. The median phase
gain is small but rescues specific songs (1fa48 0.614 → 0.975). **The human control splits it**: on
1f767 the human map wants the *same* −45 ms shift we do, so that part is an **onset-detector offset,
not our grid**. Never apply a blanket global shift — that is fitting the detector, i.e. the `h_dist`
failure. Fix only songs where the human sits fine at zero and we do not (1fa48, 1f9a0).
`data/tempo.py` already estimates phase; nothing consumes it.

### C3 — Density/rhythm tension: you cannot thin your way to human density
Re-tuning toward the human note rate costs rhythm, and the sub-metrics say why: `pulse_stability`
−0.06 → −1.11 and `ioi_cond_entropy` +0.47 → +1.61 as nps falls 4.42 → 3.63. Thinning by probability
keeps confident notes wherever they fall and breaks the runs that make a rhythm legible. Humans at
3.9 nps have a pulse; we at 3.9 nps (thinned from 4.4) do not. The lever built for exactly this
(`BEAT_IOI_PRIOR`) made everything worse. Needs a different idea, not another sweep.

### C4 — Every beat-domain result predates the tempo fix
A2 rhythm, A6 handrole and the hand-offset work were measured **and tuned** against a declared BPM
grid that was wrong on 20 of 21 songs. Their conclusions are not necessarily wrong, but they were
scored with a bad ruler and never re-checked with a good one. Re-derive before building further on
them.

### C5 — Doubles: ROOT CAUSE FOUND, untouched
Not "too many notes" — **too few distinct times**. Same note budget as human (nps ~3.9) but spread
over **467 distinct beat positions vs the human 626**; double share 0.661 vs **0.1366** (⚠️the old
0.231 was the human *p90*, not the median — we are 4.8× not 3.4×).

**Cause**: Stage-1's two hand channels correlate **0.985–0.993** — both hands get the *same*
information, run the same top-k, and pick the same slots. A 66% double share is structurally
guaranteed, not mis-tuned. This retro-explains why `BEAT_HAND_INTERLEAVE` (moved notes to worse slots,
hurt rhythm) and `BEAT_HAND_ROLE` (leaves times untouched) both failed, and why A2/A6/flow-spread are
one defect.

**Fix must RAISE the count of distinct slots**, not redistribute. Decode version: allocate the hands
over **disjoint** slot sets — take the top 2k and deal them alternately — giving ~2k positions at the
same note count, never sending a hand to a lower-probability slot. Real fix is Track B (Stage-1
emitting per-hand information); the decode version prices what is reachable without a retrain.


### C6 — `outputs/` is gitignored: mitigated, one decision still owed
All seven calibration references are snapshotted to tracked `docs/eval_references/` (28 KB).
⚠️It is a **copy, not the live path** — the suite still reads `outputs/`, so **re-copy whenever a
reference changes** or the snapshot silently drifts. `data/` is gitignored too, hence `docs/`.
**Decision owed**: move the live path into version control, or keep copy-and-remember.


## 🧭 REFERENCE

### Landmines (each cost real time at least once)
- `scripts/generate.py` takes `audio` as a **positional** arg, not `--audio`.
- Load beat checkpoints with `strict=False`.
- **Never pick inference checkpoints by `val_token_acc` / `val_f1_avg_tol`** — they anti-correlate
  with alignment and structure quality.
- Production inference: layout `version_10`, beat `version_4`, `section_gate="loud_only"`,
  temp 0.9 / top-p 0.97.
- **The single-song probe trap**: 1f333 is half-tempo and beat-domain metrics lie there. Validate on
  all 24 songs. This trap has now caught two separate hypotheses.
- `pgrep -f <name>` inside a shell script **matches its own command line** and never fires. Wait on
  an explicit PID instead.
- `eval_sweep.py --true-bpm` writes to the **same cache key** as a normal run and will silently
  overwrite a non-oracle arm. Use a distinct arm name.
- Redirecting into a path that may be a **symlink** can truncate the target — `~/.local/bin/arcviewer`
  was a symlink to the running ArcViewer binary and was saved only by `ETXTBSY`.
- Logs under `logs/` and everything in `outputs/` are artifacts, not commits (see C6).

### Explicitly deprecated (do not revisit)
| Thing | Why |
|-------|-----|
| Scratch `AudioEncoder` mel transformer | MERT knows more music than we can teach it |
| Δt tokens in Stage 2 | Timing is explicit from Stage 1; conflating WHEN and WHAT was the root failure |
| `phrase_energy_alpha` / `dt_density_alpha` losses | Symptom treatment for a missing-explicit-timing root cause |
| `bomb_hand_weight` tuning | Bomb attractor was a symptom of bad timing loss |
| Per-window Δt autoregressive inference | Replaced by beat-slot iteration from Stage 1 |
| `BEAT_IOI_PRIOR` as a density lever | Measured negative at 3 seeds: fails its own purpose, wrecks 3 axes |
| `BEAT_GRID_SUBDIV` | No-op on the v7 production path; retired before it ran |
| Tuning anti-repeat / `dir_entropy` upward | "More diversity = more human" is false — and it caused K2 |
| Near-integer BPM as a crash cause | Falsified; the ArcViewer crash was in-process GTK |

### Success criteria — **rewritten 2026-08-02 against measured human values**
The previous version targeted "NPS ≥ 5.0, Expert range 4–10". That is now known to be **wrong**: the
human Expert median is **3.91 nps**, and 6.18 is the number Kyle called unplayable. Superseded by:

1. **Alignment** — onset precision ≥ 0.93, scatter ≈ 10 ms, **and no within-song drift** (K1).
2. **Difficulty** — ≈ 3.9 nps, diagonal share ≈ 0.37 and *falling* with local speed (K2).
3. **Structure** — double share ≈ 0.23; a legible pulse at human density (C3, C5).
4. **Reproducibility** — passes across **≥ 3 seeds**, not one lucky run (P0).
5. **The real gate** — Kyle plays it and wants to keep playing. The suite has been wrong about
   "ready" twice and right zero times; it is a filter for obvious defects, not the judge.
