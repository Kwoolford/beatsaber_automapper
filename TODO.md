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

⚠️Carried into the baseline as a known cost: `BEAT_ONSET_EVIDENCE` **degrades reachability** (repaired
by `BEAT_REACH`); its only non-circular evidence is the rhythm axis.
⚠️🔴**The old note here also said it "pushes peak nps 6.25 → 6.50 against a human 5.5" and blamed W3 —
that human 5.5 is RETRACTED** (2026-08-04). It was a ~200-map corpus median compared against our
24-song eval set. Per song, against each song's own human map, **we peak LOWER** (4.00 vs 5.25). See W3.

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


### 🔴 W1 — HE CANNOT FIND THE CORE TEMPO-CARRYING INSTRUMENT ★ his biggest complaint
> *"Our model still fundamentally struggles to find the core aha tempo/instrument that a mapper
> obviously adheres to."* … SO TIRED ROCK's dooming bass ignored; its 0:14 guitar drop *"never
> generated across every model"*; 0:46 guitar+bass collision → nothing; Digital Life Hacker's chanted
> pulse unrecognised.

**MEASURED 2026-08-03/04 — full data in PROGRESS.md. Summary of what is settled:**
- ✅His **coincidence hypothesis is right**: humans map a 4-instrument collision **84.5 %** of the time
  (0.407 → 0.845 as k goes 1 → 4, n=263), and `k` is **not** a loudness proxy (conditioning retains
  110 %; `corr(k, strength)` = −0.146).
- ❌**We are not coincidence-blind** (our lift 1.915 vs human 1.732) ⇒ **do not build "weight the budget
  by coincidence count"**. We under-respond uniformly at every k (0.70×), ≈ C5's distinct-times ratio.
- 🔴**The live defect is that we play the OFFBEAT** at multi-instrument events: `halfbeat_rate`
  **0.245 vs human 0.095** (2.6×), SO TIRED ROCK worst at 0.316. **No existing axis sees it** — a note
  on a lone-stem "little sound" is still on an onset and passes A8.

**What is RULED OUT for W1a — do not retry these:**
1. ⚠️**Grid phase.** `subdiv=4` ⇒ a half beat is **two whole slots**; the grid already has one in both
   places. A phase shift moves a note ≤ half a slot. It is a **selection** defect.
2. ⚠️**Track B as already built.** B-1 (`version_8` ep 12, `--use-instr`) vs prod: paired delta
   **+0.0098 ± 0.0135 (t = 0.73)**, better on 12 of 23 songs. Knowing *which instrument* plays does not
   tell the model *where the downbeat is*.
3. ⚠️**A decode lever on the current signal.** Stage-1 prefers the right slot only **57.3 %** of the
   time — inside the band this project pre-registered as "commit nothing either way". (But
   `corr(win_rate, halfbeat_rate) = −0.494` over 23 songs, so the probability field is a real driver.)
4. ⚠️**`halfbeat_rate` may not STEER anything** — a metronome beats a human on it (0.036 vs 0.084).

★**THE LIVE HYPOTHESIS: give Stage-1 an explicit METRICAL-POSITION feature** — where each slot sits in
the beat and the bar, from the tempo fit `data/tempo.py` already computes and nothing consumes. This is
*not* idea (1): phase as an **input to the probability** differs from phase as an **offset to the grid**.
Neither `version_4` nor `version_8` encodes metrical position at all, which would explain a model that
finds the active region (2–2.9× random) but picks inside it at 57 %.
⚠️Unexplained, check first: under v8, **1f767** reports `vs_random` **64×** and **1f9a0** `p_on_event`
**0.0079** — its probabilities are far peakier on some songs.

---

### 🔴 W2 — "FALLEN KINGDOM FEELS REALLY EMPTY" — ❓CAUSE STILL UNIDENTIFIED
> *"It just feels really empty for no reason… we play like 1 out of 2/3 notes of an obvious slow beat."*
> Plus: *"maybe we should introduce a toggle that is 'How many notes do you want'."*

✅**The toggle is BUILT**: `BEAT_NOTE_BUDGET` (default 1.0 = byte-identical to baseline; monotone,
788/982/1173 notes at 1.0/1.25/1.5 on Fallen Kingdom). Ready for the UI he wants.

🔴🔴**BUT THE CAUSE IS NOT FOUND, AND FOUR INSTRUMENTS HAVE FAILED.** As a ratio to each song's *own*
human map, the map he called empty is **equal or better** than the map he graded A+ on every one:

| instrument | Hunger (**A+**) | Fallen Kingdom (**"empty"**) |
|---|---|---|
| distinct-nps / human | 0.650 | **0.781** |
| k≥3 response / human | 0.62 | **0.88** |
| >1 s phrase holes | 0.500 | 0.538 |
| pulse coverage / human | 0.72 | **0.94** |

⇒🔴**"Match human density" is REFUTED as a target** — he graded A+ a map at **0.650** of its human's
density. (My earlier advice here to target the human `used/supply` 0.854 is **withdrawn**; the
0.582-vs-0.854 measurement stands, the inference does not.)
★**Best explanation: the two verdicts are on different scales** — Hunger was *"A+ **and better than
what we had before**"* (vs our own history), Fallen Kingdom judged against *the song's obvious beat*.
No corpus-relative metric can separate those.

**Task — ★ASK KYLE, do not build a fifth metric:**
> *"Does Fallen Kingdom feel empty compared to what our model used to do, or compared to what the song
> obviously wants?"*

One sentence decides whether this is a regression or a ceiling.

✅**Separately, a REAL defect was found on the way** (independent of "empty"): on an obvious steady
beat we answer **`pulse_coverage` 0.612 vs the human 0.811**, `pulse_continuity` 0.714 vs 0.832
(`scripts/eval_pulse_consistency.py`). Worth fixing on its own merits.

✅**And the allocation mechanism is now understood** — at budget 1.30, γ 2.5→1.5→1.0 moves rhythm
0.917→0.445, playfeel 1.511→1.289, added-notes-on-k≥3 **0.9 %→10.8 %**, added-on-k=0 31.8 %→18.3 %, at
the same note count. ⇒**RULE: any lever that places more notes must flatten γ, or it buys filler.**
⚠️Not promotable as-is (γ2.5 was chosen to buy `density_corr`, unscored here).

---

### 🟠 W3 — "SOME PARTS GET REALLY INTENSE TO PLAY" ⇒ **THIS IS C5**
> *"Some parts of the song get really intense to play, even though they are not the main beat where you
> would expect the peak difficulty to be."*

🔴**Its old stated evidence is RETRACTED** — *"peak nps 6.5 vs human 5.5, Hunger 9.5"* compared our
24-song set against a ~200-map corpus median (**different populations**). Per-song on identical audio:
**we peak LOWER (4.00 vs 5.25, resolvable)**; on Hunger ours is 7.00 vs its human's **9.25**. Location
fails too (`peak_intensity` 0.727 vs 0.735), and per-window `hard_rate` on Hunger is *easier* than its
human at every quantile.

★**What he actually felt**: at Hunger 4:20–4:32, on identical 160 ms grids, the human plays **66 events
at 0.015 double share** while we play **50 at 0.640** ⇒ fewer moments, **more notes/s (6.56 vs 5.36)**,
far harder to execute. **That is C5.** ⚠️Cohort notes/s is +0.56 ± 0.56 = **noise, do not quote**;
PARTLY CONFIRMED on the named song only.

**Tasks**
1. ★**Treat W3 as a symptom of C5** — fix C5, then re-check. Do not build a separate intensity lever.
2. ⚠️**LANDMINE: any difficulty axis must count NOTES, not distinct events.** On events we look
   *easier* than human while the map plays *harder*; `peak_nps` actively hid this.

---

### 🟠 W4 — PHRASES ARE ABANDONED MID-VOCAL ✅**CONFIRMED**
> *"A few times the singer is still finishing a sentence and there's no notes."*

| metric | ours | human |
|---|---|---|
| sung phrases with a **>1 s** hole | **0.539** | **0.250** |
| with a **>2 s** hole | 0.074 | **0.000** |

**2.2×** (`scripts/eval_phrase_abandon.py`, n=60/120). ⚠️**Its first metric `tail_ratio` reported NO
defect — both cohorts exactly 1.000.** A ratio of densities cannot see a hole. Both are kept in the
script, the blunt one labelled.

**Tasks**
1. Lever: when a vocal phrase is active, do not let the note stream go silent >1 s. `vocal_phrases()`
   already exists. ⚠️Build it as **budget redistribution, not insertion** — the marginal note is drawn
   from a much worse pool (31.8 % of added notes sit near no onset), so filling a hole with filler is
   that failure again. And **flatten γ** while doing it (see W2).
2. ⚠️**`share_over_1s` may not STEER** — a metronome beats a human on it (0.200 vs 0.250), the same
   failure as `halfbeat_rate`. ★**Every metric that rewards REGULARITY is metronome-gameable**; any
   lever here needs a metronome guard (rhythm A2 / `pulse_stability`) scored alongside.

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


## 🔴🔴 P0 — THE EVAL SUITE IS NOW THE TOP PRIORITY (Kyle, 2026-08-04)

> *"I can only communicate the problem but you can see the correlation to the config of the model… if
> you had a view of all notes of the instruments and vocals the way I did, the iteration speed would
> increase dramatically. I think this should be our top priority going forward. Create a way for you to
> see the song and map in a way that gives you my vision."*

**This outranks every W-item.** The bottleneck is not ideas, it is that finding a defect currently
costs one of his listening sessions. Every hour spent here buys back many of his.

### ★ HIS DEFECT DESCRIPTION, NOW MEASURED — this is what the view must make visible
> *"It feels like every couple main beat notes were mapped instead of most of the main beats. Maybe
> this is hidden because the map still maps a lot of non main beat notes. Like it hits the main flow
> partially."*

Measured over the full songset with the robust multi-level grid (`scripts/main_beat.py`, n=24):

| | ours | human |
|---|---|---|
| main beats **covered** | **0.546** | **0.704** |
| **`main_continuity`** — P(play beat n+1 │ played beat n) | **0.523** | **0.697** |
| share of our notes ON the main beat | 0.637 | 0.617 |

★**`main_continuity` IS THE METRIC FOR HIS COMPLAINT.** *"Every couple main beat notes were mapped
instead of most of the main beats"* is exactly "given we played one, do we play the next" — ours
**0.523** vs the human's **0.697**. We drop in and out of the line; humans hold it.

⚠️**But the second half of his sentence is NOT what distinguishes us.** He guessed it was *"hidden
because the map still maps a lot of non main beat notes"* — proportionally we sit at **0.637** against
the human's **0.617**, i.e. the same. We do not play more filler than a human. The line is partial and
the filler is normal, which is presumably *why* it reads as hidden.

### 🔑 A CONSTRAINT THAT SHAPES THE WHOLE DESIGN
**I can only "see" an image by rendering it to a file and reading it back.** So the primary artifact
must be **PNG**, sized and cropped so detail survives; HTML is for *him*, not for me. A beautiful
interactive page I cannot look at would be a tool built for the wrong reader.

### THE PLAN

**P0.1 — Robust MAIN BEAT identification** *(the crux; everything else rests on it)*
Score every metrical level (½×, 1×, 2×, 4× the fitted beat, plus a downbeat phase) against the
carrier stems and pick the level the *music* supports. Report the level and a confidence. Show **all**
levels in the view so the map's chosen level is visible when it differs from the music's.

**P0.2 — The view** — per song, stacked PNG panels at a readable seconds-per-inch:
bass / drums / other / vocals onset lanes · the main-beat lane with **missed beats ringed** ·
our notes split by hand · human notes · off-main notes marked. Bars and section boundaries.

**P0.3 — Metrics panel** in the same render: main-beat coverage, non-main share, per-section, ours vs
human — so a picture is never separated from its numbers.

**P0.4 — Robustness**: any song, no human map required, missing stems tolerated, one command,
fast enough to run over all 24 songs.

**P0.5 — Close the loop**: fold `review_map.py`'s ranked timestamps into the same artifact, so
"where to look" and "what it looks like" are one output.

### Answers he gave 2026-08-04 (do not re-ask)
- **The soft outro SHOULD be mapped sparsely** (Fallen Kingdom) — not left empty, not filled.
- **"Empty" is relative to THE SONG, not our old maps** — *"they both feel empty compared to song."*
  ⇒ the human corpus is not the reference for this defect; the song's own main beat is.

### ✅ S5 — `BEAT_MAIN_BEAT_BONUS` WORKS AND NEEDS HIS EAR (2026-08-04)
Built from his "it hits the main flow partially" after the probability dumps proved the model knows
about the beats we skip. 3 seeds × 24 songs:
**alignment 0.260 → 0.087 (RESOLVABLE, 3×)**, precision 0.919 → 0.930, rhythm/idiom/handrole all
improving inside noise, **nothing regressing resolvably at mbb015/mbb025**. Main-beat coverage
0.546 → 0.596, continuity 0.523 → 0.559 (human 0.704 / 0.697).
⚠️**Helps ~1/3 of the way, does not solve.** ⚠️mbb050 turns over (idiom 0.854). ⚠️`notes_on_main` drifts
past the human from mbb025 up — metronome direction; mbb015 is the conservative pick.
**Next**: build the review pair (BEFORE vs mbb015 vs mbb025) on the four standing songs for his ear.

### 🔎 FOUND BY THE SUITE — new work items (2026-08-04)

**S4 🟠 `BEAT_TRIM_END_COINCIDENCE` IS RIGHT ON HUNGER AND MIXED ELSEWHERE.** Fires on 6/24 songs;
among the 4 scoreable it helps 2 (Hunger exactly: 272.07 → 271.76 = human) and over-cuts 2 (1fbfb,
1fa48). Cohort ending offset unmoved (0.469 s). **Do not promote as a default.** ⚠️`BEAT_END_RESOLVE`
separately worsens ending *time* (0.469 → 0.750 s) while fixing ending *shape* — two criteria, and
Kyle's ear picked shape. **Task**: make the cut conditional on the map actually having a straggler
past the last coincidence, rather than cutting whenever a coincidence is found earlier than the plain
cut.

**S1 🔴 STAGE-1's PROBABILITY FIELD IS ONE SLOT OUT OF PHASE ON 2 OF 24 SONGS.**
`1fa48` main-beat coverage **0.002**, `1f9a0` **0.000**, against human 0.734 / 0.506. Diagnosed to the
mechanism, and it is **not** a tempo or trim issue — both cohorts use the same bpm (126 / 93) and our
notes sit exactly on our own slot grid.

Stage-1's probability alternates with the beat period. Median probability by slot offset from the main
beat (0 = the beat):

| song | −2 | −1 | **0** | +1 | +2 |
|---|---|---|---|---|---|
| 1f333 (healthy) | 0.687 | 0.080 | **0.692** | 0.076 | 0.686 |
| **1fa48** | 0.117 | **0.754** | **0.117** | 0.738 | 0.117 |
| **1f9a0** | 0.038 | **0.705** | **0.036** | 0.711 | 0.035 |

⇒ the model fires confidently, on a **strict alternation, exactly one slot away from the beat**.
★**The human map is the tie-breaker and it sides with the grid**: human notes sit −8 ms (1fa48) and
−32 ms (1f9a0) from my main beat, so the grid is right and **Stage-1 is displaced**. That also rules
out the metrical-phase ambiguity I first suspected in my own fit.

⚠️I initially filed this as "our map is on a displaced grid" and as a possible C2 phase bug. Both were
wrong: the slot grid covers the main beats to within 9 ms. It is a **selection** consequence of a
**displaced probability field**.

**Task**: find why the probability lands one slot early/late on these two songs — MERT pooling phase
(`pool_to_beat_grid` anchors at t=0), or the tempo fit's unconsumed `phase_s`. ★These are the cheapest
debugging targets in the project: the error is **noise-free** (sd 0–6 ms) and there is a healthy
control song (1f333) to diff against.
**DoD**: 1fa48 and 1f9a0 coverage rise from ~0.00 into the cohort range without moving other songs.

**S2 ✅ FIXED IN-TOOL — the main-beat grid was ~18 ms early.** Symptom: on 11 of 13 songs OUR median
offset from the grid *exactly* equalled the HUMAN's (−29.6/−29.6, −52.1/−52.1 …). Two independent
cohorts cannot share a defect, so the grid was wrong. Cause: `onset_detect(backtrack=True)` moves each
onset to the preceding local minimum. Compensated by the human-corpus median (**−18.1 ms, n=13**),
calibrated on humans deliberately — calibrating on our own output would be the `h_dist` circularity.
⚠️The coverage gap is **unchanged** by the fix (ours ~0.49 vs human ~0.70), which is the reassuring
outcome: the headline defect is not a phase artifact.

**S3 🟠 FALLEN KINGDOM IS INVERTED — starved where the music is busy, dense where it is silent.**
210–230 s: **10 and 8** notes per 10 s against the human's **28 and 22**, while the music runs 73 stem
onsets. 240–250 s: **13** notes against the human's **6**, over an outro carrying **1** stem onset.
★Kyle 2026-08-04: the outro **should** be mapped, **sparsely** — so this is a *distribution* bug, not
a trim question. **Task**: check whether `section_gate="loud_only"` is suppressing the final chorus.

### What exists already — extend, do not restart
- `scripts/review_map.py` — ranked timestamped findings (STARVED / MISSED_HIT / OFFBEAT / PHRASE_HOLE /
  MAPPING_SILENCE / ENDING). Reproduced both of his 2026-08-04 ear observations unprompted.
- `scripts/view_song_vs_map.py` — stem lanes + coincidence order + notes by hand.
- `scripts/view_song_strip.py` — whole-song nps / intensity / response / offbeat.
- `scripts/view_ab_diff.py` — only what a lever changed, bucketed by coincidence order.
- `scripts/render_map.py`, `scripts/map_view.py` — lattice PNGs and the text score.

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

### C5 — Doubles: root cause found; 🔴**decode fix ATTEMPTED 2026-08-04 and it FAILED**
🔴**`BEAT_HAND_DEAL` is a measured NEGATIVE — do not revive it.** It hit every structural target
(distinct times 462→656 vs human 646, doubles 0.66→0.10, pulse coverage 0.61→0.83 vs human 0.80,
`role_asymmetry` held) **and rhythm degraded 6× (0.409 → 2.450, resolvable)**, with alignment,
flow, playfeel and precision all resolvably worse. Only handrole improved (1.148 → 0.738).
★**Mechanism**: the deal needs **2× as many distinct slots**, which means going deeper down the
probability ranking — **precision falls 0.919 → 0.893**, i.e. the added slots are off real onsets.
Identical to W2's "the marginal note is much worse than the average note".
⇒🔴**C5 IS NOT REACHABLE BY DECODE.** Our probability field does not contain ~646 slots worth
playing. **This is C1's "better probabilities, not better picking" from a FIFTH direction.** Tuning
the parameter does nothing (rhythm 2.453/2.450/2.451 at deal10/14/20) — the damage is the dealing.
★**Still also the cause of W3.**
★**2026-08-04: W3 ("some parts get really intense to play") resolves to this item.** At Hunger
4:20–4:32, on identical 160 ms grids, the human plays **66 events at 0.015 double share** while we play
**50 at 0.640** — fewer distinct moments but **more notes per second** (6.56 vs 5.36), which is what
makes it exhausting. Fixing C5 should fix W3. ⚠️Any difficulty axis must count **notes, not events**:
measured on events we look *easier* than human while the map plays *harder*.

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
- 🔴**NEVER EDIT `generate.py` (or anything it imports) WHILE A SWEEP IS RUNNING.** `eval_sweep`
  spawns a **fresh `python scripts/generate.py` per map**, so an edit takes effect mid-run and the
  arm silently becomes half one algorithm and half another. It does not crash and it still prints a
  number. Hit 2026-08-04 (the `BEAT_HAND_DEAL` strict→lead-aware fix landed mid-sweep); the deal-arm
  caches had to be deleted and the sweep relaunched. **Either wait, or copy the tree first.**

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
