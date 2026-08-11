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

## ✅ P0 — SEED LOTTERY: CLOSED *(cause + fix in PROGRESS.md)*

`generate.py --seed` / `BSA_SEED`; same seed → byte-identical map. **Three habits that outlived it:**
1. **Score every arm at ≥3 seeds and quote the sd.** ⚠️n=3 *underestimates* sd — treat it as a screen.
2. **`npass` is not a ranking statistic** (an identical config scored 4, 4, 2). Rank per-axis with error bars.
3. **Pairing helps alignment only** — it rides the postprocess `random` stream; the rest ride the torch decode.

**Open**: the spread bar (0.35) sits inside the noise. Recommendation: stop gating on it, keep a hard
alarm near 0.15. Not done unilaterally — it changes scorecard semantics.

## ✅ P1 — PROMOTED 2026-08-03 (Kyle's call, on *Hunger*) *(details in PROGRESS.md)*

Eight defaults flipped in `generate.py` / `postprocess.py`; a bare invocation reproduces the exact map
he played. **Baseline: [`docs/BASELINE_2026-08-03.md`](docs/BASELINE_2026-08-03.md).**
⚠️Carried cost: `BEAT_ONSET_EVIDENCE` **degrades reachability** (repaired by `BEAT_REACH`); its only
non-circular evidence is the rhythm axis. ⚠️Its old "peak nps 6.25→6.50 vs a human 5.5" claim is
**RETRACTED** — that human 5.5 was a cross-population number; per song we peak LOWER (see W3).

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

### 🟠 W3 — "SOME PARTS GET REALLY INTENSE TO PLAY" ⇒ **THIS IS C5** *(evidence in PROGRESS.md)*
Its old evidence ("peak nps 6.5 vs human 5.5") is **RETRACTED** — a cross-population comparison. Per
song we peak **LOWER** (4.00 vs 5.25). What he felt: at Hunger 4:20–4:32 the human plays 66 events at
0.015 doubles, we play 50 at **0.640** — fewer moments, MORE notes/s, far harder to execute.
**Task**: fix C5, then re-check. Do not build a separate intensity lever.
⚠️**LANDMINE: any difficulty axis must count NOTES, not distinct events.** On events we look *easier*
than human while the map plays *harder*; `peak_nps` actively hid this.

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


### ✅ W7 — FIXED, awaiting Kyle's ear (lever default OFF) *(full write-up in PROGRESS.md)*
> *"The final note of the song did not line up together."* — read literally: the two HANDS.

`BEAT_END_RESOLVE=0.75` takes orphaned endings **0.1528 → 0.0139** (human 0.036) at 3 seeds × 24 songs
and **costs nothing** to three decimals; it drops the orphan rather than inventing a partner.
✅`BEAT_TRIM_TAIL` is EXONERATED and there is **no general "we end late" defect** — do not reopen it.
**Only step left: he plays it.** ⚠️Do not promote unilaterally.
⚠️**Sweep-table trap**: the `delta / resolvable?` column compares the **second** arm to the control,
not the last. Difference the columns directly.

### 🛡️ Confirmed positives — protect these
- **Hand-lead alternation**: *"a giant difference maker... noticeably great impact on the flow."*
- **Density pacing**: *"when there is a slow spot we let the player breathe... we no longer have the
  monotony flood of notes."*

Any lever that regresses A6 hand-role or the density-select behaviour trades away something Kyle has
explicitly valued by ear. Check both before promoting anything.

---


## 🔴🔴🔴 P0 — "THE METRICS STILL DON'T CAPTURE THE FULL PICTURE" (Kyle, 2026-08-10)

> *"Keep working with the note that the maps need a lot more refinement. The metrics still don't
> capture the full picture. It may be time for a significantly different approach."*

★**HE IS RIGHT, AND IT IS ALREADY MEASURED IN THIS REPO — see M-F.** Ranking the songset by the mean
gap over the steer-safe axes puts **Fallen Kingdom second-best** (he called it *"really empty"*) and
**Hunger fifth-worst** (he graded it **A+** and told us to promote it). On the only two maps where
we have his verdict, the suite's ordering is close to the **reverse** of his.

⚠️⚠️**THE CONSEQUENCE THIS PROJECT HAS NOT YET DRAWN, and it is the important one.** Our headline
negative results — M-A "nothing moves a masterpiece axis", M-G "v8's gain dies at n=149", C1's six
directions — are all verdicts **measured on a ruler that demonstrably does not track his judgement**.
They are sound statements *about the axes*. They are **not** established statements about map
quality. Every "this lever does nothing" in PROGRESS.md carries that asterisk from today.

★**THE CHEAPEST GENUINELY-DIFFERENT APPROACH IS NOT A BETTER METRIC — IT IS A DIFFERENT SOURCE OF
TRUTH.** We have spent three sessions building instruments to *predict* quality from first
principles, and M-F says the best of them anti-correlates with the only judge that counts. We have
his verdicts on ~3 maps, unstructured, scattered through PROGRESS.md.
**Proposal (needs his buy-in, cheap to run): a structured A/B preference loop.** Pairs of maps on the
same song, he picks the better one and says one sentence why. 20–30 pairs is a few listening
sessions and it would, for the first time, let a lever be scored against **his ear directly** rather
than against a proxy that we have measured to be anti-correlated with it.
**DoD**: an axis (or a weighted combination) that reproduces his ordering on held-out pairs above
chance. Until something clears that bar, no axis in the suite may be called a quality metric — only
a defect detector, which is what they were validated as.
⚠️Do NOT respond to his message by building a seventh metric from first principles. That is the move
that produced the anti-correlation.

### 🔴 THREE MEASURED SENSES IN WHICH THE SUITE MISSES THE PICTURE (2026-08-10/11)
1. **It is anti-correlated in aggregate** (M-F) and **at the coin flip per axis** — 13/26 axes agree
   with his one known verdict (`scripts/preference_screen.py`). ⚠️n=1 pair, and the axes are not
   independent.
2. **The masterpiece suite is nearly blind to PLACEMENT.** M-E rewrote the position and cut direction
   of **25 % of all notes** and **12 of 15 axes moved by exactly +0.0000**. Only `harm_place`,
   `arrange`, `arrange_ami` can see it — and two of those are diagnostic-only.
3. ★**The six-axis suite cannot score a single map AT ALL.** flow/rhythm/idiom/handrole/playfeel are
   COHORT statistics comparing distributions; on one map every axis returns nan (verified). ⇒**The
   project's primary instrument is structurally incapable of the only question Kyle asks — "is THIS
   map good".** It gates a cohort; it cannot rank two maps. Not fixable by reweighting.

### 🟡 A HYPOTHESIS FOR "REALLY EMPTY" THAT SURVIVES ITS OWN CAVEAT — ask, don't build
Per-map metrics DO survive on one map. On the two he judged: nps **4.88 vs 3.21**, travel **5.94 vs
3.25**, peak_nps **9.5 vs 5.5** — he liked the denser, busier one. ⚠️Travel is mostly density wearing
a hat (1.22 vs 1.01 per note), so this is near the density story W2 already refuted as a target.
★**But the reconciliation is testable and it matters**: relative to each song's OWN human, Fallen
Kingdom is the *denser* of the two (0.781 vs 0.650) and it is the one he called empty ⇒ **he may be
judging ABSOLUTE activity, not activity relative to that song's human.** Nearly every instrument here
normalises by the human map, which would be normalising away the thing he reacts to.
⚠️**ASK HIM** — four "empty" instruments have already failed; do not build a fifth. The question:
*"is Fallen Kingdom empty compared to what our model used to do, or compared to what the song wants?"*

**The second candidate for "significantly different"** is the V8 representation direction
(`docs/architecture_v8_plan.md`, `beatsaber_v8_representation_theory` memory): symbolic per-stem
transcription as the timing backbone rather than MERT-on-a-BPM-grid. ⚠️Note it was **shelved on
evidence from these same axes** (v8's `follow_vocals` gain died at n=149) — which is exactly the kind
of verdict the paragraph above says we should not treat as settled. Worth re-opening **after** the
preference loop exists to judge it, not before.

## 🔴🔴 P0 — THE EVAL SUITE (Kyle, 2026-08-04) — ✅ BUILT, now the working instrument

> *"Create a way for you to see the song and map in a way that gives you my vision… I want to empower
> you."*

**📖 How to use it: [`docs/EVAL_SUITE.md`](docs/EVAL_SUITE.md).** Start with
`python scripts/suite_report.py --song X`.

**Status: P0.1–P0.5 all done.** `main_beat.py` (which pulse the song is on) · `view_main_beat.py` (the
picture, incl. a Stage-1 probability lane) · `review_map.py` (ranked timestamps) · `suite_report.py`
(one command) · `view_ab_diff.py` · `view_song_strip.py` · `audit_phase_metrics.py`. All 24 songs
render clean.

### ★ HIS DEFECT, MEASURED (2026-08-04)
> *"Every couple main beat notes were mapped instead of most of the main beats."*

`main_continuity` (P(play beat n+1 │ played n)) **ours 0.523 vs human 0.697** *is* his sentence; main
beats covered 0.546 vs 0.704. ⚠️But we do **not** play more filler than a human (notes-on-main 0.637
vs 0.617) — the second half of his guess is wrong.

### Answers he gave 2026-08-04 (do not re-ask)
- The soft outro **should** be mapped, **sparsely** (Fallen Kingdom).
- **"Empty" is relative to THE SONG, not our old maps** ⇒ the human corpus is not the reference for
  that defect; the song's own main beat is.

### 🔑 Design constraints that shaped it (keep)
1. **PNG is the primary artifact** — an agent can only see an image by rendering and re-reading it.
   HTML is for Kyle.
2. **Tolerance must scale with the period** or `capture` = 1.0 by construction picks a 16th grid.
3. **Calibrate against humans, never our own output** (the grid's 18 ms detector bias).
4. **Metrics that reward regularity are metronome-gameable** — diagnose, never steer.
5. ★**A property of the probability field is not a property of the map**, *and* **a map-level
   comparison between models with different note counts is not a comparison of the models.**
   Both halves cost a wrong conclusion today.

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

**S2 ✅ FIXED IN-TOOL — the main-beat grid was ~18 ms early** (`onset_detect(backtrack=True)` moves
each onset to the preceding local minimum; compensated by the human-corpus median −18.1 ms).
★The symptom that exposed it: on 11 of 13 songs OUR median offset *exactly* equalled the HUMAN's —
**two independent cohorts cannot share a defect, so the ruler was wrong.** ⚠️The coverage gap is
unchanged by the fix, which is the reassuring outcome: the headline defect is not a phase artifact.

**S6 ★★ S1 AND S3 ARE THE SAME DEFECT AT DIFFERENT SCALES — Stage-1's probability drifts off the beat.**
Bucketing 352 windows × 24 songs by main-beat coverage, against the cached probability dumps:

| bucket | coverage | **p@mainbeat** | p@window | notes/12 s |
|---|---|---|---|---|
| worst | 0.109 | **0.328** | 0.362 | **30.5** |
| mid | 0.535 | 0.714 | 0.356 | 29.7 |
| best | 0.807 | 0.717 | 0.463 | 42.1 |

★**The worst windows are not note-starved — they are BEAT-starved.** We place a normal number of notes
there (30.5 vs 29.7) and the window's overall probability is normal (0.362 vs 0.356), but the
probability **at the main beats is less than half** (0.328 vs 0.714). The model is confidently active
and pointing somewhere other than the beat.
⇒**This is S1 (probability one slot out of phase, whole-song, 2 songs) happening LOCALLY, in ~15 % of
windows on every song.** One phenomenon, two scales.
⚠️**Correct the name**: `review_map`'s `STARVED` compares our note count to the human's, which is a
different thing from these coverage buckets. Do not conflate them.
⇒**It also explains the ceiling on `BEAT_MAIN_BEAT_BONUS`**: a ×1.25 boost on 0.328 gives 0.41, still
far below the ~0.7 competitors elsewhere in the window. **A multiplicative prior cannot win a race it
starts at half distance.**
**❌ The adaptive bonus was BUILT AND LOST** (`BEAT_MAIN_BEAT_LIFT`, 3 of 3 songs, per note added) —
`max(p, α·p90)` flattens the main-beat profile and destroys the ranking *among* main beats. Selection
is a per-window RANKING, not a per-slot threshold; order beats level. Documented in PROGRESS.md.

**❌ AND THE INVERSION HAS NO SIMPLE PREDICTOR — three candidates tested, all null or weak:**

| predictor | worst | mid | best |
|---|---|---|---|
| bass % / drums % of onsets | 0.276 / 0.323 | 0.273 / 0.306 | 0.273 / 0.333 |
| **drums landing on the main beat** | **0.828** | 0.856 | 0.850 |
| onsets per second | 14.37 | 14.77 | 13.70 |
| position in song | 0.507 | 0.467 | 0.423 |
| beat-to-slot offset | 0.168 | 0.144 | 0.142 |
| slots per main beat | **2.00** | 2.09 | 2.19 |

★**The drums land on the main beat 83 % of the time in the very windows where the model peaks OFF
it.** The audio in our worst windows is indistinguishable from our best by stem mix, stem-to-beat
alignment or onset density. ⇒**the inversion is not a preprocessing artifact and not a property of the
music — it is internal to Stage-1**, which is the strongest argument yet that this needs the retrain
rather than another decode lever.
⚠️One weak signal worth a follow-up: worst windows average **2.00 slots per main beat** vs 2.19 —
i.e. they concentrate on songs whose main beat sits at the eighth level, where "off by one slot" *is*
the half-beat error. Not strong enough to act on alone.

**S3 🟠 FALLEN KINGDOM IS INVERTED — starved where the music is busy, dense where it is silent.**
210–230 s: **10 and 8** notes per 10 s against the human's **28 and 22**, while the music runs 73 stem
onsets. 240–250 s: **13** notes against the human's **6**, over an outro carrying **1** stem onset.
★Kyle 2026-08-04: the outro **should** be mapped, **sparsely** — so this is a *distribution* bug, not
a trim question. **Task**: check whether `section_gate="loud_only"` is suppressing the final chorus.

📦**FOR KYLE, 2026-08-05**: `outputs/kyle_review_2026-08-05_structure/` — four structure PNGs + a
README. **Nothing to play**; it is the night's finding as a picture, ~10 seconds each. ★Hunger is the
one to open: **our** self-similarity panel is a uniform bright blob while the **human's** is sharp
discrete squares ⇒ we do not repeat too little, **we repeat too uniformly** — a metric scoring "how
repetitive is the map" would rank us ABOVE the human there. ⚠️`outputs/` is gitignored (C6): the PNGs
are not in version control.

## 🔴 P0b — THE MASTERPIECE AXES (M1–M4, built 2026-08-04 night) — ✅ BUILT, 7 axes cleared to steer

> *"We created a model to create a playable map but now need a model to start producing masterpieces
> which we are far off from… syncing to rhythm more and making significantly more intelligent and
> intentional placements of notes."* — Kyle

**📖 How to use them: [`docs/EVAL_SUITE.md`](docs/EVAL_SUITE.md).** One command:
`python scripts/masterpiece_report.py --arm X [--vs Y]`. The picture:
`python scripts/view_structure.py --song 1f8d6`. Validity: `python scripts/audit_masterpiece.py`.

**Where we stand (paired, 13 songs with a strict Expert human map):**

| metric | ours | human | resolvable |
|---|---|---|---|
| `follow_vocals` — do we play the vocal line's figure | +0.020 | +0.149 | **yes (7×)** |
| `follow_mean` — do we play *this bar's* figure at all | +0.033 | +0.107 | **yes (3×)** |
| `rhy_rhythm` — when the groove repeats, does the map | +0.060 | +0.148 | **yes** |
| `hands_x_downbeat` — is the double used to mark the downbeat | +0.036 | +0.182 | **yes** |
| `lead_persistence` — do we stay with one instrument | 0.292 | 0.387 | **yes** |

★**These are the first steer-safe axes in this area**, because they score a CONTRAST rather than a
level: a metronome, random note times, a bar-rotated map and another song's map all score ~0 by
construction. Which ones may steer is decided by `audit_masterpiece.py`, not by argument.

⚠️**`hands_x_downbeat` has a seed sd of 0.066 (≈ its own value)** — quote it as a cohort statement,
never to rank two arms. `follow_mean`'s sd is **0.0006**, so that one ranks arms cleanly.

---

### 🔴🔴 M-A — **NOTHING WE HAVE MOVES ANY OF THESE** (updated 2026-08-05, at n=149)
mbb015 / mbb025 / endres / trimco3 all sit within **±0.008** of baseline on every M axis, against an
ours-vs-human gap of 0.082 on `follow_mean`.
🔴**And the instrument model no longer counts as an exception.** v8's `follow_vocals` gain
(+0.0082 on 13 songs) is **+0.0004 at n=148** on the same songs/seed — **RETRACTED**; its
`rhy_rhythm` loss shrank the same way. Second demonstration in one night that n=13 inflates effect
sizes.
⇒★**This extends C1**: not only will better *picking* not close the structural gap, a better
*probability field* does not either — neither changes the fact that every slot is decided on its own.
**Do not queue another decode-lever or checkpoint sweep against these axes without a mechanism
argument for why it would behave differently.**
✅v8 does two real things at n=149: `double_share` **−0.041** (resolvable, right direction for C5) and
`follow_drums` +0.010 (diagnostic-only axis). Neither is a masterpiece axis.
**`follow_vocals` is still the right acceptance metric for Track B** — it is the axis that would move
if the model learned to follow the vocal line — but **we have no arm that moves it.**

### 🟡 M-B — LEVER CANDIDATE: mark the downbeat ⚠️**DEMOTED 2026-08-05**
We spend the double — the loudest thing a map can say — on **0.667** of all events (human 0.196), so
it marks nothing, and our downbeat emphasis is 0.036 against a human 0.182. A lever that biases
double placement toward the bar's first slot is the most concrete "intentional placement" change the
suite can currently justify.
🔴**DEMOTED by the human bar (n=149)**: `hands_x_downbeat` exceedance is **9.3 %** — the rate a
cohort drawn from the *same* population produces. The human spread is enormous (p10 −0.36 to p90
+0.87, MAD 0.38): mappers disagree wildly about marking the downbeat, so being below your song's
human is unremarkable. It is a **shift inside the normal human range, not a tail defect**, and the
resolvable paired delta alone overstated it. ⇒Build M-E first; revisit this only if his ear asks for it.
**DoD (if revived)**: `hands_x_downbeat` rises toward the human at **≥5 seeds** (the axis is
seed-noisy, sd 0.066) or on the wide cohort, with `double_share` NOT rising and the six-axis suite unmoved. ⚠️Check `rhythm` (A2) and
`pulse_stability` alongside — moving emphasis onto the metrical grid is exactly the shape of change a
metronome would also make.

### ✅ M-C — REPLICATED AT n=149 (done; outcome in PROGRESS.md)
`build_wide_cohort.py` built an independent 149-song paired cohort. **Every steer-safe axis resolves
and nothing evaporated.** Use `masterpiece_report.py --wide` for any future claim; the eval songset
stays the fixed historical ruler. Human bar: `docs/eval_references/masterpiece_human.json`.

### 🔵 M-E — BUILT 2026-08-10, THREE ARMS RUNNING *(build write-up in PROGRESS.md)*
`BEAT_STRUCTURE_REUSE=<mode>[:min_sim[:min_lag[:energy_tol[:min_z]]]]`, default OFF.
`scripts/overnight_2026-08-10.sh` — `me_z20` / `me_z25` (place: position+direction only) and
`me_full25` (also the bar rhythm), paired against the 149-song prod cohort.

**✅ STEP 1 IS ALREADY ANSWERED — the mechanism engages and survives postprocess.**
`scripts/check_reuse_survives.py --arm outputs/wide_cohort_prod_me_z20`, paired over 103 songs:
placement agreement on audio-repeat bar pairs **0.0206 → 0.0633 (3.1x, resolvable, 80/103 songs
improved)**. So if `harm_place` does NOT move tonight, the lever is not the reason — look at the axis.
⚠️But the level is still **0.063, not ~1.0**: most of each copy is rewritten downstream (`fix_parity`
alone rewrites ~48% of directions). **The headroom question — how much does postprocess eat — is the
first follow-up if the arms look promising**; `BS_PREPOST_OUT` dumps the pre-postprocess map for a
direct answer on one song.
🔴**AND THE FIRST VERSION OF THAT CHECK WAS THE WRONG INSTRUMENT** — it asked whether the two bars
were IDENTICAL and read a null (0.0000 vs 0.0006), which `place` mode cannot reach by construction
since it never changes which slots play. The project's signature failure, caught in-flight this time.

## 🔴🔴🔴 P0 — BEFORE ANY M-E PROMOTION: ADD A PERIODIC-REPEAT DEGENERATE TO THE BATTERY
`view_structure.py` on アリスブルー (~71 % of bars copied) shows **our AFTER panel is a rigid periodic
CHECKERBOARD** while the music's and the human's are irregular ⇒ at high dose the lever produces
structural **regularity, not form**. The M-axes did not flag it: they are contrasts and are
degenerate-proof **against the degenerates the battery contains** (metronome, random times,
bar-rotated, other-song) — and **a periodically self-repeating map is not one of them.** Musical
repeats are often periodic themselves (8/16-bar phrases), so a fixed-lag copier can score well on
*"does the map repeat where the music repeats"* without following the music.
**TASK**: in `audit_masterpiece.py`, add a control that copies bar *i* from bar *i−k* at fixed *k*,
audio ignored, and score it.
**DoD**: if it scores near or above our arm on `rhy_rhythm` / `harm_rhythm` / `harm_place`, **those
axes are not steer-safe for this class of lever** and the 2026-08-11 headline must be re-read as
partly degenerate. Cheap, decisive, and it comes **before** anything is promoted.
⚠️Dose is the control: at ~14 % (Hunger, Fallen Kingdom) there is no checkerboard, so `min_sim` /
`min_run` are the knobs and the low-dose setting may be the shippable one.

## 📦 AWAITING KYLE'S EAR — `outputs/kyle_review_2026-08-11/` (installed, ready to play)
**8 maps deployed** as `AUTO <song> [BEFORE]` / `[AFTER]` (restart Beat Saber or SongCore-refresh).
README: [`docs/review_2026-08-11.md`](docs/review_2026-08-11.md) — leads with what to distrust.
★**Tell him: play Digital Life Hacker and アリスブルー.** They are ~71 % copied (65 % of note times
changed); Hunger and Fallen Kingdom are only ~14 % copied and will sound nearly identical.
★**The question to ask is not "is it better"** but **"does the repetition read as INTENTIONAL or as
LAZY?"** We copy a chorus and do not vary it; a mapper copies and then varies. If it reads lazy, the
next problem is variation-on-repeat — different, and easier than what was just solved.
⚠️Still unanswered from 2026-08-04: *is Fallen Kingdom empty vs what our model used to do, or vs what
the song wants?*

★★★**M-A IS RETRACTED: SOMETHING FINALLY MOVES A MASTERPIECE AXIS.** `me_full25` (copy the bar's
RHYTHM on a musical repeat) — `rhy_rhythm` **+0.0175**, `harm_rhythm` **+0.0280** against a ±0.004
seed floor (4–7×), `harm_place` **+0.0091** = **11×** what copying placement alone bought. M-A's
*"nothing we have moves any of these"* (mbb/endres/trimco3/v8 at n=149) **no longer holds.**
⚠️It pays the pre-registered price — `follow_*` −0.003…−0.005 and `lead_persistence` −0.0146, all
resolvable: a copied bar stops following THIS bar's music — and an unacceptable one: **idiom
0.40 → 2.34**, playfeel spread collapsed. ⇒**The question is no longer "can copying close the
structural gap" (yes) but "can we afford it".** ⇒Round 2's **`diag_full`** is the arm that matters:
does contiguity keep the structural gain while cutting the idiom bill?
⚠️I called full mode "dead" off the six-axis table 20 min before its M-axis report landed. **Wrong
call, corrected in PROGRESS.md** — do not read a structural lever's verdict off the playability
table alone.

🔴🔴**ROUND 1's PLACE ARMS (`me_z20`/`me_z25`) ARE A CLEAN NEGATIVE — outcome in PROGRESS.md.**
flow **0.37 → 0.75** and idiom **0.40 → 1.07**, both across their bars, for a `harm_place` gain of
**+0.0008 against a 0.0200 gap**. Every rhythm-side axis identical to 4 dp (time-neutrality held
exactly). Cause measured, not guessed: only **15.6 %** of copied bars continued the previous bar's
copy ⇒ a shuffle, and placement is not context-free. **Tie-breaking, not absent sections** — C1 one
level up. ⇒`plan_reuse_diagonal` decodes whole stripes: copy share 0.297→0.428, contiguity
0.156→**0.648**. **Round 2 is queued and self-sequencing** (`scripts/overnight_2026-08-11.sh`,
arms `diag_r4` / `diag_r6`), and it carries the kill criterion: if flow/idiom break the same way,
copying placement across contexts is the defect, place mode is DONE, **do not tune it a third time.**

⚠️**HARVEST NOTE**: the round-2 script calls `check_reuse_survives.py` WITHOUT `--diag` (the flag was
added after it launched, and the script was not edited mid-run — bash reads a script incrementally
and editing a running one corrupts it). **Re-run those two by hand with
`--diag --min-sim 0.70 --min-run 4`**; the logged numbers score a diagonal arm against per-bar pairs
and will understate it for a reason that has nothing to do with the lever.

**Harvest next session** — read `logs/overnight/me_2026-08-10.log` and `me2_2026-08-11.log`, then:
2. 🔴**`harm_place` is a MANIPULATION CHECK here, not a win.** The lever copies placement on musical
   repeats and the axis scores placement reuse on musical repeats. It answers "did the lever fire".
3. **The place arms cannot have moved alignment / rhythm(A2) / nps / precision** — no note moves in
   time (verified end-to-end, 5/5 songs). If one of those DID move, time-neutrality is broken and
   every other number in the run is void. **Check that first, not last.**
4. Guards that decide PASS vs PIVOT: six-axis suite, `hard_rate` (reachability), `follow_*`.
5. PASS ⇒ build the review set on his standing four (1f767 / 1f913 / 1f333 / 1f8d6) with the
   structure PNG beside it. **His ear decides**, per M-F.

### ✅ M-G — ANSWERED: NO. v8's vocal gain does not hold at n=149 *(outcome in PROGRESS.md)*
Paired over 148 songs, same audio and seed: `follow_vocals` **+0.0004** (songset said +0.0082),
not resolvable. Folded into M-A.

### ⚠️ M-F — THE AXES DO NOT PREDICT KYLE'S VERDICTS (yet). Do not treat them as the judge.
Ranking the songset by the mean gap over the steer-safe axes puts **Fallen Kingdom second-best
(+0.019)** — the map he called *"really empty"* — and **Hunger fifth-worst (+0.169)** — the map he
graded **A+**. So a structural gap is real and measurable and is **not** what he is reacting to.
⇒Use these axes to find and fix a defect class, never as evidence that a map is good. The success
criterion is unchanged: *he plays it and wants to keep playing*.

### 🟡 M-D — TWO INSTRUMENTS ARE TOO BLUNT TO USE
- **M4 `arrange`** fails its own control (a bar-rotated map scores 0.67× the human) ⇒ **not yet
  measurable**. Needs a section-conditioned descriptor rather than a boundary jump.
- **`harm_place`** — a 30 %-thinned human map scores 1.34× the human. Both cohorts read ~0.01, which
  is *not yet measurable*, **not** "humans don't reuse placement". Needs a pattern-level similarity
  (n-grams of position/direction) instead of the per-slot L1 agreement.

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
