# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md) — including the full session-by-session archive from 2026-06 to
2026-08-02. Evaluation-suite design rationale is in [`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule for keeping it that way:** when an item finishes, move the *outcome and what it taught* into
PROGRESS.md and delete it from here. A completed item is history, not work. Curated 2026-08-02, when
this file had reached 4,076 lines.

---

## 📍 CURRENT STATE (2026-08-13)

### ✅✅ `BEAT_GRID_PHASE=search` PASSED AT n=149 — **awaiting Kyle's ear, default OFF**
| | control | **search** | human |
|---|---|---|---|
| ★songs >0.10 below human | **39** | **21** | — |
| median precision | 0.8879 | 0.9158 | 0.9335 |
| songs moved >0.02 | — | **74 better, 0 worse** | — |
| alignment axis | 0.62 🔴FAIL | **0.35 ✅PASS** | — |

**The alignment axis has never passed before.** ★**Why it is not promoted**: the search optimises
against the same onsets the axis scores, so the gain is partly circular; the non-circular signals
(zero regressions, scatter 10.4 → 9.7, we approach but never exceed the human) are reassuring, not
conclusive. **Per the standing rule, the "why not" is: this needs his ear before a default flips.**

🔴🔴**AND THE STANDING SONGSET CANNOT TEST IT.** Built all four first, as always: the lever **declined
to touch every one of them** (best gains +0.0099 / +0.0121 / +0.0166 / +0.0081, all under `MIN_GAIN`)
and the maps came out **byte-identical** — same `.dat` SHA. **Removed them rather than spend his ear
on a pair that cannot differ.** ⇒**The defect this lever fixes does not occur on the songs he reviews**,
so the only way to hear it is on unfamiliar corpus songs. That trade is his to accept.
📦**Deployed instead**: `outputs/kyle_review_2026-08-14/` + [`docs/review_2026-08-14.md`](docs/review_2026-08-14.md)
— 6 real songs × `[BEFORE]`/`[PHASE]`, led by **BEcause** (Dreamcatcher), the largest single fix in
the cohort (+80 ms, 0.456 → 0.900). ★**The question is narrow**: does `[PHASE]` sit on the beat
better? **"Can't tell" is a real answer** — it would mean a 0.62 → 0.35 axis move is inaudible, which
is worth knowing about the axis.
⚠️**This is a new instance of the standing-songset limitation**: the 4 songs are a fixed, *well-behaved*
sample, so a lever targeting a defect they do not have is invisible there. Check "does the lever
change these maps at all" **before** building a review set on them.

⚠️**Two things this arm taught that outlive it:**
1. 🔴**`handrole` is NOT translation-invariant** — it bins on `int(n.beat // WINDOW_BEATS)`, anchored
   at beat 0, so a pure time shift moves `role_swap_rate` by up to **0.160**. It **cannot cleanly
   evaluate any lever that moves notes in time**, and its "regression" here is an instrument artifact
   (note-count delta was 0 on all 149 songs). Same class of bug as the generator's t=0 grid anchor.
2. ★**A gate whose threshold sits at its own noise floor is not a gate.** The detector check flagged
   59.5 % and a permutation null put that at chance (37.8 %, p=0.324). **Null every gate before
   trusting the alarm** — the failure it reports may be the instrument.

### 📜 SUPERSEDED — `BEAT_GRID_PHASE=1` (fitted phase) is REFUTED, do not revive
Subset 39 → 37 and the alignment gap **doubled** 0.62 → 1.32, because corr(applied, wanted) fell from
+0.367 offline to **+0.065** in production: the offline test fitted from CACHED onsets while
`generate.py` fits from Demucs stems. ★**A pre-build test run on a different input than production is
not a pre-build test.** *(Full account in PROGRESS.md.)*

⚠️**The residual is now the honest open question**: 21 songs still sit >0.10 below human, and the
original diagnostic says ~15 of them recover from **no shift at all** ⇒ that remainder is a
**SELECTION** defect (which slots we pick), not a grid defect. That is where the next alignment work
goes, and it is C1 territory.

### ✅ THE CROSSOVER GUARD IS BUILT (the P0 TASK below is done)
`flow.crossover_guard` + `scripts/calibrate_crossover.py`. Human band, n=200 strict Expert:
median **0.187**, p10 0.105, p90 0.275. Baseline **FAILS** (0.0000, 149/149 zeros); `xsep` **PASSES**
(0.1119). Reported unconditionally, gates `passed` only under `CROSSOVER_GUARD=1`.
★**Flip that default to ON when `COLOR_SEP_MODE=extreme` ships** — it is left off only because gating
would flip the *promoted* baseline to FAIL, which is Kyle's call, not a side effect of adding a metric.

**Two independent candidates are installed and waiting on Kyle's ear.** Neither is promoted; both
default OFF. `outputs/kyle_review_2026-08-11/` + [`docs/review_2026-08-11.md`](docs/review_2026-08-11.md).

| candidate | what it does | evidence |
|---|---|---|
| **`COLOR_SEP_MODE=extreme`** (`[CROSSOVER]`) | lets the map cross hands over, as every human map does | crossover 0.000 → 0.112 (human 0.183, **0/149 over human p90**); **flow 0.37 → 0.23**; `reach_p90` and `hard_rate` land **on** human values; M-axes unchanged; note count unchanged |
| **`BEAT_STRUCTURE_REUSE=diag_full:0.70:4:1.5:2.0:4:0.20`** (`[AFTER CAPPED]`) | reuses the map already written for a repeated passage | ~45–51 % of the `rhy_rhythm`/`harm_rhythm` gap **replicated at 2 seeds**; degenerate-controlled; dose-capped so variety stays human |

★**The question for him is not "is it better"** but, for the structure lever, *does the repetition read
as INTENTIONAL or LAZY?* — and for crossover, simply whether the maps play better.

**★ Three standing methodology rules:**
1. **Never calibrate the human corpus through `scorecard._load_any`** — it prefers ExpertPlus. Use
   `calibrate_playfeel.load_expert_only`.
2. **Ask "norm or aspiration?" before calibrating any axis.** His target is the **best** mappers, so
   "the human cohort passes it" is not a validity check for aspirational axes.
3. ★**When a lever passes its DoD, either flip the default or write down why not.** This project
   generated two validated levers in 2026-07 and shipped neither; re-deriving one of them was the
   single most valuable hour of 2026-08-11.

**★ And the three lessons that cost the most:**
- *A lever can pass every axis and still carry a defect no axis measures* (`BEAT_ONSET_EVIDENCE` vs
  reachability). The **sensitivity battery** (`audit_sensitivity.py`) now exists to find those.
- *A cohort mean cannot see a subset-of-songs defect.* Walked into twice on 2026-08-11 alone, on two
  different instruments. **Read the per-song minimum.**
- *The cohort `gap` moves with a distribution's spread; **paired per-song deltas** are the sensitive
  instrument.* Reading the gap alone nearly shipped a false "M-E improves alignment" claim.

⚠️**The suite's own limits, measured** (see the P0 below): it agrees with his one recorded verdict at
a coin flip, is nearly blind to placement, and its six axes **cannot score a single map at all**.

---

## 📍 PRIOR STATE — the promotion everything is measured against
**2026-08-03**: Kyle graded *Hunger* **A+** and said ship it; eight defaults were flipped and a bare
invocation reproduces the exact map he played. 📖**Full detail:
[`docs/BASELINE_2026-08-03.md`](docs/BASELINE_2026-08-03.md) — read before changing anything.**
**Song names**: `1f333`=**Hunger** · `1f8d6`=**Fallen Kingdom** · `1f913`=**Digital Life Hacker** ·
`1f767`=**アリスブルー** · plus **SO TIRED ROCK** (his motivation song).
⚠️Carried cost of the promotion: `BEAT_ONSET_EVIDENCE` **degrades reachability** (repaired by
`BEAT_REACH`); its only non-circular evidence is the rhythm axis. Its old *"peak nps 6.25→6.50 vs a
human 5.5"* claim is **RETRACTED** — cross-population; per song we peak LOWER.

## ✅ SEED LOTTERY — CLOSED. Three habits that outlived it
1. **Score every arm at ≥3 seeds and quote the sd.** ⚠️n=3 *underestimates* sd — treat it as a screen.
2. **`npass` is not a ranking statistic** (an identical config scored 4, 4, 2). Rank per-axis with error bars.
3. **Pairing helps alignment only** — it rides the postprocess `random` stream; the rest ride the torch decode.
**Open**: the spread bar (0.35) sits inside the noise — stop gating on it, keep a hard alarm near 0.15.
Not done unilaterally; it changes scorecard semantics.

## 🎯 W1–W7 — KYLE'S OBJECTIONS FROM HIS PLAY-THROUGH (2026-08-03) *(evidence in PROGRESS.md)*

★ **His standing instruction**: *"I'm hesitant to change much because we have a great foundation so we
really need to tread carefully, make isolated and tactical changes, and document like crazy."*
One lever at a time, ≥3 seeds, nothing promoted without his ear.
⚠️He declined to name exemplary mappers *yet* — *"we aren't close to exemplary"* — so the best-mapper
cohort stays blocked **by his choice**, not oversight.

| # | his complaint | status |
|---|---|---|
| **W1** | can't find the core tempo-carrying instrument (*"the core aha tempo/instrument that a mapper obviously adheres to"*) | 🔴**OPEN.** Coincidence hypothesis CONFIRMED (humans map a 4-stem collision 84.5%); the real defect is **we play the OFFBEAT** (`halfbeat_rate` 0.245 vs human 0.095). ⚠️Grid phase cannot fix it — a selection defect. Track B indicated. |
| **W2** | Fallen Kingdom *"feels really empty"* | 🔴**CAUSE UNIDENTIFIED.** Five instruments have now failed to explain it. ⚠️**ASK HIM**: empty vs what our model used to do, or vs what the song wants? New 2026-08-11 hypothesis: he may judge **absolute** activity while our instruments normalise by the human map. |
| **W3** | *"some parts get really intense to play"* | **PARTLY CONFIRMED** — it is C5 wearing a hat: fewer distinct moments, more notes/s. ⚠️Any difficulty axis must count **NOTES**, not events. |
| **W4** | phrases abandoned mid-vocal | ✅**CONFIRMED** — sung phrases with a >1 s hole: ours 0.539 vs human 0.250. |
| **W5** | dot blocks used decoratively | ⏸️**he deferred this himself.** |
| **W6** | multi-note swings missing | 🟡**missing capability** — right answer for grand low-density drops. |
| **W7** | last note *"did not line up together"* | ✅**FIXED** — `BEAT_END_RESOLVE=0.75`, orphaned ending 0.153→0.014 at no cost. **Default OFF, awaiting his ear.** |

⚠️**Protect these — he named them by ear**: A6 hand-role division, and the density pacing
(*"when there is a slow spot we let the player breathe"*). Any lever regressing either trades away
something he explicitly valued.

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

## ✅ THE EVAL SUITE — BUILT; 📖 how to use it: [`docs/EVAL_SUITE.md`](docs/EVAL_SUITE.md)
Kyle: *"Create a way for you to see the song and map in a way that gives you my vision… I want to
empower you."* One command: **`scripts/suite_report.py --song X`** (or `--all`).
`view_main_beat.py` = the picture · `review_map.py` = ranked timestamped findings ·
`view_structure.py` = whole-song structure · `audit_sensitivity.py` = what the suite cannot see.
🔑**PNG is the primary artifact — an agent can only look at an image by rendering it.**

★**HIS DEFECT, MEASURED**: main beats covered ours 0.546 vs human 0.704 (`main_continuity` 0.523 vs
0.697). ⚠️The 2nd half of his guess is **WRONG** — notes-on-main share is identical; we do NOT play
more filler. ✅**`BEAT_MAIN_BEAT_BONUS`** (default OFF, `mbb015` the conservative pick) takes alignment
0.260→0.087 with nothing regressing; it does NOT rescue starved windows. *(Full numbers in PROGRESS.)*

⚠️**READ "MAIN BEAT" CORRECTLY (2026-08-14)**: `find_main_beat` picks the finest level on 139/149
songs, and that pulse is **2× the mapper's declared bpm on 104 of 144** ⇒ the quoted numbers are
usually **eighth-note** coverage, not the mapper's beat. ✅The defect survives the check and gets
**bigger** where the grid does match the mapper's beat: **ours 0.549 vs human 0.901 (+0.345)** on
those 34 songs — the starker and more honest statement of his complaint.
⚠️**Clean NULL worth keeping**: the bpm label barely predicts coverage (`same` 0.542 vs `half` 0.500),
so the half-tempo octave problem is **not** what drives this. Two metrical-looking problems, independent.

### ✅ RESOLVED 2026-08-14 — the main-beat defect is SONG-DRIVEN; "internal to Stage-1" is RETRACTED
Stage-1's probability **inverts metrical phase** (best windows 0.301/**0.725**/0.287 vs worst
0.590/**0.320**/0.577; the human covers the main beat 0.653 and the offbeat 0.104 in those same
windows ⇒ grid right, model wrong). That part **stands**. What is retracted is the inference from
*"no predictor found"* to *"internal to Stage-1, not the audio"*.
**Seed test, n=149**: `main_covered` corr(s0,s1) **+0.9811**, median |Δ| 0.0101 against a between-song
sd of **0.1791**, and **27 of the worst 30 songs are the same at both seeds** (chance 6.0).
⇒**The same songs fail every time.** It is a property of the SONG; we have not found the feature.
★★**STANDING RULE (this inference has now been caught THREE times — alignment, grid phase, and this):
"no predictor among the features I checked" is NEVER evidence of "not driven by the audio."** The seed
test separates them and costs one paired lookup whenever a second seed cohort exists.
**Still open**: *which* audio property predicts the failing songs. 35 reproducibly-failing songs is a
named, stable target.

## ✅ THE MASTERPIECE AXES — BUILT, 7 cleared to steer *(construction in PROGRESS.md)*
`python scripts/masterpiece_report.py --arm X [--vs Y] [--wide]`. Human bar:
`docs/eval_references/masterpiece_human.json`. Validity: `audit_masterpiece.py`.
★They score a **CONTRAST, not a level**, which is why they are the first steer-safe axes here — a
metronome is identical everywhere so it scores ~0 by construction.

**Live items:**
- 🔴**M-A — nothing but structure-reuse moves them.** mbb/endres/trimco3/v8 all within ±0.008.
  ⇒**RETRACTED 2026-08-11**: `diag_full` moves `rhy_rhythm` +0.042 / `harm_rhythm` +0.054 at 2 seeds.
- 🟡**M-B — mark the downbeat: DEMOTED.** `hands_x_downbeat` exceedance is 9.3 % = the same-population
  rate; human spread is enormous (MAD 0.38). A shift inside the normal human range, not a tail defect.
- 🟡**M-D — two instruments too blunt to use**: M4 `arrange` fails its own control; `harm_place` v1
  paid you for deleting notes (rebuilt as a Jaccard over swing transitions).
- ⚠️⚠️**M-F — THE AXES DO NOT PREDICT KYLE'S VERDICTS.** They rank Fallen Kingdom (*"really empty"*)
  2nd best and Hunger (**A+**) 5th worst. **Use them to fix a defect class, NEVER as evidence a map is
  good.** The success criterion is unchanged: *he plays it and wants to keep playing.*
⚠️`hands_x_downbeat` seed sd = 0.066 ≈ its own value ⇒ cohort statement only. `follow_mean` sd
0.0006 ⇒ the most seed-stable instrument here.
⚠️`follow_mean`/`follow_best`/`follow_drums` correlate 0.65–0.84 ⇒ **ONE measurement**; report one
rhythm-fidelity finding, not three. Run `audit_axis_redundancy.py` whenever an axis is added.

## 📦📦 AWAITING KYLE'S EAR — **TWO review sets installed, 32 maps.** Nothing is promoted.

### Set A — structure reuse + crossover (2026-08-11), his 4 standing songs
📖[`docs/review_2026-08-11.md`](docs/review_2026-08-11.md) · `AUTO <song> [BEFORE] / [CROSSOVER] /
[AFTER CAPPED] / [BOTH] / [AFTER]`
★**Play `[BEFORE]` vs `[BOTH]`.** Candidate: `BEAT_STRUCTURE_REUSE=diag_full:0.70:4:1.5:2.0:4:0.20`
(default OFF) — replicated at 2 seeds, degenerate-controlled, dose-capped.
★**Dose is ONE DIAL**: gain, playability damage and loss of variety scale together smoothly, so where
to sit on it is **his taste call, not the suite's**. ⚠️`[AFTER]` is the uncapped version, kept only so
he can hear over-repetition if curious.
**① Does the repetition read INTENTIONAL or LAZY?** *Intentional* ⇒ raise the cap toward human parity
(share ~0.25–0.30). *Lazy* ⇒ next capability is **variation-on-repeat** (copy, then vary) — a
different and easier problem. *Can't tell* ⇒ leave OFF; not worth its flow/idiom cost.
**② Do the crossovers play better?** A straight quality question.

### Set B — grid phase (2026-08-14), 6 corpus songs
📖[`docs/review_2026-08-14.md`](docs/review_2026-08-14.md) · `AUTO <song> [BEFORE] / [PHASE]`
🔴**His 4 standing songs are NOT in this set and that is a finding, not an omission**: the lever
declined to touch all four (best gains +0.0099…+0.0166, under `MIN_GAIN`) and the maps came out
**byte-identical**. ⇒**The defect it fixes does not occur on the songs he reviews**, so the only way
to hear it is on unfamiliar songs. ★**Lead with BEcause** (Dreamcatcher) — the largest correction in
the cohort (+80 ms, precision 0.456 → 0.900).
**③ Does `[PHASE]` sit on the beat better than `[BEFORE]`?** ★**"Can't tell" is a real answer** — it
would mean a measured 0.62 → 0.35 axis improvement is inaudible, which is worth knowing about the axis.

**④ Still open from 2026-08-04**: is Fallen Kingdom empty vs what our model used to do, or vs what the
song wants?

⚠️**Before building any future review set: check the lever actually CHANGES those 4 maps.** They are a
fixed, comparatively well-behaved sample, so a lever aimed at a defect they lack is invisible there.

**Two standing facts** *(outcomes in PROGRESS.md)*: the **✅-but-unshipped lever sweep is CLOSED** —
exactly two existed, `COLOR_SEP_MODE=extreme` (candidate) and `LAYOUT_TRAVEL_PENALTY=1` (🔴REJECTED,
flow 0.37→0.49, superseded by `BEAT_REACH`) — and **keep `make_periodic_degenerate.py`**, the only
degenerate in the battery aimed at a *structural* lever.

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



## 🔴🔴🔴 P0 — **WE NEVER CROSS HANDS OVER. HUMANS ALWAYS DO.** (found 2026-08-11)
| | crossover share (red in cols 2-3 / blue in cols 0-1) |
|---|---|
| **human** (n=150 strict Expert) | median **0.183**, p10 0.111, p90 0.271 — **0 of 150 maps have none** |
| **ours** (n=149 wide cohort) | **0.0000 on every single map** |

Not a gap — a **missing capability**. Crossovers are a deliberate expressive device every human mapper
uses on ~18 % of notes, and our generator emits exactly zero.
**CAUSE, documented in our own code**: `enforce_color_separation` with `COLOR_SEP_MODE=full` (the
default) moves *every* wrong-side note — its docstring says outright *"this is why our maps measure
crossover == 0.000"*.
🔴🔴**AND THE FIX WAS ALREADY VALIDATED AND NEVER SHIPPED.** PROGRESS.md's 2026-07-27 lever sweep
records **`COLOR_SEP_MODE=extreme` → idiom 1.84 → 0.30 PASS**, ticked as a win. It is not in
`generate.py`'s defaults and not in `docs/BASELINE_2026-08-03.md` — **it fell through the cracks when
the eight defaults were flipped on 2026-08-03.** ⚠️That result predates the tempo fix and the wide
cohort, so it is a LEAD, not evidence. **Re-validating at n=149 now**
(`scripts/overnight_2026-08-11d.sh`).
⚠️**Never bar crossover at zero** — 0 of 150 human maps have zero crossovers, so "no crossovers" is
the *non-human* state, and it is the state we ship today.
★**HOW IT WAS FOUND** — the sensitivity battery: the suite is blind to mirroring a map left-right, and
chasing that showed `crossover` detects it perfectly (0.000→1.000) but is wired into **no axis**.
`flow.py` excludes it from the composite deliberately (it is order-independent and would dilute the
shuffled control) with the comment *"still reported, as guards"* — **nothing ever guarded it.**
✅**THE GUARD IS BUILT** (2026-08-13, see CURRENT STATE). What remains is not a build:
**ship `COLOR_SEP_MODE=extreme` and set `CROSSOVER_GUARD=1` in the same change** — the capability and
the gate that protects it belong together, and shipping either alone re-creates the failure mode.

## 🔴🔴 P0 — **AT HALF TEMPO WE PHYSICALLY CANNOT PLAY FAST.** 28 of 149 songs, capped at exactly 0.500×
*(measurement in PROGRESS.md, 2026-08-13)*

| bpm group | n | ours | human | ratio med / p10 |
|---|---|---|---|---|
| `same` | 100 | 260.0 | 273.0 | 1.000 / 0.800 |
| **`half`** | **28** | **185.0** | **369.0** | **0.500 / 0.500** |

★**p10 = 0.500 ⇒ ≥90 % sit at EXACTLY half** — a hard ceiling, not a statistical gap.
**Mechanism, measured**: our minimum swing gap is **exactly one grid slot**; `subdiv=4` at half the
true tempo makes that slot 2× the human's in real time (`20fc6` 211.3 ms vs 105.6). ⇒**No decode
lever, no selection change and no better probability field can reach it** — this is upstream of all of
them, which makes it unlike almost everything else on this list.

## ✅ MEASURED 2026-08-14 — SUBDIV 8 LIFTS IT EXACTLY. **THE BOTTLENECK IS NOW DETECTION.**
| | **HALF** (n=28, tempo wrong) | **SAME** (n=25, tempo right) |
|---|---|---|
| ebpm ratio vs human | 0.500 → **1.000** | 1.000 → **2.000** |
| onset precision | 0.9172 → **0.9189** | 0.9077 → **0.7812** |
| notes ÷ human | 0.451 → **0.838** | 0.763 → 1.140 |
| per-song **min** ratio | 0.916 | 1.000 |

★**The lever is exactly right where the tempo is wrong and exactly wrong where it is right** ⇒ the
whole value now sits in telling the two apart. ★★**Mechanism confirmed**: 2× slots *raises* precision
at half tempo (the new slots land on onsets that were unrepresentable) and *collapses* it at correct
tempo (they land between real onsets) — `BEAT_HAND_DEAL`'s death was about **where** the slots are,
not how many. **Do NOT raise subdiv globally**; a false positive costs 0.127 precision.

### 🔴 DETECTION — one route closed 2026-08-14, one named and untried
🔴**CLOSED: "share of onset gaps shorter than a slot"** (`scripts/diag_grid_too_coarse.py`) —
half 0.885 vs same 0.702 but they **overlap**: 100 % recall costs an **88.6 %** false-positive rate.
★**Cause: stem onsets are DENSE — 70 % of gaps are already sub-slot at a CORRECT tempo**, so the
statistic measures onset density, not grid coarseness. **Measuring representability via raw onset
gaps measures the wrong thing.**
🔴**CLOSED: ACF periodicity** (`scripts/diag_tempo_octave_acf.py`) — ACF(P/2)/ACF(P) on the onset
train. **Best of the three and still not usable**: half 1.041 vs same 0.810, but 75 % recall costs a
40 % false-positive rate (separation 0.350). ⚠️**Drums-only is WORSE** (0.248) than bass (0.286) or
union (0.350) — my "the pulse lives in the drums" hypothesis is refuted; the union's extra events
reinforce the periodicity more than they blur it.

✅**SOLVED 2026-08-14 — and the answer was the trivial baseline.** A cross-validated study (n=133)
put a tempogram VECTOR at AUC 0.922 / sep 0.724 — and **detected bpm ALONE at AUC 0.973 / sep 0.848**,
beating it and all three statistics. ★**The confound check I added to expose a cheat turned out to be
the best detector in the study.** ⇒**Try the trivial baseline before the clever statistic, and always
put it in the comparison table** — it is the only reason this surfaced, after a heuristic, two
statistics and a classifier had been spent on it.

### ✅✅ BUILT AND VALIDATED — `BEAT_SUBDIV_AUTO=1` (default OFF), n=149
| | HALF (n=28) | SAME (n=100) |
|---|---|---|
| fired on | **15** | **0** |
| ebpm ratio | 0.500 → **0.958** | 1.000 → 1.000 |
| precision | 0.9172 → 0.9154 | **0.8922 → 0.8922** |

★**Zero false positives across the cohort** — the 129 non-fired maps are bit-identical to baseline.
Notes 0.419 → **0.803** of human on the fired songs. **Clears its DoD.**
⚠️**Cost = 2 songs, mechanistically explained**: `30097` and `20fc6` go ratio 1.00 → **2.00** (precision
−0.115 / −0.042) because they were the only two half-tempo songs **already at the human's burst
rate** — detected an octave low, but with no ceiling to lift. ⇒**The defect set and the detector set
are not identical**, and 2 of 28 sit in the gap.
⚠️**Gap in my own pre-registration**: it gated `same`-group regressions and said nothing about
regressions *inside* the group being helped. ★**When a lever targets a subgroup, pre-register the cost
inside that subgroup too.**
🔴**REFUTED — budget compensation does NOT fix it** (tested 2026-08-14 before believing it).
`BEAT_NOTE_BUDGET=0.5` at subdiv 8: `209d2` loses the ceiling lift entirely (ratio back to 0.50) while
`30097` keeps the overshoot (still 2.00) ⇒**exactly backwards.** `ebpm_burst` is a p95 over the
*fastest* gaps, so thinning stretches a sparse song's bursts back out but leaves a dense song's intact.
★**Note count and burst ceiling are not separable by a global knob** — *"gain and damage are the same
dial"*, now on a third lever. ⚠️Mechanism worth knowing: the budget **is** `len(left_thr)` (slots
surviving threshold+NMS) and `beat_nms_radius=1` is in **slots**, so both scale with the grid.
🔴**AND THE KEEP/REVERT REFINEMENT IS CLOSED — no discriminator exists.** Tested every human-free
signal available before building the 2×-pass machinery: our `ebpm_burst` at subdiv 4 **overlaps**
(revert [142,185] vs keep [154,185]), and audio onsets/s **overlaps** at p90, p99 and median. The
precision-drop rule "separates" only by a **0.005** margin on **2** positives (`20fc6` −0.042 vs
`209d2` −0.037) — a fitted constant, not a rule. ⇒**The 2-song cost is intrinsic to the lever as it
stands.** ⚠️A probability-level test would not discriminate either: `30097` has mass on the new slots
and uses it; the problem is not that the model declines them but that taking them exceeds what the
song supports.
★**Order of operations worth repeating**: testing the discriminator on data already in hand cost
minutes and saved building machinery that could not have worked.
**Next**: needs Kyle's ear before any default flips — but note the 13 improved songs are corpus songs,
so this has the same "not on his standing four" problem as the phase lever.

### 📜 the build this came from — choose the subdivision AFTER `BEAT_TEMPO_FIT`
🔴**Do NOT ship `scripts/pick_subdiv.py` as a pre-pass.** Measured on the **raw** `detect_bpm` a
pre-pass can see, the trade is 20 songs gaining the ceiling vs **9 working songs losing 0.127
precision**, and the largest zero-false-positive threshold catches **1 of 28**.
★**On the post-fit bpm it is a different story**: the `same` group's floor rises 77.1 → **96.0**, the
groups nearly separate, and **T=95 lifts 15 ceilings with ZERO false positives.** **The tempo fit is
what makes the detector work.**
**TASK**: thread the subdivision through as a parameter instead of an import-time constant, and pick
it from `_fit.bpm` right after `BEAT_TEMPO_FIT`. Feasible — the subdivision is first used at
`pool_to_beat_grid`, already downstream of the fit. ⚠️Three call sites read `BEAT_SUBDIV` from two
modules (`generate.py` twice, plus `beat_grid`/`mert_encoder`); they must not disagree, which is why
the constant is env-read today. ⚠️Also keep `beats_per_phrase = 64 // subdiv` in step with it.
**DoD**: on the 149-song cohort, the `half` group's ebpm ratio moves off 0.500 with **zero** `same`
songs regressing on onset precision.

**Facts worth keeping from that work** *(full account in PROGRESS.md)*:
- ⚠️`BEAT_GRID_SUBDIV` is a **red herring** — a *quantisation* knob (`generate.py:457`) that does
  nothing to grid resolution. The real constant is `BEAT_SUBDIV`, now env-read from **one** place
  (`beat_grid` imports it from `mert_encoder`; they must never disagree).
- ★At half tempo `subdiv=8` restores the training-time slot **exactly** (174 bpm/subdiv-4 = 86.2 ms =
  87 bpm/subdiv-8) and, paired with `beats_per_phrase = 64 // subdiv`, the phrase's wall-clock length
  too. That pairing is **required**, not cosmetic: Stage-2's `slot_emb` is sized 97 and a literal
  `beats_per_phrase=16` at subdiv 8 indexes off the end (CUDA device-side assert).
- ⚠️Do NOT edit `beat_grid.py` / `generate.py` while a cohort build runs — `build_wide_cohort.py`
  spawns a fresh `generate.py` per map, so an edit splits the arm.
- ⚠️Do NOT read this as "make maps faster": `ebpm_burst` is a **CEILING** measurement and W3 says we
  are already too intense where we *can* be fast. **The defect is the ceiling, not the level.**

## 🔬 THE SENSITIVITY BATTERY (new 2026-08-11) — `scripts/audit_sensitivity.py`
**A different question from every audit we had.** `audit_eval_suite` / `audit_masterpiece` are
DEGENERACY batteries: build a bad map, check the suite ranks it low — i.e. can a metric be *fooled*.
This asks whether a metric can **see**: perturb a real map in a way a player would notice, and check
that *something* moves. **A lever in a dimension no axis measures is unmeasurable, not neutral** —
and silence looks exactly like safety. (M-E rewrote 25 % of all note positions and 12 of 15
masterpiece axes read +0.0000.)

**Blind spots found on the first clean run (40 wide-cohort maps, six-axis suite):**
| perturbation | verdict |
|---|---|
| `mirror_x` (reflect the whole map left-right) | 🔴**BLIND** — max \|Δ\| 0.012. Handedness geography is invisible. |
| `shift_20ms` / `shift_60ms` (global time shift) | 🔴**BLIND on this cohort** — 0.000 everywhere, *because A8 was missing* (below) |
| `swap_colors` | barely seen — only `handrole`, 0.136 |
| `flatten_rows`/`flatten_cols`/`all_dots`/`reverse_dirs`/`rows_random`/`cols_random` | seen, mostly by `idiom` |
| `drop_double_partner` | seen loudly (`handrole` 13.1) |
⚠️`rhythm` (A2) moved **0.000 for every single perturbation** — it reads inter-onset structure only,
so it is blind to *everything* positional by construction. That is correct behaviour, but it means
"rhythm unchanged" is never evidence a placement lever is safe.

## 🧭 REFERENCE
### 🔴 Landmine found 2026-08-14 — a seed re-draws the AUDIO, not just the decode
**`seed_everything(args.seed)` seeds the RNG that Demucs' random-shift augmentation uses**, so the
seed changes the STEMS → the MERT features → **Stage-1's probability field**. Measured on 1f333:
same seed twice is **bit-identical**; seed 0 vs 1 gives max \|Δ\| **0.2049** (mean 0.0264, corr
0.9915) and only **87.3 %** of the top-300 slots survive.
⇒**Every seed-based error bar in this repo contains Demucs stem variance**, including the ±0.004
"seed noise floor". The standing note that *"pairing helps alignment only — the rest ride the torch
decode"* is **wrong at the root**: the draw happens before the model runs.
⇒When you want to vary ONLY the decode, you cannot do it with the run seed as things stand.

### Landmines found 2026-08-11
- ⚠️⚠️**`copy.deepcopy` of a `scorecard._load_any` beatmap DOES NOT ISOLATE IT.** `_load_any` builds a
  local `_BM` whose `color_notes` is a **class attribute**, so the copy shares the same note list and
  the same note objects (`deepcopy(bm).color_notes is bm.color_notes` → True). Mutating "a copy"
  corrupts the original; in a loop over perturbations every row after the first is contaminated.
  **Re-read from disk per variant instead.** Caught because three perturbations agreed to 3 dp — *a
  tie to 3+ decimals is a construction, not a result.*
- ⚠️**`calibrate_playfeel.load_expert_only` returns a 2-TUPLE** (no onsets), so scoring a human map
  through `score_cohort` silently yields `alignment = nan` unless you pass
  `scorecard.onsets_for(path)` yourself. Both sides of any ours-vs-human timing comparison must use
  the **same** onsets.
- 🔴**RETRACTED 2026-08-13: `ebpm_burst` is NOT bpm-contaminated and needs no fix.** Recomputed from
  note TIMES with a wall-clock burst window it is **identical to 0.1 swings/min** on `same`- and
  `half`-tempo songs alike. The 2026-08-11 test re-scored the same beat numbers under a different bpm
  label, which is not a relabelled grid but **a different song**. ⇒The old "derive it from note times"
  fix would have changed nothing. **The real defect is below.**
- ⚠️**Never edit a running bash script** — bash reads it incrementally and a one-byte shift corrupts
  its read offset. Kill, edit, relaunch.

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
