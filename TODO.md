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

**Tasks**
1. **Coincidence detector first — it is cheap and testable today.** We already have a seeded per-stem
   onset cache for **274 songs** (`outputs/stem_onset_cache/`). Compute, per slot, how many stems have
   an onset there; test whether *multi-stem coincidences* predict human note placement better than the
   union does. **This is his hypothesis stated as a measurement, and it needs no retrain.**
2. Measure the specific failures: at SO TIRED ROCK 0:14 and 0:46, what do the stem onsets say, what
   does Stage-1's probability say, and what did the human map do? **Look at the data before theorising**
   — that has found every real cause in this project.
3. If (1) holds, the decode lever is to weight the note budget by coincidence count rather than raw
   onset density (a sharpening of `BEAT_ONSET_EVIDENCE`).
4. Track B proper: re-add an instrument projection to Stage-1 so the two hand channels and the
   probability field can differ by instrument. See `docs/stage1_instrument_rebuild.md`.

**DoD**: on SO TIRED ROCK, notes land on the bass pulse Kyle describes, and 0:14 and 0:46 are mapped.
His ear is the judge — no existing axis measures this.

---

### 🔴 W2 — FALLEN KINGDOM IS TOO EMPTY; and he wants a "how many notes" lever
> *"Its on beat, but its also an expert song and we shouldn't be afraid to play a simple beat thats
> medium tempo. Review the first minute and youll see what I mean. It just feels really empty for no
> reason."* … *"Maybe we should introduce a toggle that is 'How many notes do you want'."*

**Evidence**: we play *"like 1 out of 2/3 notes"* of an obvious slow beat. Baseline nps on Fallen
Kingdom is **3.21** against a corpus median of 3.91 — and this is a *slow* song, so a fixed global
budget under-serves it.

⚠️**Do not just raise `BEAT_DIFFICULTY_SCALE` globally** — Hunger is A+ at the current budget and he
separately complains that parts of it are *too* intense (W3). The defect is **per-song / per-section
allocation**, not the global total.

**Tasks**
1. Build `BEAT_NOTE_BUDGET` as a **user-facing multiplier** (he wants it in the UI — see
   [[feedback-levers-are-user-facing]]): clean range, monotone behaviour, default 1.0 = baseline.
2. Separately, diagnose *why* Fallen Kingdom is starved: is Stage-1's probability low on that beat, or
   is the budget being spent elsewhere? Dump probabilities against the human map's note times for the
   first minute.
3. Check whether the corpus median (3.91) is the wrong target for slow songs — human nps almost
   certainly correlates with tempo, and we apply one number.

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

**Tasks**
1. Detect phrase boundaries (vocal lines especially — the `vocals` stem is in the cache now) and
   ensure a phrase is not abandoned mid-way.
2. Re-open the shelved A5 structure axis with "cover the whole crescendo, not just its peak".

**DoD**: no phrase ends with the vocal still going; 3:20–3:28 responds at ≥0.9× the song median.

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

### 🟢 W7 — DIAGNOSED: the map ends on an ORPHANED HALF-DOUBLE (fix not built)
> *"The final note of the song did not line up together, the map was like .5 seconds late."*

**Read it literally — "together" means the two hands.** On Hunger the map plays doubles at 271.596 /
271.755 / 271.915 and then ends on a **lone red at 272.074**; the blue hand simply stops. Cohort
metric (final event not a double while ≥3 of the previous 4 were): **ours 0.159 vs human 0.036 —
4.4×**. Seed-dependent, not song-dependent (9/24 songs on ≥1 seed, **0/24 on all three**).

✅**`BEAT_TRIM_TAIL` is EXONERATED** — our last note sits **0.47 s before** the cut point, and there is
**no general "we end late" defect**: `last_onset − last_note` is 0.723 s for us vs 0.789 s for humans
over 13 songs. **Do not re-open this as a timing bug.** Full write-up in PROGRESS.md.

**Task (the only thing left)**: `BEAT_END_RESOLVE` — a postprocess rule, default OFF, that makes the
map resolve: if the final event is a single and the recent pattern was doubles, either add its partner
or drop it back to the last full double. Verify at ≥3 seeds that it moves the 0.159 toward 0.036 while
the six axes stay inside noise, then Kyle's ear.

---

### 🛡️ Confirmed positives — protect these
- **Hand-lead alternation**: *"a giant difference maker... noticeably great impact on the flow."*
- **Density pacing**: *"when there is a slow spot we let the player breathe... we no longer have the
  monotony flood of notes."*

Any lever that regresses A6 hand-role or the density-select behaviour trades away something Kyle has
explicitly valued by ear. Check both before promoting anything.

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
