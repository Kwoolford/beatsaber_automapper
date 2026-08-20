# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md); the agent-authoring trail is in
[`agent_mapper/PROGRESS.md`](agent_mapper/PROGRESS.md). Evaluation-suite rationale is in
[`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule:** when an item finishes, its *outcome and what it taught* moves to PROGRESS.md and the
item is **deleted** from here. A completed item is history, not work. Curated 2026-08-02 (from
4,076 lines) and again 2026-08-14 (from 652).

---

## 📍 CURRENT STATE (2026-08-20)

★★**Kyle's brief this session, two messages:** *"keep working on the agentic building suite until
you are confident the maps are good. **You should not need to rely on human review.** Get to a
point where **no matter what song is sent your way, you have visibility as good as a human and can
map to whatever style you want**."* Then: *"build the agent framework **so visibly that you can
confidently validate and create any map from any song**. The current note sheet looks like it could
use much more data… these **electric songs have LOTS of different note types**."*

⇒**This reorders the board.** The six defects D1–D6 and the ML levers are not the work; the work
is a build/validate loop the agent can run alone. Three legs, two now standing:

| leg | state |
|---|---|
| **1. Judge one map without him** | ✅**BUILT** — `mapjudge`, conformal, n=1, **8/8 controls rejected at 0.000**, human 0.896 |
| **2. See any song as well as a human** | 🟡**HALF** — `events.py` gives **14–20 typed note types** per song (was 4) with a per-stem trust verdict; the **alignment/audio axis is still missing from the judge** |
| **3. Map to any style** | ⬜**NOT STARTED** — `idiomize` proves the *mechanism* (hit a target profile on 6 metrics at once); a named style target does not exist yet |

### What is now true that was not yesterday
- ★**The suite can score ONE map.** `python -m beatsaber_automapper.evaluation.mapjudge <zip>`
  gives PASS/FAIL, a conformal p, and a **ranked list of what is furthest from human, in
  percentiles**. `scorecard.py` remains the cohort tool; they answer different questions.
- ★**It reproduces the one Kyle ordering the old suite got wrong** (BEFORE > AGENT on Hunger) and
  names his stated reason (`idiom_coverage` 0.4th pct, `angle_change` 95.8th).
- ★**The judge's density ceiling (6.10 nps rejected) lands on the 6.18 he called unplayable**,
  un-fitted. PARTLY CONFIRMED.
- ★**`idiomize` fixes the agent map's flow defect**: FAIL→PASS, 6 metrics onto the human median,
  **notes/times/hands byte-identical**, 0 parity violations across 33 maps.
- ★**`events.py` + the notesheet TYPES block** — the song as 14–20 typed events, accent in dB,
  ring time, register; untrusted stems collapse to one lane.

### 🔴 The three limits that bound all of it
1. 🔴🔴**THE JUDGE HAS NO AUDIO AXIS.** `alignment` is calibrated but not wired in, so the judge is
   note-attributes-only and **structurally cannot see D2/D3/D4** (anything music-relative). This
   killed a finding mid-session: `[PHASE]` ranked worse on 6/6 songs, and the control showed the
   two arms are **identical on 5/6 once `offgrid_frac` is excluded** — a global phase shift moves
   every note off a beat-0-anchored grid **by construction**. ⇒**Top open item.** The onset cache
   is being expanded (254 → ~1150) to calibrate it.
2. ⚠️⚠️**`idiomize` was tuned against the judge** — "the judge scores it higher" is **circular** and
   must not be quoted as a win. Non-circular: the isolation invariant, parity, and that the defect
   named matches his words.
3. 🔴**The judge certifies NOT DEFECTIVE, not GOOD.** It gates against the human corpus *median*,
   and his standing instruction is *"my target is the best mappers"* ⇒ a corpus median is a
   **floor**. `rank_score` is a distance-from-typical and **minimising it Goodharts toward the
   average map** — it is a defect detector, never an optimisation target. Leg 3 is what closes this.

---

### ▶️ START THE NEXT SESSION HERE
1. 🔴🔴**Wire `alignment` into `mapjudge`** once the onset cache finishes. Needs its own
   calibration slice + a SEPARATE conformal calibration set (23 metrics vs 21 breaks the
   guarantee). Without it the judge cannot see the music.
2. ⬜**Leg 3 — style targets.** `idiomize` already demonstrates hitting a 6-metric profile; give it
   a *named* target (not the corpus median) and let a style be selected. ★Ties to the standing
   memory that **levers are user-facing**.
3. ⬜**Ask him for the best-mapper list** — blocked by his choice, and it is what leg 3 needs.
4. ⬜**`[CROSSOVER]` is still unjudged** and the judge now independently ranks the CROSSOVER arms
   **top of 34**. Strongest unjudged candidate, unchanged.
5. ⬜The notesheet was sent to him for the first browser render — **the audio player has still
   never been confirmed working**.

## ✅ P0 — THE VISIBILITY SUITE: BUILT (V1–V5). Only his looking remains
`notesheet.py` (score) · `overlay.py` (HIT/MISSED/WASTED) · `flowview.py` (hand paths + located
bursts) · `chords.py` (polyphonic LEAD, adopted per song by an in-key gate) · `review.py defect`
(located defects) · one page per song per arm in `outputs/notesheets/<Song>_{BASE,DENSER}.html`.
**Numbers and caveats in PROGRESS.** Still open:
- 🔴🔴**THE PAGE PLAYER HAS NEVER RUN IN A BROWSER** (none on this box). Open one
  `outputs/notesheets/*.html` before building anything on it.
- 🔴**The two published pages are STALE** (old lyrics, no audio); republish needs his approval.
  Overlay `claude.ai/code/artifact/34bf3922-080b-4c9b-bcd6-90792bb1a6b9`, score `.../f47350fb-…`.
- ⬜Only `1f8d6` was ever published; the other three render locally.
- 🔴**Word ACCURACY is unaddressed** and sung-coverage cannot see it ⇒ forced alignment against
  supplied lyrics is the only route (the bigger-ASR route is closed). **Ask him first** whether
  roughly-right landmark words are enough.
- ⬜**V5's ledger holds zero LOCATED defects** — that half is his. `review.py defects` prints the
  six standing unlocated ones as a backlog.

## 🔴🔴 P0.5 — THE SIX DEFECTS HE NAMED (2026-08-17), AS MEASURED

★**The status table is in CURRENT STATE above; the numbers are in PROGRESS.** What remains *open*
per defect:

| # | his words | what is still open |
|---|---|---|
| **D1** | *"very slow"* | **= D4's budget lever.** `--beat-threshold 0.25` is the candidate (P0.6); beyond it, training-side (P0.8) |
| **D2** | *"slightly off beat"* | 🔴Refuted as note placement. ⚠️**Live but untested**: we may be *too* quantised for a song with groove — nothing measures that yet |
| **D3** | *"drops at the wrong time"* | Confirmed (0.347 vs human-human 0.49). 🔴**Its obvious fix is refuted** — `structure.py` locates the human's moves *worse* than our map does. **A better route is unknown.** |
| **D4** | *"not following the main vocals"* | ★**The main line of work.** See P0.8 — training-side density conditioning |
| **D5** | *"random bursts"* | 🔴Refuted/inverted. **Do not build a burst-suppressor.** ⚠️Ask him at a shaded bar in the FLOW view whether *that* is what he meant |
| **D6** | ★*"nps wasted on non main notes"* | = **C5 doubles**, priced at 21 % of the vocal budget. ★**Still gated on HIS definition of "main"** |

⚠️**D6 must not become another invented axis.** Show him `overlay.MAIN_DEFAULT` and let him
correct the *definition* — a metric he has endorsed by looking at it is a different object from
one built from first principles, and building those from first principles is exactly what produced
the anti-correlation.

## 🟢 P0.55 — ★★THE BIGGEST LEVER FOUND: `BEAT_SUBDIV_AUTO=1` (2026-08-19d)
On the 15 half-tempo songs its trigger fires on: **vocal coverage 0.275 → 0.487 (+0.216, 15/15
songs, p=6e-05)**, notes 432 → 855, **precision unchanged**. Those songs go from worst-in-cohort to
better than the correct-tempo group. **alignment 0.589 → 0.237** (crosses its bar), rhythm and flow
much better. ✅**AND `idiom`'s collapse IS AN ARTIFACT (2026-08-19e)** — `idiom` buckets `dt` in the map's own
beats, so a halved bpm shifts every gap one bucket down. **Rescaled to true tempo the ON arm's
timing distribution matches the human's almost exactly** (1/4: 0.499 vs 0.435; 1/16: 0.001 vs
0.000). ⇒**The notes are fine; the labels were wrong. The only objection to this lever falls.**
🔴**NEW LANDMINE: every beat-domain axis is unreliable on the 28 half-tempo songs.**
✅**SEED-VALIDATED 2026-08-19g at 3 seeds: +0.222 = 49× the seed sd** (OFF 0.272±0.0027,
ON 0.493±0.0058) — the most robust result of the session by a wide margin.
⚠️**The lever has been OFF by default the whole time** despite passing its DoD.
⬜**Do NOT flip it unilaterally — it needs his ear**, and ⬜`idiom`'s collapse needs its own
investigation (at subdiv 8 the note-type mix changes with the density; the axis may be reading
that). ⚠️**The 13 half-tempo songs with bpm ≥ 95 stay missed, and that is NOT fixable by the trigger
(2026-08-19f)**: raw bpm is the best feature available (AUC 0.978, 16/28 at zero false fires);
adding a drum-gap rule buys ~3 songs for ~1 false fire, and widening to bpm<110 costs 10 false
fires for 8 songs. ⇒**Do not widen it.** Those songs need the **real tempo model** (P1).

## 🟢 P0.6 — ★THE POSITIVE LEVER, VALIDATED AND AWAITING HIS EAR: `--beat-threshold 0.25`
Production v4, threshold 0.40 → **0.25**. **At 3 seeds: vocal coverage +0.029 (8.6× the seed sd),
playfeel +0.122 worse (4.2× sd, still inside its 1.00 bar); everything else — flow, idiom,
handrole, alignment — is seed noise.** Median **4.06 nps**, **0/23 songs above the 6.18 he called
unplayable**. ★**0.25 is the operating point**: lower buys no more notes and costs precision.
✅**Installed as `[BASE]` vs `[DENSER]`** — ⚠️**not** `[BEFORE]` (stale). **Fallen Kingdom is the
song to judge.**
⬜**The only open question is his ear.** If it is promoted, flip the default in
`scripts/generate.py` and re-baseline the cohort.

## ✅ P0.7 — TRACK B: CLOSED. Do not promote v7/v8
At matched budget v8 scores **0.417 vs production 0.420** on vocal coverage — a coin flip. Its
allocation is better and it doubles less; none of it reaches coverage. ⚠️Not a clean instr-only
ablation (v7/v8 also carry `struct_proj`). Full numbers in PROGRESS.

## 🔴 P0.8 — D4's ONLY REMAINING ROUTE IS TRAINING-SIDE (chain closed 2026-08-18z)
Every alternative is now eliminated by measurement: decode saturates at thr 0.25 (**5× lower buys
+4 notes**), the 1/4-beat grid is only **26-33 %** full, Track B at matched budget is **parity**,
and we sit **244 notes (29 %) below the human even at the floor threshold**. **Stage-1's
probability field is nearly BINARY — no mass below 0.25.** ⇒**Make Stage-1 propose more notes**
(loss/target/threshold-during-training), not another decode knob. ★**SCOPED 2026-08-18aa**: we
emit **0.217** positives per (slot,hand) against a **corpus label mean of 0.245** and these songs'
humans at **0.294** ⇒ **we are below our own training distribution AND we do not modulate per
song.** 🔴Dead ends already checked: training is Expert/E+ only (not diluted), and the `_NPS_RANGES`
cap is legacy-only. ✅Span is not the gap (0.988 vs 0.990 of the music covered).
★★**MECHANISM NAMED 2026-08-18ab: STAGE-1 DOES NOT MODULATE DENSITY PER SONG.** Our nps vs the
human's on the same song is **r = 0.046, slope 0.026** (n=144) — no measurable tracking — while
**crude audio features predict the human's nps at R² = +0.185** (drums/s 0.414, bpm 0.396). We emit
**3.82 ± 0.77** nps where humans span **3.75-7.00**. ⇒**The signal exists and we are not using it.**
⬜**PROPOSED RETRAIN, deliberately NOT queued** (it is a retrain, and the density lever is still
unjudged): an auxiliary per-song **density target** / FiLM conditioning on the song's own label
rate. **DoD: per-song nps correlation rises from 0.046 toward the demonstrated ≈0.43 floor AND
vocal coverage rises, at ≥3 seeds.**
⚠️Whatever is built must not
regress the density Kyle already accepted — he called **6.18 nps unplayable** and the current
lever sits at 4.06.

## 🔵 P1 — W1 + W4 + `follow_vocals` are ONE defect (now inside P0.8)
Three views of one cause: Stage-1 cannot hear the melodic instruments. ⚠️**Sized 2026-08-18j and
it is NOT sufficient** — we reach **0.581** of the human even where a drum marks the vocal, vs
0.456 where none does. Use them as **one** acceptance target; the route is **P0.8**.

## 📦 AWAITING KYLE'S EAR — sets A and B, now answered globally

`for_review/` holds 32 maps; `python scripts/review.py next` is the shortlist and
`review.py list` the full pending list. **He has now answered both sets at the level of
defects rather than arms**, so:
- ★⚠️**`[CROSSOVER]` HAS NEVER BEEN PLAYED.** He reviewed *"the before and after as well
  as the before vs phase"* — crossover was not among them. It remains the **strongest
  unjudged candidate** (crossover 0.000 → 0.112 against a human 0.183 with 0 of 150 human
  maps at zero; flow 0.37 → **0.23**, i.e. it improves the exact axis he complains about).
  🔴Do not let it get swept into "numeric wins that failed his ear" — it has not been tried.
- ⬜**`[AFTER]` vs `[BEFORE]` (set A) is still unresolved per-arm** — the intentional/lazy
  question was never answered, and `mapctl reuse --vary 0.15` is still a guess.
- ⚠️**`[PHASE]` did not remove *"slightly off beat"*** ⇒ do NOT flip
  `BEAT_GRID_PHASE=search` on this evidence. Either the lever is inaudible or the
  off-beat feel has a different cause (tempo, D2).

**Also still open from 2026-08-04**: is Fallen Kingdom empty vs what our model *used to
do*, or vs what the *song wants*?

---

## 🔵 P1 — AGENT MAP AUTHORING (`agent_mapper/`) — demoted 2026-08-17

Was P0; **visibility is now above it**, because the agent map's verdict was
*"expecting… much better"* + *"the notes flow in a really odd way"* and we cannot fix
flow we cannot see. The perception tools built for it (`melody`/`percussion`/`structure`)
are exactly what the visibility suite is built from, so this is a demotion in priority,
not a retreat.
🔴**The judged map predates all of them** — it was planned off `brief.py`'s 8-bar onset
densities alone and knew no pitch, no sections and no kit. It is the BEFORE for the
perception work, not a test of it.
- Rebuild a map using melody + structure + percussion, on **Fallen Kingdom, not Hunger**
  (Hunger's vocals are genuinely unpitched — pYIN voiced-on-loud 0.19 vs 0.91–0.99).
- 🔴`travel` is a **sequence** property — per-note pitch placement is REFUTED (made it
  worse, 4.77 → 3.56). Do not try another per-note rule.
- Wire `percussion.py` into doubles (crashes/snare accents, not "strong beat + 2 stems").

---

## 🔵 P1 — TEMPO: now PRICED, and much smaller than its reputation
Our tempo is right on **70.5 %** of songs (n=149). ✅Half-tempo is the big group and
**`BEAT_SUBDIV_AUTO` already recovers 16 of its 28 songs** (see P0.55).
★**PRICED 2026-08-19h/i — read this before building a tempo model:**
- `BEAT_SUBDIV_AUTO` as it stands is worth **+0.030 cohort-wide** on vocal coverage, for one env var.
- Catching the remaining **12 half-tempo songs is worth a further +0.025** — *that* is what a real
  tempo model buys: about **a fortieth of the human gap**.
- 🔴**Cheap detection is exhausted**: raw bpm is the best feature (AUC 0.978, **16/28 at zero false
  fires**); a drum-gap rule adds ~3 for ~1 false fire; widening to bpm<110 costs **10** false fires
  for 8 songs; post-hoc features add only 4 of the remaining 12. ⇒**Do not widen the trigger.**
  ⚠️`notes per second` scores AUC 0.903 and catches **zero** at an affordable FP rate — ★*AUC is not
  an operating point.*
- ⚠️**Every cheap route to the 2:3 misreads was already closed** (onset-energy balance, onset-gap
  density, ACF periodicity, a CV tempogram classifier, the `fit_tempo` tie-break).
🔴Tempo error hurts **HOW MANY** notes we place, not **WHERE** (placement AUC 0.543): at half the
bpm the 1/4-beat grid has half the slots, so the map is budget-starved before a note is chosen.

## 🎯 STANDING CONSTRAINTS + the three older objections D1–D6 does NOT cover

★**His standing instruction, still binding**: *"tread carefully, make isolated and tactical
changes, and document like crazy."* One lever at a time, ≥3 seeds, nothing promoted without his ear.
⚠️He declined to name exemplary mappers *yet* — the best-mapper cohort is blocked **by his choice**.
*(The W1–W7 table and its mapping onto D1–D6 is history and lives in PROGRESS.md. Work D1–D6.)*

**Still open and NOT covered by D1–D6:**
- **W2** — Fallen Kingdom *"really empty"*. 🔴Cause unidentified; five instruments have failed.
  ⚠️**Ask him**: empty vs what our model *used to do*, or vs what the *song wants*?
- ★★**W6 — WIDER THAN RECORDED (measured 2026-08-19j): we ship ONE of the five elements human maps
  use.** **Walls: 93 % of human maps, median 86 each — we emit ZERO** (Fallen Kingdom: human 124,
  ours 0). **Arcs: 88 % of the maps that CAN have them** (v3 only — 97 of 147 corpus maps are v2,
  which is why they look rare at 30 %). Chains 50 % of v3. Bombs 18 % vs our 6/146. ⚠️**We already
  write v3 (3.3.0), so we CAN emit all of them.**
  ★**This gives W2 (*"Fallen Kingdom is really empty"*) its first untested candidate cause** — five
  instruments have failed to explain it and every one of them looked at notes.
  ✅**BUILT 2026-08-19k — `agent_mapper/walls.py`**, a post-processor (notes byte-identical, so the
  A/B isolates one thing). Copies the measured human vocabulary: 84 walls, width 1, outer lanes,
  median 0.16 beats, 62 % crouch, **0 of 84 colliding with a note**. ✅**Installed as
  `AUTO … [WALLS]` for Fallen Kingdom and アリスブルー**, built on the DENSER map.
  ⬜**Ask: does `[WALLS]` still feel "really empty" next to `[DENSER]`?** ⚠️A candidate cause, not
  a measured defect — "no difference" is a good answer too. ✅**Arcs BUILT 2026-08-19l** (`agent_mapper/arcs.py`, additive — notes byte-identical; 48 arcs,
  median span 1.00 beat, 48/48 anchored on real notes, per-hand 24/24). Installed as `[FULL]`
  (= DENSER + walls + arcs) for Fallen Kingdom and アリスブルー.
  ⇒**Review ladder: `[BASE]` → `[DENSER]` → `[WALLS]` → `[FULL]`, one change per step.**
  ✅**CHAINS BUILT 2026-08-19m** (`agent_mapper/chains.py`) — **they ARE additive** (678/678 human
  chain heads coexist with a note), 16/map, span 0.062, 4-5 slices, notes byte-identical, installed
  as `[CHAINS]`. 🔴**But `swing_sim` IGNORES `burstSliders`** — it returns 913 swings and 0
  violations with *and* without them ⇒ **the chains are UNVALIDATED for playability**; only the
  notes were checked. ✅**DONE 2026-08-19n — `swing_sim` models chains** (a chain lengthens its head swing to the tail;
  verified firing, and our 16 chains give **0 parity violations** under it). 🔴**DEFAULT OFF**
  (`BEAT_SIM_CHAINS=1` to opt in) because **25/51 v3 human maps have chains**, so enabling it
  shifts the human reference — measured at `travel` **+2 %**, other axes unchanged.
  ⬜**Flip it on and recalibrate the human reference in the SAME change**, when chains are adopted.
  ⬜Original reason to build them: a chain is one swing carrying
  4 segments — **density with no new distinct time**, which is exactly what we buy with doubles
  (39.6 % vs human 20.7 %, costing 21 % of the vocal budget). ⚠️Chains alter notes, so they must
  clear the swing simulator before shipping.
- **W5** — dot blocks as decoration: ⏸️**he deferred this himself.** Do not revive unprompted.

⚠️**Protect these — he named them by ear**: A6 hand-role division, and the density pacing
(*"when there is a slow spot we let the player breathe"*). ★And new on 2026-08-17: *"there is a
good deal of notes that are on beat and I can tell play part of the song"* — **that half works.
Do not regress it while chasing D1–D6.**

---

## 🔵 P2 — CARRIED FORWARD

### C1 — Precision sits at the greedy optimum; gains need better probabilities, not better picking
Three decode levers moved onset precision off ~0.90 by nothing (density, γ allocation, a
probability floor); the IOI prior moved it *down* to 0.769. **Stop hunting decode knobs.**
✅**2026-08-14 sharpened it**: the ~10 correct-tempo alignment failures are a **pure selection
defect**, established by elimination — not tempo, not phase, **not onset supply** (4.5 onsets
available per note we emit), and **not difficulty** (the human scores 0.943 on exactly those songs
vs 0.934 on the ones we handle fine).

### C2 — Grid PHASE ⚠️resolved ON THE METRIC ONLY — and its successor suspect is now REFUTED
`BEAT_GRID_PHASE=search` fixed ~18 of 39 failing songs **by the alignment axis**, and Kyle still
reported *"slightly off beat"* after playing `[PHASE]`. This item then pointed at **tempo** as the
better suspect. 🔴**2026-08-18g refutes that too**: on all four maps he played the bpm is
**exactly** the human's, the offset is 0, and our note times match a human's **better than two
humans match each other** (0.87/0.92/0.73–0.82/0.67–0.71 vs human-human **0.676**). Cohort-wide we
are **more** on-beat than a human and place **fewer** 16ths.
⇒**Do not flip `BEAT_GRID_PHASE` on the axis alone, and do not attribute D2 to tempo.** What is
left is untested: we may be *too* rigidly quantised for a song with groove. ⚠️Never apply a blanket
global shift — on `1f767` the human wants the same shift we do, so that part is an **onset-detector
offset**, and "fixing" it is the `h_dist` failure.

### C3 — Density/rhythm tension: you cannot thin your way to human density
Re-tuning toward the human note rate costs rhythm (`pulse_stability` −0.06 → −1.11). Humans at
3.9 nps have a pulse; we at 3.9 (thinned from 4.4) do not. `BEAT_IOI_PRIOR` made it worse. Needs a
different idea, not another sweep.

### C4 — Every beat-domain result predates the tempo fix, and beat-domain axes LIE on tempo errors
A2 rhythm, A6 handrole and the hand-offset work were tuned against a bpm that was wrong on 20 of
21 songs. Re-derive before building further on them.
🔴🔴**2026-08-19e gives this teeth**: every beat-domain axis buckets by the **map's own beats**, so
on the **28 half-tempo songs** every interval lands one bucket off. `idiom` showed it loudest
(0.663 → 2.955 under `BEAT_SUBDIV_AUTO`, entirely an artifact — rescaled to true tempo the timing
distribution matches the human almost exactly). ⇒**When a beat-domain axis moves on a cohort
containing tempo errors, check whether the BPM moved first.**

### C5 — Doubles: root cause found · decode fix FAILED · ★now PRICED against D4
**Not too many notes — too few distinct times.** Stage-1's two hand channels correlate
**0.985–0.993**, so both hands get the same information and pick the same slots; the double rate is
structurally guaranteed. `BEAT_HAND_DEAL` hit every structural target and degraded rhythm 6× ⇒
**not reachable by decode.**
★★**2026-08-18k prices it in the units of the biggest defect**: **39.6 % of the notes we spend on
the vocal line are doubles** onto an onset the other hand already covered (human 20.7 %), so
**fixing doubling alone lifts vocal coverage 0.385 → ≈0.487 — a quarter of the D4 gap, with no
extra notes.**
★**And 2026-08-19l names the human's alternative**: a **chain** is one swing carrying 4–5 segments
— *density with no new distinct time*, which is exactly what we buy with doubles. `chains.py` now
builds them; they are installed and unjudged.
⇒This justifies costing a **representational** fix, not reviving the decode one.

---

## 🧭 REFERENCE
### 🔴 Landmines — a seed re-draws the AUDIO, not just the decode
**`seed_everything(args.seed)` seeds the RNG that Demucs' random-shift augmentation uses**, so the
seed changes the STEMS → the MERT features → **Stage-1's probability field**. Measured on 1f333:
same seed twice is **bit-identical**; seed 0 vs 1 gives max \|Δ\| **0.2049** (mean 0.0264, corr
0.9915) and only **87.3 %** of the top-300 slots survive.
⇒**Every seed-based error bar in this repo contains Demucs stem variance**, including the ±0.004
"seed noise floor". The standing note that *"pairing helps alignment only — the rest ride the torch
decode"* is **wrong at the root**: the draw happens before the model runs.
⇒When you want to vary ONLY the decode, you cannot do it with the run seed as things stand.

- 🔴🔴**AXIS GAPS ARE COHORT-SIZE DEPENDENT (2026-08-19r).** The **same maps** score **flow 1.260
  at n=5 → 0.446 at n=50** and **alignment 1.062 → 0.341**: a small cohort estimates its own
  distribution noisily and the noise reads as distance, so **small cohorts look worse**.
  ⇒**NEVER compare cohorts of different sizes**, and the **bars do not transfer across n**.
  ⚠️Also: **all six axes are `nan` at n=1 and n=2** — the suite is a cohort statistic and **cannot
  score one map**, which is the structural reason a passed DoD says nothing about a map.
- 🔴**`alignment` SILENTLY RETURNS `nan` IF THE FILENAME DOES NOT START WITH THE SONG ID.**
  `scorecard.song_id()` parses the id from the filename: `1f8d6_WALLS.zip` → `'1f8d6_WALLS'` → no
  cached onsets → `alignment = nan`, **no error, five axes scored instead of six**.
  ★**Name generated maps `<arm>__<songid>.zip`, never `<songid>_<arm>.zip`** (2026-08-19p).
- 🔴**THE SUITE IS BLIND TO WALLS, ARCS AND CHAINS.** Adding 84 walls + 48 arcs + 16 chains moves
  **every axis by exactly 0.000** — it scores notes and nothing else. ⇒**No axis can justify or
  reject the element work; only his ear can.** (Chains: the model works, but 16 in 913 swings is
  1.7 % and `travel` is a median — under-powered at human chain density.)
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

### Habits that outlived the seed lottery
1. **Score every arm at ≥3 seeds and quote the sd.** ⚠️n=3 *underestimates* sd — treat it as a screen.
2. **`npass` is not a ranking statistic** (an identical config scored 4, 4, 2). Rank per-axis with error bars.
3. **Open**: the spread bar (0.35) sits inside the noise — stop gating on it, keep a hard alarm near 0.15.
   Not done unilaterally; it changes scorecard semantics.
