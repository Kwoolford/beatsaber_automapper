# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md); the agent-authoring trail is in
[`agent_mapper/PROGRESS.md`](agent_mapper/PROGRESS.md). Evaluation-suite rationale is in
[`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule:** when an item finishes, its *outcome and what it taught* moves to PROGRESS.md and the
item is **deleted** from here. A completed item is history, not work. Curated 2026-08-02 (from
4,076 lines) and again 2026-08-14 (from 652).

---

## 📍 CURRENT STATE (2026-08-20, close)

> ★★**FOCUS, set by Kyle 2026-08-20 (mid-session):** *"I want to focus solely on building an
> agentic map building suite. The objective is to build a suite so good you are highly confident
> and can manually map any song you'd like and even customize them to whatever style you want."*
> ⇒**Everything above the backlog is agent-path work.** ML-pipeline work (retraining Stage-1, tempo
> detection, decode levers) is **BACKLOGGED** at the bottom of this file — do not pick it up, and do
> not let a shared metric pull the loop back into it. ★The test of an item is: *does it make the
> AGENT build or validate a map better?*


★★**Kyle's brief redefined the work this session:** *"keep working on the agentic building suite
until you are confident the maps are good. **You should not need to rely on human review.** Get to a
point where **no matter what song is sent your way, you have visibility as good as a human and can
map to whatever style you want**."* Plus: *"build the agent framework **so visibly that you can
confidently validate and create any map from any song**. The note sheet could use much more data —
these **electric songs have LOTS of different note types**."*

⇒**The work is a build/validate loop the agent runs alone.** The ML levers below are now
secondary. Full numbers for everything here are in PROGRESS.md (2026-08-20).

| leg | state |
|---|---|
| **1. Judge one map without him** | ✅**BUILT** — `mapjudge`, conformal, n=1. Human 0.884, **all 8 degenerate controls 0.000** |
| **2. See any song as a human does** | 🟡**HALF** — `events.py` gives **14–20 typed note types** (was 4) + a per-stem trust verdict. 🔴**The judge still has no audio axis** |
| **3. Map to any style** | 🟡**DEMONSTRATED** — `style.py`; ordering checks hold but magnitudes are modest (n=1 song) |

**One command runs the whole loop:** `python agent_mapper/autobuild.py <audio> --name X`
→ SEE (`events`+`structure`) → PLAN → BUILD (`mapctl`) → DRESS (`idiomize`) → JUDGE (`mapjudge`).
**23/23 eval-songset songs produce a PASSing map.**

### 🔴 The four limits that bound all of it — read before trusting any of the above
1. 🔴🔴**THE JUDGE HAS NO AUDIO AXIS** ⇒ note-attributes-only, **structurally blind to D2/D3/D4**.
   It killed a finding mid-session: `[PHASE]` ranked worse 6/6, but with `offgrid_frac` excluded the
   arms are **identical on 5/6** — a global shift moves every note off a beat-0 grid *by
   construction*. **This is P0.**
2. ⚠️⚠️**`idiomize` was tuned against the judge** ⇒ "the judge scores it higher" is **circular**.
   Non-circular: the isolation invariant, parity, and that the defect named matches his words.
3. 🔴**A PASS = NOT DEFECTIVE, not GOOD.** It gates at the corpus **median**, which his standing
   *"target is the best mappers"* makes a **FLOOR**. ★**`rank_score` is a distance-from-typical:
   minimising it Goodharts toward the average map. NEVER optimise it.**
4. 🔴**Nothing here has reached his ear.** Every 2026-08-20 result is a number.

---

### ▶️ START THE NEXT SESSION HERE
★**Kyle's mid-session redirect (2026-08-20) makes this list agent-only.** ML items are in the
BACKLOG section near the bottom; do not start one.
1. 🔴🔴**Give the agent's validator ears — wire `alignment` into `mapjudge`.** Without it the agent
   can build a map and has **no way to check the notes are on the music**, which is the one thing
   Kyle asked for ("visibility as good as a human"). Run
   **`bash scripts/build_onsets_calib_spans.sh`** (~55 min GPU — Demucs onset extraction, **not**
   model training), then `python scripts/calibrate_mapjudge.py --n 1100`, then
   `python scripts/audit_mapjudge.py --n 250`. Dual calibration sets are **already plumbed** (21 vs
   23 metrics must not share one).
2. ★★**Attack the pulse defect (P0.5)** — the biggest measured defect the agent path has, and it is
   purely agent-side (no model in the path). `scripts/diag_pulse_union.py` has already split it.
3. ⬜**Get an autobuilt map in front of his ear.** They are unheard. Suggest 1f767 or 1f8d6 against
   the human map. Record with `scripts/record_verdict.py`.
4. ⬜**`[CROSSOVER]` is still unjudged** — and `mapjudge` independently ranks the CROSSOVER arms
   **top of 34**. Oldest open question on the board, now with a second line of evidence.
5. ⬜**The best-mapper list** — leg 3 needs it to aim above the corpus median. ⚠️Ask only when he is
   AT the machine; never block an overnight run on it.

---

## ✅➡️ P0 — THE JUDGE CAN HEAR (closed 2026-08-20 evening), WITH ONE GAP
`mapjudge` now scores **23 metrics** including **`onset_precision`** and **`offset_mad_ms`**.
Audit **DoD MET**: human accept 0.884, all eight controls 0.000. Verified live on a real map —
`scored 23 metrics`, no banner. Full numbers in PROGRESS.md.
🔴**THE REMAINING GAP: the audio axis is CALIBRATED but NOT AUDITED.** The control battery runs in
no-audio mode (its per-metric table has 21 rows, not 23), so `onset_precision` has never had to
discriminate a control. The timing controls are rejected by the **beat-domain** metrics instead.
**Task**: give `audit_mapjudge.py`'s perturbed variants their song's cached onsets and re-run.
**DoD**: the discrimination table has 23 rows, and `onset_precision` scores AUC > 0.8 on at least
one timing control.
⬜**Re-judge the songset against the NEW reference.** The 23/23 PASS was measured on the 21-metric
ruler; every percentile in this file from before tonight moved.
★**It already found something**: `onset_precision` **0.823, 10.1st percentile** on `PULSE__1f767` —
our notes land on an audible event less often than 90 % of humans. **D2/D4 on the agent path**,
invisible to every agent instrument until tonight.

## ✅➡️ P0.5 — PULSE: FIXED AND VALIDATED, AWAITING HIS EAR
**Built 2026-08-20 evening: `agent_mapper/pulse.py`, `--pulse` on `mapctl auto` / `autobuild`.**
Full numbers in PROGRESS.md. n=23: `pulse_stability` 0.329 (6th pct) → **0.637 (71st)**,
`dominant_share` 0.362 → **0.495**, `ioi_switch_rate` 25.9 → **12.5**, `ioi_cond_entropy` 0.680 →
**0.505**, lift over each map's own shuffle 0.074 → **0.273**, `n_notes` 641 → **711** (rose).
**23/23 still PASS.** All three DoD terms met.
⬜**THE ONLY THING LEFT IS HIS EAR.** Installed as **`AUTO <song> [AUTOBASE]` vs `[AUTOPULSE]`** on
all four standing songs. Record with `scripts/record_verdict.py`.
⬜**Then decide the default.** Left OFF deliberately — it has not been heard.
⚠️**It overshoots to the RIGID side**: lift 0.273 vs the human 0.167, `pulse_stability` past the
95th pct on 5/23 songs. If he says "too mechanical", that is the knob — `MAX_EMPTY_RUN` and the
syncopation-restore threshold in `pulse.py`, not the phrase length.

## ✅➡️ P0.6 — HAND ROLE: PRICED AND SETTLED, AWAITING HIS EAR
**`mapctl auto --lead-bias 0.3`** (default 0 = old behaviour). Judged, n=23, new 23-metric ruler:
`role_asymmetry` **1.1st → 49.9th percentile** (the human median), `ebpm_burst` 6.9th → 46.4th,
**p median 0.222 → 0.626**. Full table in PROGRESS.md.
★**Picked two-sided**: bias 0.5 gives `role_asymmetry` 97.8 %, as wrong as the 1.1 % we started at.
⚠️**Real cost**: `role_swap_rate` 58.0 → 81.2 % (we hand the lead over too often). bias 0.4 trades
it back at the price of asymmetry. ⬜If his ear says the hands feel jumpy, try 0.4.
✅**Isolation invariant holds**: `nps` / `onset_precision` / `pulse_stability` identical across all
four arms — the lead hand changes who plays a note, never when.
⬜**Installed as `AUTO <song> [AUTOLEAD]`** (pulse + lead 0.3) on the four standing songs.

## ✅ P0.9 — DENSITY: TWO ARITHMETIC BUGS OF OURS, FIXED
`nps` **3.43 (19th pct) → 4.02 (44th)**, human median 4.17, 23/23 PASS. (1) the energy curve was
never normalised — its duration-weighted mean was 0.919, so 4.17 silently delivered 3.83; (2) the
accent filter counted EVENTS while the budget is spent in DISTINCT SLOTS, so a plan budgeting 840
delivered ~604. `--target-notes` now searches the accent percentile for surviving slots.
⚠️**4.17 is the TARGET, not a floor** — 6.18 is unplayable per Kyle and the judge rejects near 6.10.
★**Attribution first saved tuning the wrong stage**: the pulse quantiser and the per-hand floor were
both innocent (pulse is note-neutral; nothing is skipped at placement).

## 🔴 P1.0 — `1f9a0` STILL FAILS, AND THE OBVIOUS REMEDY IS REFUTED
**The constraint is real**: below 150 bpm a 1/4-beat slot is `15000/bpm` ms, so the grid snap alone
is worst-case ±`7500/bpm` — outside the axis' 50 ms tolerance. `1f9a0` (93 bpm) fails on
`onset_precision` 0.474.
🔴🔴**BUT A FINER GRID IS REFUTED** (`--adaptive-subdiv`, kept default-OFF as the measurement):
`onset_precision` falls on **10 of 10** affected songs (0.899 → 0.846) and `pulse_stability`
0.591 → 0.376. ★**`1f9a0`'s FAIL→PASS is NOT the defect being fixed** — its `onset_precision` moved
only 0.474 → 0.499. **A PASS without the named defect moving is not a fix.**
⇒**The binding constraint is note SELECTION, not what the grid permits** — agreeing with the
grid-representability result (+0.106 headroom still unused at SUBDIV 4).
**Next candidates** (none tried): choose events by *distance to a scored onset* rather than by accent
alone; or let the pulse lattice prefer phases whose points carry onsets.
**DoD**: `onset_precision` rises on the affected songs **without** `pulse_stability` leaving 25–75 %.

## 🟡 P0.7 — DETECTOR MISMATCH: HALF FIXED, RESIDUAL IS A COVERAGE BIAS IN THE AXIS
✅**Built**: `agent_mapper/refonsets.py` + `--snap-onsets` (opt-in). `onset_precision`
**0.856 → 0.890** (human 0.919), p median 0.655 → 0.680, 23/23 PASS, nothing else regressed.
🔴**DoD term 1 FAILED**: placed events within 50 ms of a scored onset **0.832 → 0.867**, target 0.97
— because **13.3 % of our events have no scored onset within 60 ms at all**. A snap fixes TIMING
disagreement, not COVERAGE.
★★**`events.py` uses `htdemucs_6s` (guitar + piano stems); the scoring cache is the 4-stem union.**
⇒**Part of this metric may be the AXIS being blind rather than the map being wrong** — a guitar note
we can see and the scorer cannot is scored as a miss.
⬜**The principled fix — rebuild the onset cache from the 6-stem separation.** ⚠️**This moves the
human baseline** (`build_onset_cache.py` says so) ⇒ it means recalibrating `mapjudge` and
re-measuring every alignment number recorded so far. **A deliberate decision, not a chore.**
⬜**Generalise `--snap-onsets` to any song** — it needs cached reference onsets, which exist only for
corpus songs, so it cannot be the default while "map any song" is the goal.
⚠️Cost: `nps` 3.43 → 3.29; snapping collapses events sharing one onset (88 of 833 on 1f767).

## ✅ P0.8 — PULSE OVERSHOOT: FIXED by `SYNC_FRAC = 0.3`
`pulse_stability` **88.1st → 37.2nd percentile** (human 0.514, ours 0.515) and `onset_precision`
0.829 → 0.856 **at the same time** — not a trade, because everything the syncopation-restore puts
back is a REAL detected onset, so it breaks the lattice and lands on audible events together.
🔴**The fill knob (`MAX_EMPTY_RUN`) was the WRONG lever and the sweep refuted my mechanism**: fill 2
scored BETTER `onset_precision` than fill 1 (0.842 vs 0.829), so "more fill = more invented notes =
worse precision" is false — the count-matched period search reacts to the fill, and WHICH PERIOD is
chosen matters more than how many gaps are bridged.
⚠️`--pulse-sync` 0.5 and 0.4 are **identical**: the period is an integer number of slots, so both
round to the same "≥1 slot off" test at the common period of 2. 0.2 overshoots the other way.

## 🟢➡️ P1 — LEG 3: REPLICATES ON 10 SONGS (4 of 6 orderings), AWAITING HIS EAR
`scripts/sweep_style.py`, n=10 songs, scored as orderings: `nps` calm<dense **10/10**, `crossover`
flowing<technical **10/10**, `angle_change` **9/10**, `peak_nps` **8/10**. **All five styles PASS
10/10** — a style is reached without making defective maps.
★★**`ebpm_burst` calm<dense is 3/10 with a median gap of EXACTLY 0.000 — the system correctly
REFUSING.** It is pinned by the 150 ms per-hand floor (31 723 human gaps, p5 = 148 ms); a style
asking for faster bursts asks to swing faster than a human ever does. ⇒**The style space is bounded
by playability. Do not "fix" this by lowering the floor.**
⚠️**`travel` 7/10 = NOT RESOLVABLE at this n**, not refuted — one short of the bar, positive median
gap. More songs or seeds before a verdict.
⬜**Remaining**: 3 seeds per song (this was 1), and his ear on whether the presets are named right —
**the names are conventions over a continuum, not discoveries.**

## 📦 AWAITING KYLE'S EAR
- ★⚠️**`[CROSSOVER]` HAS NEVER BEEN PLAYED** — crossover 0.000 → 0.112 vs a human 0.183, `flow`
  0.37 → 0.23, and **`mapjudge` now ranks the CROSSOVER arms top of 34 independently.**
- ⬜**The autobuilt maps** (`outputs/autobuild_*.zip`, and the songset run in the scratchpad).
- ⬜**Two ML levers that passed their DoDs and were never flipped**: `BEAT_SUBDIV_AUTO=1`
  (**+0.222 vocal coverage at 49× the seed sd** on the 15 half-tempo songs it fires on, notes
  432→855, no precision cost, 0/121 false fires) and `--beat-threshold 0.25` (+0.029 at 8.6× sd,
  median 4.06 nps, 0/23 songs above the 6.18 he called unplayable).
- ⚠️**`[PHASE]` did not remove *"slightly off beat"*** ⇒ do not flip `BEAT_GRID_PHASE=search`.
- 🔴**The notesheet page player has still never been confirmed working in a browser.** One page was
  sent to him 2026-08-20 for its first render; no verdict yet.

## 🔴 P1 — THE SIX DEFECTS HE NAMED (2026-08-17): what is still open
| # | his words | still open |
|---|---|---|
| **D1** | *"very slow"* | = D4's budget lever. ⇒P0.5 + `--beat-threshold 0.25` |
| **D2** | *"slightly off beat"* | 🔴Refuted as note placement (our times match a human's better than two humans match each other). ⚠️**Untested**: we may be *too* quantised for a song with groove |
| **D3** | *"drops at wrong time"* | Confirmed (0.347 vs human-human 0.49). 🔴Its obvious fix is refuted — `structure.py` locates his moves *worse* than our map does |
| **D4** | *"not following the main vocals"* | ★**Biggest ML-side defect**: we play **0.385** of the sung line vs human **0.743**. Route is training-side (P1 below) |
| **D5** | *"random bursts"* | 🔴Refuted/inverted — the human bursts *more*. **Do not build a burst-suppressor** |
| **D6** | *"nps wasted on non main notes"* | = **C5 doubles**, priced at **21 %** of the vocal budget. ★Gated on **his** definition of "main" — change `overlay.MAIN_DEFAULT`, not the map |

## 🔵 P1 — W2 / W6, still open and NOT covered by D1–D6
- **W2** — Fallen Kingdom *"really empty"*. 🔴Cause unidentified; five instruments failed, and every
  one looked at notes. ⚠️**Ask him**: empty vs what our model *used to do*, or vs what the *song
  wants*?
- **W6** — **we ship one of the five elements human maps use.** Walls 93 % of human maps (median
  86, we emit 0), arcs 88 % of v3-capable maps, chains 50 % of v3. ✅All three now **built** as
  post-processors, notes byte-identical, installed as a ladder `[BASE]→[DENSER]→[WALLS]→[FULL]→
  [CHAINS]`. 🔴**The eval suite is BLIND to all three** (adding 84+48+16 moves every axis by
  **0.000**) ⇒ only his ear can judge them. ⬜`BEAT_SIM_CHAINS=1` must be flipped **and** the human
  reference recalibrated in the same change.
- **W5** — dot blocks as decoration: ⏸️**he deferred this himself.** Do not revive unprompted.

⚠️**Protect these — he named them by ear**: A6 hand-role division; the density pacing (*"when there
is a slow spot we let the player breathe"*); and *"there is a good deal of notes that are on beat
and I can tell play part of the song"* — **that half works.**

---

## 🔵 P2 — CARRIED FORWARD

### C1 — Precision sits at the greedy optimum; gains need better probabilities, not better picking
Three decode levers moved onset precision by nothing; the IOI prior moved it *down* to 0.769.
**Stop hunting decode knobs.** The ~10 correct-tempo alignment failures are a **pure selection
defect**, established by elimination — not tempo, not phase, **not onset supply** (4.5 onsets
available per note we emit), and **not difficulty**.

### C2 — Grid PHASE: resolved ON THE METRIC ONLY, and its successor suspect is REFUTED
`BEAT_GRID_PHASE=search` fixed ~18 of 39 failing songs by the alignment axis, and he still reported
*"slightly off beat"*. Tempo is refuted as the cause too: on all four maps he played the bpm is
**exactly** the human's and our note times match a human's **better than two humans match each
other**. ⇒**Do not flip it on the axis alone.** ⚠️Never apply a blanket global shift — that part is
an **onset-detector offset**, and "fixing" it is the `h_dist` failure.
★**2026-08-20 adds**: `mapjudge` cannot adjudicate this at all — its only response to a global shift
is `offgrid_frac`, which moves **by construction**.

### C3 — ★NOW PROMOTED TO P0.5. You cannot thin your way to human density
Humans at 3.9 nps have a pulse; we at 3.9 do not, and **2026-08-20 reproduced this on 23 maps built
with no ML in the path at all**. See P0.5.

### C4 — Beat-domain axes LIE on tempo errors
Every beat-domain axis buckets by the **map's own beats**, so on the **28 half-tempo songs** every
interval lands one bucket off. `idiom` showed it loudest (0.663 → 2.955 under `BEAT_SUBDIV_AUTO`,
**entirely an artifact** — rescaled to true tempo the timing distribution matches the human almost
exactly). ⇒**When a beat-domain axis moves on a cohort containing tempo errors, check whether the
BPM moved first.**

### C5 — Doubles: root cause found · decode fix FAILED · priced against D4
**Not too many notes — too few distinct times.** Stage-1's two hand channels correlate
**0.985–0.993**. **39.6 % of the notes we spend on the vocal line are doubles** onto an onset the
other hand already covered (human 20.7 %), so fixing doubling alone lifts vocal coverage
**0.385 → ≈0.487** with no extra notes. `BEAT_HAND_DEAL` hit every structural target and degraded
rhythm 6× ⇒ **not reachable by decode**; cost a **representational** fix instead.
★A **chain** is the human's alternative — one swing carrying 4–5 segments, *density with no new
distinct time*. `chains.py` builds them; they are installed and unjudged.



---

## 🧊 BACKLOGGED — ML PIPELINE (deprioritised by Kyle, 2026-08-20)
**Not dead, not being worked.** These are the model-training items; Kyle redirected the loop to the
agentic suite mid-session. Each keeps its measured evidence so it can be resumed without re-deriving
anything. ⚠️**Do not queue any of these from `/todo`.** If an agent-path item needs one of these to
progress, say so and stop — that is a decision for Kyle, not a silent re-prioritisation.

### 🧊 D4 — the ML generator does not follow the vocal line (training-side only)
Every alternative is eliminated by measurement: decode saturates (5× lower threshold buys **+4
notes**), the 1/4-beat grid is only **26–33 %** full, Track B at matched budget is **parity**, and
**Stage-1 does not modulate density per song at all (r = 0.046)** while crude audio features reach
**R² = 0.185**. We emit **0.217** positives per (slot,hand) against a corpus label mean of **0.245**
and these songs' humans at **0.294**.
⬜**PROPOSED RETRAIN, deliberately NOT queued**: an auxiliary per-song **density target** / FiLM
conditioning on the song's own label rate.
**DoD**: per-song nps correlation rises from 0.046 toward the demonstrated ≈0.43 floor **AND**
vocal coverage rises, at ≥3 seeds. ⚠️Must not regress the density he accepted (6.18 = unplayable;
current lever sits at 4.06).

### 🧊 TEMPO — priced, and smaller than its reputation
Right on **70.5 %** of songs (n=149). `BEAT_SUBDIV_AUTO` already recovers 16 of the 28 half-tempo
songs and is worth **+0.030 cohort-wide**; the remaining 12 are worth a further **+0.025** — about
**a fortieth of the human gap**. 🔴**Cheap detection is exhausted** (raw bpm AUC 0.978, 16/28 at
zero false fires; widening to bpm<110 costs 10 false fires for 8 songs). ⇒**Do not widen the
trigger.** ⚠️`notes per second` scores AUC 0.903 and catches **zero** at an affordable FP rate —
★*AUC is not an operating point.*

### Also backlogged
- **The two validated-but-unflipped ML levers** — `BEAT_SUBDIV_AUTO=1` (+0.222 vocal coverage at
  49x the seed sd on the 15 half-tempo songs it fires on) and `--beat-threshold 0.25` (+0.029 at
  8.6x sd). Both passed their DoDs; both change the **ML generator**, not the agent, so they wait.
- **C1 / C2 / C4 / C5 below** are ML-pipeline diagnoses, kept for their landmines only. C3 is the
  exception: it reproduced with **no ML in the path at all** and is live as **P0.5**.

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
