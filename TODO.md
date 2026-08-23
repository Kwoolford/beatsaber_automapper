# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md); the agent-authoring trail is in
[`agent_mapper/PROGRESS.md`](agent_mapper/PROGRESS.md). Evaluation-suite rationale is in
[`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule:** when an item finishes, its *outcome and what it taught* moves to PROGRESS.md and the
item is **deleted** from here. A completed item is history, not work. Curated 2026-08-02 (from
4,076 lines) and again 2026-08-14 (from 652).

---

## 📍 CURRENT STATE (2026-08-22, after the agent-suite session)

> ★★**FOCUS (Kyle, 2026-08-20):** *"focus solely on building an agentic map building suite… map any
> song you'd like and customize them to whatever style you want."* ML work is in `🧊 BACKLOGGED`.
> ★**2026-08-22 goal:** *"be confident people will want to play the maps."*

**One command:** `python agent_mapper/autobuild.py <audio> --pulse --lead-bias 0.2 [--walls 89
--arcs 90 --chains 16]`  → 📖`agent_mapper/WORKFLOW.md` · 📖`agent_mapper/READING.md` (how to JUDGE
by reading) · 📖`LISTENING.md` (what to play).

**Measured, 23-song songset, 23-metric judge:** 23/23 PASS · 0 parity violations · 565 tests pass.

★★**THE BAR IS THE SPREAD BETWEEN TWO HUMAN MAPPERS** (163 different-mapper pairs), not the human
map. Two humans agree on the climax only 43 % of the time — much of "different" is taste.

| property | ×human-human spread | |
|---|---|---|
| legibility / recurrence / crossover | 0.72 – 0.81× | ✅ inside |
| travel | 1.12× | 🟡 marginal |
| `vertical_share` | 1.32× | 🟡 marginal, largest non-circular gap |
| doubles | 1.40× | 🟡 near the taste floor |
| density | — | ceiling is the song's own event supply |

**Nothing exceeds 1.5×.** All five map elements now ship (notes + walls + arcs + chains + bombs-capable).

### 🔴 The limits that bound all of it
1. 🔴🔴**A PASS DOES NOT MEAN THE NOTES ARE ON THE MUSIC** — the judge accepts **65 %** of maps
   shifted a quarter-beat off. Read `onset_precision` yourself. ⇒**P0.2, needs Kyle.**
2. 🔴**A PASS = NOT DEFECTIVE, not GOOD**; a **FAIL can mean NOT TYPICAL, not bad** (P0.1).
   Never rank by `p` — human maps score p median 0.455, ours 0.5–0.75; **high p means blander**.
3. 🔴**Walls, arcs and chains are invisible to all 23 metrics** — only his ear can judge them.
4. 🔴**Nothing here has reached his ear.** Every number above is a number.
5. ⚠️**THE OUTPUT DIRS ARE NOT INTERCHANGEABLE — check the date before quoting or queueing one.**
   `outputs/review_2026-08-22/V2__*` (newest, **the build to judge**), `outputs/final_verify2/`
   (2026-08-21), `outputs/kyle_review_2026-08-19/*_FULL` (2026-08-19). On the same 4 songs these
   differ **2–4×** on a note-level property (measured 2026-08-22 via `diag_pendulum`), so a
   number taken from one **does not describe** another. ★Name the directory, never just the arm.

### ▶️ START THE NEXT SESSION HERE
1. ⬜**His verdict on `[V2]` vs `[FULL]`** (Fallen Kingdom) — see `LISTENING.md`. Settles a
   three-day-old question no metric can touch.
2. 🔴**P0.1 and P0.2 are his decisions** — both change what a PASS means.
3. 🟡**P0.4** — a grid-phase change reliably selects a better map on 16/23 songs, cause unknown.
4. ⬜**Approve the map cleanup** in `LISTENING.md` — 78 arms are installed, ~40 superseded.

---

## 🔴 P0.1 — THE JUDGE CANNOT VALIDATE A DELIBERATELY-HARD MAP (needs Kyle's call)
`--nps 9.0` on `1f65d` produces a map whose only problem is **`nps` at the 97.1st human
percentile** — 0 violations, nothing else out of range. The gate asks *"is this typical?"*, so a
map Kyle would enjoy is one it calls defective.
**Human corpus (n=1100):** 5th 2.73 · median 4.16 · 95th 6.77 · 99th 8.53 nps; the gate starts
rejecting near **7.18**.
★**His stated taste is the conflict**: *"I like playing fast songs… map whatever difficulty we
want."* ⇒**A FAIL here means NOT TYPICAL, not BAD** — the mirror of the standing rule that a PASS
means not-defective rather than good.
**Options** (his choice, not mine): (a) keep gating `nps` against the corpus; (b) exempt
`nps`/`peak_nps` when a density is explicitly requested; (c) gate conditioned on the requested
difficulty. ⚠️**(b) is easiest and most dangerous** — it removes the only guard against runaway
density, and 6.18 nps is the number he once called unplayable.
**DoD**: a requested-difficulty map is judged on everything EXCEPT the density it was asked for,
without that exemption silently applying to maps where no density was requested.

## 🟡 P0.2 — SOLVED IN PRINCIPLE, ONE DECISION LEFT (Kyle's)
**The judge accepts 65 % of maps a quarter-beat off the music** (`offbeat` control; 21 of 23 metrics
score AUC 0.500, only alignment sees it). ✅**The fix is an UNDILUTED alignment floor** — the one
thing pooling structurally cannot do.
**Measured trade (real combined gate, n=200):**

| alignment floor | human | `offbeat` |
|---|---|---|
| none (today) | 0.870 | 0.650 |
| 93rd pct | 0.850 | 0.315 |
| **90th pct** | **0.825** | **0.080** |

⇒**A floor at the 90th human percentile takes `offbeat` 65 % → 8 %, and costs human accept
0.870 → 0.825.** My bar was human ≥0.85, so it misses by **0.025**.
★★**THAT 0.025 IS THE DECISION**: reject ~17.5 % of human maps instead of 13 %, to catch 92 % of
off-beat maps instead of 35 %. ⚠️His *"target is the best mappers"* makes the corpus median a FLOOR,
so rejecting a few more median-ish human maps may be right — **but it changes what a PASS means, so
it is his call.**
🔴**FOUR ALTERNATIVES ELIMINATED** (all worse than today's gate on `offbeat`): per-metric bound,
per-axis mean, per-axis max, **Fisher / higher criticism**. ★The current aggregate beat every
replacement I built — the fix is to ADD a term, not replace the combination.
⚠️**Not Goodharting**: alignment answers a *qualitatively different* question the other 21 metrics
cannot; `alignment.py` records five axes passing a map 5/5 while it sat off the beat.

## 🟡➡️ P0.6 — HAND ROLE: LEVER CONFIRMED, OPERATING POINT UNDER-CONSTRAINED
✅**The lever is real at 3 seeds**: `role_asymmetry` arm gap 60.8 pts vs a 47.5-pt seed spread;
`handedness` 3.1 → ~32 (gap 2.7× spread). `mapctl auto --lead-bias 0.3`, default 0.
⚠️**But the OPERATING POINT is not reliable**: at bias 0.3 `role_asymmetry` lands between the **37th
and 85th percentile** depending on seed. The "39.6 %, human median exactly" I reported was **seed 0,
the lowest of three**. Honest claim: *moves it into the human body*, not *onto the median*.
🔴**A cost I reported is REFUTED**: `role_swap_rate` 58 → 81 % is **inside the seed spread**
(gap 8.8, spread 10.1) ⇒ not a result. Do not treat it as a trade.
✅**DONE — `--lead-mode cyclic` is the default and the seed spread is 0.0.** Re-tuned:
**`--lead-bias 0.20`** (not 0.30 — that was tuned against the sampled lead and overshoots to the
77.9th percentile under `cyclic`). ★**An operating point is not portable across a change in how the
knob works.** 🔴0.30 has the higher `p` median and is still the wrong choice: `p` is a
distance-from-typical and ranking by it Goodharts toward the average map.

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
**0.856 → 0.890** (human 0.919), p median 0.655 → 0.680, ng every alignment number recorded so far. **A deliberate decision, not a chore.**
⬜**Generalise `--snap-onsets` to any song** — it needs cached reference onsets, which exist only for
corpus songs, so it cannot be the default while "map any song" is the goal.
⚠️Cost: `nps` 3.43 → 3.29; snapping collapses events sharing one onset (88 of 833 on 1f767).

## 🟡 P0.4 — A DIFFERENT GRID PHASE SELECTS A BETTER MAP (cause unknown)
**Effect CONFIRMED n=23**: shifting the bar grid +0.05–0.10 beats improves `onset_precision`
0.866 → 0.916 **and** agreement with the human's own note times 0.645 → 0.686, on **16/23 songs**,
inverted-U on both references.
🔴🔴**BUT IT IS NOT A REALIGNMENT — RETRACTED.** Only **45.9 %** of note beat positions survive the
shift and counts move ±14 %: the shift changes **which events snap to which slot**, so this is two
different maps rather than one map realigned. **"Every note is placed early" was wrong.**
🔴**And the phase criterion is not biased**: `R`-optimal vs minimum-snap-distance-optimal differ by
a median **signed** +0.002 beats (random directions). ⇒**a fix aimed at the phase estimator would be
aimed at nothing.**
⬜**What is left**: a reproducible grid-dependent selection effect with **no known mechanism**. Worth
understanding before acting — a per-song phase search is the obvious next probe, but it optimises a
selection rather than an alignment, which is a different thing to justify.
⚠️`--phase-shift` stays TEST ONLY.

## 🟡 P1.2 — WE USE SIX CUT DIRECTIONS, HUMANS USE NINE
**Largest non-circular gap** (1.32× the human-human spread — marginal, not urgent).
`vertical 0.773 vs 0.480 · diagonal 0.223 vs 0.415 · horizontal 0.004 vs 0.094`.
🔴**Four explanations refuted**: the vocabulary HAS the diagonals (34.4 % stationary); 0 fallbacks;
`_reparity` changes 0 directions; doubles are not the cause (doubles and solo notes have the same
diagonal share). ⇒**the loss is inside the sampler and is NOT yet explained.**
★Reachability gradient: from a vertical only 18–31 % of reachable idioms are diagonal; from a
diagonal, 42–66 %. **Diagonals sustain diagonals — the question is why we never accumulate them.**
**Next candidates** (untested): the crossover filter and `REPEAT_P`, both of which re-weight after
the `d_from` match.
🔴**RETRACTED: the horizontal gesture is NOT a miss.** The 9.4 % figure came from ONE song; across
109 human maps it is **3.2 %**, and on the 23-song cohort human 0.019 vs ours 0.001 = **|Δ| 0.016**,
negligible. ★*The page proposes, only the cohort disposes* — generalised from one song again, one
iteration after writing that rule down.
🔴**AND THE DIAGONAL GAP IS INSIDE THE HUMAN SPREAD**: |Δ| 0.099 vs a 0.106 spread over 163 pairs.
⇒**The ONLY real cut-direction quantity is vertical over-use** (|Δ| 0.164 vs spread 0.125 = 1.32×,
marginal). Do not build a horizontal or diagonal lever.
**DoD**: diagonal share inside the human-human spread (|Δ| ≤ 0.125) without `viol` rising.

## 🟢➡️ P1 — LEG 3: 4 OF 6 ORDERINGS, AND `technical` IS OVER-SPECIFIED
🔴**`travel` and `angle_change` anti-correlate on `width`** (width 1 → p51/p1; width 12 → p19/p73)
**and `technical` asks for both at p75.** Three width mappings measured on 10 songs: none beats
leaving `width` unset (4 of 6). ⇒**needs a SECOND lever that moves travel without touching cell
variety** — lane spread is the candidate. Do not re-attempt a width mapping; it has been measured
three times.
⚠️`ebpm_burst` remains 0/10 by design — the 150 ms per-hand floor correctly refuses.

## ⚠️ SEEDS ON THE AGENT PATH — READ BEFORE QUOTING ANY "n SEEDS" NUMBER
**10 of 23 metrics are seed-INVARIANT by construction** — every time-domain one (`nps`, `peak_nps`,
`pulse_stability`, `dominant_share`, `ioi_*`, `ebpm_burst`, `onset_precision`, `offset_mad_ms`,
`offgrid_frac`). The agent builds from **cached events**, so note TIMES are deterministic.
★**This is the opposite of the ML path**, where a seed re-draws the Demucs stems.
⇒**Running 3 seeds for a time-domain metric is wasted GPU and a false sense of rigour.** Seeds matter
only for **geometry and hand-role**.
✅**Fixed 2026-08-21**: `--seed` reached `idiomize` only, never `mapctl auto`, so the lead-hand RNG
always ran at seed 0. Hand-role numbers recorded before this are single-seed.

## 📦 AWAITING KYLE'S EAR
- ★⚠️**`[CROSSOVER]` HAS NEVER BEEN PLAYED** — crossover 0.000 → 0.112 vs a human 0.183, `flow`
  0.37 → 0.23, and **`mapjudge` now ranks the CROSSOVER arms top of 34 independently.**
- ⬜**The autobuilt maps** (`outputs/autobuild_*.zip`, and the song (our times match a human's better than two humans match each other). ⚠️**Untested**: we may be *too* quantised for a song with groove |
| **D3** | *"drops at wrong time"* | Confirmed (0.347 vs human-human 0.49). 🔴Its obvious fix is refuted — `structure.py` locates his moves *worse* than our map does |
| **D4** | *"not following the main vocals"* | ★**Biggest ML-side defect**: we play **0.385** of the sung line vs human **0.743**. Route is training-side (P1 below) |
| **D5** | *"random bursts"* | 🔴Refuted/inverted — the human bursts *more*. **Do not build a burst-suppressor** |
| **D6** | *"nps wasted on non main notes"* | = **C5 doubles**, priced at **21 %** of the vocal budget. ★Gated on **his** definition of "main" — change `overlay.MAIN_DEFAULT`, not the map |

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
- 🔴**DO NOT BUILD A "PENDULUM LOCK" AXIS (tested 2026-08-22).** Two-state alternations
  (one hand oscillating between exactly two `(x,y,dir)` states) are real and readable in our
  maps — `map_view --bars 105-108 --idioms` on 1f333 shows ~8 beats of it — but **humans
  produce them at the same rate**: our median share sits at human **percentile 0.57**, 0/23
  above p90. A zero-spike looked decisive (29 % of humans emit zero, 0/23 of ours,
  p=0.0025, survives a sparsity control) but **the gap exists only at the run-length
  threshold I picked and reverses sign by MINRUN=10**, where our maps are cleaner than
  human. Tool kept at `scripts/diag_pendulum.py` as a reading aid, **not** as an axis.
  ★Generalisable: sweep the free parameter before believing any threshold-defined metric.
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
