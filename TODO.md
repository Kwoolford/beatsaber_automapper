# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md); the agent-authoring trail is in
[`agent_mapper/PROGRESS.md`](agent_mapper/PROGRESS.md). Evaluation-suite rationale is in
[`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule:** when an item finishes, its *outcome and what it taught* moves to PROGRESS.md and the
item is **deleted** from here. A completed item is history, not work. Curated 2026-08-02 (from
4,076 lines) and again 2026-08-14 (from 652).

---

## 📍 CURRENT STATE (2026-08-27)

> ★★**FOCUS (Kyle, 2026-08-24):** *"a big focus on the manual agent building skill… beefing up the
> tooling so the agent calling the skill has as much visibility as a person playing and listening at
> the same time note sheet level… complete when you call the skill and are confident in the map you
> build and feel you can evaluate the map correctly and don't need to rely on me to audit."*
> ML work stays in `🧊 BACKLOGGED`.

**Build:** `python agent_mapper/autobuild.py <audio> --pulse --lead-bias 0.2 [--walls 89 --arcs 90
--chains 16]`, or the per-section loop in 📖`agent_mapper/WORKFLOW.md` (which beats autobuild).
**Judge:** `python scripts/audit_map.py <map.zip> [--audio <song>]` — every channel, one verdict.
**Read:** `python scripts/map_view.py <map.zip> --bars a-b --align --elements --stems`.
📖`agent_mapper/READING.md` (judge by reading) · 📖`LISTENING.md` (what to play).

**Measured:** 23/23 PASS on the songset · 0 parity violations · **573 tests**.

### ✅ THE VISIBILITY SUITE IS BUILT (2026-08-24, PROGRESS 2026-08-24f/g)
All six "what a player perceives" gaps are closed, each scored against a human corpus:
elements (**2 688** maps) · alignment · emptiness (**975**) · effort (**224 485** transitions) ·
6-stem lanes · one assembled verdict.
✅**DoD tested for real** on a COLD-CACHE song (`24e6c`): everything except *"is it fun"* was
judgeable alone — PASS p=0.438, 91.3 % on-sound, zero gaps, all elements in range.
🔴**What still needs Kyle: taste.** That is not a tooling gap.

★★**THE BAR IS THE SPREAD BETWEEN TWO HUMAN MAPPERS** (163 pairs), not the human map. Two humans
agree on the climax only **43 %** of the time — much of "different" is taste. Nothing we ship
exceeds 1.5× that spread.

### 🔴 The limits that bound all of it
1. 🔴🔴**A PASS DOES NOT MEAN THE NOTES ARE ON THE MUSIC** — the judge accepts **65 %** of maps a
   quarter-beat off. ⇒**Read the ALIGNMENT block of `audit_map` yourself.** (P0.2 needs Kyle.)
2. 🔴**A PASS = NOT DEFECTIVE, not GOOD**; a **FAIL can mean NOT TYPICAL, not bad** (P0.1).
   Never rank by `p` — human band 0.28–0.57; **high `p` means BLANDER**.
3. ⚠️**The 23 note metrics are still blind to walls/arcs/chains** — `audit_map`'s ELEMENTS block is
   what sees them now, but it scores *typicality*, not whether they feel good.
4. ⚠️**`onset_precision` scores our notes against OUR onsets**, both downstream of our grid, so a
   self-consistent-but-shifted grid can score well. The human-note-time check catches that on a
   corpus song and **does not exist on a new one**.
5. ⚠️**OUTPUT DIRS ARE NOT INTERCHANGEABLE** — name the directory, never just the arm.
   Newest: `outputs/phase_ab_2026-08-24/` · `outputs/width_ab_2026-08-24/` · `outputs/dod_test/`.

### ▶️ START THE NEXT SESSION HERE
1. 🔴**FOUR THINGS NEED KYLE'S EAR** — all installed and playable:
   `[V2]` vs `[FULL]` (walls/arcs/chains) · `[W3]` vs `[W5]` (cut directions, play `1f767` first) ·
   `[PBASE]` vs `[PCAL]` (grid phase, play Hunger first) · `AUTO Shut Up [DOD]` (cold-cache build).
2. 🔴**P0.1 and P0.2 are his decisions** — both change what a PASS means.
3. 🟡**DOUBLES: a real defect, now visible.** `double_share` **0.000** vs human median **0.144**;
   only **2.0 %** of human maps have none. See P1.3.

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

## ✅ P0.7 — CLOSED 2026-08-24 (see PROGRESS 2026-08-24)
Snap **validated as a real realignment** (human note times, +shifted/random controls), **generalised
to any song**, **priced** (`onset_precision` 0.866→0.891, and the recorded `nps` cost was an artifact
of the pre-two-stream build), and its obvious remedy — a **bpm-relative window — REFUTED** (keeps
only 73 % of the gain; the reselection IS the gain).
⬜**Only open**: `--snap-onsets` stays **opt-in**; DoD not met because PASS fell 23→21.
🔴**Do not rebuild**: bpm-relative window, or a "snap fixes cut directions" story (cohort: 0.000).

## ✅ P0.4 — ANSWERED AND SHIPPED 2026-08-24 (see PROGRESS 2026-08-24c)
**Our bar grid sits 0.053 beats EARLY of a human mapper's** (18/18 songs, against a reference our
detector never touched), and that is exactly the empirically optimal shift. Held-out validated
(`onset_precision` 0.888→0.917, human agreement 0.642→0.709, **99 % of the per-song oracle from one
constant**). Shipped as **`--phase-calibrate`**, `mapctl.PHASE_BIAS_BEATS = 0.053`.
🔴**DEFAULT OFF — Kyle's call.** Every note moves ~20 ms and this targets his *"slightly off beat"*
complaint directly. A/B installed: `AUTO <song> [PBASE]` vs `[PCAL]` — **Hunger 0.768→0.917**.
⚠️`PHASE_BIAS_BEATS` is a property of the CURRENT tempo/phase fit — re-derive with
`scripts/sweep_phase_heldout.py` if that fit changes.
⚠️**A weak `grid_r` does NOT imply bad alignment** (24e6c: r=0.32, phase 43 % of a slot from the
human's, yet 91.3 % on-sound) — **comparing grid ANCHORS is not comparing where notes LAND.**
🔴**Do not rebuild**: the negative-shift sign test (arms displace a full slot; it measures
quantisation), or a blanket "+0.05 everywhere".

## 🔴 P1.3 — THE TWO BUILD PATHS DISAGREE ON DOUBLES, AND THE HAND PATH SHIPS ZERO
**CONFIRMED — a defaults divergence, verified by running both:**

| | `--doubles` | `--accent-slots` | result |
|---|---|---|---|
| `autobuild.py` | **ON** (exposes `--no-doubles`) | `0,2,4,6,8,10,12,14` | 10.2 %–20.4 % ✅ human range |
| `mapctl auto` (what `WORKFLOW.md`/`SKILL.md` document) | **OFF** (`store_true`) | **`0,8`** only | **0.000** 🔴 |

⇒**Following the skill by hand ships a map with a whole gesture missing**, while `autobuild` on the
same repo lands in the human band (p25–p75 **0.089–0.212**; only **2.0 %** of 250 human maps have
none).
✅**Docs fixed 2026-08-24** — `WORKFLOW.md` and `buildmap/SKILL.md` now pass
`--doubles --doubles-rate 0.3`.
⚠️**PARTLY CONFIRMED, do not over-claim**: adding the flag **plus** `--accent-slots
"0,2,4,6,8,10,12,14"` on `24e6c` yields only **0.010** (2 doubles), not the human 0.137. The gate is
`slot ∈ accent_slots AND ≥2 stems agree`, so on a song whose stems rarely co-onset the doubles
cannot be placed at all. **Whether the agent path can reach human doubles on THIS song is NOT
established.**
**Tasks**: make `mapctl auto`'s defaults match `autobuild`'s so the paths cannot diverge again;
then measure `double_share` across the songset to see whether the `≥2 stems agree` gate is the real
ceiling.
**DoD**: a hand-built map following the documented command lands `double_share` inside p25–p75 on
the songset median, `viol` unchanged and `nps` not inflated — doubles must not buy density.
★**How this was found**: `audit_map`'s ABSENCE block flagged it, then Step-2 validation corrected the
cause **twice** — first from "generator defect" to "docs omission", then to "defaults divergence
plus a stem-agreement gate". ***A defect a tool finds still needs its cause verified before it is
written down.***

## 🟡➡️ P1.2 — MECHANISM FOUND, FIX BUILT, AWAITING KYLE'S EAR (PROGRESS 2026-08-24b)
**The `width` truncation strips the diagonals.** The candidate pool is already at the HUMAN diagonal
share (0.404 vs 0.415) on all 23 songs; `_pick` then keeps only the `width`(=3) highest-COUNT
candidates and human idiom frequency is vertical-dominated.
🔴**Both previously-named suspects are dead**: crossover filter **exactly 0.000**, `REPEAT_P` 40 % of
width's effect.
✅**`width=5` meets the DoD** (|Δdiagonal| 0.112) **and is closer to human on BOTH vocabulary axes**
than the default, which overshoots (`idiom_coverage` 0.990 vs human 0.909). 23 songs × 3 seeds.
🔴**Default stays 3 — Kyle's call**: it was chosen by READING two maps side by side, and an
eye-derived decision is not reversed on axis numbers. A/B installed: `AUTO <song> [W3]` vs `[W5]`,
identical times/notes/elements. ⇒**play `1f767` first — w5 overshoots there (0.223→0.492).**

## ✅ P1 — LEG 3: SECOND TRAVEL LEVER BUILT 2026-08-24 (PROGRESS 2026-08-24f)
**`--travel-target`** (default 4.167) re-weights candidates by HOW FAR each move travels, where
`width` sets how MANY are considered. On 1f767 `travel` **2.667 → 4.400 → 5.181** while
`angle_change` moves only 11.4 → 14.7 — the orthogonal pair LEG 3 needed.
⚠️`ebpm_burst` remains 0/10 by design — the 150 ms per-hand floor correctly refuses.
🔴**Do not re-attempt a width→STYLE mapping**; measured three times, dead.

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
- 🔴🔴**KNOBS ARE DEAD UNTIL PROVEN OTHERWISE — THREE IN ONE SESSION (2026-08-24).**
  `idiomize --width` (accepted, threaded, never used for weeks), `--travel-target` (added to
  `idiomize_zip`'s signature, never passed to `idiomize()` — caught because three arms printed
  IDENTICAL rows), and `mapctl auto --doubles` (works, but gated behind `--accent-slots` defaulting
  to `0,8` where `autobuild` uses eight positions, so it produced nothing).
  ★***Change the knob and DIFF THE OUTPUT before building on it. A knob whose arms agree to three
  decimals is not a weak lever, it is an unwired one.***
  ⚠️And when two entry points wrap the same engine, **diff their defaults** — `autobuild` and
  `mapctl auto` disagreed on both `--doubles` and `--accent-slots`.
- 🔴🔴**NEVER NAME A MODULE AFTER AN INSTALLED PACKAGE.** `agent_mapper/emptiness.py` was called
  `coverage.py` for one commit and broke **8 tests in four unrelated files**: `agent_mapper/` is on
  `sys.path`, so it SHADOWED the `coverage` package for every importer, and numba's
  `coverage_support` died on *"module 'coverage' has no attribute 'types'"*. **The traceback named
  numba and no file of ours.** Same trap for `types`, `json`, `parser`, `test`.
- 🔴**`pytest -q 2>&1 | tail -2` HIDES THE EXIT CODE** — the pipeline returns `tail`'s status, so
  `pytest … | tail && git commit` COMMITS ON A RED SUITE. It did, 2026-08-24. Redirect to a file and
  check `$?`. ★This is landmine "my filters hide the error I need to see", caught again.
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
