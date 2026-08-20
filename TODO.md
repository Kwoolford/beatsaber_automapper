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
1. 🔴🔴**Wire `alignment` into `mapjudge`.** Run **`bash scripts/build_onsets_calib_spans.sh`**
   (~55 min GPU), then `python scripts/calibrate_mapjudge.py --n 1100`, then
   `python scripts/audit_mapjudge.py --n 250`. Dual calibration sets are **already plumbed** (21 vs
   23 metrics must not share one). **State: DIST 907/1415 cached, CALIB 10, HELDOUT 6.**
2. ★★**Attack the pulse defect (P0.5)** — the biggest measured defect the agent path has.
3. ⬜**Ask him to play an autobuilt map.** They are unheard. Suggest 1f767 or 1f8d6 against the
   human map. Record with `scripts/record_verdict.py`.
4. ⬜**`[CROSSOVER]` is still unjudged** — and `mapjudge` independently ranks the CROSSOVER arms
   **top of 34**. Oldest open question on the board, now with a second line of evidence.
5. ⬜**Ask him for the best-mapper list** — blocked by his choice, and leg 3 needs it to aim at
   anything above the corpus median.

---

## 🔴🔴 P0 — THE AUDIO AXIS (blocking everything music-relative)
**Evidence**: five of six axes were once blind to a map sitting off the beat, and A8 was added the
day he said *"it's painfully obvious the notes are off beat"*. `mapjudge` currently repeats that
hole: 21 metrics, none of which load the audio.
**Tasks**: cache the CALIB/HELDOUT onset spans → recalibrate → re-audit → confirm the audio-mode
p-value is not coarse (needs ≥100 calibration maps with onsets; it warns if not).
**DoD**: `mapjudge` reports 23 metrics without the `⚠️NO AUDIO AXIS` banner, human accept stays
≥0.85, all eight controls stay ≤0.10, **and the two timing controls are still rejected**.

## ★★ P0.5 — OUR MAPS DO NOT HOLD A PULSE (the biggest agent-path defect)
**Evidence (n=23 autobuilt songs, PROGRESS 2026-08-20 §8)** — four metrics, one story:
`pulse_stability` **0.329 vs human 0.560** · `dominant_share` **0.362 vs 0.512** ·
`ioi_switch_rate` **25.9 vs 13.5** · `ioi_cond_entropy` **0.680 vs 0.536**.
★**This is C3 arriving from a completely different direction** — no Stage-1, no decoder; note times
come from **merging two accent-filtered event streams**, and **the union of two rhythms is not a
rhythm**. ⇒C3 is a deep property of how we choose note TIMES.
**Tasks**: pick a subdivision per section and *hold* it, breaking it deliberately, instead of taking
the union of two independent streams. The typed events now make "which stream defines the pulse" a
choosable thing.
**DoD**: `pulse_stability` and `dominant_share` medians move inside the human interquartile range
across ≥20 songs **without** `nps` falling below the p20 the judge accepts.

## 🟡 P1 — LEG 3: style is real but weak
**Evidence**: all four ordering checks hold, but **n=1 song, 1 seed**, and `dense` reaches only
4.10 nps against a human median 4.17. `ebpm_burst` is identical (320) across all five styles —
it is set by the 150 ms per-hand floor and **no style knob moves it**.
🔴**Named style CLUSTERS are REFUTED** (silhouette 0.153 vs null 0.105, n=1098) — style is a
**continuum**; do not rebuild clusters.
**Tasks**: replicate the ordering checks across ≥10 songs × 3 seeds; find why density saturates
(the budget is not event-supply-limited — only 3 of 16 accent slots were at "keep all").
**DoD**: each ordering holds on ≥8 of 10 songs, and `dense` clears the human median nps.

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

## 🔴 P1 — D4's ONLY REMAINING ROUTE IS TRAINING-SIDE
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

## 🔵 P1 — TEMPO: priced, and smaller than its reputation
Right on **70.5 %** of songs (n=149). `BEAT_SUBDIV_AUTO` already recovers 16 of the 28 half-tempo
songs and is worth **+0.030 cohort-wide**; the remaining 12 are worth a further **+0.025** — about
**a fortieth of the human gap**. 🔴**Cheap detection is exhausted** (raw bpm AUC 0.978, 16/28 at
zero false fires; widening to bpm<110 costs 10 false fires for 8 songs). ⇒**Do not widen the
trigger.** ⚠️`notes per second` scores AUC 0.903 and catches **zero** at an affordable FP rate —
★*AUC is not an operating point.*

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
