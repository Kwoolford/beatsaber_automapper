# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md); the agent-authoring trail is in
[`agent_mapper/PROGRESS.md`](agent_mapper/PROGRESS.md). Evaluation-suite rationale is in
[`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule:** when an item finishes, its *outcome and what it taught* moves to PROGRESS.md and the
item is **deleted** from here. A completed item is history, not work. Curated 2026-08-02 (from
4,076 lines) and again 2026-08-14 (from 652).

---

## 📍 CURRENT STATE (2026-08-14)

📖★**START HERE: [`docs/overnight_2026-08-14.md`](docs/overnight_2026-08-14.md)** — the brief,
leading with the questions that need Kyle's ear.

**Promoted**: the 2026-08-03 baseline (8 defaults, `docs/BASELINE_2026-08-03.md`). Unchanged.
**Built, validated, and default OFF — all three await his ear**:

| lever | evidence |
|---|---|
| `COLOR_SEP_MODE=extreme` | crossover 0.000 → 0.112 (human 0.183); flow 0.37 → 0.23 |
| `BEAT_STRUCTURE_REUSE=diag_full:…:0.20` | ~45–51 % of the `rhy_rhythm`/`harm_rhythm` gap, 2 seeds |
| `BEAT_GRID_PHASE=search` | songs >0.10 below human **39 → 21**; **alignment axis 0.62 FAIL → 0.35 PASS**, first ever pass; 74 better / 0 worse |
| `BEAT_SUBDIV_AUTO=1` | fired on 15 songs with **zero false positives**; burst ceiling 0.500 → 0.958 |

**Song names**: `1f333`=Hunger · `1f8d6`=Fallen Kingdom · `1f913`=Digital Life Hacker ·
`1f767`=アリスブルー · plus SO TIRED ROCK.

---

## 🔴🔴🔴 P0 — AGENT MAP AUTHORING (`agent_mapper/`) ★KYLE'S PRIORITY, 2026-08-14

> *"I'd really like to see an agentic way to manually build a map… if you or another LLM has a
> longitudinal view with notes by breakdown and importantly with when lyrics are said you could
> create some amazing maps."*

**Why it is P0 and not a side quest.** W1, W4 and `follow_vocals` are **one defect**: Stage-1's
representation does not carry the melodic instruments. An agent reading timestamped lyrics and a
per-stem timeline **does not have that defect**, so this answers a question the ML track cannot:
*if the model could hear the vocal line and see the whole song, would the maps be good?* A good
hand-built map justifies Track B by demonstration; a bad one says the problem is elsewhere.

**Built** (`agent_mapper/`, workflow in `WORKFLOW.md`, skill `/buildmap`):
`brief.py` (song as a text score + **lyric repeat map**) · `lyrics.py` (Whisper on the separated
vocals) · `mapctl.py` (`init/auto/add/view/check/status/clear/export`).
**First map**: Hunger, 1 261 notes from a 20-line plan, `ebpm_burst` **376 = human exactly**,
nps 4.66, zero parity violations. Installed as `AUTO Hunger [AGENT]`.

### Tasks, in order
1. ★**Get his verdict on `AUTO Hunger [AGENT]`.** Everything else here is unfalsifiable until a
   human plays it — onset precision is **circular** (`auto` places notes on the onsets the metric
   scores), so the suite cannot judge this map. Record with `scripts/record_verdict.py`.
2. **Close the two measured gaps**: `travel` **4.60 vs human 12.53** (hands barely move — `auto`
   uses two columns and two rows per hand) and `double_share` **0.034 vs 0.146** (needs a broader
   accent model than "strong beat with ≥2 stems agreeing").
3. **Map a second song**, ideally one with a different shape (Fallen Kingdom — he called it
   *"really empty"*, so it is the sharpest test of whether a longitudinal view fixes emptiness).
4. **Add a `--crossover` placement mode** — human maps cross hands over on ~18 % of notes and
   both our generator *and* `auto` emit exactly zero. It would move `travel` too.
5. **Teach `auto` the lyric repeat map**: map a chorus once and reuse it deliberately, which is
   what `BEAT_STRUCTURE_REUSE` infers from an SSM and an agent can simply read.

**DoD**: Kyle plays an agent-built map and says it is better than the generator's on the same
song. That is the only bar that matters here; every suite number on these maps is either
circular or known not to track his ear.

⚠️**Landmines already paid for** (details in `agent_mapper/PROGRESS.md`): a guard that checks only
one neighbour looks applied and does nothing (70 floor violations survived, metric unmoved);
`postprocess.fix_parity` already solves parity — hand-rolling it cost 380 notes and still left 5
violations; and doubles cannot be bolted onto a dense pass because both hands are always busy —
**place accents first, then fill**.

---

## 📦 AWAITING KYLE'S EAR — 3 review sets, 33 maps installed

| set | maps | the question |
|---|---|---|
| **A** structure + crossover (`docs/review_2026-08-11.md`) | `[BEFORE]/[CROSSOVER]/[AFTER CAPPED]/[BOTH]/[AFTER]` × his 4 songs | ★Play `[BEFORE]` vs `[BOTH]`: **does the repetition read INTENTIONAL or LAZY?** And do the crossovers play better? |
| **B** grid phase (`docs/review_2026-08-14.md`) | `[BEFORE]/[PHASE]` × 6 corpus songs | ★Lead with **BEcause**: does `[PHASE]` sit on the beat better? **"Can't tell" is a real answer.** |
| **C** agent-built | `AUTO Hunger [AGENT]` | Is it better than the generator's Hunger? |

🔴**His 4 standing songs are absent from set B by measurement, not oversight** — the lever left
them byte-identical, so there was nothing to hear. ⚠️**Check that a lever changes those 4 maps at
all before building a review set on them**; they are a well-behaved sample and a lever aimed at a
defect they lack is invisible there.

**Also still open from 2026-08-04**: is Fallen Kingdom empty vs what our model *used to do*, or vs
what the *song wants*?

---

## 🔴🔴 P0 — "THE METRICS STILL DON'T CAPTURE THE FULL PICTURE" (Kyle, 2026-08-10)

The masterpiece axes rank the map he called *"really empty"* **second-best** and the one he graded
**A+** **fifth-worst**. Three measured senses in which the suite misses: it is at a coin flip per
axis (13/26 on his one known verdict), nearly **blind to placement** (M-E rewrote 25 % of note
positions and 12 of 15 axes moved by +0.0000), and **the six-axis suite cannot score a single map
at all** — they are cohort statistics and return `nan` on one map.

⇒**The answer is a different SOURCE OF TRUTH, not a seventh metric** — building metrics from first
principles is what produced the anti-correlation.
✅**The blocker is removed**: `scripts/record_verdict.py` + `docs/eval_references/preference_verdicts.json`
now exist, so verdicts accumulate instead of living in prose. `preference_screen.py` reads them.
**DoD**: an axis (or weighting) that reproduces his ordering on held-out pairs above chance. Until
something clears that, no axis may be called a quality metric — only a defect detector.

---

## 🔵 P1 — TEMPO, the largest quantified upstream defect

**Our tempo is right on 70.5 % of songs** vs the mapper's declared bpm (n=149) — the old "30 %
wrong" confirmed on a 6× larger cohort. ✅Half-tempo is **handled** (`BEAT_SUBDIV_AUTO`; the `half`
group is now the *best-performing* group at 0.2× the base alignment failure rate).
🔴**Open: the 2:3 / odd-ratio misreads** — `three_halves` fails alignment at **4.6×** the base rate
and `other` at **5.5×**, together 48 % of the remaining failures from 11 % of the cohort. A
half-tempo error gives a coarse but *aligned* grid; a 2:3 misread puts beats in the **wrong
places** and drifts, so no global shift helps.
⚠️**Every cheap route is closed**: onset-energy balance (2026-07-27, made it worse), onset-gap
density, ACF periodicity, a cross-validated tempogram classifier, and the `fit_tempo` tie-break
(+1 song of 149). ⇒**What remains is a real tempo model**, with free supervision from the declared
bpm in 5,373 corpus maps.

## 🔵 P1 — TRACK B: three defects are one defect

**W1** (can't find the tempo-carrying instrument), **W4** (phrases abandoned mid-vocal, 2.75× the
human at n=123) and **`follow_vocals`** (0.020 vs 0.149, **7×**) are three views of one cause:
**Stage-1 carries only `drum_proj` + `mix_proj`** and cannot hear the melodic instruments.
★The clincher: redistributing the note budget toward an abandoned phrase **does not fill it**
(γ 2.5 → 1.0 closes ~17 %, note count unchanged) ⇒**you cannot select what the model does not
propose.** ⚠️Does *not* revive v8 as built (its `follow_vocals` gain died at n=149) — it says the
**target** is right. Use W1 + W4 + `follow_vocals` as **one** acceptance target, not three.

---

## 🎯 W1–W7 — KYLE'S OBJECTIONS *(evidence in PROGRESS.md)*

★**His standing instruction**: *"tread carefully, make isolated and tactical changes, and document
like crazy."* One lever at a time, ≥3 seeds, nothing promoted without his ear.
⚠️He declined to name exemplary mappers *yet* — the best-mapper cohort is blocked **by his choice**.

| # | complaint | status |
|---|---|---|
| **W1** | can't find the core tempo/instrument | 🔴**OPEN → Track B.** The real defect is we play the OFFBEAT (`halfbeat_rate` 0.245 vs 0.095); a selection defect grid phase cannot fix. |
| **W2** | Fallen Kingdom *"really empty"* | 🔴**CAUSE UNIDENTIFIED**, five instruments have failed. ⚠️**ASK HIM** (above). |
| **W3** | *"parts get really intense"* | **PARTLY CONFIRMED** — C5 wearing a hat. Any difficulty axis must count **notes**, not events. |
| **W4** | phrases abandoned mid-vocal | ✅**CONFIRMED n=123 and it GREW** (0.500 vs 0.182; 109/123 paired) 🔴density weighting **refuted** as the cause ⇒**Track B**. |
| **W5** | dot blocks decorative | ⏸️**he deferred this himself.** |
| **W6** | multi-note swings missing | 🟡**missing capability** — right answer for grand low-density drops. **Untouched; a good `agent_mapper` target.** |
| **W7** | last note didn't line up | ✅**FIXED** — `BEAT_END_RESOLVE=0.75`, 0.153 → 0.014 at no cost. Default OFF, awaiting his ear. |

⚠️**Protect these — he named them by ear**: A6 hand-role division, and the density pacing
(*"when there is a slow spot we let the player breathe"*).

---

## 🔵 P2 — CARRIED FORWARD

### C1 — Precision sits at the greedy optimum; gains need better probabilities, not better picking
Three decode levers moved onset precision off ~0.90 by nothing (density, γ allocation, a
probability floor); the IOI prior moved it *down* to 0.769. **Stop hunting decode knobs.**
✅**2026-08-14 sharpened it**: the ~10 correct-tempo alignment failures are a **pure selection
defect**, established by elimination — not tempo, not phase, **not onset supply** (4.5 onsets
available per note we emit), and **not difficulty** (the human scores 0.943 on exactly those songs
vs 0.934 on the ones we handle fine).

### C2 — Grid PHASE ✅ largely resolved
`BEAT_GRID_PHASE=search` fixed ~18 of 39 failing songs. Remaining phase work is inside the tempo
item above. ⚠️Never apply a blanket global shift — on `1f767` the human wants the same shift we do,
so that part is an **onset-detector offset**, and "fixing" it is the `h_dist` failure.

### C3 — Density/rhythm tension: you cannot thin your way to human density
Re-tuning toward the human note rate costs rhythm (`pulse_stability` −0.06 → −1.11). Humans at
3.9 nps have a pulse; we at 3.9 (thinned from 4.4) do not. `BEAT_IOI_PRIOR` made it worse. Needs a
different idea, not another sweep.

### C4 — Every beat-domain result predates the tempo fix
A2 rhythm, A6 handrole and the hand-offset work were tuned against a bpm that was wrong on 20 of
21 songs. Re-derive before building further on them.

### C5 — Doubles: root cause found; 🔴decode fix FAILED, do not revive
Not too many notes — **too few distinct times** (467 vs the human's 626). Stage-1's two hand
channels correlate **0.985–0.993**, so both hands get the same information and pick the same slots;
66 % doubles is structurally guaranteed. `BEAT_HAND_DEAL` hit every structural target and degraded
rhythm 6× — **C5 is not reachable by decode.**
★**2026-08-14 adds the flip side**: doubles are how a human buys density *without* speeding either
hand up (the human plays 8.35 nps at the same `ebpm_burst` as our 4.00).

### C6 — `outputs/` is gitignored: one decision still owed
All calibration references are snapshotted to tracked `docs/eval_references/`. ⚠️It is a **copy**,
so re-copy whenever a reference changes. **Decision owed**: move the live path into version
control, or keep copy-and-remember.

---

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

### Habits that outlived the seed lottery
1. **Score every arm at ≥3 seeds and quote the sd.** ⚠️n=3 *underestimates* sd — treat it as a screen.
2. **`npass` is not a ranking statistic** (an identical config scored 4, 4, 2). Rank per-axis with error bars.
3. **Open**: the spread bar (0.35) sits inside the noise — stop gating on it, keep a hard alarm near 0.15.
   Not done unilaterally; it changes scorecard semantics.
