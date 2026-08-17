# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md); the agent-authoring trail is in
[`agent_mapper/PROGRESS.md`](agent_mapper/PROGRESS.md). Evaluation-suite rationale is in
[`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule:** when an item finishes, its *outcome and what it taught* moves to PROGRESS.md and the
item is **deleted** from here. A completed item is history, not work. Curated 2026-08-02 (from
4,076 lines) and again 2026-08-14 (from 652).

---

## 📍 CURRENT STATE (2026-08-17)

★★**Kyle has judged the maps and the answer is a DEFECT LIST, not a preference.** He
reviewed the agent map, set A and set B and reported, across *every* song:

> *"it varys from very slow, slightly off beat, doing drops at the wrong time, not
> following the main vocals or having random bursts of really fast non flowy notes…
> **I think the nps is generally wasted on every few non main notes**… there is a good
> deal of notes that are on beat and I can tell play part of the song, but **they aren't
> hitting that main flow that mappers can generally see**."*

And the directive that follows from it:

> *"I think we need to make the visibility suite the top priority… so visibly great that
> you can go back through and evaluate through me instead of making another evaluation
> metric."*

🔴**Three levers with strong numeric evidence have now failed to reach his ear**
(`COLOR_SEP_MODE` flow 0.37→0.23; `BEAT_GRID_PHASE` 74 better/0 worse; the agent map at
human `ebpm_burst` and human nps). ⇒**A passed DoD is evidence about the METRIC, not
about the map.** The answer is not a seventh axis — it is to make the map legible enough
that *he* is the evaluator. That is what P0 now is.

⚠️**And the review framing was wrong.** The A/B pipeline collects *preferences between
arms*; he produces *defects located in songs*. Preference is second, defects are first.

---

## 🔴🔴🔴 P0 — THE VISIBILITY SUITE ★KYLE'S PRIORITY, 2026-08-17

**The goal, in his words: "so visibly great that you can evaluate through me."** Every
item below is judged on whether it lets him point at a moment and name what is wrong —
not on whether it produces a number.

**What already exists to build on**: `melody.py` (pitch per onset), `percussion.py`
(kit labels), `structure.py` (sections + which repeat, CONFIRMED held-out p=0.019),
`brief.py` (per-bar stem grid + lyrics), `map_view.py` (map as text), ArcViewer (play
preview). ⚠️**None of it is time-aligned into one picture**, which is the whole gap.

### V1 — The NOTESHEET: the song as a readable score
One time axis, one lane per voice: vocal pitch line + lyric, bass line, the `other`
lead/chords, four kit lanes, with the section banner above it.
- ⚠️**Terminal text will not carry this.** 4 minutes × 6 lanes does not fit usefully;
  this must render (HTML/SVG page or PNG), because the whole point is that he looks at it.
- ❌**`bass` has no transcription at all** — build it first, it is the cheapest possible
  win (bass is monophonic, so pYIN works directly, exactly as for vocals).
- ⚠️`other` is currently ONE salience peak per frame, not chords. Real polyphony needs a
  model — `basic-pitch` installs here via its ONNX backend (the TF wheel blocks Python
  3.12; `--no-deps` + onnxruntime resolves). **De-risk with a PoC on one song first.**
**DoD**: he opens one page and can read what the song is doing without the audio.

### V2 — ★THE OVERLAY: our map drawn ON the notesheet
The single most important view, because it makes his central complaint visible without
inventing a metric for it. Every placed note is drawn against the musical event under it
and falls into exactly one of three classes:
- **HIT** — a note on a main musical event
- **MISSED** — a main event with no note ⇒ *"not following the main vocals"*
- **WASTED** — a note with no main event under it ⇒ ★*"the nps is generally wasted on
  every few non main notes"*
**DoD**: he can look at one section and agree or disagree with the three colours. If he
disagrees, the *definition of "main"* is what is wrong — and that is a far better thing
to argue about than an axis score.

### V3 — The FLOW view: play-level clarity
Hand paths over time (L/R column+row), cut directions, and **bursts marked where they
happen**. His flow complaint has been unlocatable so far: *"random bursts of really fast
non flowy notes"* names a symptom with no timestamp.
**DoD**: given his complaint, we can point at the bar it refers to.

### V4 — One page per song
V1+V2+V3 on a single scrollable page, published so he can open it on any device and
scrub to a timestamp. This is the surface that replaces metric invention.
**DoD**: he reviews a map from the page + ArcViewer and never needs a scorecard.

### V5 — Defect capture, replacing arm-preference as the primary record
`scripts/review.py` collects `--better/--worse`. He produces *"drop is late at 2:10"*.
Add `review.py defect --song X --at 2:10 --kind drop_timing --quote "…"`, and make the
defect list the thing that drives work.
**DoD**: his review of a song lands in the ledger as located, typed defects.

---

## 🔴🔴 P0.5 — THE SIX DEFECTS HE NAMED (2026-08-17)

★**Unifying hypothesis, and the most valuable thing he has said in weeks:**
> **Our nps problem is an ALLOCATION problem, not a budget problem.**

*"The nps is generally wasted on every few non main notes"* + *"not following the main
vocals"* + *"I'd like the general beat parts to be faster and play more main notes"* are
plausibly **one defect**: we do not distinguish the MAIN musical line from incidental
onsets, so the note budget is spent on filler and the map simultaneously feels **slow**
(the notes you want are absent) and **busy** (notes you do not want are present).
⚠️This is a hypothesis stated in his words, not a measured finding. **Test it before
building on it**: if true, reallocating a *fixed* note budget onto main-line events
should improve his verdict with nps held constant.

| # | defect, his words | first read | DoD |
|---|---|---|---|
| **D1** | *"very slow"* | ours ~4.96 nps vs human 8.35 on Hunger — but see the allocation hypothesis before simply raising density | he stops calling it slow **without** a raw nps increase |
| **D2** | *"slightly off beat"* | survives `BEAT_GRID_PHASE`; tempo is right on only **70.5 %** of songs (n=149) and 2:3 misreads are open | he stops calling it off-beat on a song whose tempo we call correct |
| **D3** | *"doing drops at the wrong time"* | ⭐**directly actionable now** — `structure.py` finds sections and marks `DROP`/`build`/`breakdown`, and the generator does not use any of it | detected drop bar == where he says the drop is, on 4 songs |
| **D4** | *"not following the main vocals"* | `follow_vocals` ours **0.020** vs human **0.149**; root cause known — Stage-1 carries no melodic instruments (Track B) | `follow_vocals` ≥ 0.10 **and** he agrees the vocal line is being played |
| **D5** | *"random bursts of really fast non flowy notes"* | two joined problems: bursts are not musically motivated, **and** they break flow. V3 must locate them first | he cannot find a burst he calls random |
| **D6** | ★*"nps wasted on non main notes"* | the allocation hypothesis above; needs a definition of **main** that he endorses via V2 | main-line recall up with nps flat, and he agrees |

⚠️**D6 is the one that must not become another invented axis.** Define "main" in V2,
show it to him, and let him correct the definition. A metric he has endorsed by looking
at it is a different object from one built from first principles — building those from
first principles is exactly what produced the anti-correlation.

---

## 📦 AWAITING KYLE'S EAR — sets A and B, now answered globally

`for_review/` holds 32 maps; `python scripts/review.py next` is the shortlist and
`review.py list` the full pending list. **He has now answered both sets at the level of
defects rather than arms**, so:
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

### C2 — Grid PHASE ⚠️resolved ON THE METRIC ONLY — reopened 2026-08-17
`BEAT_GRID_PHASE=search` fixed ~18 of 39 failing songs **by the alignment axis**.
🔴**Kyle still reports *"slightly off beat"* across every song after playing `[PHASE]`**, so
either the lever is inaudible or the off-beat feel is not phase at all — see **D2**, where tempo
(right on only 70.5 % of songs) is the better suspect. Do not flip the default on the axis alone;
the axis is the thing the lever optimises. Remaining phase work is inside the tempo item above. ⚠️Never apply a blanket global shift — on `1f767` the human wants the same shift we do,
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
