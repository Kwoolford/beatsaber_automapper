# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md) — including the full session-by-session archive from 2026-06 to
2026-08-02. Evaluation-suite design rationale is in [`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule for keeping it that way:** when an item finishes, move the *outcome and what it taught* into
PROGRESS.md and delete it from here. A completed item is history, not work. Curated 2026-08-02, when
this file had reached 4,076 lines.

---

## 📍 CURRENT STATE (2026-08-03)

**Three lever candidates are built, validated and waiting on Kyle's ear. Nothing is promoted.**
The session's real yield was not the levers though — it was finding that **three separate numbers the
suite depended on were measurement artifacts** (a loader preferring ExpertPlus, `BPMInfo.dat` read as
`Info.dat`, and Demucs never seeded), plus **P0 closed**: the generation path is now seeded and a run
is byte-reproducible.

| | |
|---|---|
| Best arm | `tf_hl014_ds048` + `BEAT_TRIM_TAIL=0.5` + `BEAT_ONSET_EVIDENCE=0.3` + `BEAT_REACH=3:0.3:0.5` |
| Promoted | **nothing.** `generate.py` defaults untouched; every lever default-off |
| Candidates | **`BEAT_TRIM_TAIL=0.5`** (endings; tail defect closed), **`BEAT_ONSET_EVIDENCE=0.3`** (density follows audio; ⚠️degrades reachability, see K2), **`BEAT_REACH=3:0.3:0.5`** (hard_rate 0.123→0.059 = human, no shrinkage) |
| Stylistic knob | `BEAT_SPEED_DIAG=6:0.8` — **not a fix.** Kyle wants levers exposed as UI controls; see [[feedback-levers-are-user-facing]] |
| Suite | 6 axes + A8 drift term; **A4 musical-role built but must not gate anything** (measures the wrong thing for K5) |
| Seeding | **fixed** — `generate.py --seed` / `BSA_SEED`; `eval_sweep --seeds N` scores mean ± sd; Demucs seeded in the cache builders |
| Review maps | `outputs/kyle_review_2026-08-03/` — BEFORE / AFTER / **AFTER2** per song; README leads with what to be suspicious of |
| Caches | onsets `outputs/onset_cache/` (24); **per-stem `outputs/stem_onset_cache/` (274, seeded)** |
| Viewer | fixed; `arcviewer` self-heals its file-dialog plugin, `arcview <map.zip>` skips the dialog |
| Tooling | ArcViewer fix `tools/arcviewer_sfb_fix/`; calibration refs snapshotted to `docs/eval_references/` |

**★ Two standing methodology rules learned 2026-08-03:**
1. **Never calibrate the human corpus through `scorecard._load_any`** — it prefers ExpertPlus. Use
   `calibrate_playfeel.load_expert_only`. Three separate human references were wrong because of this.
2. **Ask "norm or aspiration?" before calibrating any new axis.** Kyle's target is the **best**
   mappers, so for aspirational axes "the human cohort passes it" is *not* a validity check.
   See [[feedback-target-is-best-mappers]].

**Human reference values worth memorising** — **re-verified 2026-08-03 on a strictly-Expert cohort
after two loader bugs were found** (see PROGRESS.md). Anything not on this list should be re-measured
before it is trusted:

| value | figure | status |
|---|---|---|
| onset precision | **0.930 ± 0.032** | ✅ confirmed (0.9366 on the clean cohort) |
| timing scatter | **10.35 ± 1.30 ms** | ✅ confirmed (10.20) |
| Expert nps | **3.91** | ✅ confirmed (3.931) |
| Expert **peak** nps | **5.5** | ✅ confirmed — ⚠️*not* 6.5, which was a contaminated reading |
| diagonal share | **0.354** | ⚠️corrected from 0.370 |
| **double share** | **0.1366** median (p90 0.2505) | 🔴corrected from 0.231 — **that was the p90** |
| drift (first→last fifth) | **−0.006** median, p90 **0.068** | ✅ new; **humans do not drift** |

⚠️**Never calibrate the human corpus through `scorecard._load_any`** — it prefers ExpertPlus. Use
`calibrate_playfeel.load_expert_only`.

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


## 🟠 P1 — PROMOTION (needs Kyle's ear; he is playing `outputs/kyle_review_2026-08-03/`)

**Five things would flip together**, and they are not independent:

| | why it is in the bundle |
|---|---|
| `BEAT_TEMPO_FIT=1` | the 2026-08-02 fix he called *"genuinely beautiful"* |
| `BEAT_DIFFICULTY_SCALE=0.48` | **must** ship with tempo-fit — it changes slots/second, and 0.55 was fitted to the wrong grid |
| `BEAT_TRIM_TAIL=0.5` | strongest evidence; tail defect closed across 3 seeds, costs nothing |
| `BEAT_ONSET_EVIDENCE=0.3` | rhythm improves resolvably ⚠️but **degrades reachability** and pushes peak nps 6.25→6.50 (human 5.5) |
| `BEAT_REACH=3:0.3:0.5` | `hard_rate` 0.123 → 0.059 = human, with **no** loss of reach distance; repairs the above |

⚠️**The suite will not endorse the bundle** (npass ~4/6). That is promoting on his ear over the
scorecard — defensible, since the scorecard has been wrong about "ready" twice and right zero times,
but it must be a deliberate choice.

⚠️**`BEAT_ONSET_EVIDENCE` is the one to interrogate**: half its supporting evidence is circular
(`density_corr` and A8 both reference librosa onsets, which the lever consumes), and it is the lever
that made reachability worse. **`rhythm` is its only independent evidence.**

**Tasks when the answer is yes**: flip the five defaults in `generate.py`, keep env-var overrides to
disable, re-run the regression at ≥3 seeds, record before/after in PROGRESS.md.

---

## 🎯 K1–K5 — FROM KYLE'S PLAY-THROUGH (2026-08-02)

Each claim was checked against data before being written down; measurements are in PROGRESS.md.
**K1–K3 confirmed with numbers. K4 partly confirmed. K5 not yet measurable.**

### K1 — Timing degrades toward the END of a song
**Findings + all measurements: PROGRESS.md.** Summary: the tail half is **closed** (`BEAT_TRIM_TAIL`,
post-music notes 37.5% → ~11%); the **decay half is open**. Humans do **not** drift (quintile
precision flat 0.937→0.947, drift p90 **0.0677**), so every drift number of ours is our defect. Cause
is a note-SELECTION defect, **not timing** — no offset ramp exists; ⚠️do not resurrect the tempo
hypothesis. Stage-1 gives dead outros the same window probability as the body of a song, so nothing
computed from that probability can fix it.

**Open work**
1. Close the decay: 37.5% of maps still exceed the human drift p90 (10% expected). `BEAT_ONSET_EVIDENCE`
   cut *severity* hard (p90 0.391 → 0.205) but not the count.
2. ⚠️Only **77 of 400** Expert maps are drift-scorable (need cached onsets) — widen the onset cache
   before treating that p90 as precise.
3. The human control splits the corpus: 1f8d6 and 1f8ce drift for humans too, so they are partly
   song/detector. **Ours alone: 1f336, 1f3d7, 1f767, 1f65d, 1f333** — fix only those.

### K2 — REFRAMED BY KYLE 2026-08-03: the defect is REACHABILITY, not diagonals
> *"I don't like the global thin diagonal, they can be fun in fast passages, but not outside corner
> in swings followed by another swing that's hard to reach... They should still be playable though
> that's the core problem not that they are diagonal."*

**Measured immediately and he is right** (`scripts/eval_reachability.py`; a cut carries the hand to
`p + direction`, so the next note's cost is the travel from *there*):

| metric | ours | diag-thinned | human |
|---|---|---|---|
| reach_median | 2.83 | 2.83 | 2.83 |
| reach_p90 | 3.16 | 3.16 | **3.61** |
| **hard_rate** (≥3 units within 0.3 s) | **0.1364** | **0.1364** | **0.0592** |
| corner_exit_rate | 0.243 | 0.233 | 0.185 |
| **hard_given_diagonal** | **0.0867** | 0.0754 | **0.0773** |

1. **We make 2.3× the human rate of hard transitions** — the real defect.
2. **Diagonals are blameless**: conditional on a diagonal our hard rate (0.0867) ≈ human (0.0773).
3. **The diagonal thin moves `hard_rate` by exactly zero.** It was never going to fix this.
4. ★ **Humans reach FURTHER than us** (p90 3.61 vs 3.16) — they make bigger movements and **give them
   time**. The defect is distance-per-*time*, not distance.

**Open work**
1. Build a reachability-aware lever: when placing the next note, disincentivise positions that are
   far from the previous follow-through point *given the time available*. Target `hard_rate`
   0.136 → ~0.059, and do **not** target distance — shrinking travel would move us away from human.
2. `corner_exit_rate` 0.243 vs 0.185 is a genuine but secondary excess (1.3× vs the 2.3× on hard_rate).
3. Diagonal share is still high overall (0.45 vs 0.354) — a **style** difference, not a playability one.

**`BEAT_SPEED_DIAG` is reclassified as a STYLISTIC KNOB, not a fix.** Kyle wants the levers exposed in
a UI so players can customise (e.g. *more* diagonals). Keep it, keep it parameterised and monotone,
do not promote it as a defect fix. See [[feedback-levers-are-user-facing]].

### K3 — We enter later than a human mapper (and should end on the song's last beat)
**Evidence**: our first note is 1.9–14 ms from a real onset — so it is *not* misaligned — but we start
late: 1f333 human 1.91 s vs ours 2.39 s; 1f8d6 1.74 s vs 2.17 s.

**Kyle explicitly does not want this hardcoded**: *"I'm ok with the first note not being a note every
time, as sometimes it's just backlight filler that isn't the 'real' start."* The stronger preference
is the **ending**: *"the ending note should be obvious and grand to give the player that satisfaction
of beating it on a good note."*

**Tasks**: measure first-note and last-note offsets against the human map across the corpus and close
the gap where the audio supports it. Prime suspect for the late entry is `section_gate="loud_only"`
suppressing a quiet intro.

**DoD**: first/last-note offsets inside the human corpus range, achieved through better section
handling rather than a hardcoded "always place a note on the first onset".

### K4 — Under-response during build-ups (partly confirmed)
**Evidence**: notes-per-onset against each song's own median — 1f333 1:30–1:33 at **0.67×** (*"a
really sick building guitar, but only catches the end and cuts it short"*) and 3:20–3:28 at
**0.74×**. But 3:05 (1.19×) and 1f767 2:20 (1.46×) carry normal note counts, so those are **not**
density failures — they belong to K5.

**Tasks**: detect build-ups/crescendos and respond across the whole phrase rather than its tail.
Kyle's framing — *"phrase-aware maybe needs better sectioning of when the beginning and the end of
the phrase is"* — matches the **A5 structure axis that was built and shelved as a negative result**
(2026-07-27). Re-open it with "cover the whole crescendo, not just its peak" as the target, which is
sharper than A5 had the first time.

**DoD**: 1f333's 1:30–1:33 window responds at ≥0.9× the song median without inflating density
elsewhere.

**Re-checked 2026-08-03 — tonight's levers nudge but do NOT fix it** (table in PROGRESS.md).
`BEAT_ONSET_EVIDENCE` moves the 1:30–1:33 build-up 0.83× → 0.87× against a 0.9× DoD, and **3:20–3:28
sits at 0.54×** — design against *that* window, not 1:30–1:33. 3:05 is above the median, confirming it
is K5 and not density. ⚠️This binning differs from the one behind the 0.67×/0.74× recorded on
2026-08-02; compare arms, not sessions.

### K5 — "the average of all of them" (REFRAMED — do not close as refuted)
**Kyle's target is the BEST mappers**, and his usable form of the claim is *"a great mapper would
have at least addressed the main instrument when it comes into play."*

**Three operationalisations, all measured, none showing a defect against the MEDIAN human**
(numbers in PROGRESS.md): per-section instrument commitment, winner-take-all attribution, and
entry-events (ours 1.216 vs human 1.167 over 127 maps). A4 passes its own control battery, so it
detects lead-following — both cohorts simply score like the "average of all stems" control.

🔴 **Blocked on data, and it is a small ask.** The median is a *floor*, not the target, so none of the
above is an answer. A best-mapper cohort cannot be built from disk: `data/raw/manifest.json` (5,373
maps) has only `category` (mod requirements), `genre`, `genre_tags` and `downloaded_at` — **no
rating, downloads, ranked or curated flag anywhere.**

**Open work**
1. ★ **Ask Kyle to name exemplary mappers/maps** → build the reference cohort from those. Minutes of
   his time; unblocks *every* aspirational axis, not just K5.
2. Alternatives if he would rather not: pull BeatSaver ranked/curated flags (network, needs his nod),
   or drop human references for aspirational axes and state absolute targets.
3. ⚠️**A4 must not gate anything** meanwhile — it works, but both cohorts score like the union
   control, so it is not measuring what K5 is about. Reading 3 (lead-by-onset-activity ≠ musical
   lead) has positive evidence: human commitment sits *below* the union control at every granularity.
   Next signal to try: pitch salience / melodic contour (`--use-contour` already extracts it).


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
