# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md) — including the full session-by-session archive from 2026-06 to
2026-08-02. Evaluation-suite design rationale is in [`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule for keeping it that way:** when an item finishes, move the *outcome and what it taught* into
PROGRESS.md and delete it from here. A completed item is history, not work. Curated 2026-08-02, when
this file had reached 4,076 lines.

---

## 📍 CURRENT STATE (2026-08-02)

**The tempo fix landed and Kyle heard it.** Axis A8 (audio alignment) found the note grid was built
on a tempo wrong on 20 of 21 songs; a tempo+phase fitter took `alignment_gap` **5.41 → 0.554** and
timing scatter **17.4 → 10.2 ms** (human 10.35). His verdict: *"genuinely beautiful... the foundation
is now complete."* First time in the project's history that his ear and a measurement agreed in the
**positive** direction.

| | |
|---|---|
| Best arm | `tf_hl014_ds048` — tempo fit + hand-lead 0.14 + difficulty scale 0.48 |
| Promoted | **nothing.** `generate.py` defaults untouched; `BEAT_TEMPO_FIT` and `ds048` default-off |
| Suite | 6 axes: A1 flow, A2 rhythm, A3 idiom, A6 handrole, A7 playfeel, A8 alignment |
| Best score | alignment 0.436 ± 0.113 (bar 0.39); npass 4,4,2 across 3 seeds — rank on gaps, not passes |
| Seeding | **fixed** — `generate.py --seed` / `BSA_SEED`; `eval_sweep --seeds N` scores mean ± sd |
| Candidates | `BEAT_TRIM_TAIL=0.5` (tail defect closed) + `BEAT_ONSET_EVIDENCE=0.3` — **both default-off, both need Kyle's ear** |
| Review maps | `outputs/kyle_review_2026-08-02/` — 1f767, 1f913, 1f333, 1f8d6, SO TIRED ROCK |
| Viewer | fixed; `arcviewer` self-heals its file-dialog plugin, `arcview <map.zip>` skips the dialog |
| Tooling | ArcViewer fix source `tools/arcviewer_sfb_fix/`; skill backups `docs/skills-backup/` |

**Human reference values worth memorising** (measured, not assumed): onset precision **0.930 ±
0.032**, timing scatter **10.35 ± 1.30 ms**, Expert **3.91 nps**, diagonal share **0.370**, double
share **0.231**.

---

## 🔴 P0 — SEED LOTTERY: **CLOSED**, with three habits to keep

Cause and fix in PROGRESS.md (commits `7a5544d`, `7869115`): nothing in the generation path was ever
seeded. `generate.py --seed` / `BSA_SEED` + `eval_sweep --seeds N`. **DoD met** — regenerating at
seed 0 from a fresh process, after a whole sweep ran in between, reproduced the swept maps
**byte-identically** on all three probe songs.

**What the run also settled:**
- **Pairing helps exactly one axis.** sd(paired) vs sd(unpaired): alignment **0.033 vs 0.143**
  (4.3× tighter) because it rides the postprocess `random` stream; the other five ride the torch
  decode, which diverges once configs differ, and show no benefit. ⚠️n=3, so the per-axis sd
  estimates are themselves noisy — "pairing helps alignment" is the only safe claim.
- **★ `npass` is not a ranking statistic.** Even seeded, `tf_hl014_ds048` scored **4, 4, 2** across
  three seeds. Rank on per-axis gaps with error bars, never on the pass count.

**Tasks**
1. Decide whether the spread bar (0.35) is the right gate. **Evidence gathered 2026-08-03**: the bar
   is not miscalibrated — the held-out human cohort scores `min_spread` **0.923** and 0.35 was set
   deliberately low as a *mode-collapse alarm*, not a human-likeness demand. The problem is that **we
   sit right on it** (0.39–0.46, about half the human value) with sd up to 0.09, so pass/fail is a
   coin flip. **Recommendation: stop using spread as a hard pass/fail gate** — report it as a number
   with its sd, and keep a hard alarm only at a much lower threshold (~0.15) where crossing really
   does mean collapse. ⚠️Not done unilaterally: it changes scorecard semantics and would make tonight's
   numbers incomparable with everything recorded before it. Wants a deliberate call.
   ★Note this is the same conclusion as "`npass` is not a ranking statistic", reached from the other
   end — a threshold sitting inside the noise cannot gate anything.
2. ~~Retire the fake seed arms~~ — **DONE.** The 11 re-roll arms (`prod_rep`, `hl014_seed*`,
   `*_s1`…`*_s4`) are now in `DEPRECATED_ARMS`: excluded from a bare sweep, flagged in `list-arms`,
   but **not deleted** — PROGRESS.md quotes noise-floor numbers derived from their cached maps, and
   deleting them would orphan that evidence. Name one with `--arms` to reproduce the old measurement.
3. Score every future arm at **≥3 seeds** and quote the sd. Any single-run comparison is now a
   choice, not a limitation.

---

## 🟠 P1 — PROMOTION DECISION (needs Kyle; one answer unblocks it)

The gate was "stays off until Kyle plays it and says it sounds on-beat." **He has.** Two things to
know before it flips:

- **It is a pair.** `BEAT_TEMPO_FIT=1` changes how many 1/4-beat slots exist per second, so it needs
  `BEAT_DIFFICULTY_SCALE=0.48` alongside it. The old 0.55 was fitted to the wrong grid; promoting one
  without the other yields a map that is on-beat and a tier too dense.
- **The suite will not endorse it.** That config scores 4/6 on a good seed and 0/5 across seeds. This
  is promoting on his ear over the scorecard — defensible, since the scorecard has been wrong about
  "ready" twice and right zero times, but it should be a deliberate choice, not a slip.

**Tasks when the answer is yes**: flip both defaults in `generate.py`, keep env-var overrides to
disable, re-run the regression, record before/after in PROGRESS.md.

---

## 🎯 K1–K5 — FROM KYLE'S PLAY-THROUGH (2026-08-02)

Each claim was checked against data before being written down; measurements are in PROGRESS.md.
**K1–K3 confirmed with numbers. K4 partly confirmed. K5 not yet measurable.**

### K1 — Timing degrades toward the END of a song ★ highest confidence, worst symptom
**Evidence**: onset precision per fifth — 1f333 0.973→0.856, 1f767 0.985→0.783, **1f8d6
1.000→0.518**; 1f913 stable, so it is not universal. No notes fall outside the audio file, but 1f333
places 5 and 1f8d6 places 10 notes **after the last detected onset** — after the music has stopped,
which is exactly what Kyle heard as "notes playing about 5 seconds after the song ends".

**Why we missed it**: A8 reports ONE precision per map, so drift *within* a song averages away. **A
song-level metric cannot see a song-shaped defect** — the same blind-spot shape as the audio-blind
suite itself.

**MEASURED 2026-08-02** (`scripts/eval_align_drift.py`, results `outputs/align_drift_2026-08-02.json`).
The metric exists and is calibrated. Calibrating on humans first was the right call — **humans drift
too** (quintile precision 0.950 0.942 0.949 0.914 0.920, median drift 0.0385), so a bar at zero drift
would have repeated the `h_dist` error.

🔴 **THE K1 HUMAN BAR IS SUSPENDED.** It was calibrated through a loader that silently prefers
**ExpertPlus** (see PROGRESS.md). On a strictly-Expert cohort the human drift distribution is far
wider — median **0.0073**, **p90 0.4618** (not 0.1451) — and humans place post-music notes in
**32.5%** of maps with `tail_after_secs` p90 **19.25 s** (not 0.0). Against that p90 our maps sit
*inside* the human range nearly everywhere. **Every "% past the human p90" figure for K1 is
withdrawn** pending a rebuild on a documented, strictly-Expert, adequately-sized cohort — the 77
drift-scorable maps there may also be too few.

**What still stands, because it never referenced the bar**: 1f8d6 q1 1.000 → q5 0.571 with **11 notes
running 4.43 s past the last onset** — Kyle's "about 5 seconds after the song ends", measured — and
the per-song pattern reproducing his report exactly, including 1f913 *not* drifting. Absolute
before/after effects of the trim lever (11 → 1 notes, 4.43 → 0.53 s) are likewise unaffected.

★ **Keep the methodological lesson**: a cohort-median metric cannot see a subset-of-songs defect —
the same blind-spot shape as a song-level metric missing a song-shaped defect, one level up. Rank by
**exceedance over the human p90** and name the songs.

**Tasks**
0. 🔴 **REBUILD THE K1 BAR FIRST** on a strictly-Expert cohort via
   `calibrate_playfeel.load_expert_only`, and check how many maps are actually drift-scorable (only
   77 of 200 had cached onsets). Until then no K1 pass/fail claim means anything.
1. ~~Control battery~~ — **DONE** (`scripts/audit_align_drift.py`; the standard battery would have
   flattered it, so drift got its own with a POSITIVE control as well as negatives). ⚠️Its *human*
   row shares the ExpertPlus contamination; the degenerate/positive controls are relative to that
   same cohort so the qualitative verdict holds, but re-run it after the bar is rebuilt:

   | variant | precision | drift med | share > human p90 |
   |---|---|---|---|
   | human | 0.920 | 0.030 | 10.0% |
   | timing_random | **0.601** | −0.007 | 10.0% |
   | timing_jitter | 0.794 | 0.047 | 5.0% |
   | decay_0.25b | 0.870 | 0.235 | **85%** |
   | decay_1.00b | 0.852 | 0.293 | **95%** |

   ⚠️**Drift is CONDITIONAL and must never be read without A8's precision gate beside it**: a map with
   randomised times loses a third of its precision and drift does not notice (−0.007). Confirmed, not
   assumed. Conversely the `decay_*` controls sit at near-human precision (0.85–0.87) yet drift
   catches 85–95% of them — so it adds information precision alone does not.
2. ~~Fit tempo per-segment~~ — **DONE, and the answer is neither branch**
   (`scripts/diag_align_drift_cause.py`). **K1 is a note-SELECTION defect, not a timing one.** There
   is no offset ramp: median match offset across quintiles spans −14.6 to +5.5 ms, non-monotone, and
   the *human* maps wobble the same (−2.9 to +3.8); scatter barely grows. Accumulated tempo error
   would ramp tens of ms. The notes that land are as accurately placed at the end as at the start —
   **the lost precision is notes matching nothing.** A piecewise fit or BPM events would not touch it.
   ⚠️Do not resurrect the tempo hypothesis for K1.
3. **The decay — allocator exonerated, Stage-1 probabilities indicted** (measured 2026-08-03 via
   `BEAT_PROBS_DUMP`; table in PROGRESS.md). On 1f8d6's dead outro the onset count is **0** across
   windows where `wmean` is 0.28–0.42 — *as high as the body of the song* — so the formula allocates
   ~35 notes to a region with ~2 real onsets. **A decode ceiling computed from `wmean` cannot fix
   this**, because `wmean` is exactly what is wrong. Two mechanisms, not one:
   (a) 1f8d6/1f336 — music thins, probability does not follow it down;
   (b) 1f333/1f3d7 — music does **not** thin, but probability *rises* at the end, so we allocate more
   notes into the final section. (b) also explains why 1f3d7 was the exception to the earlier
   notes/onsets mechanism.
   **`BEAT_ONSET_EVIDENCE` BUILT AND MEASURED at 5 seeds — β=0.3 is the candidate.** Weights the
   per-window budget by librosa onset density instead of Stage-1's belief. At β=0.3, paired against
   `trim`: **rhythm −0.073 (sd 0.029, resolvable)**, alignment −0.099 (sd 0.047, resolvable),
   `peak_nps` 6.25 → **6.50 = the human median exactly**, everything else flat. Full write-up and two
   important caveats in PROGRESS.md:
   ⚠️**Half the evidence is circular** — `density_corr` and A8 both reference librosa onsets, which is
   what the lever consumes. `density_corr` must not be cited at all. **Rhythm is the independent
   result.**
   ⚠️**n=3 lied twice**: idiom's "resolvable" gain did not replicate and is non-monotone, and n=3
   underestimated the sds. Treat n=3 as a screen, not a verdict.
4. ~~Stop emitting notes after the last musical onset~~ — **DONE and validated.** `BEAT_TRIM_TAIL`
   (grace in seconds after the last librosa onset, default OFF, `0.5` tested). Over 24 songs it takes
   `tail_after_secs` p90 **2.37 s → 0.019 s** and tail exceedance **37.5% → 12.5%** (10% is what you
   get by construction), improves drift exceedance 29.2% → 20.8%, and **costs nothing** — all five
   non-alignment axes inside noise over 3 seeds. Alignment −0.055, resolvable *only* under paired
   comparison (sd 0.023 vs unpaired 0.113). Details in PROGRESS.md.
   **Confirmed on all 3 seeds** (control 37.5/37.5/37.5% maps with tail notes vs trim 12.5/12.5/8.3%;
   `tail_secs` p90 2.37/2.16/2.06 s vs 0.019/0.019/0.000 s). **The tail defect is closed** and this is
   now a promotion candidate alongside the P1 pair.
   ⚠️Landmine recorded: energy is the WRONG cut criterion (1f8d6's energy runs 4.7 s past its last
   onset, so an energy cut removed one note); and the lever must live in `generate_v7_level`, not
   `predict_onsets`, which only the legacy path calls.

**The human control splits the set — fix only what is ours** (the C2 lesson): 1f8d6 and 1f8ce drift
for the human map too (0.147, 0.208), so part of those is the song or the onset detector and
"fixing" them is fitting the detector. **Ours alone: 1f336, 1f3d7, 1f767, 1f65d, 1f333.**

**DoD**: A8 reports drift, the human corpus passes it, degenerate controls fail it, and the five
*ours-alone* songs fall inside the human drift range. ⚠️1f8d6 is the wrong target for the DoD — its
human map drifts 0.147 too, so "1f8d6's final fifth is no longer half as accurate as its first" was
partly asking us to beat the detector.

### K2 — Diagonal cuts INCREASE with speed; they should decrease
**CORPUS-VALIDATED 2026-08-03** (`scripts/eval_diagonal_vs_speed.py`, 54 human maps vs 24 of ours;
results `outputs/diag_vs_speed_2026-08-03.json`). The TODO evidence had been **one song**, and 1f333
is the half-tempo song the landmine list warns about — so this was the trap, and K2 survived it:

| local nps | 0–4 | 4–7 | 7–10 | **10+** | overall | median slope |
|---|---|---|---|---|---|---|
| human (n=200, **strict Expert**) | 0.355 | 0.346 | 0.301 | **0.236** | 0.354 | **−0.01141** |
| ours | 0.466 | 0.476 | 0.536 | **0.631** | 0.477 | **+0.00226** |

★ Humans fall **monotonically** as passages speed up; we rise monotonically. At 10+ nps we use
**2.7×** the human share — precisely Kyle's framing ("wanted" when slow, "not preferred" when fast).
Target the **10+ band** (0.631 → ~0.24) and the overall level (0.477 → 0.354).

⚠️First measured on a cohort contaminated with ExpertPlus, which *muddied* K2 into a non-monotone
human curve and made slope look like a weak discriminator. On the clean cohort it is both a slope
defect and a level defect. Only 16 human maps reach the 10+ band, so that cell is the least certain.

**Superseded single-song evidence**: 1f333 diagonal share by local note rate — 0.516 / 0.477 / 0.530 / 0.653 across
0–4 / 4–7 / 7–10 / 10+ nps, against a human Expert average of **0.370**. Diagonal-heavy everywhere,
and *most* diagonal exactly where they punish hardest. The correlation should be negative; ours is
positive.

Kyle's framing should survive into the fix: broad "outside-in" swings are **wanted** in slow sections
and on drops — *"they get the player moving and feel like they are playing a grand orchestra"* — and
only become a problem in fast passages, where they are *"difficult but possible, and not preferred"*.
Specific cases: 1f333 at 2:11 and 2:17.

**Tasks**
1. A decode lever scaling diagonal probability down as local note rate rises (flag-gated,
   default-off, like every other lever here).
2. Watch `dir_entropy` while doing it. **Over-diversifying direction created this defect in the
   first place** (2026-07-27): spreading entropy across all nine cut directions yields mostly
   diagonals, because six of the nine are diagonal or dot.

**DoD**: diagonal share correlates *negatively* with local nps; A7 `diagonal_share` moves toward
0.370 without collapsing direction variety; A1 flow does not regress beyond its 0.23 two-sigma floor.

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

### K5 — "It does the average of all of them" (NOT MEASURABLE YET — do not close as refuted)
Kyle heard 1f913 as never committing: *"it doesn't seem to stick to one beat or one flow, it's kinda
trying to do the average of all of them."* And on 1f333 at 3:05 a guitar solo enters where *"a good
mapper 100% would have played notes to accentuate this change... most if not all notes would have
changed to be this guitar solo, the lead hand would have played a lot of the solo, and on consistent
beat drops a spare hand would have come in."* He suspects phrase-similarity scoring is too tight and
should be **loosened** so a real change of character is allowed to change the map.

**Measured, and it did not reproduce**: whole-song stem commitment matches the human (1f333 0.382 vs
0.420; 1f913 0.297 vs 0.285), and our lead-instrument switch rate is as high or higher (1f333 0.129
vs 0.067; 1f913 0.250 vs 0.292).

**But the metric is too blunt to be believed**: both cohorts read as drum-led because drums carry the
most onsets, so the argmax is nearly predetermined. **His ear has been ahead of the metrics twice —
treat a null from a blunt instrument as a metric problem, not a refutation.**

**Tasks**: build the **A4 "musical-role correctness"** axis that `docs/eval_suite_v2.md` planned and
never built — weight stems by salience rather than raw onset count, measure per-section which layer
the map follows, then re-test the claim. Then revisit the phrase-similarity threshold.

**DoD**: a metric that separates "follows the section's lead instrument" from "spreads across
everything", validated on the human corpus, and an answer to his claim either way.

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

### C5 — The 4× double share, still the largest untouched structural defect
Ours 0.73–0.79 against a human **0.231**. Sits upstream of A2, A6 and the flow spread. Untouched
since identified.

### C6 — `outputs/` is entirely gitignored (needs a decision) — **this already bit us once**
`git ls-files outputs/` returns **zero**. A commit message on 2026-08-02 stated the ArcViewer fix
was "copied here for version control" into `outputs/`; it was not, and the only two copies were both
untracked. Caught at close and moved to `tools/arcviewer_sfb_fix/`. The same trap still applies to
every calibration reference the suite depends on. All seven calibration artifacts — every axis's human
reference plus `ioi_human_model.json` — exist only on this machine. The suite's bars are meaningless
without them, and A8 fails closed when its reference is missing, so a rebuild would look like a
regression rather than a missing file. Fixing it (track the seven small JSONs, or move them under
`data/`) changes a project convention, so it wants a call rather than a unilateral commit.

---

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
