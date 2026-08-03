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
| Best score | alignment 0.554 ± 0.092 (bar 0.39); 4/6 on a good seed; **0/5 seeds pass all six** |
| Seeding | **fixed** — `generate.py --seed` / `BSA_SEED`; `eval_sweep --seeds N` scores mean ± sd |
| Review maps | `outputs/kyle_review_2026-08-02/` — 1f767, 1f913, 1f333, 1f8d6, SO TIRED ROCK |
| Viewer | fixed; `arcviewer` self-heals its file-dialog plugin, `arcview <map.zip>` skips the dialog |
| Tooling | ArcViewer fix source `tools/arcviewer_sfb_fix/`; skill backups `docs/skills-backup/` |

**Human reference values worth memorising** (measured, not assumed): onset precision **0.930 ±
0.032**, timing scatter **10.35 ± 1.30 ms**, Expert **3.91 nps**, diagonal share **0.370**, double
share **0.231**.

---

## 🔴 P0 — THE SEED LOTTERY BLOCKS EVERYTHING DOWNSTREAM

Five runs of an **identical** configuration scored 4, 2, 1, 3 and 5 of six axes. Measured per-axis
standard deviations: **flow 0.116, handrole 0.317, alignment 0.092, rhythm 0.076, idiom 0.065,
playfeel 0.040**.

**Why this is P0 rather than a caveat:** with those floors, most single-run differences this project
has ever reported are unresolvable — including several made during the session that measured them.
Every ranking, promotion decision and K-item below is untrustworthy until this is addressed. It is
cheap to work around (score means over ≥3 seeds) and worth understanding properly.

**The cause is found and fixed** (commit 7a5544d; write-up in PROGRESS.md). Nothing in the
generation path was seeded — not decode sampling, not post-processing. `generate.py --seed` /
`BSA_SEED` now seeds all three RNGs, and `eval_sweep --seeds N` scores an arm as a mean ± sd. What
remains:

**Tasks**
1. **Read `logs/overnight/seedrepro_2026-08-02.log`** (running as of 2026-08-02 23:00, ~1.6 h,
   `scripts/overnight_2026-08-02h.sh`). It answers two things: is a *score* reproducible, not just a
   map; and is sd(paired, matched seeds) meaningfully below sd(unpaired)? If it is, small levers can
   be ranked with ~3 seeds. If it is not, **stop ranking small levers** — say so and move the effort
   to defects big enough to clear the floor.
2. Decide whether the spread bar (0.35) is the right gate. Identical configs land 0.39–0.46 on it
   with sd up to 0.09 — the bar sits *inside* the noise, which is mechanically why the pass count
   swings. Either the bar moves or spread stops being a hard gate. **Seeding does not fix this**: it
   makes each run repeatable, not the seeds agree.
3. Retire the fake seed arms in `eval_sweep.py` (`*_s1`…`*_s4`, which vary `BEAT_HAND_LEAD_SEED` and
   were only ever a way to force a re-roll). `--seeds N` replaces them; leaving both invites someone
   to average a seed replicate together with a genuinely different config.

**DoD**: an arm's verdict is reproducible — the same config scored twice gives the same pass count,
or the report says plainly that it cannot. *(Map-level determinism is confirmed; the score-level
half is what the running job settles.)*

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

**K1 is confirmed, but not in the shape it was written.** Our cohort *median* drift (0.062) sits
*inside* the human p10–p90 — the median does not separate us at all. The defect is in the **upper
tail**: 7/24 maps past the human p90 for drift (29.2%, vs 10% by construction) and 8/24 for notes
after the last onset (33.3%). Worst: 1f8d6 q1 1.000 → q5 0.571 with **11 notes running 4.43 s past
the last onset** — Kyle's "about 5 seconds after the song ends", measured. Per-song it reproduces his
report exactly, including 1f913 *not* drifting.

★ **Keep the methodological lesson**: a cohort-median metric cannot see a subset-of-songs defect —
the same blind-spot shape as a song-level metric missing a song-shaped defect, one level up. Rank by
**exceedance over the human p90** and name the songs.

**Tasks**
1. Run `scripts/audit_eval_suite.py` on the drift metric before it steers anything. Note it is a
   **conditional** metric: a randomised map has uniformly low precision and therefore ~no drift, so
   it must be gated on overall precision or a degenerate control will "pass" it.
2. Diagnose before fixing. Fit tempo per-segment: if the *song's* tempo genuinely moves (live
   playing, a ritardando) the answer is BPM events, not a better global fit. If the song is constant
   and our single fitted tempo accumulates error, the answer is a piecewise fit.
3. Stop emitting notes after the last musical onset.

**DoD**: A8 reports drift, the human corpus passes it, degenerate controls fail it, and 1f8d6's final
fifth is no longer half as accurate as its first.

### K2 — Diagonal cuts INCREASE with speed; they should decrease
**Evidence**: 1f333 diagonal share by local note rate — **0.516 / 0.477 / 0.530 / 0.653** across
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
