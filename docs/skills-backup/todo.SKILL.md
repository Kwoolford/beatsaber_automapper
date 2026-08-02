---
name: todo
description: Drive a beatsaber_automapper research session — work the TODO stack in order (remaining architecture items → run completion → output evaluation → queue the next overnight experiment). Use when the user says /todo, "continue the TODO", "what's next on the automapper", or asks to pick up the overnight research loop.
---

# /todo — Beat Saber Automapper research-session driver

A session loop for the ML research project at `/home/kyle/repos/beatsaber_automapper`.
The project runs in overnight cycles: queue an experiment → it trains/generates while
the user sleeps → next session evaluates the outputs and queues the next one. This skill
is that next-session routine. Work the steps **in order** and stop to report between any
that change the plan.

## Cold start (always first)
```bash
cd /home/kyle/repos/beatsaber_automapper && source .venv/bin/activate
```
Read context before acting:
- `MEMORY.md` index + the `project_beatsaber_automapper.md` / `beatsaber_v8_representation_theory.md`
  memories (they hold the live status + the "key notes we identified" — currently: the
  ~13–15s **silent drop**, the **flat ~8 NPS** density that ignores song structure, the
  **late-song / final-chorus collapse**, and "for-sport diagonal swings").
- `TODO.md` — forward-looking work only: CURRENT STATE, priorities, work items with DoDs,
  REFERENCE. The history of what was already done lives in `PROGRESS.md`.

## Step 1 — Remaining architecture items
Read the **top of `TODO.md`** and the current "Scoped V8" TASK stack. Classify each TASK as
DONE / DEAD / live. Report the *live* ones only. (As of 2026-06-05: TASK 0 done; TASK 1 & 4
dead; **TASK 2 = inference-DoD pending**; **TASK 3 = Stage-2 pitch-contour, untouched** — the
two live build items. TASK 5 stretch.) Don't re-litigate dead tasks.

## Step 2 — Is the last run complete?
```bash
ps aux | grep -E "python|train|preprocess|overnight" | grep -v grep
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
ls -t logs/overnight/*.log | head; tail -30 <newest overnight log>
```
GPU idle + no train/generate proc + the overnight script printed its `COMPLETE` line ⇒ done.
If still running, report ETA and stop (don't queue on top of a running GPU job).

## Step 3 — Have the outputs been evaluated?
Find the run's artifacts (usually `outputs/<date>/`, `outputs/task*/`, `outputs/v8_poc/`,
`experiments/leaderboard_v7.jsonl`). For each, check whether the **evaluation** (not just the
raw map) exists — `align_*.json`, `density_*.json`, `*_summary.log`. A run is "evaluated" only
when its DoD metric has a number AND a verdict.

## Step 4 — Evaluate + review the key notes
Open the per-section eval and check the key notes explicitly:
- **Drop @ ~13–15s** has notes (silent-drop fixed by `section_gate="loud_only"`).
- **Density tracks structure?** Per-section/`density_*.json` — flat ~8 NPS everywhere = FAIL;
  varying with the music = the goal. Decisive DoD metric: `scripts/eval_density_corr.py`
  (Spearman of generated vs reference onset density over uniform windows, **≥ 0.41**).
- **Late-song collapse** (final chorus ~160–164s) — still present?
Eval tools:
```bash
python scripts/eval_alignment.py    --audio <song> --map <zip> --difficulty Expert --tolerance-ms 50 --json <out>
python scripts/eval_density_corr.py --audio <song> --map <zip> --difficulty Expert --json <out>
```
Write the **outcome** into `PROGRESS.md`, update/remove the corresponding `TODO.md` item, and
update the project memory. See the DOC CONVENTION section at the end of this file.

## Step 5 — Queue the next overnight experiment
Pick the highest-value live item (Step 1) whose DoD is still unproven. Pattern:
1. Make the minimal code change (gate it behind a flag; keep prior behavior the default).
2. Smoke-test the new path fast (load ckpt + one forward) before committing a GPU night.
3. Write `scripts/overnight_<date>.sh` with explicit **arms + a control**, each generating +
   evaluating, ending in a `python - <<'PY'` summary that prints the **verdict logic**
   (what result means DoD-met vs pivot). Log to `logs/overnight/<name>.log`.
4. Launch detached: `nohup bash scripts/overnight_<date>.sh >/dev/null 2>&1 &`, confirm the
   first arm clears the new code path, then report and let it run.

## Step 6 — Enter the autonomous research loop (do NOT stop after one experiment)
After Step 5 has something running (or if Step 5's item was CPU-only and already finished),
**do not end the session** — enter a self-paced research loop and keep going until the user
runs `/close`. Discovery rate is the bottleneck; a session that queues one experiment and
stops wastes the hours in between.

Invoke it with the `loop` skill, **no interval** so the model self-paces:
```
Skill(skill="loop", args="/todo — autonomous research iteration: advance the live task stack")
```
Each iteration of the loop:
1. **Harvest** — if a run finished, evaluate it against its DoD and write the verdict to
   `PROGRESS.md`, prune the finished `TODO.md` item, and update memory. Never leave a finished
   run unevaluated.
2. **Advance** — take the next live item off the stack and build it. Prefer CPU-only work
   (eval axes, corpus mining, analysis) while a GPU job is running so the two overlap.
3. **Generate new hypotheses** — when the stack runs low, propose new ones from evidence
   already in hand (blind spots the control battery exposes, metric gaps vs the human corpus,
   surprising numbers in the last run). Add them to `TODO.md` with an explicit DoD before
   building.
4. **Re-pace** — `ScheduleWakeup` with a delay matched to what is actually being waited on:
   a long fallback (1200s+) when a tracked job will notify on completion, a shorter one only
   when polling something the harness cannot see.

Loop discipline:
- **One GPU job at a time.** Never queue on top of a running job; use the wait for CPU work.
- **Commit + push each completed unit of work** so nothing is lost if the box restarts.
- **Every new metric passes the control battery** (`scripts/audit_eval_suite.py`) *before* it
  is used to steer the generator, and is scored by cohort shift/spread, never per-map
  distance-to-median (that reproduces the h_dist saturation — see `docs/eval_suite_v2.md`).
- **Report at each iteration** what finished, what it showed, and what is now running.
- Stop only on `/close` (which calls `ScheduleWakeup(stop=true)`), or if a decision genuinely
  needs the user — surface it and keep working on everything that does not depend on it.

## Conventions (learned the hard way — see TODO "Bugs found")
- `scripts/generate.py` takes `audio` as a **positional** arg, not `--audio`.
- Load beat checkpoints with `strict=False` (field additions break strict load).
- **Don't pick inference checkpoints by `val_token_acc` / `val_f1_avg_tol`** — they
  anti-correlate with alignment/structure quality. Use alignment F1 / density-corr.
- Production inference defaults: layout `version_10` (ctx16 + song-memory ON), beat
  `version_4`; `section_gate="loud_only"`.
- Logs dirs under `logs/` are artifacts, not commits.
- Track multi-step work with the Task tools; update PROGRESS.md + TODO.md + memory at the end,
  per the DOC CONVENTION below.

## Output
Finish with: live architecture items, run status, what the eval showed on the key notes,
and what experiment is now queued (with its DoD + how to read the verdict next session).

---

## 📁 DOC CONVENTION (read before writing to any markdown)
**`TODO.md` is forward-looking ONLY; `PROGRESS.md` is the trail of what was done and how it worked
out.** When an item finishes, its outcome and what it taught goes to `PROGRESS.md` and the item is
**deleted** from `TODO.md`. Never prepend a dated session retro to `TODO.md` — doing that every
session is what grew it to 4,076 lines before it was curated on 2026-08-02.

`TODO.md` structure to maintain: CURRENT STATE → priorities (P0/P1…) → work items, each with
**evidence / tasks / DoD** → REFERENCE (landmines, deprecated, success criteria). If it passes
~300 lines or gains dated session logs, migrate them to `PROGRESS.md` in that session.

Also: **validate before recording.** A single observation is not a confirmation, a delta inside 2sd
of the measured noise floor is "not resolvable", and a null from a metric you suspect is blunt is
"not yet measurable" rather than refuted. Label findings CONFIRMED / PARTLY CONFIRMED /
NOT REPRODUCED.
