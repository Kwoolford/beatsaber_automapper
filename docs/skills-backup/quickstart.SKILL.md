---
name: quickstart
description: Drive a beatsaber_automapper research session hands-free — same routine as /todo (architecture items → run completion → output evaluation → queue the next experiment → autonomous loop) but NEVER asks the user a question and never blocks on an answer. Use when the user says /quickstart, or wants to kick off the automapper research loop and walk away.
---

# /quickstart — Beat Saber Automapper research session, hands-free

Identical to `/todo` in what it does, with one hard rule layered on top:

> **NEVER ask the user a question. Never block waiting for a human.**
> The user launches this and walks away. A question left on screen costs a whole
> night of GPU time — that is the exact failure this skill exists to prevent.

## The no-questions rule (read this before anything else)

- **Do not call `AskUserQuestion`. Ever.** Not for lever promotion, not for arm
  selection, not for "which of these two candidates". Not even at the very end.
- **Do not end a turn on an open question.** Never write "let me know which you
  prefer" / "should I proceed?" / "waiting on your call" and stop. If your next
  sentence would be a question to the user, you have already failed the skill.
- **Do not stop to report between steps.** `/todo` says to pause between steps
  that change the plan; here you keep going and report as you pass.
- When you hit a genuine fork, resolve it yourself with the **DECIDE-AND-LOG**
  procedure below, then keep working. There is always more work that does not
  depend on the answer — do that work.

### DECIDE-AND-LOG (what to do instead of asking)
1. Pick the option that is **reversible and default-OFF-preserving**. Between a
   change to production defaults and a new gated arm, always take the arm.
2. If the fork is a tie on evidence, pick the one that **serves Kyle's stated
   priority** (currently: handrole / "not Expert+" difficulty / no for-sport
   diagonals), and say so.
3. Write the fork into `TODO.md` under a **`## ❓ DECISIONS TAKEN WITHOUT KYLE`**
   heading near the top: the fork, the option taken, the assumption, and what
   would reverse it. `/close` surfaces this block so he can override in one read.
4. Keep working on everything that does not depend on it.

### The one thing you still may NOT do alone
**Do not bake a lever into `generate.py` / `layout_model.py` production defaults
without Kyle having played the maps.** That rule predates this skill (2026-07-27:
a lever that scored well on paper was unplayable in practice) and hands-free mode
does not relax it — it is a *quality* rule, not an approval formality.
Instead: render the candidate, `SendUserFile` it (a file drop is not a question —
it does not block), log it under DECISIONS TAKEN WITHOUT KYLE as "awaiting ears,
NOT promoted", and **go work on something else**. Never idle waiting for a reply.

---

## Cold start (always first)
```bash
cd /home/kyle/repos/beatsaber_automapper && source .venv/bin/activate
```
Read context before acting:
- `MEMORY.md` index + the `project_beatsaber_automapper.md` /
  `beatsaber_v8_representation_theory.md` memories (live status + the key notes:
  the ~13–15s **silent drop**, **flat ~8 NPS** density ignoring song structure,
  **late-song / final-chorus collapse**, "for-sport diagonal swings").
- `TODO.md` — forward-looking work only (history is in `PROGRESS.md`); the live task stack,
  including any `⏭️ NEXT SESSION` handoff and `❓ DECISIONS TAKEN WITHOUT KYLE`
  block from a previous `/quickstart` run.

## Step 1 — Remaining architecture items
Read the top of `TODO.md` and the live task stack. Classify each item DONE / DEAD
/ live; report the live ones only, don't re-litigate dead ones. If a previous
`/quickstart` left decisions in the DECISIONS block and Kyle has since responded
in the conversation, fold his answer in; if he hasn't, the logged decision stands
— do not re-raise it.

## Step 2 — Is the last run complete?
```bash
ps aux | grep -E "python|train|preprocess|overnight" | grep -v grep
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
ls -t logs/overnight/*.log | head; tail -30 <newest overnight log>
```
GPU idle + no train/generate proc + the script printed `COMPLETE` ⇒ done.
**If a job is still running: do NOT queue on top of it and do NOT stop either** —
go straight to CPU-only work (Step 6.2) and set a `ScheduleWakeup` fallback.

## Step 3 — Have the outputs been evaluated?
Find the run's artifacts (`outputs/<date>/`, `outputs/task*/`, `outputs/v8_poc/`,
`experiments/leaderboard_v7.jsonl`). A run counts as evaluated only when its DoD
metric has **a number AND a verdict** (`align_*.json`, `density_*.json`,
`*_summary.log`). Any finished-but-unevaluated run is the highest-priority work.

## Step 4 — Evaluate + review the key notes
```bash
python -m beatsaber_automapper.evaluation.scorecard <zips> --label <arm>   # 5-axis verdict
python scripts/eval_density_corr.py --audio <song> --map <zip> --difficulty Expert --json <out>
python scripts/eval_alignment.py    --audio <song> --map <zip> --difficulty Expert --tolerance-ms 50 --json <out>
```
Check the key notes explicitly: drop @ ~13–15s; does density track structure
(flat ~8 NPS = FAIL, DoD `eval_density_corr` **≥ 0.41**); late-song collapse.
Use `scorecard.py` directly for the 5-axis read — `eval_sweep.py`'s own rhythm
table is broken and silently prints `nan`.
Write the **outcome** into `PROGRESS.md`, prune the finished `TODO.md` item, **and** update the
project memory as you go,
not at the end — the session may be interrupted.

## Step 5 — Queue the next experiment
Pick the highest-value live item whose DoD is unproven. Don't ask which one; rank
by (evidence value ÷ GPU cost) and state the ranking in your report.
1. Minimal code change, **gated behind a flag, prior behavior stays the default**.
2. Smoke-test the new path fast (load ckpt + one forward) before spending a night.
3. `scripts/overnight_<date>.sh` with explicit **arms + a control**, each
   generating + evaluating, ending in a `python - <<'PY'` block that prints the
   **verdict logic** (what result means DoD-met vs pivot). Log to
   `logs/overnight/<name>.log`.
4. Launch detached: `nohup bash scripts/overnight_<date>.sh >/dev/null 2>&1 &`,
   confirm the first arm clears the new code path, then let it run.

## Step 6 — Autonomous research loop (the point of this skill)
As soon as Step 5 has something running — or immediately, if the next item is
CPU-only — enter the self-paced loop and keep going until `/close`:
```
Skill(skill="loop", args="/quickstart — autonomous research iteration: advance the live task stack, never ask")
```
Each iteration:
1. **Harvest** — evaluate any finished run against its DoD; write the verdict to
   `PROGRESS.md`, prune the finished `TODO.md` item, and update memory. Never leave a finished
   run unevaluated.
2. **Advance** — take the next live item and build it. Prefer CPU-only work (eval
   axes, corpus mining, analysis, renders) while a GPU job runs so they overlap.
3. **Generate new hypotheses** when the stack runs low — from evidence in hand
   (control-battery blind spots, metric gaps vs the human corpus, surprising
   numbers in the last run). Add to `TODO.md` **with an explicit DoD** first.
4. **Re-pace** — `ScheduleWakeup` matched to what is actually being waited on:
   1200s+ fallback when a tracked job will notify, shorter only when polling
   something the harness cannot see.

Loop discipline:
- **One GPU job at a time**; use the wait for CPU work, never for idling.
- **Commit + push each completed unit of work** so a restart loses nothing.
- **Every new metric passes the control battery** (`scripts/audit_eval_suite.py`)
  before it steers the generator, scored by cohort shift/spread — never per-map
  distance-to-median (that reproduces the h_dist saturation, `docs/eval_suite_v2.md`).
- **Report each iteration**: what finished, what it showed, what is now running.
- **Stop only on `/close`.** Not on a fork, not on a null result, not on a
  candidate awaiting Kyle's ears — those all go in the DECISIONS block and the
  loop continues.

## Conventions (learned the hard way)
- `scripts/generate.py` takes `audio` as a **positional** arg, and needs `--v7`
  or it silently uses untrained models.
- Load beat checkpoints with `strict=False` (field additions break strict load).
- **Don't select checkpoints by `val_token_acc` / `val_f1_avg_tol`** — they
  anti-correlate with quality. Select by the v2 suite / alignment F1 / density-corr.
- Production defaults: layout `version_10` (ctx16 + song-memory ON), beat
  `version_4`, `section_gate="loud_only"`.
- **Never run two sweeps against one cache** (`.sweep.lock` in `outputs/eval_sweep_cache/`).
- **Validate every lever on the full 24-song set** — single-song probes lie
  (1f333 is half-tempo; beat-domain axes are distorted there).
- Noise floor: flow ±0.03 / rhythm ±0.08 / idiom ±0.09 / handrole ±0.29.
- Logs dirs under `logs/` are artifacts, not commits.

## Output
Every report ends with: live architecture items, run status, what the eval showed
on the key notes, what is now queued (with its DoD and how to read the verdict),
and any **DECISIONS TAKEN WITHOUT KYLE**. It ends with a **statement**, never a
question — if the session has anything left to run, the last line says what is
running and when you will check it, not what you need from him.

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
