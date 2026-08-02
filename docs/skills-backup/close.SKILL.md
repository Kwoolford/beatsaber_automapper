---
name: close
description: Safely close out the current work session before a computer restart — pause/save running work without losing it, flush in-context findings into the project's progress/TODO markdown under the forward-only TODO convention, then commit and push. Use when the user says /close, "I need to restart", "wrap up before I reboot", "safe shutdown", or similar.
---

# /close — safe session close before a restart

The mirror of `/todo`. `/todo` *picks up* a session; `/close` *puts one down safely* so a
reboot loses nothing and the next `/todo` starts from a clean, written-down state. Work the
steps **in order**. This skill touches running processes and does a `git push`, so **confirm
with the user before anything destructive or irreversible** (killing a job that can't resume,
force-pushing, committing secrets).

The default project is the beatsaber_automapper repo, but this skill works for **whatever repo
the current session was working in** — detect it from context / `git rev-parse --show-toplevel`
rather than assuming.

## Step 0 — Orient (always first)
- Identify the repo(s) touched this session and `cd` to the root. Activate the venv if there is one.
- Skim the current conversation for **anything only in the context window** and not yet on disk:
  decisions, findings, numbers, a verdict, a half-written plan, file paths, "we should next…".
  That volatile context is the single most important thing to save — it's what a reboot erases.
- `git status` + `git stash list` so you know the starting state.

## Step 0.5 — Stop the autonomous research loop (if one is running)
`/todo` Step 6 leaves a self-paced research loop running that will otherwise keep firing
wakeups after the machine comes back. **End it first**, before touching any running jobs, so it
cannot re-enter mid-shutdown and queue new work on top of the close:
```
ScheduleWakeup(stop: true)
```
Say in the final report whether a loop was stopped, so the user knows `/todo` must be re-run to
restart research after the reboot.

## Step 1 — Pause / save running work without losing it
Find live work: `ps aux | grep -E "python|train|generate|overnight|node|nohup" | grep -v grep`
and `nvidia-smi` (GPU jobs). For each running job, pick the **least-lossy** option and confirm
with the user before killing anything:
- **Writes checkpoints / logs to disk** (training, an overnight sweep) → let it reach the next
  safe checkpoint if quick, else note the last checkpoint + exact resume command. Detached
  `nohup` jobs survive a terminal close but **not a reboot** — capture how to relaunch.
- **In-memory only, not resumable** → save what you can (dump partial results to the scratchpad
  or an `outputs/` dir), record where it stopped, then stop it. Never silently drop work.
- **Uncommitted edits mid-refactor** → either finish the thought or leave a clear TODO marker.
Record, for every paused job: what it was, how far it got, the artifact/log path, and the
**one-line command to resume it**.

## Step 2 — ★ VALIDATE before you write anything down
A close is where this session's claims become the next session's premises, so **check them
first**. A wrong number written confidently costs more than an unwritten one.
- Re-read what you're about to assert. Is each number **measured**, or inferred from one run?
- **A single observation is not a confirmation.** This project has twice had a hypothesis
  "confirmed" by one data point and falsified the next day (near-integer BPM as a crash cause;
  lower density raising onset precision).
- Know the **noise floor** before reporting a delta. If a difference is inside ~2sd of the
  measured per-axis sd, write "not resolvable", never a ranking.
- If a claim came from a metric you suspect is too blunt, record it as **not yet measurable**
  rather than refuted — especially when it originated from the user's ear, which in this project
  has been ahead of the metrics more than once.
- Label findings explicitly: **CONFIRMED / PARTLY CONFIRMED / NOT REPRODUCED.**

## Step 3 — Flush findings into the docs, under the forward-only convention
**★ THE RULE (beatsaber_automapper, and a good default anywhere):**

> **`TODO.md` is forward-looking ONLY. `PROGRESS.md` is the trail of what was done and how it
> worked out.** When an item finishes, its *outcome and what it taught* goes to PROGRESS.md and
> the item is **deleted** from TODO.md. A completed item is history, not work.

This exists because the previous version of this skill said "add a dated session retro to the top
of TODO.md" every close — and TODO.md reached **4,076 lines / 275KB** of interleaved history,
superseded handoffs, and success criteria that contradicted later measurements. Do not
reintroduce that.

**PROGRESS.md** — the session's narrative goes *here*:
- What was tried, the numbers, what worked and (more usefully) what did not.
- Negative results are first-class: record the lever that failed and why, so nobody rebuilds it.
- Mark validation status per claim (Step 2).

**TODO.md** — edit in place, never prepend a retro:
- Update the **CURRENT STATE** block (best config, what is promoted, what is running, key
  measured reference values).
- **Delete** finished items once their outcome is in PROGRESS.md.
- **Add / re-prioritise** work items. Each deserves a real breakdown written while the context is
  fresh: **evidence** (the numbers that justify it), **tasks**, and a **DoD**.
- Fold anything durable into REFERENCE: a new landmine, a newly deprecated approach.
- **Correct stale content you pass over.** Curation is the moment to catch a target or success
  criterion that later measurements have contradicted — fix it, don't copy it forward.

**README** — only if something user-facing changed (new flag, new default, new command).

**Persistent memory** (`~/.claude/projects/**/memory/`) — update the project memory file + its
MEMORY.md index line with the new live status. One fact per file; update, don't duplicate.

Convert relative dates to absolute. Write numbers and verdicts, not vibes.

## Step 3.5 — Curation check (cheap, do it every close)
```bash
wc -l TODO.md PROGRESS.md
```
If **TODO.md is over ~300 lines**, or contains dated session logs, "SUPERSEDED" blocks, or more
than one "NEXT SESSION" heading, it has started silting up again. Migrate the history to
PROGRESS.md **in this close** rather than letting it compound — it is minutes now and hours later.

## Step 4 — Commit and push
- `git status` and **review the diff** before staging — never commit blindly. Watch for secrets,
  large artifacts, or logs that belong in `.gitignore` (logs/outputs dirs are usually artifacts,
  not commits — check the repo's conventions).
- If on the default branch and the repo's convention is to branch, branch first; otherwise commit
  in place per how the repo has been operating.
- Stage intentionally (the doc updates + real code changes), commit with a message that summarizes
  the session — **including what was disproven** — then **push**. Use the commit/push footers from
  the main system prompt (Co-Authored-By / Claude-Session lines; PR footer if a PR is opened).
- If push fails (auth etc.), **say so explicitly**, leave the commit in place locally, and give the
  user the exact `git push` command to run via `!`. A committed-but-unpushed state still survives a
  reboot — an uncommitted one may not.

## Step 5 — Report: safe to restart
Finish with a short checklist the user can trust before hitting reboot:
- Running jobs: paused/saved (with resume commands) or still-running-and-reboot-will-kill-them (flagged).
- Docs updated: which files, and **what moved from TODO.md to PROGRESS.md**.
- TODO.md line count (proof it stayed forward-only).
- Open decisions that need the user — stated plainly, since these block the next session.
- Git: committed ✅ / pushed ✅ or ⚠️ pushed pending `<command>`.
- **One-line verdict: "Safe to restart" or "Not safe yet because X."**

## Conventions
- Confirm before killing a non-resumable job or force-pushing. Everything else, just do it and report.
- Prefer editing existing running-doc files over creating new ones.
- Don't invent work status — if a job's progress is unknown, say "unknown, last log line was X".
- This skill is about *not losing things*; when in doubt, over-save (scratchpad dumps are cheap).
- **Never let a close inflate TODO.md.** If the session produced a lot, that means PROGRESS.md
  should grow — not TODO.md.
