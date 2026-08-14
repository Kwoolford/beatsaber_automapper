# Overnight 2026-08-13 → 14 — what happened, and what needs you

**Nothing was promoted.** Every lever below is default OFF. Two levers were built and validated,
three plausible fixes were refuted, and three separate defects turned out to be one.

---

## 1. WHAT NEEDS YOU — 32 maps installed, three questions

Both sets are in the live BSManager instance (verified playable).

### Set A — structure reuse + crossover · your 4 standing songs
`AUTO <song> [BEFORE] / [CROSSOVER] / [AFTER CAPPED] / [BOTH] / [AFTER]` ·
📖 [`review_2026-08-11.md`](review_2026-08-11.md)

★**Play `[BEFORE]` vs `[BOTH]`.**

> **① Does the repetition read as INTENTIONAL or LAZY?**
> We copy a chorus and never vary it; a mapper copies and then varies.
> *Intentional* ⇒ raise the cap toward human parity. *Lazy* ⇒ the next capability is
> **variation-on-repeat**. *Can't tell* ⇒ leave it OFF; it is not worth its flow cost.

> **② Do the crossovers make it play better?** A straight quality question.

### Set B — grid phase · 6 corpus songs (NOT your standing four)
`AUTO <song> [BEFORE] / [PHASE]` · 📖 [`review_2026-08-14.md`](review_2026-08-14.md)

🔴**Your four standing songs are deliberately absent, and that is a finding.** The lever declined to
touch all four and produced **byte-identical** maps, so there was nothing to listen to. **The defect
it fixes does not occur on the songs you review** — which means the only way to hear it is on
unfamiliar songs. That trade is yours to accept or reject.

★**Lead with `BEcause` (Dreamcatcher)** — the largest correction in the cohort (+80 ms, onset
precision 0.456 → 0.900).

> **③ Does `[PHASE]` sit on the beat better than `[BEFORE]`?**
> Not "is it a good map" — these are corpus songs you have no baseline for.
> ★**"Can't tell" is a real answer**: it would mean a measured 0.62 → 0.35 axis improvement is
> inaudible, which is worth knowing about the axis.

**Still open from 2026-08-04**: is Fallen Kingdom empty compared to what our model *used to do*, or
compared to what the *song wants*?

### Your answers now become data
Verdicts used to live hardcoded in a script and as prose in `PROGRESS.md`. There is now a tracked
ledger, so the P0 preference loop can actually accumulate:

```bash
python scripts/record_verdict.py --song 2c352 --name BEcause \
    --better PHASE --worse BEFORE --quote "locks onto the beat, the other drifts"

python scripts/record_verdict.py --song 2c352 --name BEcause \
    --tie PHASE BEFORE --quote "can't tell them apart"     # a REAL verdict
python scripts/preference_screen.py                        # which axes agree with you
```
It refuses to record a verdict about a map that is not on disk, and treats *"can't tell"* as a
first-class answer rather than a non-answer.

---

## 2. The biggest finding — three defects are one defect

**W1** (*"can't find the core tempo/instrument a mapper adheres to"*), **W4** (*phrases abandoned
mid-vocal*) and **`follow_vocals`** (ours 0.020 vs human 0.149, **7×**) are three views of one root
cause: **Stage-1's representation does not carry the melodic instruments.**

The missing piece arrived last night. W4 reproduced at n=123 and *grew* (2.75× the human rate;
worse on **109 of 123** songs paired). Its obvious mechanism was written in our own docstring —
the density weighting literally says *"quiet ones thin out"*. **Measured, that is not the cause**:
removing almost all of it closes ~17 % of the gap, leaves 9 of 10 songs worse, and does not change
the note count. The notes go somewhere else entirely.

⇒**If redistributing the budget toward the abandoned phrase does not fill it, the model is not
proposing notes there. You cannot select what the model does not propose.**

This is the strongest evidence yet for the **Track B / representation** direction, and it came from
three independent measurements rather than architecture reasoning. ⚠️It does *not* revive v8 as
built — that arm's gain died at n=149. It says the **target** is right.

---

## 3. What was built (both default OFF)

| lever | result at n=149 | cost |
|---|---|---|
| **`BEAT_GRID_PHASE=search`** | songs >0.10 below human **39 → 21**; **alignment axis 0.62 FAIL → 0.35 PASS**, the first time it has ever passed; **74 better, 0 worse** | gain is partly circular — it optimises the same onsets the axis scores |
| **`BEAT_SUBDIV_AUTO=1`** | fired on **15 songs, zero false positives**; burst-rate ceiling **0.500 → 0.958**; other 129 maps bit-identical | 2 songs overshoot — the only two half-tempo songs that had no ceiling to lift |

The grid was anchored at `t=0` and the tempo fit's phase was computed, logged, and thrown away. At
half tempo the finest slot is twice as long in real time, so 28 songs were **hard-capped at exactly
half the human's burst rate** — no decode lever could ever have reached it.

---

## 4. What is now closed (so it does not get re-asked)

- **Predicting the grid phase** — refuted at n=149. Searching for it works; predicting it does not.
- **Octave detection**, four routes: energy balance, onset-gap density, ACF periodicity, and a
  cross-validated tempogram classifier. The best detector turned out to be **thresholding the
  detected bpm itself** — but only *after* the tempo fit.
- **Budget compensation** for the subdiv lever — exactly backwards: it removes the benefit where we
  want it and keeps the harm where we do not.
- **A keep/revert rule** for the 2 harmed songs — no human-free signal separates them.
- **The `fit_tempo` tie-break** as the cause of 2:3 errors — right mechanism, +1 song of 149.
- **The density weighting** as the cause of W4 (above).

---

## 5. Numbers worth carrying forward

- **Our tempo is right on 70.5 % of songs** measured against the mapper's own declared bpm (n=149),
  confirming the old "30 % wrong" figure on a 6× larger cohort. **Tempo is the largest quantified
  upstream defect**, and every cheap route into it is now closed — what remains is a real tempo model.
- **The `half`-tempo group is now the best-performing group in the cohort** (0.2× the base failure
  rate), down from being the defect that started the thread.
- **A seed re-draws the audio, not just the decode** — `seed_everything` seeds Demucs' random shift,
  so every seed-based error bar in the repo contains stem variance.
- **`handrole` is not translation-invariant** and cannot evaluate any lever that moves notes in time.
- **"Main beat coverage" usually means eighth notes** (2× the declared bpm on 104/144 songs). On the
  songs where it *is* the mapper's beat, humans cover **0.90** and we cover **0.55**.

---

## 6. The alignment defect is now fully accounted for

It started at **39 songs** more than 0.10 below their human. As of this morning every one is
attributed:

| | how many | what it needs |
|---|---|---|
| fixed by grid phase | ~18 | ✅ done (`BEAT_GRID_PHASE=search`, awaiting your ear) |
| 2:3 / odd-ratio tempo misreads | ~10 | a real tempo model |
| pure selection defect | ~10 | Track B |

The last group is established by **elimination**, not assumption: not tempo, not phase, not onset
supply (4.5 onsets available per note we emit), and **not song difficulty — the human scores 0.943 on
exactly those songs, better than the 0.934 they score on the ones we handle fine.** Same note count,
same onset budget; we just match less of it.

---

## 7. What I would do next, in order

1. **Your three answers above.** Two levers and the next capability all hinge on them.
2. **The 2:3 / odd-ratio tempo misreads** — 4.6–5.5× the base alignment failure rate, from 11 % of
   the cohort. Needs a real tempo model, not a fifth statistic.
3. **Track B**, with W1 + W4 + `follow_vocals` as one acceptance target rather than three.

⚠️**Both 2 and 3 are multi-day builds whose direction you flagged as yours to set** — the v8 /
representation direction is explicitly *"worth re-opening after the preference loop exists to judge
it, not before"*. I have not started either. The unblocked stack is genuinely thin now; the next
real move is your ear.

*Full evidence in `PROGRESS.md`; forward work in `TODO.md`. Everything committed and pushed;
557 tests pass.*
