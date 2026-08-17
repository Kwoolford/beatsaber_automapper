# Pre-registered prediction: what Kyle will say about sets A and B

**Written before he told me.** Committed deliberately, because a prediction made after
the fact is worthless and this project's whole P0 is that the suite does not track his
ear. If my model of his ear is any good it should survive being written down first.

He has reviewed `[BEFORE]` vs `[AFTER]` (set A) and `[BEFORE]` vs `[PHASE]` (set B).

---

## ★ The headline prediction, and the one I care most about

**His dominant complaint will be FLOW again — not the thing either lever changes.**

He just said of the agent map: *"the main problem is the notes flow in a really odd
way."* Neither of these levers touches flow. Structure reuse changes *what pattern* is
written where; grid phase changes *when* the whole grid sits. Both are about placement
in time and space; neither is about how one note leads into the next.

So I expect the specific A/B answers to be lukewarm, and the real signal to be a repeat
of the flow complaint cutting across all of them. **If he comes back mostly talking
about flow rather than about repetition or timing, that is the finding**, and it says
the next work is flow and not another placement lever.

Confidence: **65 %**.

## Set A — `[BEFORE]` vs `[AFTER]`

**Prediction: he dislikes `[AFTER]`.** Repetitive, samey, possibly "lazy".

`[AFTER]` is structure reuse with the **dose cap OFF** and our own review doc marks it
🔴 *diagnostic only — overdosed*: flow 0.55 and idiom 1.21, both **failing** axes,
against `[AFTER CAPPED]`/`[BOTH]` which pass. It is in the set as the deliberate
"this is what too much looks like" arm.

⚠️If he actually played `[AFTER CAPPED]` or `[BOTH]` and said "after" loosely, I expect
"hard to tell" instead — capped reuse only bites where the music repeats.

Confidence he is negative on `[AFTER]`: **70 %**.

## Set B — `[BEFORE]` vs `[PHASE]`

**Prediction: mixed and mostly subtle, not the clean win the numbers suggest.**

- most likely outcome: *"can't tell"* or *"maybe slightly tighter"* on most songs — **50 %**
- clear preference for `[PHASE]` on most songs — **30 %**
- **`[PHASE]` actively WRONG on at least one song** — **20 %**, and this is the outcome
  worth watching for

**Why I am not predicting the clean win the numbers imply.** Two reasons.

1. **The lever left his four standing songs byte-identical.** On the songs whose grids
   were already right it does nothing — which means every song in set B is one where the
   grid fit was *poor to begin with*. A shaky grid is not rescued by shifting it; both
   arms may feel off, and the comparison comes out muddy.
2. **The failure mode is specific and audible.** A phase search shifts every note by one
   constant offset. If it locked onto an offset that scores well on onsets but is
   musically the *offbeat* — an eighth out — that song will feel distinctly wrong rather
   than slightly worse. The 74-better/0-worse result was measured on the alignment
   metric, which is exactly the metric this lever optimises, so it cannot rule this out.

## The base rate I am reasoning from

Numeric wins on this project have a **poor record** of reaching his ear:
- the masterpiece axes rank the map he called *"really empty"* **second best** and the
  one he graded **A+** fifth worst;
- the agent map matched the human `ebpm_burst` exactly, sat on the human nps median and
  had zero parity violations — and he found it disappointing.

So when a lever's evidence is entirely numeric, the prior should be **"he will not hear
much"**, and I am applying it here rather than talking myself out of it.

## What each outcome changes

| if he says | then |
|---|---|
| flow, again, across both | **flow becomes P0**, above notesheet work — and the notesheet is for arrangement decisions, not note-to-note motion |
| `[AFTER]` too repetitive | `mapctl reuse --vary 0.15` is **too low**; raise it, and cap reuse dose by default |
| `[AFTER]` reads intentional | reuse ships; `--vary` may drop toward 0 |
| `[PHASE]` clearly better | flip `BEAT_GRID_PHASE=search` ON — the circularity objection is answered by an ear outside the loop |
| `[PHASE]` can't tell | the alignment axis is measuring something **inaudible**; stop treating it as a quality axis |
| `[PHASE]` wrong on a song | the search has an **octave/offbeat failure mode**; constrain it to sub-beat offsets |
