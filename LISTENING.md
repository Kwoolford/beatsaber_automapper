# What to play, in what order, and what each answers

> Written 2026-08-22. **78 agent maps are installed across six sessions** — this file says which
> four matter and what question each one settles. Everything else is superseded or from an older
> line of work; a proposed cleanup is at the bottom.

## ⭐⭐ ADDED 2026-08-24 — `[PBASE]` vs `[PCAL]`: **is the map ON THE MUSIC?**

★**Play this one first.** It is the biggest measured change in the session and it targets the thing
you named by ear three sessions ago — *"slightly off beat"*.

**What was found**: our bar grid sits **0.053 beats EARLY** of the grid a human mapper used for the
same song — **18 of 18 songs**, measured against their maps, which our onset detector never touched.
`[PCAL]` corrects that measured bias; `[PBASE]` is today's default.

| song | `onset_precision` | agreement with the human's own note times |
|---|---|---|
| **Hunger** | 0.768 → **0.917** | 0.484 → **0.667** |
| アリスブルー | 0.890 → **0.937** | 0.547 → 0.588 |
| Digital Life Hacker | 0.914 → **0.941** | 0.656 → 0.689 |
| Fallen Kingdom | 0.825 → **0.866** | 0.600 → 0.661 |

★**Held-out validated**: the constant was fitted on half the songs and applied to the *other* half,
where it still lifts `onset_precision` 0.888 → 0.917 and human agreement 0.642 → 0.709 — capturing
**99 %** of what a per-song oracle achieves. ★**And it is not a corpus artifact**: padding the audio
by 137 ms moves the fitted phase *with the music* (6/6 songs), so it should work on any song you
bring.

⚠️**Every note moves by ~20 ms**, so unlike the `[W3]`/`[W5]` A/B below this is not a
same-notes comparison — note counts shift slightly too as more events land inside slots.
🔴**Default is still OFF** (`--phase-calibrate`). Every number above is a number, and
`BEAT_GRID_PHASE` once "fixed" 18 songs on the axis while you still heard the defect. **Start with
Hunger** — it moves furthest.

---

## ⭐ ADDED 2026-08-24 — `[W3]` vs `[W5]`: the cut-direction A/B

**The question**: we play **vertical** where humans play **diagonal** (vertical 0.77 vs human 0.48;
diagonal 0.24 vs 0.42) — P1.2, the largest non-circular gap. **Do the diagonals actually feel
better, or only measure better?**

**What was found**: the cause is located. The candidate pool the sampler draws from is already at
the **human** diagonal share (0.404 vs 0.415) on all 23 songs — then `_pick` keeps only the
`width`(=3) **most FREQUENT** candidates, and human idiom frequency is vertical-dominated, so the
truncation strips diagonals **by construction**. Raising `width` puts them back.

`outputs/width_ab_2026-08-24/` — four songs, **`W3__` = today's default, `W5__` = the candidate**.
★**The note TIMES, the note COUNT and all 89 walls / 90 arcs / 16 chains are identical between the
two.** Only cut directions differ, so anything you feel is the directions.

| song | diagonal share W3 → W5 |
|---|---|
| `1f767` | 0.223 → **0.492** (biggest change — and it overshoots human 0.415) |
| `1f8d6` | 0.250 → 0.362 |
| `1f913` | 0.200 → 0.326 |
| `1f333` | 0.252 → 0.281 (barely moves — read this one last) |

⚠️**Play `1f767` first**: it is where the knob does the most, and where it overshoots. If W5 feels
*worse* there but better on `1f8d6`, the answer is an intermediate or per-song setting, not a new
default.

🔴**Why the default was NOT changed**: `width=3` was chosen by **reading two maps side by side** —
by eye — and a decision made by eye should not be reversed by axis numbers alone. On the numbers
`width=5` is the better arm (closest to human on **both** `idiom_top50` 0.449 vs human 0.404 and
`idiom_coverage` 0.929 vs 0.909, where `width=3` overshoots to 0.557/0.990 — the *"more human than
human"* over-purity the vocabulary depth was tuned to avoid). Its only cost is `idiom_local`
p15 → p27. ★**Reproducible**: 23 songs × 3 seeds, arm gaps far larger than the ±0.005–0.021 seed
spread. **But that is still only numbers — this A/B is the decision.**

Try it on any song: `--width 5`.

---

## The 4 that matter, in order

### 1. `AUTO Fallen Kingdom (2022 Remap) [V2]` vs `[FULL]` ← **most valuable**
**The question**: is *"Fallen Kingdom feels really empty"* about missing NOTES or a missing
PHYSICAL LAYER?

`[V2]` is notes only. `[FULL]` is the identical notes plus **89 walls, 90 arcs, 16 chains**.
The note data is **byte-identical** between them — only the other elements differ.

★**Five separate instruments failed to explain "empty", and every one of them looked at notes.**
Until 2026-08-22 we shipped zero walls while **96 % of human maps have them** (median 89).
🔴**No metric can judge this** — adding walls/arcs/chains moves every axis by exactly 0.000, and the
judge's p-value is identical for both. **Your ear is the only instrument that works here.**

### 2. `AUTO Hunger [V2]` — the current best on a song you know well
Everything measured this session, at the shipped defaults: pulse held per phrase, lead hand per
passage, doubles at the human rate, density normalised. **1222 notes, 4.50 nps, 0 parity
violations.** The question is simply: **is this a map you would keep playing?**

### 3. `AUTO driveaway [AGENTBUILT]` — proof it works on a song nobody tuned for
Built cold, off-corpus, hand-planned section by section. **274 notes, 6 of 7 axes within a few
points of the human median.** The question: **does it hold up on a song the suite has never seen?**

### 4. `AUTO アリスブルー [AUTOPULSE]` vs `[AUTOLEAD]` — one knob, if you want to feel it
Identical except the lead-hand bias. `[AUTOLEAD]` lets one hand carry a passage instead of strictly
alternating. ⚠️Smallest difference of the four — skip if short on time.

## What I most need to hear
1. **Does `[FULL]` feel less empty than `[V2]`?** Yes/no settles a three-day-old question.
2. **Where does `[V2]` on Hunger stop being fun?** A timestamp is worth more than any metric —
   `python scripts/record_verdict.py` captures it, or just tell me the moment.
3. **Two decisions only you can make** (both change what a PASS *means*):
   - **P0.1** — the judge rejects above ~7.18 nps because humans rarely go there. You play faster
     than that. Should a requested difficulty be exempt from the density gate?
   - **P0.2** — the gate accepts **65 %** of maps shifted a quarter-beat off the music. An alignment
     floor fixes it at the cost of rejecting ~4 % more human maps. Worth it?

## ⚠️ Proposed cleanup — needs your OK, nothing deleted yet
**Superseded by today's build** (safe to remove, 40+ maps):
`[BEFORE]` `[AFTER]` `[AFTER CAPPED]` `[BOTH]` `[BASE]` `[DENSER]` `[CHAINS]` `[AUTOBASE]`
`[AUTOPULSE]` `[AUTOLEAD]` `[READ]` `[WALLS]` `[B v8_plus_bonus]`, plus the whole
`DigitalLifeHacker`/`FallenKingdom` spelling (an older naming convention that duplicates the
current one).

**Keep regardless**:
- `[CROSSOVER]` (4 maps) — never judged, oldest open question on the board
- `[PHASE]` (6 maps) — the grid-phase arm, tied to live work
- `AUTO Hunger [AGENT]` — the 2026-08-14 hand-built map, not reproducible from current code
- The four listed above

Say the word and I'll remove the superseded set.
