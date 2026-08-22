# What to play, in what order, and what each answers

> Written 2026-08-22. **78 agent maps are installed across six sessions** — this file says which
> four matter and what question each one settles. Everything else is superseded or from an older
> line of work; a proposed cleanup is at the bottom.

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
