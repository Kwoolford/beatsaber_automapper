# SO TIRED ROCK (NUEKI): Expert. Build log and rationale

**Deliverable:** `outputs/sotired/SO_TIRED_ROCK_Expert.zip`: 840 notes, 2 walls, 2 arcs, 123 BPM,
0 parity violations, 0 resets, 0 collisions. Built 2026-09-18 from `spec.txt` (the rhythm, bar by
bar) + `compose.py` (figures, and the hand plan between hits) + `mapedit.py` (elements).
Rebuild: `agent_mapper/sessions/sotired/build.sh`, then the four `mapedit` ops at the bottom.

Kyle was away and asked for no questions ("just start with a vision, and build it out"), so every
open question in PLAN.md was decided and is logged here for him to argue with.

## What the map is, section by section

| section | bars | notes | nps | what the player does |
|---|---|---|---|---|
| intro | 4-8 | 26 | 2.7 | band lands on bar 4. **Left hand alone** on the guitar stabs (5), **right hand alone** as the drums pick up (6), and the drums-out guitar lick (8) is a **left-hand solo** |
| verse 1 | 9-16 | 64 | 4.1 | the singer's rhythm on 8ths, 16th pairs only where he sings two. Hits on the crashes at 9 and 13 |
| pre-chorus | 17-24 | 72 | 4.6 | vocal-led, diagonals and top-row chops; bar 24 = first 16th run (guitar + kick fill) into the chorus |
| chorus 1 | 25-32 | 90 | 5.8 | **a down-hit on every bar's downbeat**; 26-30 follow the guitar break (no vocal); bar 29's crash = down-hit then up-hit to both top corners |
| **HYPE 1** | 33-40 | 114 | **7.3** | 33-34 build (drums thin), then **a 4-bar and a 2-bar 16th stream** following the fast picking, with a hit to open each; the breath at 40.12 is exactly where the guitar breaks |
| breath | 41 | 4 | 2.1 | one hit, both hands **arc** through the guitar's silence, side **walls** frame it |
| verse 2 | 42-48 | 60 | 4.4 | vocal-led, simple |
| bridge | 49-56 | 68 | 4.4 | top-row chops; 53 = left hand carries the guitar through the vocal rest; 55 two crash hits; 56 drums out = 16th run-up |
| chorus 2 | 57-64 | 72 | 4.6 | lighter than chorus 1, as the song is (guitar 7/bar vs 10); opens and closes on hit pairs |
| **HYPE 2** | 65-72 | 120 | **7.7** | **two 4-bar streams**, a top-row "windmill" figure (inner-top chop, outer-mid up); the kick fill at 72 rides the end |
| final chorus | 73-80 | 86 | 5.5 | the busiest chorus: down-hit every bar, the last vocal run at 79 |
| outro | 81-89 | 64 | 3.6 | right hand alone to open; 87 = the four kicks as down / up / down hits; **final slam** from the top on the last chord |

**Whole file 4.77 nps, played span (bars 4-89) 5.0 nps. Hype sections 7.3-7.7.**

## Decisions made without Kyle, and why

1. **Density: 5.0 nps where there are notes, not ~6.** At 123 BPM, 6 nps averaged over the whole
   song is 12.2 notes a bar everywhere including the intro. That means 16ths running through the
   body, which is the opposite of "simple flows". I kept the body at 4-6 so the two hype sections
   stand out at 7.3-7.7 (1.6× the body), and stayed well under the 6.18 Kyle once called
   unplayable. **Lever if it's too easy:** chorus 2 and verse 2 (4.4-4.6) take 10-single bars like
   chorus 1 without breaking the double grammar below.
2. **The hype sections are 33-40 and 65-72.** Chosen from the guitar stem (the only stem whose
   control passes and which separates the sections): short stab events at 104-128 ms, a 16th at
   123 BPM, in exactly those bars. Not heard, read.
3. **The difficulty in the drops is speed + travel, never crossovers.** Each hand stays in its own
   half all song; the streams reach from the bottom row to the top row at 244 ms per swing per hand.
4. **Verses follow the singer, heavy sections follow the guitar.** The lyrics are unreadable (Whisper
   filler), but the vocal onsets are real and they are what a player hears in the verses.

## What I learned building it (tooling, for the next map)

- ★★**A double is a hit, and the hit dictates the phrase.** Both hands must arrive at it in the same
  phase, which means the number of singles between two doubles must be **even** (≡ 2 mod 4 for a
  down-hit, or ≡ 0 with a quarter-note gap where one hand takes two in a row). My first draft put
  doubles on the snare backbeat with 1 or 3 8ths between: **75 of 99 doubles were mixed (one hand
  up, one down) or both up**, with 0 parity violations and 0 resets. **Nothing in `mapctl check`,
  `verdict.py` or mapjudge reads it.** `compose.py` now plans hands phrase by phrase and reports any
  phrase that can't land its hit; the finished map has **0 mixed doubles**, and its 6 up-hits are
  all deliberate (hit pairs, down then up).
  🔴**2026-10-04 correction (W5 control, 4 737 human maps): humans write mixed doubles CONSTANTLY** —
  median map 41 % mixed / 26 % both-up / 33 % both-down, on the downbeat as often as off it. The
  draft's 76 % mixed was the 98th percentile; the shipped map's **88 % both-down sits beyond 2 human
  maps in 4 737**. The fix overshot into a monotone double vocabulary. The "singles between hits must
  be even" rule is a choice, not a playability law. If the doubles feel samey in play, this is why.
- **A double inside a 16th stream needs a free 16th on each side**, or one hand swings twice in
  122 ms. A 2-bar stream phrase is therefore hit + 26 sixteenths + a 3-slot breath; a 4-bar one is
  hit + 58.
- **Writing a syncopation, I drifted onto the "e"/"a"**: after a 16th pair, the rest of the bar
  followed on odd slots. `verdict.py`'s FLOW caught two bars of it, and a spec-wide scan found 45 notes.
  All were moved one 16th earlier onto the 8th grid, which also removes the "nothing leading in" defect.
- **Strict alternation never gives one hand a passage** (verdict ABSENCE: 0 vs corpus median 10). The
  song has natural places for it (stabs, a drums-out lick, vocal rests): 5 passages now.
- The onset lane is saturated (every slot) from bar 9 on for this song, so it can't judge alignment
  past the intro. The per-stem lanes (`lanes.txt`) were what I authored against.

## Backstop (last, not the reason to ship)
`verdict.py`: SHIP? YES, nothing located. FLOW ✅, D2 ✅, playability ✅ (incl. the new collision
check), lead-hand passages 5 ✅, judge PASS p=0.162 (deaf: no onset cache for this song). Nine of
the eleven codes are ⚪ because there's no human map of this song, so **the page says very little
here**. What I actually read is above.

## 2026-10-04 — v2: the blindspot sweep's first finding on this map
`scripts/outlier_sweep.py local` (8-bar windows vs 19 414 human windows at the same local density)
flagged **all three choruses** (25-32, 57-64, 73-80): the chorus figure `C` ended each hand's cycle on
an **inward up-diagonal** (red UR at (1,0), blue UL at (2,0)) — 24 % of swings there, where the human
window median is **0 %**, and it made every red swing in the chorus a 135° turn instead of the 180°
down/up. 60 notes, all chorus. **v2 = those 60 cut straight UP; nothing else changed**
(`outputs/for_kyle_2026-10-04/SO_TIRED_ROCK_Expert_v2.zip`, sub-name "v2"; figure `C` in compose.py
updated so a rebuild reproduces it). Local windows flagged: 3 → **0**; parity/resets/collisions 0.
Still outside humans map-wide (deliberate "simple flows" choices, kept for Kyle's ear): hands never
cross the centre (R col1 = L col2 = 0: 0-1 of 400 humans), alternation 97 % (human p95 89 %), every
double same-row and 88 % both-down, red uses 10 shapes (human p5 16 — v2 narrowed it from 11).

**v3 (2026-10-05)** = v2 + varied doubles. The new `dbl_mirror` feature read v2's doubles as **96 %
exact mirror images** (blue = red reflected; 0 of 399 humans; human median 17 %) — Kyle's ML-era
words for it were *"both hands do the same thing"*. First attempt (move notes outward) widened the
doubles to dx 2.20 (1 of 399 humans) — the sweep caught that too. Final rule: the 14 outer pairs
(0,0)/(3,0) move one hand inward, alternating; every other inner pair (1,0)/(2,0) turns one hand's
down into an OUTWARD down-diagonal, alternating. 24 notes, positions/diagonals only, parity
untouched. Mirror and width flags clear; windows 0/11; verdict UNNAMED ✅.
`outputs/for_kyle_2026-10-04/SO_TIRED_ROCK_Expert_v3.zip` (sub-name "v3").

## What is not known
- **Nobody has played it.** Kyle's DoD is that he plays it and wants to keep playing.
- Two 4-bar 16th streams (7.8 s each) in hype 2 is the most demanding thing in the map; if it's
  tiring, split 65-68 / 69-72 into 2-bar phrases (the grammar allows it; costs 4 notes).
- The figures repeat within a section by design (lock-in), but the verse figure `A` is plain.
  If the verses feel flat, that's the first place for vocabulary.
