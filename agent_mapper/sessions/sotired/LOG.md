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

## What is not known
- **Nobody has played it.** Kyle's DoD is that he plays it and wants to keep playing.
- Two 4-bar 16th streams (7.8 s each) in hype 2 is the most demanding thing in the map; if it's
  tiring, split 65-68 / 69-72 into 2-bar phrases (the grammar allows it; costs 4 notes).
- The figures repeat within a section by design (lock-in), but the verse figure `A` is plain.
  If the verses feel flat, that's the first place for vocabulary.
