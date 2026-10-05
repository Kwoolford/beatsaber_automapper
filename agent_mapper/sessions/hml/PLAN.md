# HEAVY METAL LOVE (twocolors) — Expert, hand-built BLIND. PLAN (charter step 1)

Written 2026-10-05 before any note. **No brief from Kyle for this song**: it is the blind test of the
agent mapper (Kyle, 2026-10-04: *"Rigorously evaluate your outputs"*). Two humans mapped this song
(`2f1e5`, `2ef6a`); I have NOT opened either map and will not until the build is finished. Only
leakage: the corpus listing printed the 2f1e5 map's BPM (125) and nps (4.3) while I picked the song.

## What I could and could not read
- **Tempo 125.0 BPM**, bar = 1.92 s. Grid fit r=0.27 ("weak"), but kick lands on slots 0/4/8/12 every
  chorus bar, so the phase is right.
- **Lyrics: readable** (en p=0.90, 263 words) — the best perception this project has had on a song.
- **Stems trusted** except `other` (z=2.0 → read as one lane). Kick is clean (219 hits, +9 dB accents).

## What the song is
A dance-pop / rock-flavoured track (twocolors are a German dance duo): a quiet sung verse over guitar
and piano stabs, then a **four-on-the-floor chorus** ("I want your heavy metal love / I just can't,
just can't get enough / so gimme, gimme all you got"), a quiet bridge, and the chorus twice to the end.

| bars | time | song | map should follow |
|---|---|---|---|
| 1 | 0:00 | silence | nothing |
| 2-6 | 0:02 | verse 1a: vocal over guitar stabs (slots 10/15), no kick | the VOCAL, sparse 8ths/quarters |
| 7-12 | 0:11 | verse 1b: same, guitar stab pairs at slots 10/15 | vocal + the guitar pair as a recurring hit |
| 13-16 | 0:23 | verse 1c: guitar 16th stabs build (XXXXX) under the vocal | rising: 8th streams with the guitar |
| 17-18 | 0:30 | pre-chorus "I want your heavy": kick at 18.8 | breath then the pickup |
| 19-33 | 0:34 | **CHORUS 1**: kick on every quarter, bass stab on 1 and the last 16th, vocal hook | the KICK as the backbone + the hook; doubles on the downbeat |
| 34-35 | 1:03 | "So babe": kick drops, guitar stabs | a short release |
| 36-41 | 1:07 | verse 2: quarter kick, busy vocal | vocal 8ths, kick quarters underneath |
| 42-48 | 1:18 | **bridge**: sparse, kick gone by 46, "when the smoke clears" | slow, wide, breathing; 48 crash 8ths = the build |
| 49-51 | 1:32 | "oh my god" → pickup "I want your heavy" (kick 51.8) | the pickup |
| 52-67 | 1:37 | **CHORUS 2 ×2**: same groove, vocal hook repeats more densely 60-67 | as chorus 1, busiest at 64-67 |
| 68 | 2:08 | guitar run ..XXXX...XXX + kick fill | the final run |
| 69 | 2:10 | end | a final hit |

## Where the difficulty comes from
**Groove, not speed.** 125 BPM 8ths are 240 ms — comfortable. The chorus is the hard part: a downbeat
double every bar, 8th motion between, wide diagonal reaches on the hook ("HEA-vy ME-tal LOVE"), and
one hand carrying the kick while the other answers the vocal in places. Target ~4.5 nps in choruses,
~2.5-3 in verses, ~1.5-2 in the bridge (choice from the song's energy, not from the human's number).

## Rules this map takes from the blindspot sweep (things every earlier map got wrong)
1. **No inward up-diagonals** (red UR / blue UL): human window median 0 %.
2. **Doubles are not mirrors** (human median 17 % mirror): double-pairs are written out per section,
   with mixed rows, outward diagonals, and the occasional both-up / mixed hit (humans: 41 % mixed).
3. **Hands may cross the centre** column (red in col 2, blue in col 1 ~10-20 % of the time) where the
   other hand is out of the way.
4. **One hand gets passages** (lead-hand runs), not 97 % strict alternation.
5. **No notes in the centre middle cells** (1,1)/(2,1) when anything follows within 0.5 s (vision block).
6. Run `outlier_sweep.py map` + `local` after EVERY section, not at the end.

## Done means
Built note by note via `agent_mapper/compose.py` (W6, promoted from `sessions/sotired`), verdict clean
of reds, UNNAMED ✅, THEN compared — first time — against both human maps: verdict `--vs`, the
fingerprint distance ours↔human vs human↔human, songprint, and Kyle's play when he is back.
