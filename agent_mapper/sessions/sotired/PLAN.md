# SO TIRED ROCK (NUEKI): Expert, hand-built. PLAN (charter step 1)

Written 2026-09-17, before any note was placed. Kyle's brief: *"medium tempo expert level map with
~6 nps and simple flows with the hype fast drop sections to still be hard and fast tempo."*

## What I could NOT see (listed first so none of it steers the map)
- **Lyrics: unusable.** Whisper returned "Thank you." ×6 at language p=0.25, which is its stock
  filler. The vocal stem *is* active (7-12 onsets/bar from bar 10 on), so there is a singer, but I
  have no words and will not map to phrases I can't read.
- **Section letters: unusable.** `structure.py` calls the silent intro and the drops the same
  section ("A ×4"), and its lyric control couldn't run. Sections below come from the drum kit and the
  guitar, not from the letters.
- **Energy: flat.** It reads 0.75-0.82 on nearly every bar from 9 to 88, so it can't locate a drop.
- **Which drum is which: partly.** The kit hits land on the quarter-note slots every bar, so **the
  grid phase is correct**, but "snare on all four beats, kick every other bar" is almost certainly a
  labelling error. I trust *where* the drums hit, not *which* drum it was.
- **No human map of this song**, so `--vs` is ⚪ and there is no tutor. This plan is the only reference.

## What I trust
- **Tempo 123.0 BPM, constant.** An independent comb scan puts 123.0 at 3.1× the mean onset
  strength, with every neighbour ±0.5 BPM at ~1.05. The `r=0.24 ⚠️weak` flag is the fit metric's
  scale, not a wrong tempo.
- **Guitar stem:** its control passes (z=+9.6). It is the instrument that tells the sections apart.
- **Drum stem:** control passes (z=+19.9). Gives the pulse, the crashes (phrase starts) and two fills.

## What the song is
A mid-tempo (123 BPM) J-rock track, ~2:56. There is a steady quarter-note drum backbeat all the way
through, and **the guitar decides how intense each section feels**: palm-muted chops in the verse,
ringing 8ths in the pre-chorus, dense stabs in the chorus, and **two bursts of fast staccato picking
at 16th speed**. Those two bursts are the "hype" sections Kyle means.

| bars | time | what the song does (guitar events/bar, gap between hits) | name |
|---|---|---|---|
| 1-3 | 0:00 | near silence, bass/piano swell | intro swell |
| 4-7 | 0:05 | band enters, sparse guitar stabs, drums partial | intro riff |
| 8 | 0:13 | **drums drop out, guitar plays a 15-note run** | pickup lick |
| 9-16 | 0:15 | backbeat, guitar chops ~5/bar (342 ms), vocal enters bar 10 | verse 1 |
| 17-24 | 0:31 | guitar rings in 8ths ~9/bar (209 ms); bar 24 = guitar + kick fill | pre-chorus |
| 25-32 | 0:46 | chorus: guitar ~10/bar (139 ms), crashes at 29 and 31 | chorus 1 |
| 33-40 | 1:02 | **HYPE 1: staccato picking 16.8/bar (104 ms)**; drums thin at 33-34, full from 35 | drop 1 |
| 41 | 1:18 | guitar nearly stops (3 hits) | breath |
| 42-55 | 1:20 | steady stabs ~11/bar (139 ms), vocal throughout | verse 2 / bridge |
| 56 | 1:49 | drums out, guitar 14-note run | pickup |
| 57-64 | 1:49 | chorus, lighter guitar ~7/bar (232 ms) | chorus 2 |
| 65-71 | 2:04 | **HYPE 2: continuous 16ths, 16/bar (128 ms)**, every bar | drop 2 |
| 72 | 2:20 | **kick fill (8 kicks)** | fill |
| 73-80 | 2:22 | final chorus: guitar ~12/bar, crash at 76 | chorus 3 |
| 81-86 | 2:38 | guitar thins to ~9/bar | outro |
| 87-88 | 2:49 | kick fill, snare out, last guitar run | ending fill |
| 89-91 | 2:53 | ring-out | end |

⚠️I have not *heard* any of this; it all comes from reading the stems. The 16th-speed picking in the
two hype sections is the call I'm most confident about (short stab events, spaced tighter than a
16th at 123 BPM is 122 ms). The chorus/verse boundaries are the least certain.

## Where the difficulty comes from
**Speed, only in the two hype sections.** Everywhere else is **simple flow**:
- **Body (verses, choruses, bridge):** notes on 8ths and quarters following the guitar's attacks.
  Strict down/up alternation, each hand stays in its own half (left in columns 0-1, right in 2-3),
  mostly rows 0-1, angles change at most 45° per swing, **no crossovers, no resets, no awkward
  angles**. A double (both hands at once) on crash downbeats to mark phrase starts, and not much
  else. It should read at a glance.
- **Hype 1 & 2:** **16th-note streams that follow the picking**, alternating hands, anchored with a
  double on each bar's downbeat. The hard part is speed plus **wider lateral travel** (streams that
  sweep across all four columns). Still parity-clean: fast, not awkward.
- The two **pickup licks (bar 8, bar 56)** and the **fills (24, 72, 87)** get short 16th runs as a
  taste of what's coming. They're the only 16ths outside the hype sections.

## Density budget: ~6 is the *average*, spent unevenly on purpose
At 123 BPM, 8ths alternating between hands = **4.1 nps** and a 16th stream = **8.2 nps**.

| section | bars | target nps | ~notes |
|---|---|---|---|
| intro swell + riff | 1-7 | 0 → 2.5 | 20 |
| pickup lick | 8 | 4 | 8 |
| verse 1 | 9-16 | 4.5 | 70 |
| pre-chorus + fill | 17-24 | 5 | 78 |
| chorus 1 | 25-32 | 6 | 94 |
| **HYPE 1** | 33-40 | **8.5** | 133 |
| breath | 41 | 1 | 2 |
| verse 2 / bridge + pickup | 42-56 | 5.5 | 161 |
| chorus 2 | 57-64 | 5.5 | 86 |
| **HYPE 2** + fill | 65-72 | **9** | 140 |
| final chorus | 73-80 | 6.5 | 101 |
| outro + fill | 81-88 | 5 | 78 |
| end | 89-91 | — | 2 |
| **total** | | **5.6 over the whole file, ~5.9 over the played span (bars 4-88)** | **~975** |

That keeps the average **just under 6** (6.18 is where Kyle once said "unplayable") while the drops
run at 8.5-9. **For Kyle to argue with:** if the body feels too easy, the lever is verse 1 and the
bridge (4.5 / 5.5). Raising them to 5.5 / 6 adds ~40 notes and pushes the average to ~6.1.

## Elements
Walls: a few only, to mark the breath at 41 and the pickup at 56, and nothing in the hype sections.
Arcs: on the guitar's ringing 8ths in the pre-chorus (17-23), where a held swing reads naturally.
Chains: none on the first pass.

## How I'll build it
One section at a time, in song order: place notes with `mapctl add` / `mapedit.py` at `bar.slot`,
read them back with `score.py --bars a-b`, and write one line of rationale per section in
`LOG.md` beside this file. Every segment gets read and hand-edited before I move on. Backstop at the
end: `verdict.py` (it now reds note collisions) + `mapjudge`. With no human map most of that page
will be ⚪, which is expected.
