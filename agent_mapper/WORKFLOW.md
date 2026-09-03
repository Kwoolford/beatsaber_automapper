# Building a map as an agent — the loop

Repo: `/home/kyle/repos/beatsaber_automapper`. `source .venv/bin/activate` first.
All commands are `python agent_mapper/<tool>`.

> **This document was rewritten 2026-08-21 by building a map with it**, on a song outside
> the eval songset with cold caches (`1ddd1` "Drive Away Lite", 130 bpm, instrumental).
> Every ⚠️ below is something that actually went wrong during that build, not something
> anticipated. The previous version claimed things that are not true — see §9.

---

## 0. PERCEIVE — and read each tool's own verdict on itself

```bash
python agent_mapper/events.py     <audio>              # ★START HERE: 6 stems, typed
python agent_mapper/structure.py  <audio> --validate   # sections, and which repeat
python agent_mapper/percussion.py <audio> --validate   # which drum is hitting
python agent_mapper/melody.py     <audio>              # pitch: register + direction
python agent_mapper/lyrics.py     <audio> --lines      # timestamped words (slow, cached)
python agent_mapper/brief.py      <audio>              # 8-bar timeline, 4 stems
```

★**`events.py` is the most informative and the previous workflow did not mention it.**
It runs `htdemucs_6s` — **six** stems including **guitar and piano** — and names 14–20
typed note classes by physics. On the dogfood song it found `piano` 244 events and
`guitar` 159; **`brief.py`'s four-stem view cannot see either**, and piano was the real
lead. **If you plan from `brief.py` alone you will miss the melody on any song with a
keyboard or guitar.**

⚠️**`melody.py` does NOT take `--validate`.** The old doc said "each takes `--validate`";
it does not, and an agent following that gets an argparse error. `structure.py` and
`percussion.py` do.

### 🔴 Believe the verdicts, including the negative ones
- `percussion --validate` → `z = +19.3 ✅ labels carry real repeating structure` — trust it.
- `structure --validate` → `⚠️ not better than chance — do not trust these letters here`
  — then **§3b is unusable on this song** and you must plan by density instead (§1).
- 🔴🔴**`lyrics.py` HALLUCINATES ON INSTRUMENTALS. It does NOT "return nothing".** The
  dogfood song returned *"Thank you for watching"* ×3 at `lang p=0.253` — Whisper's
  signature filler. **It then propagated into `brief.py`'s LYRIC REPEATS block and was
  presented as the song's structural backbone.** ★**Check `language_probability`: below
  ~0.5 on a song with no singing, the words are invented. Discard them.**

---

## 1. READ THE WHOLE SONG, THEN WRITE THE PLAN IN PROSE

```bash
python agent_mapper/brief.py <audio>              # timeline + energy + lyric repeats
python agent_mapper/brief.py <audio> --bars 17-24 # zoom: one row per stem per bar
python agent_mapper/notesheet.py <audio>          # the full score as a page
```

**Write the plan before placing anything**: which instrument carries each section, how
dense, where the player breathes. **This is the part the ML pipeline cannot do** — it
decides one slot from a 16-note window and cannot know bar 33 is an outro.

🔴🔴**DO NOT PICK THE CLIMAX FROM THE `vocals` STEM.** Measured against 197 different-mapper pairs:
two humans agree on which sixth of the song holds the density peak **43.1 %** of the time, our best
audio rule (the phrase where the most stems peak) reaches **0.26**, and **following the vocals peak
scores 0.17 — exactly chance.** On an instrumental that stem is Demucs leakage. **Peak placement is a
judgement call with a weak prior; say so in the plan rather than reading it off a density column.**

★**Start the plan with what you cannot see.** On the dogfood song three of six signals
were unusable (lyrics hallucinated, structure untrusted, energy flat `####` on every
phrase) — writing that down first stopped all three from silently steering the map.

⚠️**`init` prints a grid fit `r`, and two different numbers matter.**
- **`R_TRUST = 0.10`** (`tempo.py`) is the real floor: below it *"the grid has no real relationship
  to the music"* and the map is not worth building. **Every songset song clears it** (min 0.171).
- **`brief.py` flags `⚠️weak` below 0.35**, which is a soft display warning, NOT a correctness
  threshold. The songset median is **0.303**, so most songs show it. 🔴An earlier version of this
  doc reported the 0.35 as "the bar lines are wrong" — that was a display string mistaken for a
  specification.

★**Low `r` does predict a weaker map, though**: across 23 songs `corr(r, onset_precision) = +0.575`,
and the `r < 0.30` half scores **0.781** against **0.882** for the rest. ⇒**Expect a poorly-fitted
song to land its notes less precisely on the music, and say so in the plan** rather than treating it
as a failure. ⚠️Correlational at n=23; fit quality and hit-ability may share a cause.

---

## 2. BUILD, section by section

```bash
python agent_mapper/mapctl.py init <audio> --name drive --fresh

python agent_mapper/mapctl.py auto drive --bars 9-16 \
    --follow "drums,piano/mid-stab" --wide \
    --pulse --lead-bias 0.2 --target-notes 62 \
    --doubles --doubles-rate 0.3 --accent-slots "0,2,4,6,8,10,12,14"
```

★**DOUBLES — the truth as of 2026-09-02 (P4).** `mapctl auto` defaults doubles OFF, and
`autobuild.py` passes `--doubles` ONLY inside its `--pulse` branch — its default two-pass path
ships 0 % and its `--no-doubles` help used to claim "ON by default". Two settings have both been
wrong: 51–70 % (08-03 maps, D6 "nps wasted") and 0 % (2 % of human maps have none). Measured on
the songset with the P3 queries + the tutor: default path 43 hits / 25 of 99 situations his way;
`--pulse --lead-bias 0.2` 53 hits (FLOW 21, D2 8 — notes on the odd 16th between the lead's
8ths) / 19 of 99. So `--pulse` is a request, not the default, and doubles go **where the tutor
puts them** (drums-in / bass-in → `D` on the bar line; vox-in → none): `--doubles --doubles-rate
0.3` on a section that wants them, or `mapedit.py double <bar>.1.0` at the entry. Run
`scripts/verdict.py` either way — it is the gate now.

⚠️**`--fresh` is not optional for a rebuild.** Without it `init` KEEPS existing notes and
a second build silently appends a whole second map onto the first. No error, just a worse
score. It faked a PASS-rate regression on 2026-08-20.

### The four placement levers, with their measured operating points

| lever | what it does | operating point | why |
|---|---|---|---|
| `--follow "a,b"` | ONE pass over merged streams | comma-separated | two layered passes cost the pulse: drums alone 0.387, drums+carrier union 0.329 (human 0.514) |
| `--pulse` | each phrase holds ONE interval; the only path with doubles + a lead hand | a request (off in `autobuild`) | `pulse_stability` 6th → 56th percentile, but FLOW/D2 jitter 21+8 spans vs 8+1 on the songset (2026-09-02) |
| `--lead-bias` | a hand leads a passage | **0.2** | `role_asymmetry` 1st → 33rd pct. **0.3 overshoots**; mode is `cyclic` (deterministic) |
| `--target-notes N` | search the accent percentile until N **distinct grid slots** survive | the section's budget | the budget is spent in SLOTS, not events; colliding events collapse |
| `--doubles-rate` | fraction of eligible accent slots that become doubles | **0.3** | rate and placement are separate: `--accent-slots` sets WHERE (eighth-notes match the human 0.635 on-beat share), this sets HOW MANY |
| `--nps N` | the density target | 4.17 (human median) | ⚠️**bounded by the song, not the code** — see below |

★**`--snap-onsets`** moves each event onto the nearest onset the JUDGE recognises. It needs
cached onsets (§4) and only helps where they exist.

### 🔴 A stem's CLASSES can be refused, per song, and that is correct
```
--follow guitar/hi-stab: the class labelling of `guitar` FAILED its control on this
song (labels repeat no better than shuffled) ... Follow `guitar` as one lane instead.
```
This happened twice in the dogfood build (`bass`, `guitar`). **Follow the bare stem name
instead.** ⚠️`events.py` PRINTS those class names without flagging that `mapctl` will
refuse them — the summary and the follow-check disagree, so expect it.

---

## 3. REUSE — when structure is trustworthy
```bash
python agent_mapper/mapctl.py plan  drive
python agent_mapper/mapctl.py reuse drive --label D    # --vary defaults to 0.15
```
⚠️**When `structure --validate` fails, `plan` reports ONE section for the whole song** and
there is nothing to reuse. That is not a bug; plan by density instead.

---

## 3c. THE OTHER FOUR MAP ELEMENTS — a map is not only notes

```bash
python agent_mapper/autobuild.py <audio> --pulse --lead-bias 0.2 \
    --walls 89 --arcs 90 --chains 16
```

🔴🔴**Until 2026-08-22 every map this agent produced contained notes and NOTHING ELSE**, while
**96 % of human maps have walls** (median 89) and, among v3 maps where they can exist, **100 % have
arcs** (median 90) and **71 % have chains** (median 16). `walls.py`, `arcs.py` and `chains.py` were
all built, measured against the human corpus, and never wired in.

⚠️**NO METRIC IN THE SUITE CAN SEE ANY OF THIS.** Adding 84 walls + 48 arcs + 16 chains moves every
axis by **exactly 0.000** — the judge scores notes only. The p-value is *identical* with and without
them. ⇒**Only his ear can say whether they help**, and a `[NOTES]` vs `[FULL]` pair is the way to ask.

★**The counts are a request, not a promise.** A short or dense song gets fewer: on a 39-second song
`--walls 89 --arcs 90` produced **56 walls and 73 arcs**, because a wall may only go where no note
is and the song ran out of room. That is correct behaviour, not a failure.

🔴**A wall a note sits inside is unplayable, and nothing else checks it.** `walls.py` places walls
in lanes no note occupies — but `idiomize` REDRAWS every note's column afterwards, so walls must be
added **after** it. Doing it the other way produced **12 trapped notes** on a map whose lane,
duration and width statistics all still matched the human idiom perfectly.
`tests/test_walls_playable.py` pins this; run it after touching wall placement.

## 4. VALIDATE — verdict first, then the raw checks

```bash
python scripts/verdict.py out.zip          # queries (wrong) + tutor (like him) + judge (typical); SHIP?
```
★**Since 2026-09-02 this is the gate.** Every 🔴 names the bars to open in the score and the
tool that fixes them; go back to the section, fix, run it again. The lines below are what it
reads underneath.

```bash
python agent_mapper/mapctl.py check  drive     # parity, doubles, violations
python agent_mapper/mapctl.py status drive     # coverage, and what is still empty
python agent_mapper/mapctl.py export drive --out out.zip
python agent_mapper/idiomize.py out.zip --out dressed.zip
python -m beatsaber_automapper.evaluation.mapjudge dressed.zip
```

★**A parity violation is an unplayable map, and `mapjudge` FAILs on `viol > 0` regardless
of the p-value.** That is correct behaviour, not a quirk.

### Getting the audio axis on a song outside the corpus
```bash
python scripts/build_onset_cache.py --audio-dir /path/to/dir --songs <audio-stem>
```
⚠️**Without this the judge prints `⚠️NO AUDIO AXIS` and scores 21 metrics instead of 23** —
it cannot tell you the notes are on the music. The cache is keyed by the **audio file
stem**, not the corpus id. This flag existed and was undocumented.

---

## 5. TAILOR — the loop Kyle asked for

Feedback is about a *place* and a *feeling*. Translate, redo that range only, re-judge.

| he says | what to do |
|---|---|
| "the chorus is too empty" | `clear --bars a-b`, re-`auto` with a higher `--target-notes` |
| "too busy there" | same, lower `--target-notes` |
| "it ignores the melody" | `--follow` the carrier `events.py` names, not the one `brief.py` shows |
| "it feels mechanical" | lower `--lead-bias`, or `--pulse-sync 0.2` to break the pulse more |
| "let me breathe before the drop" | `clear` those bars — emptiness is a choice |
| "the hands don't move" | known gap: `travel` sits ~15th pct. Hand-place with `add` |

**Measured on the dogfood song**: densifying one section moved `peak_nps` 4.50 (10th pct)
→ 6.00 (49th) and `p` 0.604 → 0.878, with everything else held. **Tailoring a part works
and is measurable.**

---

### ⚠️ `--nps` IS BOUNDED BY THE SONG'S OWN EVENTS
You cannot place a note where no event is. The real ceiling is **how many distinct grid slots the
song's events occupy**, which ranges **4.69 – 9.22 nps** across the songset — well below the grid's
own `bpm/60 × 4`. Asking for 9 nps on a song whose events support 5 gets you 5, and that is not a
bug. ★Above a song's own event density, more notes must come from **subdivision or invention**,
which is a different feature.
★**Judge the map by nps, not by note count.** A 39-second song at a correct 4.59 nps has only 179
notes, which looks alarming and is not.

## 5b. DIFFICULTY IS A TARGET YOU SET — AND IT IS NOT JUST NPS

★★**Kyle, 2026-08-21:** *"I wouldn't worry too much [about nps]. It's something we can
tailor to whatever we want of the song. I like playing fast songs… The objective is to be
able to map whatever difficulty we want. **Difficulty isn't always just NPS, it's how hard
are the notes to get to from the last note as well.**"*

⚠️**So 6.18 nps is NOT a ceiling.** It is one verdict on one map from 2026-08-11, and his
taste has moved since. **Treat it as a data point, not a bound.** (An earlier version of
this section said "his number wins" — that was wrong.)

**Difficulty has two independent axes, and the suite measures both:**

| axis | metrics | dial |
|---|---|---|
| **RATE** — how often you swing | `nps`, `peak_nps`, `ebpm_burst` | `--target-notes` per section |
| **TRANSITION COST** — how hard the next note is to reach | `travel`, `angle_change`, `crossover` | `idiomize --width`, `--crossover` |

**Measured on the dogfood song (same note times, cells redrawn):**

| `--width` | `travel` | `angle_change` | `idiom_local` |
|---|---|---|---|
| 1 (greedy) | **4.21 (51st pct)** | 1.66 (1st) | 0.61 (2nd) |
| 3 | 3.78 (35th) | 12.1 (13th) | 0.80 (15th) |
| 12 | 3.23 (19th) | **24.5 (73rd)** | **0.89 (65th)** |
| `--crossover 0.30` | 3.28 (20th) | 26.9 (81st) | — |

🔴**`travel` and `angle_change` pull in OPPOSITE directions.** Greedy picking sends the
hands far between repeated identical figures but never rotates the wrist; a wide draw
rotates constantly but keeps the hands near home. **There is no single setting that is
high on both**, so "harder" has to be a *direction you choose*, not a slider.
⬜**Open**: reaching human `travel` (51st pct) currently costs `idiom_local` (2nd pct).
Whether a human map achieves both, or trades the same way, is unmeasured.

---

## 5c. READ THE MAP — `agent_mapper/READING.md`
📖**The metrics cannot tell you if it is fun. Reading can.** `READING.md` is the process: what to
look at on the page and in what order, the habits to watch for in yourself, and the human-human
spread that tells a real defect apart from a valid interpretation.
★**Headline**: the legibility defect (23/23 maps) was invisible to all 23 metrics and obvious on
the page in one passage.

## 6. WHAT TO DISTRUST IN YOUR OWN MAP

🔴🔴**A PASS DOES NOT MEAN THE NOTES ARE ON THE MUSIC.** Measured 2026-08-21: the judge
accepts **65 %** of maps shifted a quarter-beat off the song. The alignment axis sees it
(`onset_precision` AUC 0.898) but the 23-metric aggregate dilutes two moving metrics among
twenty-one silent ones. **Read `onset_precision` yourself; do not rely on the verdict.**

🔴**A PASS = NOT DEFECTIVE, not GOOD.** It gates at the corpus **median**, and Kyle's
standing *"my target is the best mappers"* makes that a **FLOOR**.

🔴**Never rank by `p` or `rank_score`** — they are distance-from-typical, so minimising
them Goodharts toward the average map.

⚠️**`idiomize` is partly circular** — it was tuned against `idiom_local`/`idiom_jsd`.

**Known gaps, still open, measured on the dogfood build:**
- `travel` **3.06 at the default, 15th pct** — but this is a DIAL, not a defect: it
  reaches the 51st percentile at `--width 1`. See §5b for what that costs.
- `double_share` **0.000 vs a human 0.137** — it never plays both hands at one instant.

---

## 7. A COMPLETE WORKED EXAMPLE
`1ddd1` "Drive Away Lite", 130 bpm, 40 bars, instrumental, cold caches, built with this
document: **274 notes, 3.95 nps, PASS p=0.878 on 23 metrics, 0 parity violations**, with
`onset_precision` 0.945 (55th pct), `pulse_stability` 0.577 (55th), `role_asymmetry` 0.108
(48th), `ebpm_burst` 260 (46th), `peak_nps` 6.00 (49th).
★**The hand-planned map beat `autobuild.py` on `onset_precision` by 55th vs 15th
percentile** — choosing the carrier per section from `events.py` is what did it.

## 8. `autobuild.py` IS NOT THIS WORKFLOW
`python agent_mapper/autobuild.py <audio>` (`--pulse --lead-bias 0.2` on request) drives these same tools
with **fixed heuristics** in place of the prose plan. It is a consumer of the framework,
useful as a baseline and for bulk runs. **It scores worse on alignment than a planned build
because it cannot choose the carrier by ear.**

## 9. WHAT THE PREVIOUS VERSION OF THIS DOC GOT WRONG
1. *"Each takes `--validate`"* — `melody.py` does not.
2. *"Instrumental songs legitimately return nothing"* — they return **hallucinated filler**.
3. It never mentioned `events.py`, `notesheet.py`, `mapjudge`, or any placement lever.
4. It never mentioned that a stem's classes can be refused per song.
5. Its "known gaps" numbers were from 2026-08-14 and stale.
★**A workflow doc decays silently. Rebuild a map with it before trusting it.**
