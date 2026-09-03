# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md); the agent-authoring trail is in
[`agent_mapper/PROGRESS.md`](agent_mapper/PROGRESS.md). Evaluation-suite rationale is in
[`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule:** when an item finishes, its *outcome and what it taught* moves to PROGRESS.md and the
item is **deleted** from here. A completed item is history, not work. Curated 2026-08-02 (from
4,076 lines), 2026-08-14 (from 652), and **2026-09-02 (re-planned around the audit in
[`docs/audit_2026-09-02_buildmap.md`](docs/audit_2026-09-02_buildmap.md))**.

---

## 📍 CURRENT STATE (2026-09-02)

> ★★**THE GOAL (Kyle, 2026-09-02):** *"a tool suite that empowers the LLM to create a map like the
> best mappers, and a user can make requests to have specific mapping styles… the eval suite still
> to this day requires my approval and oversight… the errors are pretty obvious from my
> perspective."* And the fix, in his words: *"**The model doesn't have the visibility that I do**
> when evaluating a map. Convert the map to text or code or a numpy array where the rows are
> possible note placements and the columns are the notes, matched with another text or number
> array of the song in note-sheet format with lyrics and all. This granular visibility with deep
> timings is what the model does not have. This would catch the obvious errors more than a metric.
> **This is the eval suite.**"*

**What the audit found** (`docs/audit_2026-09-02_buildmap.md`): the build half works; the judge
half measures *typicality*; his verdicts were never kept as labels; and — the finding this plan is
built on — **the model-facing view of a map is sparse, song-blind and partly broken** while the
rich score (VOX pitch + lyric, LEAD, BASS, KIT, sections) exists only as HTML drawn for Kyle's eye
(`notesheet.py`). Every perception cache the score needs is already on disk
(`outputs/{event,percussion,melody,lyrics,structure,chords,onset}_cache/`). The missing thing is
**one join: song and map on ONE time lattice, as text and as arrays.**

**Build:** `python agent_mapper/autobuild.py <audio> --pulse --lead-bias 0.2` ([FULL] walls/arcs/
chains and phase-calibrate are the defaults since P0; `--notes-only`, `--no-phase-calibrate`), or the
per-section loop in 📖`agent_mapper/WORKFLOW.md` (beats autobuild).
**Judge (today):** `python -m beatsaber_automapper.evaluation.mapjudge <map.zip> [--nps N]` — parity
→ alignment floor → requested density → typicality, and `why:` names which gate failed;
`scripts/audit_map.py` for the handover page.
**Judge (now):** `python scripts/verdict.py <map.zip>` (P4 ✅ — queries + tutor + judge on one
page; every red = bars + tool; `SHIP?`; exit 1) then READ `python agent_mapper/score.py <map.zip>
--song <id> --vs auto --bars a-b` where it fires and fix with `agent_mapper/mapedit.py` (`tutor.py
--copy/--thin` emit the ops, `mapedit resets` names the parity leftovers). Loop until SHIP? YES.
📖`agent_mapper/READING.md` · 📖`LISTENING.md`.

### ★★ THE DoD FOR THE WHOLE PLAN
> The agent opens the **score** of its own map — every slot of the song, with what the song is
> doing (kick/snare/hat, bass and lead pitch, vocal pitch + lyric syllable, section, energy) beside
> what the map is doing (both hands, walls, arcs, chains) at the **same** timestamp — and, reading
> it the way Kyle listens, names the same defects at the same places he does on the labelled bench.
> A map ships when the read is clean. Kyle spot-checks; every disagreement becomes a bench label.
> **And the target above "clean": the map wins a blind A/B against the top human map of the same
> song (P4b) — "competes with top mappers" is a win rate, not a PASS count.**

### ★★ HOW THE AGENT IS EXPECTED TO WORK (Kyle, 2026-09-02)
> *"It does not need to one-shot build a map, or build it quickly. I want to give it the tools so
> it can achieve a great map in the end. If it doesn't call all the tools off start that's fine.
> It may not need extremely granular details for every note, but should be able to call more
> tools and know it can, and recognise which parts need more tooling and attention to detail."*

⇒ The loop is **many passes, no clock**: build coarse → read the overview → **triage** which
sections deserve slot-level attention (main-instrument entries, drops, vocal phrases, anything a
query flags, anything the human map treats differently) → zoom the score there → edit at the
slot (P1b) → re-read → next section. Sections that read clean at the overview stay coarse.
⇒ The agent must **know its toolbox**: `SKILL.md` carries a one-screen manifest of every tool
with *"reach for this when…"* (P4), and the verdict names the tool for every red line.
⇒ Time and token budget are spent where the score says the map is weak, not evenly.

### 🔴 Limits that still bound everything
1. 🟡**A PASS now means the notes are at least ON the music** (P0.2 floor: `offbeat` 68 %→7.7 %
   accepted) — but only when the song has cached onsets; without them the report says
   `no floor applied`. The score still shows *where* per slot; the floor only says *whether*.
2. 🔴**A PASS = NOT DEFECTIVE, not GOOD; a FAIL can mean NOT TYPICAL.** Never rank by `p`.
3. ⚠️**"Humans do it too" is NOT a no-defect verdict** for anything Kyle named by ear — the
   corpus median is a floor (`feedback-target-is-best-mappers`).
4. ⚠️**OUTPUT DIRS ARE NOT INTERCHANGEABLE** — name the directory, never just the arm.
5. ⚠️The score's song side is only as good as the caches: `melody` coverage (printed in the
   header) and whisper's language probability decide whether a blank VOX lane is the song or the
   tool. Read the header before reading the page.

### ▶️ START THE NEXT SESSION HERE — no GPU, no questions for Kyle
**P1 ✅ · P1b ✅ · P0 ✅ · P2 ✅ · P2b ✅ · P3 ✅ · P4 ✅ · P4b ✅ shipped** (the score, it is writable, the judge has its floor, the bench reads every verdict, the tutor is on the lattice, six queries answer with addresses, one page says SHIP? and one map went through the loop, and THREE blind pairs wait in `for_review/compete/` — `compete.py table` is the headline) → **P4b leftover: loop 1f333 to SHIP? YES and stage it** (1f913 done 2026-09-02: `tutor --fill` + `--thin` + 4 one-note reconciles, SHIP YES, staged) → **P5** (`review.py defect --at` already exists; the DoD is the bench growing without JSON edits) → **P6**.
🔴**DECIDE-AND-LOG.** Nothing below may block on Kyle.

---

## ✅→🔵 P1 — THE SCORE shipped 2026-09-02 (`agent_mapper/score.py`) — what is left of it
Read `PROGRESS.md 2026-09-02b` for the DoD read. Commands: `--sections` (triage, one row per bar)
· `--bars a-b` (every 1/16 slot: KIT/BASS/LEAD/VOX+lyric/gt pn/E/ON/MAIN │ L/R/B/W/A/C/DBL/TRV/
ROT/ALT/±ms) · `--vs auto` (human answer key + `H±ms`) · `--npz` (song[T,F], map[T,C], human[T,C]).
📖`READING.md §2.0` is how to read it. `map_view --stems/--audio` are retired (print a pointer).

**Still open inside P1 (do alongside P3, none blocks P1b/P0/P2):**
- **FLOW has no column that names it.** Hunger AGENT (odd flow) vs A+ (preferred): the page shows
  it (bar 34: six singles on off-beat 16ths vs kit-aligned strikes) but bar-level counts barely
  differ (on-KIT 85 % vs 78 %); what differs is on-beat share 36 % vs 57 % and the per-hand cut
  sequence. Candidate columns before any metric: the hand's **gap-ms since its previous note**
  (swing speed) and a **3-cut path glyph** per hand (`↙→↗`). P2's read (2026-09-02d) named it
  from `bench.py stats`: no hand role + bursts-between-rests. A column is still wanted so it is
  visible at the bar, not only in the whole-map histogram.
- `notesheet.py --map` should draw Kyle's HTML from `score.to_arrays()` so the two never disagree
  (today notesheet has its own join).
- `--sub 12` (triplets) is untested on a triplet song; `--perceive` runs the tools but the
  `lyrics` entry point name is guessed (`transcribe`) — verify on a cache-less song.
- `E` is whole-mix RMS; a per-stem energy (drums vs rest) would make "the drop" a two-column
  step rather than one. Add only if a D3 read needs it.

## ✅→🔵 P1b — THE SCORE IS WRITABLE shipped 2026-09-02 (`agent_mapper/mapedit.py`) — leftovers
`mapedit.py <map.zip> place|move|flip|delete|double|mirror|wall|bomb|arc|chain|from|undo|log`
in the score's `bar.beat.sub` addresses; refuses NEW parity violations / <150 ms same-hand gaps /
notes in walls / cell collisions with the reason (`--force`); snapshots + `edits.log` in
`<dir>/.mapedit/<stem>/`; `--song` re-prints the touched bars. **Edits the ZIP, not the session**
(decided: everything after `mapctl export` is zip surgery, so the zip is the map).
DoD met on Hunger AGENT bar 34 (`PROGRESS.md 2026-09-02b`).

**Leftovers:** v2 (human) maps are refused — add a v2→v3 convert so a tutor map can be edited
into a variant; `mapctl edit` alias is NOT provided (sessions have no dressed map); no
`--dry-run` (undo is the dry run); chain/arc ops don't validate that the tail cell is reachable.

## ✅→🔵 P0 — PARKED DECISIONS CLEARED 2026-09-02 (decide-and-log; nothing waited on an ear)
All seven shipped; `PROGRESS.md 2026-09-02c` has the numbers. What each became and how to undo it:

| item | shipped as | reverse with |
|---|---|---|
| **P0.2 alignment floor** | `mapjudge` FAILs any map with `onset_precision` below the **human 10th pct (0.822)** regardless of the pooled p — `reference["align_floor"]`, written by `scripts/calibrate_align_floor.py` (one-sided, one metric: two-sided / +`offset_mad_ms` measured worse). Priced on 300 held-out humans: human 0.877→**0.793**, `offbeat` 0.680→**0.077** | `MAPJUDGE_ALIGN_FLOOR=0` or `judge(..., align_floor=False)` |
| **P0.1 requested density** | `judge(nps_request=…)` / `mapjudge --nps` / autobuild `--nps` or a style preset: `nps` gated ±15 % against the request, `nps`/`peak_nps` leave the pool; report prints `density … met/MISSED` | omit the request |
| **`[PCAL]`** | `--phase-calibrate` **default ON** in `mapctl init` and `autobuild` | `--no-phase-calibrate` |
| **`[FULL]`** | autobuild default `--walls 89 --arcs 90 --chains 16`; crossover stays a knob at 0 | `--notes-only` |
| **width** | default 3 unchanged; `--width` is the cut-variety knob for P6 | — |
| **`for_review/` cleanup** | set A (20 maps) filed to `outputs/reviewed/A_structure_crossover/` via `review.py done A --force`, CATALOG annotated; `review.py next` now lists **only set B** (the PCAL A/B, optional) | — |
| **`[DOD]` cold-cache map** | no listening; P2 reads it and files it as a bench negative | — |

**DoD read** (23-song set, `outputs/p0_songset_2026-09-02/{OLD,NEW}__<sid>.zip`): under the new gate
the OLD defaults pass **13/23** (10 "off the music", pooled gate said 23/23) and the NEW defaults
pass **21/23**; `onset_precision` median 0.866→**0.901**; `offbeat` control 68 %→**7.7 %**. The two
NEW fails are `1f335` (0.735) and `1f9a0` (0.475, the known P1.0 song) — both real, both now named
by the judge instead of hidden by it.

**Leftovers:** ⬜`audit_mapjudge` human bar reads the `no-floor` column (documented) — the pooled
guarantee is unchanged, the floor's cost is on top. ⬜`calibrate_mapjudge.py` drops `align_floor`
on a rebuild and says so; re-run `calibrate_align_floor.py` after it. ⬜The floor needs cached
onsets: `autobuild` now warns loudly when a song has none (`scripts/build_onset_cache.py`).
⬜`1f335` deserves a score read (P3 locator candidate: which sections are off?).

## ✅→🔵 P2 — THE BENCH shipped 2026-09-02 (`scripts/bench.py`, `docs/eval_references/labelled_maps.json`)
`PROGRESS.md 2026-09-02d` has the reads. **17 rows** (8 strong, 9 weak, 4 unreadable set-B
corpus songs), every path verified, a written read for every strong row + set A + the DoD map in
`docs/eval_references/bench_reads.json`. Commands: `bench.py list` · `read <id> [--bars a-b]`
(score at the labelled bars, human alongside) · `note <id> --codes … --text …` · `stats`
(events / doubles / hand roles / streams per row) · `score module:func` (any P3 query →
HIT / MISS / FALSE / VIOLATION per row, counts not rates, same-notes pair must agree).

**What the reads found — P3 starts from these, not from the old locator list:**
- ★★**EMPTY, D1 "very slow" and D6 "nps wasted" are ONE number: doubles.** Every agent map on
  the bench is 51–70 % two-hand doubles (humans 7–34 %); doubles double the note count while
  halving the *events* — 1f8d6 has MORE notes than its human (788 vs 725) and fewer things to do
  (497 vs 646 rows); Hunger A+ has 1328 notes / 809 events vs human 1434 / 1242 = half the
  event rate at equal nps. This is why 69.5 % onset coverage was blind to "empty".
- ★**FLOW (Hunger AGENT) = no hand role + bursts-between-rests.** Both hands have identical
  16th-position histograms (29–30 % on 16th-offbeats each) where the human's blue hand takes
  the beat (62 % on-beat, 6 % 16ths) and red fills (14 %); L-R-L 80 ms bursts at 34.2/34.4/36.1
  are each followed by ≥ a beat of nothing, where the human streams 8ths.
- ★**D4 sits inside EMPTY**: 1f8d6 bars 121–130 (the sung climax) answer 1–3 of 5–8 vox
  onsets per bar as doubles on the kick; the human puts 4–6 notes on the words.
- ⚠️**The A+ and "very slow" verdicts are the SAME NOTES** (1f333_AFTER2 = Hunger_BEFORE):
  A+ was relative to the older builds, D1–D6 relative to the human. Labels record both; only
  FLOW is forbidden on the A+ row.
- ★**Grid phase must be read against the KIT columns, never the map lattice**: `DOD__24e6c`
  (built before `[PCAL]`) has kick/snare on `.3` of its own grid — 78 % of its notes "on
  16th-offbeats" while the judge (rightly) says on the music.
- Clean-side facts a query must survive: a human map with 34 % doubles (1f767), one with 18
  gaps > 1 s (1f8d6, a ballad), one whose kick sits on the offbeat 8th (1f913), hand asymmetry
  everywhere.
- `smoke:density` (bars with < ½ the human's notes) is the worked example: fires 22× on the maps
  he liked vs once on the empty map — the bench says so in one line.

**Leftovers:** ⬜set B ids for FLASH/GOCRYGO unresolved (rows deliberately absent); set-B rows
are unreadable until corpus songs get perception caches. ⬜`bars_from: "kyle"` exists on no row
yet — P5 is where his finger lands on a bar. ⬜`24e6c-dod` stays UNLABELLED (clean note-level
read, grid-phase fact recorded); flip to CLEAN when a P3 pass agrees.

## ✅→🔵 P2b — STUDY MODE shipped 2026-09-02 (`scripts/tutor.py`) — leftovers
```bash
python scripts/tutor.py 1f8d6                      # situation → the tutor's pattern (2 bars from each)
python scripts/tutor.py 1f8d6 --map x.zip          # ours beside it: SAME / differs, count at the end
python scripts/tutor.py 1f8d6 --bars 102-103       # the rows, to copy cells + cuts with mapedit
python scripts/tutor.py --vocab [ids…]             # pooled situation → pattern counts (default: songset)
python agent_mapper/score.py x.zip --song 1f8d6 --vs 1f8d6   # --vs now takes a corpus id
```
**Decided-and-logged:** the corpus stores NO per-map rating (manifest has category/requirements/
genre only) and holds ONE map per id; the crawl was rating-sorted with upvote ratio ≥ 0.8 ⇒
`data/raw/<sid>.zip` IS the top-rated human map we have. "Highest-rated" = that. Logged in
`tutor.py`'s docstring and `score.resolve_vs`.
**DoD met (PROGRESS 2026-09-02e):** `NOTUTOR__1f8d6` (autobuild defaults) answers **4/15**
situations the tutor's way; `TUTOR__1f8d6` (same build + `mapedit.py from` at bars 62-63,
102-103, 110-111, 135-136 copied from the tutor's rows) answers **9/15**. Both PASS the judge
(p 0.572 → 0.604). READING.md §0b "Studying a map" written.
**What the tutors teach (`--vocab`, 99 situations over 1f333/1f767/1f913/1f8d6):** vox-in →
alt-8ths at 0 % doubles; drums-in / bass-in → DOUBLES (the entry accent — 4/6 and 3/5);
E-drop → rest 5/10; the bar before an E-jump is rest 4/5; repeats echo the first figure; first
answer is `+0b` almost everywhere. ⚠️The current autobuild defaults produce **0 % doubles**
(699 notes / 699 rows on 1f8d6; human 7 %, and the human puts them exactly at entries) — the
08-03 maps' 51–70 % and today's 0 % are two wrong settings of one lever; P3's doubles query
should locate BOTH (doubles where no stem enters; no double where drums/bass enter).
**Leftovers:**
- The copy is manual (`mapedit.py from`); no `tutor.py --copy 102-103` yet. Parity has to be
  re-footed by hand (reverse the tutor's cut sequence when our hand enters on the other foot).
- Nothing feeds `idiomize.py` / `mapctl reuse` yet. The recurring patterns are now countable
  (`--vocab -v`), so the feed is a table lookup: situation kind → pattern word → lever
  (`--doubles-rate` at entries, rest before a jump). Do it after P3 names the queries.
- Situations miss: chorus-repeat *variation* (glyph-identical at 1f913 43/83 — measure the
  echo distance), phrase starts inside a section (lyric line breaks), fills before a drop.
- `lead-in/out` is noisy on sparse melody stems (coverage < 0.5) — the 15 % stay-rule trims
  it; a header caveat would be better.
- `1f767`'s tutor answers `vox-in` with 1–2 events/bar inside walls (sparse×3) — a whole
  different style from 1f333's alt-8ths. `--vocab` pools styles; a per-song read is the
  vocabulary, the pool is the prior.

## ✅→🔵 P3 — QUERIES OVER THE ARRAYS shipped 2026-09-02 (`scripts/queries.py`) — leftovers
`PROGRESS.md 2026-09-02f` has the measurements. **Six queries, one file, every hit is an address**
(`code mm:ss bar why`): `python scripts/queries.py <map.zip> [--song id] [--vs auto|id|zip]`
(a zip, or a `score.py --npz` file; `--query q_flow` for one). Bench: `python scripts/bench.py
score queries:q_all` → **not refuted: 2 strong hits (1f8d6-empty EMPTY, Hunger AGENT FLOW at the
labelled bars), 0 false fires, 0 violations, 4 humans CLEAN, all 4 set-A rows hit, the A+/BEFORE
pair agrees, AGENT>BEFORE ordering holds (19 % vs 2 % of bars FLOW).**
| query | codes | reads |
|---|---|---|
| `q_events` | EMPTY D1 D6 | events (a double = 1) per 4 bars vs the human's: < 0.6× → EMPTY; ≥ 2× → D6 over-dense; map-wide doubles ≥ 50 % and 20 pts over the human → D6; median ratio < 0.7 → D1 |
| `q_flow` | FLOW D2 | 2-bar sliding: ≥ 30 % of events start on an odd 16th from silence (and ≥ 20 pts over the reference's share) while ≥ 30 % sit on the 8th grid → FLOW; ≥ 80 % odd 16ths where the reference has < 35 % → D2 shifted grid |
| `q_vocals` | D4 | per 4 bars: vox-MAIN slots answered (note within ±1 slot) ≥ 25 pts under the human, human ≥ 60 % |
| `q_drops` | D3 | at song E-jumps (≥ 0.25/bar): first note > 1 beat after the human's, or step < 0.8× his AND density after < 0.8× his; at E-drops: he halves, we don't |
| `q_elements` | ELEMENTS | 0 walls where the human has ≥ 5 (arcs/chains reported, never fired: songset humans are v2, 0 of either) |
**Decided-and-logged (measured, 2026-09-02):** ★**every reference is the same song's human map**
(or the song's onsets without one) — absolute rules were refuted twice in one afternoon: an
absolute D3 step floor fired on human 1f913 (its own jumps are 0.6–1.1× steps), and an absolute
"odd-16th = shifted" called 20 spans of `1f335` shifted when the human sits on the odd 16th 35 % of
the time (195 bpm: the odd 16th IS the felt 8th). Bench rows grew two fields: `tolerance` (Kyle
said "the VAST MAJORITY is A+" ⇒ FLOW may cover ≤ 10 % of bars — q_flow finds real jitter at A+
152-157, 2 %) and `worse_than` (AGENT must draw more FLOW than BEFORE — the comparison he made);
`1f333-agent-flow` bars widened 33-36 → 29-47 (agent-read; the jitter surrounds the fold).
**Refuted reads (do not retry without a new column):** hand-role histogram distance (AGENT 0.54 >
A+ 0.16 — backwards); 16th bursts followed by rest (0 at the labelled bars); `±ms` alignment
(humans 4–13 % of notes > 40 ms, agents 2–15 % — no separation, D2 is the shifted grid); D5
"random bursts" as 16th-runs-without-onsets (humans 4–7, AGENT 1) or isolated fast clusters
(3–5 on every map) — **D5 has no locator**; notes-inside-walls (human 1f767 has 4 under crouch
walls — the arrays carry no wall height).
**First customers:** `NEW__1f335` (judge 0.735): EMPTY ×16 incl. **bars 77-88: 1 event vs 31**,
FLOW ×16, D2 ×6 (reference 0–20 %), D3 ×4, D4 ×2. `NEW__1f9a0` (0.475): **D6 over-dense ×5 —
2.0–2.6× the human's events in every chorus** (657 vs 271 map-wide), D2 at 2-3 and 74-80. Both
songs have **empty KIT columns** (no percussion cache) — the score's header should say so.
`NOTUTOR__1f8d6` and `TUTOR__1f8d6` (today's defaults): **0 hits** — event ratio 0.6–1.5, walls,
vox 84 % vs 88 %: by these six reads the default build has no located defect, while the tutor
says it answers 4/15 situations his way. Queries measure *wrong*; the tutor measures *like him*.
**Leftovers:** ⬜D5 no locator (three reads refuted). ⬜wall HEIGHT into the arrays, then
notes-under-full-walls as an ELEMENTS playability read. ⬜arc/chain playability needs a v3 human
reference (songset humans have none). ⬜KIT-empty songs: `q_flow` falls back to onsets only when
there is no human; add "no percussion cache" to the score header. ⬜`bench.py` "fires MORE on the
maps he liked" now counts forbidden codes only. ⬜`q_events` D1 is map-wide (one fire) — a
per-section D1 needs the request (P6). ⬜ONBEAT_MAIN/HANDROLE/BREATHING codes exist in the label
file with no query — they were the typicality instruments; only write one if a verdict names it.

## ✅→🔵 P4 — THE VERDICT AND THE SIX-STEP LOOP shipped 2026-09-02 (scripts/verdict.py, /buildmap) — leftovers
Shipped (PROGRESS 2026-09-02g): `verdict.py <map>` = queries + tutor + judge on one page, every red
= bars + tool, `SHIP?`, exit 1; `/buildmap` = THE TOOLBOX + THE LOOP (study → coarse → triage →
zoom & edit → verdict → compete); doubles contradiction root-caused (two entry points; default
stays non-pulse, `--pulse` draws FLOW/D2 jitter 21+8 spans vs 8+1); `mapedit resets`, `tutor
--copy/--thin`; **AliceBlue 1f767 went through the loop to SHIP? YES in 172 guarded ops** and is
staged as `for_review/D_verdict_loop` (`review.py next` #2).
**Leftovers** (decide-and-log candidates, none blocking P4b):
- **Reset reconciliation is manual**: `mapedit resets` names the pair; the fix (ONE note per hand:
  place / delete / flip … X) is chosen by reading the score. Flipping the second note cascades.
  A `mapedit reconcile <bar>` that tries the three one-note fixes and keeps the first with 0 new
  resets and no guard refusal would close the last hand step of the thinning path.
- `--thin` keeps the survivor's cut direction; after a move the note may want re-cutting (the
  human's arrow at that slot is known — `_cells(h[j])`).
- The verdict does not read D5 (no locator, P3), and does not yet say "no percussion cache" when
  KIT is empty (1f335, 1f9a0) — the score header should.
- `mapctl clear` + `auto` on a section is still the SESSION workflow; once `export` has run the zip
  is the artefact (idiomize / walls / arcs / chains are zip surgery), so a section rebuild today
  means re-running the whole dress. A `mapctl reauto --zip z --bars a-b` that rebuilds one range
  and re-dresses only those bars would give the loop its second section tool.
- The pulse path's odd-16th interval choice (1f8d6 bars 2-4, notes on `.3` between the lead's
  8ths) is a bug to locate in `mapctl auto --pulse`, not a reason the path exists.
- Tutor `same_way` is coarse (pattern word + ev/bar ±35 % + first answer within a beat): bar 85
  reads `differs` with both first beats copied verbatim because our 8ths at 85.3-85.4 are on the
  other foot. A per-beat cell match would make the TUTOR line an edit list.
- The verdict is judged only by the bench's 13 rows + one loop; Kyle's answer on set D ("name the
  first WRONG bar in LOOP") is the next locator — record it with `review.py defect`.

## ✅→🔵 P4b — THE COMPETE TEST shipped 2026-09-02 (`scripts/compete.py`) — leftovers
`stage <sid> | --songset` (blind X/Y pair, same Info.dat skeleton, one difficulty each, key in
`for_review/compete/.key.json`) · `list` · `verdict <sid> X|Y|tie --because … [--code] [--bars]`
(unblinds, ledger `kind: compete`, **a loss with a reason = a bench row the same day**, pair
filed to `outputs/reviewed/compete/`) · `table` (★the headline: win rate, n, every loss with its
reason). Staged: **AliceBlue (LOOP) and Fallen Kingdom (NOPULSE)** — `review.py next` lists them
FIRST because set D unblinded would spoil them (AliceBlue_HUMAN moved out of set D).
**Leftovers:**
- ✅ 1f913 looped (PROGRESS 2026-09-02i) and staged; ⬜ 1f333 (2 red: EMPTY, D1) still REFUSED by `stage` — its
  page is red and losing with a known red teaches nothing. Run the six-step loop on each
  (`/buildmap`), then `compete.py stage --songset` again. DoD "four staged" is 2/4 until then.
- ⬜ Note counts differ visibly in ArcViewer (771 vs 632 on AliceBlue) — a tell only to someone
  who knows our count; accepted, logged. The audio bytes differ too (ours re-encoded).
- ⬜ `table` has no per-song "ours page at staging" column yet — the key carries it; print it
  when n > 0 so a loss can be read against what the page already said.
- ⬜ The DoD's target (what win rate is "competes") is Kyle's to set; until then the number is
  reported, not gated. ⚠️One pair per listening session (P5 rule).

## 🟡 P5 — THE LABEL CHANNEL: his remaining oversight, cheap and cumulative
`review.py next` asks **one thing per session** — *"play X; if anything is wrong, give me the
timestamp and the word"*; `review.py defect --at` appends to the P2 bench automatically; `/close`
rule: a Kyle verdict that disagrees with the agent's read is a bench row AND a P3 task, never a
TODO opinion. **DoD**: bench grows ≥ 1 row per listening session with no JSON editing; pending
list ≤ 4 maps.

## 🟡 P6 — STYLE REQUESTS: "make it more X" as a lever table + presets
`docs/style_levers.md` — one row per request (*faster · harder · more diagonals · more doubles ·
one hand leads · follow the piano · breathe before the drop · more walls*) with the lever, its safe
range, and the score column that shows it moved; `mapctl auto --style {flow,tech,dance}`;
`verdict.py --style` judges against the preset where P0.1 uses the nps request.
**DoD**: three presets build clean on the songset with the named column moved. ⚠️Levers stay
monotone and default-off — they ship in a UI (`feedback-levers-are-user-facing`).

---

## 🔵 CARRIED FORWARD — still live, lower than P0–P6

### P1.3 — the two build paths disagree on doubles; the hand path can still ship ~zero
`autobuild` (doubles ON, 8 accent slots) lands 10–20 %; `mapctl auto` with the documented flags
reached **0.010** on `24e6c` because the gate is `slot ∈ accent_slots AND ≥ 2 stems agree`.
**Tasks**: make `mapctl auto`'s defaults equal `autobuild`'s; measure `double_share` across the
songset. **DoD**: the documented command lands p25–p75 (0.089–0.212), `viol` unchanged.

### P1.0 — `1f9a0` (93 bpm) fails `onset_precision` 0.474; a finer grid is REFUTED
Binding constraint is note **selection**, not the grid (`--adaptive-subdiv` hurt 10/10). Untried:
choose events by distance to a scored onset; pulse lattice prefers onset-carrying phases.
**DoD**: `onset_precision` rises without `pulse_stability` leaving 25–75 %.

### P0.6 — hand role: `--lead-bias 0.20` under `cyclic`. Landmine only
An operating point is not portable across a change in how the knob works.

### W6 — walls/arcs/chains built and installed; no metric sees them
`[FULL]` IS the autobuild default since P0 (2026-09-02); P3 adds playability. ⬜`BEAT_SIM_CHAINS=1` must be flipped
**and** the human reference recalibrated in the same change.

### The six defects (2026-08-17) → P3 queries
D1 · D2 · D3 · D4/D6 · D5 · FLOW · EMPTY. ⚠️**Protect — he named them by ear**: hand-role
division; breathing pacing; *"notes on beat that play part of the song"*. They are the bench's
must-not-flag rows.

### Doc debt from the audit (do in the P4 session)
`CLAUDE.md` V6-era, no `agent_mapper/`; `buildmap/SKILL.md` contradicts itself on doubles;
`/todo` Step 4 checks ML-era key notes.

## 🔵 C — ML-SIDE DIAGNOSES CARRIED FORWARD (landmines only; not being worked)

### C1 — Precision sits at the greedy optimum; gains need better probabilities, not better picking
Three decode levers moved onset precision by nothing; the IOI prior moved it *down* to 0.769.
**Stop hunting decode knobs.** The ~10 correct-tempo alignment failures are a **pure selection
defect**, established by elimination — not tempo, not phase, **not onset supply** (4.5 onsets
available per note we emit), and **not difficulty**.

### C2 — Grid PHASE: resolved ON THE METRIC ONLY, and its successor suspect is REFUTED
`BEAT_GRID_PHASE=search` fixed ~18 of 39 failing songs by the alignment axis, and he still reported
*"slightly off beat"*. Tempo is refuted as the cause too: on all four maps he played the bpm is
**exactly** the human's and our note times match a human's **better than two humans match each
other**. ⇒**Do not flip it on the axis alone.** ⚠️Never apply a blanket global shift — that part is
an **onset-detector offset**, and "fixing" it is the `h_dist` failure.
★**2026-08-20 adds**: `mapjudge` cannot adjudicate this at all — its only response to a global shift
is `offgrid_frac`, which moves **by construction**.

### C3 — You cannot thin your way to human density
Humans at 3.9 nps have a pulse; we at 3.9 do not, and **2026-08-20 reproduced this on 23 maps built
with no ML in the path at all**. Now the D1 query's mechanism.

### C4 — Beat-domain axes LIE on tempo errors
Every beat-domain axis buckets by the **map's own beats**, so on the **28 half-tempo songs** every
interval lands one bucket off. ⇒**When a beat-domain axis moves on a cohort containing tempo errors,
check whether the BPM moved first.**

### C5 — Doubles: root cause found · decode fix FAILED · priced against D4
**Not too many notes — too few distinct times.** Stage-1's two hand channels correlate
**0.985–0.993**. **39.6 % of the notes we spend on the vocal line are doubles** onto an onset the
other hand already covered (human 20.7 %). `BEAT_HAND_DEAL` hit every structural target and degraded
rhythm 6× ⇒ **not reachable by decode**. ★A **chain** is the human's alternative — one swing carrying
4–5 segments; `chains.py` builds them.

### ⚠️ SEEDS ON THE AGENT PATH — read before quoting any "n seeds" number
**10 of 23 metrics are seed-INVARIANT by construction** (every time-domain one). The agent builds
from **cached events**, so note TIMES are deterministic — the opposite of the ML path, where a seed
re-draws the Demucs stems. ⇒Seeds matter only for **geometry and hand-role**; `--seed` reaches
`mapctl auto` since 2026-08-21.

---

## 🧊 BACKLOGGED — ML PIPELINE (deprioritised by Kyle, 2026-08-20)
**Not dead, not being worked.** These are the model-training items; Kyle redirected the loop to the
agentic suite mid-session. Each keeps its measured evidence so it can be resumed without re-deriving
anything. ⚠️**Do not queue any of these from `/todo`.** If an agent-path item needs one of these to
progress, say so and stop — that is a decision for Kyle, not a silent re-prioritisation.

### 🧊 D4 — the ML generator does not follow the vocal line (training-side only)
Every alternative is eliminated by measurement: decode saturates (5× lower threshold buys **+4
notes**), the 1/4-beat grid is only **26–33 %** full, Track B at matched budget is **parity**, and
**Stage-1 does not modulate density per song at all (r = 0.046)** while crude audio features reach
**R² = 0.185**. We emit **0.217** positives per (slot,hand) against a corpus label mean of **0.245**
and these songs' humans at **0.294**.
⬜**PROPOSED RETRAIN, deliberately NOT queued**: an auxiliary per-song **density target** / FiLM
conditioning on the song's own label rate.
**DoD**: per-song nps correlation rises from 0.046 toward the demonstrated ≈0.43 floor **AND**
vocal coverage rises, at ≥3 seeds. ⚠️Must not regress the density he accepted (6.18 = unplayable;
current lever sits at 4.06).

### 🧊 TEMPO — priced, and smaller than its reputation
Right on **70.5 %** of songs (n=149). `BEAT_SUBDIV_AUTO` already recovers 16 of the 28 half-tempo
songs and is worth **+0.030 cohort-wide**; the remaining 12 are worth a further **+0.025** — about
**a fortieth of the human gap**. 🔴**Cheap detection is exhausted** (raw bpm AUC 0.978, 16/28 at
zero false fires; widening to bpm<110 costs 10 false fires for 8 songs). ⇒**Do not widen the
trigger.** ⚠️`notes per second` scores AUC 0.903 and catches **zero** at an affordable FP rate —
★*AUC is not an operating point.*

### Also backlogged
- **The two validated-but-unflipped ML levers** — `BEAT_SUBDIV_AUTO=1` (+0.222 vocal coverage at
  49x the seed sd on the 15 half-tempo songs it fires on) and `--beat-threshold 0.25` (+0.029 at
  8.6x sd). Both passed their DoDs; both change the **ML generator**, not the agent, so they wait.
- **C1 / C2 / C4 / C5 below** are ML-pipeline diagnoses, kept for their landmines only. C3 is the
  exception: it reproduced with **no ML in the path at all** and is live as **P0.5**.

## 🧭 REFERENCE
### 🔴 Landmines — a seed re-draws the AUDIO, not just the decode
**`seed_everything(args.seed)` seeds the RNG that Demucs' random-shift augmentation uses**, so the
seed changes the STEMS → the MERT features → **Stage-1's probability field**. Measured on 1f333:
same seed twice is **bit-identical**; seed 0 vs 1 gives max \|Δ\| **0.2049** (mean 0.0264, corr
0.9915) and only **87.3 %** of the top-300 slots survive.
⇒**Every seed-based error bar in this repo contains Demucs stem variance**, including the ±0.004
"seed noise floor". The standing note that *"pairing helps alignment only — the rest ride the torch
decode"* is **wrong at the root**: the draw happens before the model runs.
⇒When you want to vary ONLY the decode, you cannot do it with the run seed as things stand.

- 🔴🔴**AXIS GAPS ARE COHORT-SIZE DEPENDENT (2026-08-19r).** The **same maps** score **flow 1.260
  at n=5 → 0.446 at n=50** and **alignment 1.062 → 0.341**: a small cohort estimates its own
  distribution noisily and the noise reads as distance, so **small cohorts look worse**.
  ⇒**NEVER compare cohorts of different sizes**, and the **bars do not transfer across n**.
  ⚠️Also: **all six axes are `nan` at n=1 and n=2** — the suite is a cohort statistic and **cannot
  score one map**, which is the structural reason a passed DoD says nothing about a map.
- 🔴**`alignment` SILENTLY RETURNS `nan` IF THE FILENAME DOES NOT START WITH THE SONG ID.**
  `scorecard.song_id()` parses the id from the filename: `1f8d6_WALLS.zip` → `'1f8d6_WALLS'` → no
  cached onsets → `alignment = nan`, **no error, five axes scored instead of six**.
  ★**Name generated maps `<arm>__<songid>.zip`, never `<songid>_<arm>.zip`** (2026-08-19p).
- 🔴**THE SUITE IS BLIND TO WALLS, ARCS AND CHAINS.** Adding 84 walls + 48 arcs + 16 chains moves
  **every axis by exactly 0.000** — it scores notes and nothing else. ⇒**No axis can justify or
  reject the element work; only his ear can.** (Chains: the model works, but 16 in 913 swings is
  1.7 % and `travel` is a median — under-powered at human chain density.)
- 🔴**DO NOT BUILD A "PENDULUM LOCK" AXIS (tested 2026-08-22).** Two-state alternations
  (one hand oscillating between exactly two `(x,y,dir)` states) are real and readable in our
  maps — `map_view --bars 105-108 --idioms` on 1f333 shows ~8 beats of it — but **humans
  produce them at the same rate**: our median share sits at human **percentile 0.57**, 0/23
  above p90. A zero-spike looked decisive (29 % of humans emit zero, 0/23 of ours,
  p=0.0025, survives a sparsity control) but **the gap exists only at the run-length
  threshold I picked and reverses sign by MINRUN=10**, where our maps are cleaner than
  human. Tool kept at `scripts/diag_pendulum.py` as a reading aid, **not** as an axis.
  ★Generalisable: sweep the free parameter before believing any threshold-defined metric.
- ⚠️⚠️**`copy.deepcopy` of a `scorecard._load_any` beatmap DOES NOT ISOLATE IT.** `_load_any` builds a
  local `_BM` whose `color_notes` is a **class attribute**, so the copy shares the same note list and
  the same note objects (`deepcopy(bm).color_notes is bm.color_notes` → True). Mutating "a copy"
  corrupts the original; in a loop over perturbations every row after the first is contaminated.
  **Re-read from disk per variant instead.** Caught because three perturbations agreed to 3 dp — *a
  tie to 3+ decimals is a construction, not a result.*
- ⚠️**`calibrate_playfeel.load_expert_only` returns a 2-TUPLE** (no onsets), so scoring a human map
  through `score_cohort` silently yields `alignment = nan` unless you pass
  `scorecard.onsets_for(path)` yourself. Both sides of any ours-vs-human timing comparison must use
  the **same** onsets.
- 🔴**RETRACTED 2026-08-13: `ebpm_burst` is NOT bpm-contaminated and needs no fix.** Recomputed from
  note TIMES with a wall-clock burst window it is **identical to 0.1 swings/min** on `same`- and
  `half`-tempo songs alike. The 2026-08-11 test re-scored the same beat numbers under a different bpm
  label, which is not a relabelled grid but **a different song**. ⇒The old "derive it from note times"
  fix would have changed nothing. **The real defect is below.**
- 🔴🔴**KNOBS ARE DEAD UNTIL PROVEN OTHERWISE — THREE IN ONE SESSION (2026-08-24).**
  `idiomize --width` (accepted, threaded, never used for weeks), `--travel-target` (added to
  `idiomize_zip`'s signature, never passed to `idiomize()` — caught because three arms printed
  IDENTICAL rows), and `mapctl auto --doubles` (works, but gated behind `--accent-slots` defaulting
  to `0,8` where `autobuild` uses eight positions, so it produced nothing).
  ★***Change the knob and DIFF THE OUTPUT before building on it. A knob whose arms agree to three
  decimals is not a weak lever, it is an unwired one.***
  ⚠️And when two entry points wrap the same engine, **diff their defaults** — `autobuild` and
  `mapctl auto` disagreed on both `--doubles` and `--accent-slots`.
- 🔴🔴**NEVER NAME A MODULE AFTER AN INSTALLED PACKAGE.** `agent_mapper/emptiness.py` was called
  `coverage.py` for one commit and broke **8 tests in four unrelated files**: `agent_mapper/` is on
  `sys.path`, so it SHADOWED the `coverage` package for every importer, and numba's
  `coverage_support` died on *"module 'coverage' has no attribute 'types'"*. **The traceback named
  numba and no file of ours.** Same trap for `types`, `json`, `parser`, `test`.
- 🔴**`pytest -q 2>&1 | tail -2` HIDES THE EXIT CODE** — the pipeline returns `tail`'s status, so
  `pytest … | tail && git commit` COMMITS ON A RED SUITE. It did, 2026-08-24. Redirect to a file and
  check `$?`. ★This is landmine "my filters hide the error I need to see", caught again.
- ⚠️**Never edit a running bash script** — bash reads it incrementally and a one-byte shift corrupts
  its read offset. Kill, edit, relaunch.

- `scripts/generate.py` takes `audio` as a **positional** arg, not `--audio`.
- Load beat checkpoints with `strict=False`.
- **Never pick inference checkpoints by `val_token_acc` / `val_f1_avg_tol`** — they anti-correlate
  with alignment and structure quality.
- Production inference: layout `version_10`, beat `version_4`, `section_gate="loud_only"`,
  temp 0.9 / top-p 0.97.
- **The single-song probe trap**: 1f333 is half-tempo and beat-domain metrics lie there. Validate on
  all 24 songs. This trap has now caught two separate hypotheses.
- `pgrep -f <name>` inside a shell script **matches its own command line** and never fires. Wait on
  an explicit PID instead.
- `eval_sweep.py --true-bpm` writes to the **same cache key** as a normal run and will silently
  overwrite a non-oracle arm. Use a distinct arm name.
- Redirecting into a path that may be a **symlink** can truncate the target — `~/.local/bin/arcviewer`
  was a symlink to the running ArcViewer binary and was saved only by `ETXTBSY`.
- Logs under `logs/` and everything in `outputs/` are artifacts, not commits (see C6).
- 🔴**NEVER EDIT `generate.py` (or anything it imports) WHILE A SWEEP IS RUNNING.** `eval_sweep`
  spawns a **fresh `python scripts/generate.py` per map**, so an edit takes effect mid-run and the
  arm silently becomes half one algorithm and half another. It does not crash and it still prints a
  number. Hit 2026-08-04 (the `BEAT_HAND_DEAL` strict→lead-aware fix landed mid-sweep); the deal-arm
  caches had to be deleted and the sweep relaunched. **Either wait, or copy the tree first.**

### Explicitly deprecated (do not revisit)
| Thing | Why |
|-------|-----|
| Scratch `AudioEncoder` mel transformer | MERT knows more music than we can teach it |
| Δt tokens in Stage 2 | Timing is explicit from Stage 1; conflating WHEN and WHAT was the root failure |
| `phrase_energy_alpha` / `dt_density_alpha` losses | Symptom treatment for a missing-explicit-timing root cause |
| `bomb_hand_weight` tuning | Bomb attractor was a symptom of bad timing loss |
| Per-window Δt autoregressive inference | Replaced by beat-slot iteration from Stage 1 |
| `BEAT_IOI_PRIOR` as a density lever | Measured negative at 3 seeds: fails its own purpose, wrecks 3 axes |
| `BEAT_GRID_SUBDIV` | No-op on the v7 production path; retired before it ran |
| Tuning anti-repeat / `dir_entropy` upward | "More diversity = more human" is false — and it caused K2 |
| Near-integer BPM as a crash cause | Falsified; the ArcViewer crash was in-process GTK |

### Success criteria — **rewritten 2026-08-02 against measured human values**
The previous version targeted "NPS ≥ 5.0, Expert range 4–10". That is now known to be **wrong**: the
human Expert median is **3.91 nps**, and 6.18 is the number Kyle called unplayable. Superseded by:

1. **Alignment** — onset precision ≥ 0.93, scatter ≈ 10 ms, **and no within-song drift** (K1).
2. **Difficulty** — ≈ 3.9 nps, diagonal share ≈ 0.37 and *falling* with local speed (K2).
3. **Structure** — double share ≈ 0.23; a legible pulse at human density (C3, C5).
4. **Reproducibility** — passes across **≥ 3 seeds**, not one lucky run (P0).
5. **The real gate** — Kyle plays it and wants to keep playing. The suite has been wrong about
   "ready" twice and right zero times; it is a filter for obvious defects, not the judge.

### Habits that outlived the seed lottery
1. **Score every arm at ≥3 seeds and quote the sd.** ⚠️n=3 *underestimates* sd — treat it as a screen.
2. **`npass` is not a ranking statistic** (an identical config scored 4, 4, 2). Rank per-axis with error bars.
3. **Open**: the spread bar (0.35) sits inside the noise — stop gating on it, keep a hard alarm near 0.15.
   Not done unilaterally; it changes scorecard semantics.
