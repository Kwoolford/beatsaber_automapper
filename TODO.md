# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md); the agent-authoring trail is in
[`agent_mapper/PROGRESS.md`](agent_mapper/PROGRESS.md). Evaluation-suite rationale is in
[`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule:** when an item finishes, its *outcome and what it taught* moves to PROGRESS.md and the
item is **deleted** from here. A completed item is history, not work. Curated 2026-08-02 (from
4,076 lines), 2026-08-14 (from 652), **2026-09-02 (re-planned around the audit in
[`docs/audit_2026-09-02_buildmap.md`](docs/audit_2026-09-02_buildmap.md))**, and **2026-09-10
(from 629: ten shipped P-sections collapsed to one toolbox table, one query table, and the open
items grouped by tool)**, **2026-09-13 (from 874: the day's nine measured levers collapsed to one table, the seven landmines to one list)**, and **2026-09-12 (from 671: the day's ten shipped findings collapsed into P0.7 / P0.8 / P0.9, three live builder items)**, and **2026-09-16 (from 801: the 09-13 dated narrative collapsed to a P0.4 table and a what-is-left list; the ML backlog moved to `docs/ml_backlog.md`)**. ⚠️Roughly a third of what is left is the permanent **REFERENCE** tail —
landmines that each cost a session. Curate the WORK half; leave those.

📖**A read of any map is one command:** `python scripts/verdict.py <map.zip>`. Before believing a
clean one, read `START THE NEXT SESSION HERE` below.

---

## 📍 CURRENT STATE — goal and audit framing (2026-09-02); live status is in START HERE below

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

**Build (best known, 2 of 4 ship):** `autobuild <audio> --pulse --lead-bias 0.2 --lead-in
--drop-orphan --carrier-bias 2.0` then `repeat.py <zip> --out <zip> --song <id>`.
**Build (shipped defaults):** `python agent_mapper/autobuild.py <audio> --pulse --lead-bias 0.2` ([FULL] walls/arcs/
chains and phase-calibrate are the defaults since P0; `--notes-only`, `--no-phase-calibrate`), then
`agent_mapper/repeat.py <zip> --out <zip> --song <id>` (2026-09-12, not yet wired in) — or the
per-section loop in 📖`agent_mapper/WORKFLOW.md`, which **beats autobuild** and by a wide margin:
fresh autobuild + repeat gives 1f333 **5 reds** where the curated chain gives 1.
⚠️**Never compare a fresh autobuild against a staged map** — they are different processes.
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

### ▶️ START THE NEXT SESSION HERE — no GPU, no questions for Kyle (updated 2026-09-16)
**The toolbox is done** (P1 · P1b · P0 · P2 · P2b · P3 · P4 · P4b · P5b · P5c); what is left is the
builder and Kyle's ear.

▶️**Read these three rules before believing any read.**
1. **A clean page is evidence about the QUERIES, not the map** — ask what is ABSENT.
2. **A clean bench row is evidence only if the row COULD HAVE FAILED** — the rows with teeth are
   `humanplus-*` / `humanexp-*`, the human-vs-human panel, and one zip's two difficulties.
3. **Four rows can refute a norm, never establish one** — take mechanism claims to ≥ 100 maps.

**Songset** — best build `outputs/best_2026-09-13b/` (`autobuild --pulse --lead-bias 0.2 --lead-in
--drop-orphan --carrier-bias 2.0` → `repeat.py` → `answer.py`), staged compete zips untouched:
`1f913` **SHIP? YES — nothing located** · `1f8d6` **SHIP? YES** · `1f767` **1 red** (D3 E-drop) ·
`1f333` **2 red** (BREATHING · SCATTER). **2 of 4 ship**, and every shipping map has a code with no
margin. ⚠️Four of 09-13's clearances were THRESHOLD changes made because the code fired on humans;
only `answer.py` changed a map.
**Both 1f333 reds are real and, as of 2026-09-16, neither has a builder fix with evidence behind it**
(see *What is left*): BREATHING is a mapper's choice the song does not announce, SCATTER is
vocabulary width, which is mapper style.

❓**THE DECISIONS THAT ARE KYLE'S** (the pairs are staged either way; nothing blocks on them):
1. **P5 — play ONE pair** (`compete.py verdict <sid> X|Y|tie --because …`). It is the only thing
   that moves the headline, and the only test of whether BREATHING/SCATTER matter to the ear.
2. Stage red maps (tests the codes) or rebuild first (tests the builder)?
3. The FLOW fix is a trade (`1f335` staged as its own A/B); `--taper` and `--palette` are style
   levers waiting on his ear.
🔴**DECIDE-AND-LOG.** Nothing below may block on Kyle.

⚠️**Read THE BUILDER'S LEVERS before touching any of them.**

---

## ✅→🔵 SHIPPED — the toolbox (P1 · P1b · P0 · P2 · P2b · P3 · P4 · P4b · P5b · P5c)
Every one of these is DONE; what it taught is in `PROGRESS.md`. This section keeps only the
**pointer, the live leftovers, and the rules that were paid for**. Curated 2026-09-10.

| tool | what it is | history |
|---|---|---|
| `agent_mapper/score.py` | song + map on ONE 1/16 lattice: `--sections` · `--bars a-b` · `--vs auto` · `--npz` | 2026-09-02b |
| `agent_mapper/mapedit.py` | the score is WRITABLE, in `bar.beat.sub` addresses, guarded, with `undo` | 2026-09-02b |
| `mapjudge` gates | alignment floor 0.822 · requested density · parity — see the reverse table below | 2026-09-02c |
| `scripts/bench.py` | **23 rows** of Kyle's verdicts + the controls; `score module:func` is the only validation that counts | 2026-09-02d |
| `scripts/tutor.py` | the top human map of the same song, on the lattice: situation → pattern, `--vocab`, `--copy/--thin` | 2026-09-02e |
| `scripts/queries.py` | **eight** named defects as numpy, every hit an address | 2026-09-02f · 09-03b · 09-10 |
| `scripts/verdict.py` | queries + ABSENCE + tutor + judge on ONE page, `SHIP?`, exit 1; gates `/buildmap` | 2026-09-02g · 09-10 |
| `scripts/compete.py` | blind X/Y vs the top human map; ⚠️**never print anything per letter to Kyle** | 2026-09-02h |
| `scripts/make_bench_fixtures.py` | the difficulty controls — the only human negatives that can fail | 2026-09-10 |

### The eight queries (`bench.py score queries:q_all` → not refuted, 4 strong hits, 0 false fires)
| query | codes | reads |
|---|---|---|
| `q_events` | EMPTY D1 D6 | events (a double = 1) per 4 bars vs the human's: < 0.6× → EMPTY; ≥ 2× → D6 over-dense; map-wide doubles ≥ 50 % and 20 pts over him → D6; median ratio < 0.7 → D1 |
| `q_flow` | FLOW D2 | 2-bar sliding: ≥ 30 % of events start on an odd 16th from silence (≥ 20 pts over the reference) while ≥ 30 % sit on the 8th grid → FLOW; ≥ 80 % odd 16ths where he has < 35 % → D2 |
| `q_vocals` | D4 | per 4 bars: vox-MAIN slots answered (±1 slot) ≥ 25 pts under the human, human ≥ 60 % |
| `q_drops` | D3 | at song E-jumps: first note > 1 beat after his, or step < 0.8× his AND density after < 0.8× his; at E-drops he halves and we do not |
| `q_elements` | ELEMENTS | 0 walls where the human has ≥ 5 |
| `q_breathing` | BREATHING | a run of ≥ 2 bars he leaves empty in which we play ≥ 4 events and ≥ 2/bar (09-03b) |
| `q_scatter` | SCATTER | map-wide: mean 4-bar-block echo ≥ 0.15 under his — how much of each block the map has already played (09-10) |

★★**THE THREE RULES THE READS ARE BUILT ON — every one of them cost a session:**
1. **Every reference is the SAME SONG'S human map** (or the song's onsets without one). Absolute
   norms have now been refuted **four** times: a D3 step floor fired on human 1f913; "odd-16th =
   shifted" called 20 spans of `1f335` shifted (at 195 bpm the odd 16th IS the felt 8th); an
   absolute scatter floor would have called 1f767's human a scatter; and `_tutor_ok`'s 50 %
   called a top mapper's own map 🟡.
2. **A LEVEL claim cannot be made across two declared difficulties** (`queries.cross_difficulty`)
   — EMPTY · D1 · D4 · D6 over-dense · the density step at a drop are skipped and show ⚪.
   SHAPE claims are fair and still asked. 2026-09-10c has the eleven false fires that proved it.
3. **A clean page is evidence about the queries; a clean bench row is evidence only if the row
   could have failed.** Run `humanplus-*` / `humanexp-*` before believing any clean side.

🔴**REFUTED READS — do not retry without a NEW COLUMN or new evidence:** hand-role histogram
distance (backwards: AGENT 0.54 > A+ 0.16) · 16th bursts followed by rest (0 at the labelled
bars) · `±ms` alignment (no separation; D2 is the shifted grid) · D5 as 16th-runs-without-onsets
or isolated fast clusters — **D5 still has NO locator** · notes-inside-walls (the arrays carry no
wall height) · vocabulary breadth per hand-window (fired on a human's own harder map) ·
normalising a window ratio by the map-wide ratio (13 of 16 windows still fired) · **the entry
accent** as a doubles locator (n = 1-9 per song per kind, and difficulty-dependent: 2/9 vs 7/9) ·
**the mirror of BREATHING** ("he plays a run of bars, we play nothing" — zero on all four maps) ·
**the late-song collapse** (all four builds hold 0.85-1.24 across every quintile).

### How to reverse each P0 default
| item | shipped as | reverse with |
|---|---|---|
| **P0.2 alignment floor** | `mapjudge` FAILs below the human 10th pct (**0.822**) regardless of pooled p; `scripts/calibrate_align_floor.py` writes it. Priced on 300 humans: human 0.877→0.793, `offbeat` 0.680→**0.077** | `MAPJUDGE_ALIGN_FLOOR=0` |
| **P0.1 requested density** | `judge(nps_request=…)` / `mapjudge --nps` / `autobuild --nps`: gated ±15 %, `nps`/`peak_nps` leave the pool | omit the request |
| **`[PCAL]`** | `--phase-calibrate` default ON in `mapctl init` and `autobuild` | `--no-phase-calibrate` |
| **`[FULL]`** | autobuild default `--walls 89 --arcs 90 --chains 16`; crossover a knob at 0 | `--notes-only` |
| **width** | default 3; `--width` is the cut-variety knob for P6 | — |

### ⬜ OPEN LEFTOVERS, by tool
**score.py** — ⬜**FLOW has no column that names it**: the page shows it (Hunger AGENT bar 34, six
singles on off-beat 16ths vs kit-aligned strikes) but bar counts barely differ (on-KIT 85 % vs
78 %); what differs is on-beat share 36 % vs 57 % and the per-hand cut sequence. Candidate columns
*before* any metric: **gap-ms since that hand's previous note**, and a **3-cut path glyph**
(`↙→↗`). ⬜`notesheet.py --map` should draw from `score.to_arrays()` so the two cannot disagree.
⬜`--sub 12` (triplets) untested; `--perceive`'s `lyrics` entry point name is guessed. ⬜`E` is
whole-mix RMS; per-stem energy would make a drop a two-column step. ⬜The header should say
**"no percussion cache"** when KIT is empty (1f335, 1f9a0).

**mapedit.py** — ⬜v2 (human) maps are refused; a v2→v3 convert would let a tutor map be edited
into a variant. ⬜**Reset reconciliation is manual**: `mapedit resets` names the pair, the fix is
ONE note per hand (place / delete / flip … X) chosen by reading the score — flipping the second
note cascades. A `mapedit reconcile <bar>` trying the three one-note fixes and keeping the first
with 0 new resets would close the thinning path. ⬜`--thin` keeps the survivor's cut direction;
after a move the note may want re-cutting (his arrow at that slot is known). ⬜no `--dry-run`
(undo is the dry run); chain/arc ops do not check the tail cell is reachable.

**mapjudge** — ⬜`audit_mapjudge`'s human bar reads the `no-floor` column (documented). ⬜
`calibrate_mapjudge.py` drops `align_floor` on a rebuild — re-run `calibrate_align_floor.py`
after it. ⬜The floor needs cached onsets (`scripts/build_onset_cache.py`); autobuild warns.
⬜The two real fails are still unfixed: **`1f335` (0.735)** — EMPTY ×16 incl. bars 77-88 (1 event
vs 31), FLOW ×16, D2 ×6 — and **`1f9a0` (0.475)**, D6 over-dense ×5, 657 events vs his 271.

**bench.py** — ⬜set-B ids for FLASH/GOCRYGO unresolved; set-B rows unreadable until those corpus
songs get perception caches. ⬜**`bars_from: "kyle"` exists on no row yet** — P5 is where his
finger lands on a bar. ⬜`24e6c-dod` stays UNLABELLED; flip to CLEAN when a pass agrees.

**tutor.py** — ⬜Nothing feeds `idiomize.py` / `mapctl reuse`; the patterns are countable
(`--vocab -v`), so the feed is a table lookup: situation kind → pattern word → lever. ⬜Situations
miss chorus-repeat *variation* (glyph-identical at 1f913 43/83), phrase starts inside a section,
and fills before a drop. ⬜`lead-in/out` is noisy on sparse melody stems (coverage < 0.5). ⬜No
tool prints a **reference** tutor score beside ours, and the number is per-song (09-10e).

**queries.py** — ⬜**D5 has no locator** (three reads refuted). ⬜wall HEIGHT into the arrays,
then notes-under-full-walls as an ELEMENTS playability read. ⬜arc/chain playability needs a v3
human reference (songset humans have none). ⬜`q_events` D1 is map-wide; a per-section D1 needs
the request (P6). ⬜ONBEAT_MAIN / HANDROLE have no query — same rule as BREATHING and SCATTER:
**only write one when a verdict names it.**

**verdict.py / `/buildmap`** — ⬜`mapctl clear` + `auto` is still the SESSION workflow; after
`export` the zip is the artefact, so a section rebuild re-runs the whole dress. A
`mapctl reauto --zip z --bars a-b` would give the loop its second section tool. ⬜The pulse path's
odd-16th interval choice (1f8d6 bars 2-4) is a bug to locate in `mapctl auto --pulse`. ⬜Tutor
`same_way` is coarse — a per-beat cell match would make the TUTOR line an edit list. ⚠️**Never
compare two maps by the verdict's HIT COUNT** (a merged SPAN count); `RED_SHARE` is the
comparable number. ⚠️`--density-search` is **NOT proven for quality and must not become the
default**; 1f8d6's tutor drops 4 → 1/15 under it, unexplained, and its density still misses −24 %.

**compete.py** — ⬜note counts differ visibly in ArcViewer (771 vs 632 on AliceBlue) and the audio
is re-encoded: a tell only to someone who knows our count. Accepted, logged. ⬜`table` has no
per-song "what the page said at staging" column; the key carries it — print it when n > 0.
✅**The pair is harmonised** (2026-09-10f): both sides fly the HUMAN's difficulty label, NJS and offset, because
ours emits a constant 16.0/0.0 that named our side in every pair — and 1f913 had gone out as ExpertPlus at NJS 19
against Expert at 16. Each side's original values are kept as `own` in the key. ⬜The target win rate is Kyle's to
set; until then the number is reported, not gated. ⚠️**One pair per listening session.**
🔴⚠️**`BEST` resolves a bare song id and it silently downgraded 1f333 on a restage** (p4b_loop was missing from it:
1 red → 5, tutor 31/49 → 15/49, with nothing in the output saying so). **Any new output dir holding a looped map
goes at the FRONT of `BEST` the day it is made**, and a restage should be diffed against the key.


## 🟡 P5 — THE LABEL CHANNEL: his remaining oversight, cheap and cumulative
`review.py next` asks **one thing per session** — *"play X; if anything is wrong, give me the
timestamp and the word"*; `review.py defect --at` appends to the P2 bench automatically; `/close`
rule: a Kyle verdict that disagrees with the agent's read is a bench row AND a P3 task, never a
TODO opinion. **DoD**: bench grows ≥ 1 row per listening session with no JSON editing; pending
list ≤ 4 maps.

## 🔵 P5b/P5c — SHIPPED 2026-09-10. Open leftovers only
`PROGRESS.md 2026-09-10 · b–g`. Shipped: `q_scatter`/SCATTER · ABSENCE on the verdict page ·
the `humanplus-*` / `humanexp-*` difficulty controls + `make_bench_fixtures.py` · `MapData.difficulty`
through to `queries.cross_difficulty` · the TUTOR line uncoloured.

- 🔴**No density claim is possible about `1f913`** — its only human map is an ExpertPlus and every build
  we make declares Expert, so its page prints ⚪ EMPTY · ⚪ D1 · ⚪ D4 and its blind pair asks Kyle to
  compare two difficulties (`.key.json` carries the caveat). ⬜Build 1f913 at his difficulty, or find an
  Expert human map of it.
- 🔴**The doubles gap has no locator.** `1f913` plays 4.2 % doubles against his 32.9 % and every read says
  ✅. ⛔The entry-accent locator is REFUTED. ⚠️Whatever comes next must not be "distance to his number".
- ⬜**Walls**: `q_elements`' 0.5× coverage red asks a map to match an unpredictable quantity — wall coverage
  has **R² = 0.089** against the song over 600 maps, so it is mapper style. The *systematic* under-walling was
  real and is fixed (`walls.py`, 2026-09-12). 🔴**The threshold was NOT loosened.** DoD to revisit it: a
  human-vs-human negative — two mappers' Expert maps of one song — showing how often 0.5× fires between two
  people who both did it right. Centring the level also bought the opposite failure: over-walled on 24 %.
- ⬜`audit_map.py`'s ABSENCE reference (250 corpus maps) does not separate difficulties; bites only on a song
  with no human map.

## 🔧 THE BUILDER'S LEVERS — every one measured 2026-09-12/16, all default OFF but two
`PROGRESS.md 2026-09-12…2026-09-13g` has the full working. Control arm rebuilds **byte-identical**
at every default; that check is mandatory before reading any sweep.

| lever | default | what it does | measured |
|---|---|---|---|
| `walls.py` corridors | **ON** | wall durations in the human's 3 modes, corridors where onsets thin | coverage 0.40× → **0.98×** his, ELEMENTS fires 59 % → 21 % |
| `repeat.py` | **ON** (manual) | a section's figure comes back when the song repeats | SCATTER clears on 3 of 4; **1f913 = "nothing located"** |
| `--lead-in --drop-orphan` | off | take the on-grid note before an odd 16th, else drop it | **FLOW 62 hits → 0 across 16 songs**; ⚠️−0.022 typicality |
| `--carrier-bias` | off (1.0) | vocals can win the carrier ranking, not just the busiest stem | D4 **29 → 23**; ⚠️no song clears, 3 regress |
| `--nps-from-song` | off | density from bpm + onset rate, not the fixed 4.17 | our nps sd 0.48 → 0.64 (his 1.00); reds flat |
| `--palette` | off | commit the map to N landing shapes/hand | SCATTER room +0.23-0.29, coverage intact since STRICT_PARITY (09-16b) · ⚠️judge p −0.17, ~0.5 local per echo |
| `--hand-run-p` | off | a takeover sometimes holds a run of 4+ | runs≥4 0 → 9–21 · ⚠️costs `idiom_coverage` |
| `--energy-slope` | off (0.60) | how hard energy scales the budget | 🔴NULL on D6 |
| `--vocal-keep` | off (1.0) | gentler accent cut on a vocal spec | 🔴NULL — the cut was never binding |
| `--taper` | off (0) | move budget from the bars before an energy rise to the bars after | D3 11 → 7 hits · 1f767 ships · ⚠️1f8d6 ship → 2 red (zero-margin crossings) |
| `--map-memory` | off (0) | boost landings this map already played | SCATTER room +0.2-0.3, stable since STRICT_PARITY · ⚠️local cost ~2x human, judge p −0.2 |

**Best known build** and where each songset map stands: see START HERE.

### ❓ THE ONE DECISION THAT IS KYLE'S — the FLOW fix is a TRADE
**FLOW → 0 on all 16 songs, against 23-metric typicality worsening 0.511 → 0.533 (worse on 11/16).**
The repo's own rules say a PASS is *not defective, not good*, typicality is a floor never a rank,
and his ear is the arbiter — on that reading the fix is right, but it is his call.
✅**Staged**: `1f335` is a blind A/B of the fix against its own baseline (`compete.py --against`,
does not count toward the win rate). Default stays **off** until he plays it.

### ⬜ Is LOCAL VARIETY aspirational or merely typical? (outcome in `PROGRESS.md` 2026-09-13aa-ab)
`REPEAT_P` is now 0.25, which lands `idiom_local` on the human median; 0.00 lands at the 70th-86th
percentile and **nothing else moves** (echo, coverage, typicality, hits all unchanged). Both are
inside the human range, so the median is the defensible default and 0.00 has no evidence behind it.
⬜**Only Kyle's ear can settle which**: the standing rule says the corpus median is a FLOOR for
aspirational axes, and `mapjudge` scores typicality so it cannot tell these apart. Not staged as a
pair — five already wait on P5 and a sixth adds nothing.
★**The rule it paid for: check what a knob measurably MOVES, not what its pass is named after.**
### 🔧 P0.4 — every per-song threshold against a real human control (history: PROGRESS 2026-09-13ad-aj, 09-16a)
Controls: the **human-vs-human panel** (`outputs/dup_songs_2026-09-13.json`, 172 songs, map-only codes;
`exp_human_panel.py --breathing` aligns pairs by density envelope) and **one zip's two difficulties**
(exact same audio). DoD per code: fires on < ~5-10 % of human pairs, `bench.py` not refuted.
| code | control | result |
|---|---|---|
| ELEMENTS | panel | 0.50x fired on 34 % → **0.10x** |
| EMPTY | panel | 0.60 (25.9 %) → **0.30** (8.4 %) |
| D6 · SCATTER | panel | kept (8.6 % · 7.1 %) |
| BREATHING | panel (aligned) + difficulties | kept (3.4 % of pairs; but a 2nd mapper plays through 16-25 % of a 1st's rests) |
| D3 absolute | 120 human maps | step claim **dropped** (85 % red), lag kept at 2.0 beats |
| D3 human-ref | difficulties | passes (2-5 % red) ⇒ both standing D3 reds are genuine |
⬜**D3's LEVEL clause** — uncontrolled (skipped across difficulties; the panel has no shared audio).
⬜**D4** — untested: no reference-free branch, and the panel's audio does not align. Per-pair audio
cross-correlation would open it (the map-envelope aligner is validated, 09-16a).
★**A row that never runs the code is not evidence at all** (`bench.py` cannot reach D3's no-human branch).

### ✅ `agent_mapper/answer.py` (2026-09-13ai) — land a note when the energy rises
Moves (never adds) the first note of a late energy rise onto the earliest onset after the bar line.
1f333 3 red → 2. Runs after `idiomize`/`repeat.py`, before walls — the only pass that moves a TIME.
⬜Not wired into `autobuild`; like `repeat.py` it is a manual step in the best-build chain.

### 🔴 What is left on the songset, and why each is hard
- **BREATHING (1f333 bars 163-169)** — ✅CLOSED as a builder item 2026-09-16a: no song feature
  (energy band, quiet runs, drum dropouts) predicts where a human rests (≤ 18 % of runs). A pass
  copying his rests clears the code by construction ⇒ **not built**; Kyle's ear / P6 "breathe".
- **SCATTER (1f333 bar 81)** — the gap is **figure-vocabulary width**, uniformly (PROGRESS 09-13r,
  09-16c-e): his echo is the same in returning and first-occurrence blocks, from near and far
  sources, with no drift over the song; ours is ~0.15 lower everywhere. In humans local variety
  tracks width too (r +0.625), so narrowing costs local for them as well.
  Levers that narrow: `--palette` (stable since STRICT_PARITY, room +0.23-0.29, judge p −0.17) and
  `--map-memory` (room +0.2-0.3, local cost ~2x human). **Neither clears it; both are STYLE levers.**
  🔴Refuted mechanisms: distance-return · per-section memory · narrow-figures-varied-order.
  ✅**DoD for any new one**: `idiom_local` no further below humans AT THE RESULTING ECHO (±0.05,
  `exp_vocabulary.stats`) than the control is — palette passes on 1f8d6, fails on 1f333. Never a
  pooled rate (the old "0.23" was endpoint-selected; OLS says 0.47).
  ⚠️Nothing in the song predicts width (r² 0.039) — it is mapper style, so an unfixable red is a
  live possibility; **never optimise echo directly**.
- **D3 (1f767 E-drop bar 42; 1f333 lag bar 170, fixed by `answer.py`)** — D3 is *"breathe before
  the drop"*: he empties the 2 bars before a jump (ours ÷ his 1.47) and floods after (0.85).
  `autobuild --taper` (v2, default 0) moves budget across the boundary: D3 11 → 7 hits, 1f767
  ships, **but 1f8d6 goes ship → 2 red** — only because it was passing with **zero margin**.
  ⬜Decide the taper default once margins are wider, or ship it as a P6 style lever.
- **The density family (EMPTY · D6 · D4)** — one defect (our per-song nps sd 0.48 vs his 1.00) at
  its ceiling: the best song predictor reaches R² 0.231. ⛔"Tune the density" is retired.
- **The pulse pass never holds a pulse**, and `--pulse` is a trade (FLOW for ABSENCE), not a defect.

### ✅ Margins — every code reports its own (PROGRESS 2026-09-13l-q)
`room` = 1 + signed slack / line, **1.00 is the line**, invariant `room < 1.00` iff fired —
🔧`python scripts/check_margins.py` (exit 1) after touching any query or threshold. A margin is a
second implementation of the query: copy its GATE and POPULATION, not just its threshold.
★**A map passing AT a threshold is not really passing**: at the 09-13 close, 1f8d6 (walls 0.50x,
lead-hand 1), 1f767 (lead-hand 1) and 1f913 (echo +0.118 of 0.150; D6 1.65x of 2x; D3 exactly on
the line) all shipped with no margin — so any lever that nudges them looks like it "broke" a map.
★A margin says how close a map is to a LINE, not how close it is to being right.

### ★★★ THE SHAPE EVERYTHING TODAY HAD — read this before building anything
**Eight times in one session the builder reproduced the corpus MEAN and missed the per-song,
per-window or per-SECTION VARIATION**: wall duration (pooled marginal, not the 3 modes) · the pulse
(an interval sometimes, never held) · hand runs (the mean run length, never the tail) · the
vocabulary (no per-map palette) · block echo · per-song density · D4's windows · and the
within-section taper (D3).
⇒**When a read says we are inside the human range on the mean and outside on the spread, the fix is
never a rate knob — it is a hold/commit mechanism, and a rate knob will saturate trying.**

### 🔴 AND THE LANDMINES THIS SESSION PAID FOR
1. **Verify the CONTROL ARM reproduces the known baseline BYTE-FOR-BYTE before reading a sweep.** An
   indentation slip made every later build flag conditional on a new one; the sweep looked monotone
   and the control was a different builder.
2. **Measure a DoD with the TOOL THAT WILL JUDGE IT**, not a reimplementation — mine disagreed 3×.
3. **Measure the AGGREGATE as well as the named axis.** A DoD naming one metric passed a change that
   moved 23-metric typicality the wrong way on 11 of 16 songs.
4. **Scrape the judge's FULL table (`--top 30`), never its worst-N list** — an absent flag reads as a
   value and silently confounds the column.
5. **A share-of-OURS and a share-of-MISSED are different measurements.** Conflating them made me
   retire the vocal framing, wrongly, for two iterations.
6. **Check which ARTIFACT a red belongs to** before building a mechanism for it — one belonged to a
   three-week-old map, not the builder.
7. 🔴🔴**FIVE documented-but-unwired things found in one day**: `width`, `travel_target` (×2),
   `PERIODS`, and `MELODIC`'s preference order. **This codebase's comments describe intent the code
   does not implement, and only a measurement ever catches it.**


## 🟡 P6 — STYLE REQUESTS: "make it more X" as a lever table + presets
`docs/style_levers.md` — one row per request (*faster · harder · more diagonals · more doubles ·
one hand leads · follow the piano · breathe before the drop · more walls*) with the lever, its safe
range, and the score column that shows it moved; `mapctl auto --style {flow,tech,dance}`;
`verdict.py --style` judges against the preset where P0.1 uses the nps request.
**DoD**: three presets build clean on the songset with the named column moved. ⚠️Levers stay
monotone and default-off — they ship in a UI (`feedback-levers-are-user-facing`).

---

## 🔵 CARRIED FORWARD — still live, lower than P0–P6

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

### Doc debt from the audit
`CLAUDE.md` V6-era, no `agent_mapper/`; `buildmap/SKILL.md` contradicts itself on doubles.
✅`/todo` Step 4 rewritten 2026-09-10 around the agent path — and the note it was checking, the
**late-song collapse, is NOT REPRODUCED**: all four current builds hold their event ratio to the
end (0.85-1.24 across every quintile), and the sag is visible only on the August maps, on top of a
map-wide deficit of 0.65-0.77 that is EMPTY/D1's job. `PROGRESS.md 2026-09-10d` has the table.

## 🧊 ML PIPELINE — BACKLOGGED, and its C1-C5 diagnoses → [`docs/ml_backlog.md`](docs/ml_backlog.md)
Deprioritised by Kyle 2026-08-20; **do not queue from `/todo`**. Kept there with their evidence and
landmines (C3 — *you cannot thin your way to human density* — reproduced with no ML in the path).

### ⚠️ SEEDS ON THE AGENT PATH — read before quoting any "n seeds" number
**10 of 23 metrics are seed-INVARIANT by construction** (every time-domain one). The agent builds
from **cached events**, so note TIMES are deterministic — the opposite of the ML path, where a seed
re-draws the Demucs stems. ⇒Seeds matter only for **geometry and hand-role**; `--seed` reaches
`mapctl auto` since 2026-08-21.

---

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
