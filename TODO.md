# Beat Saber Automapper — what we are working on next

<!-- ══════════════════════════════════════════════════════════════════════════════════════════
     IMMUTABLE CHARTER — DO NOT EDIT, SUMMARISE, CURATE OR DELETE THIS BLOCK.
     Kyle, 2026-09-17. Every session reads this FIRST and checks its plan against it.
     ══════════════════════════════════════════════════════════════════════════════════════════ -->

## ⛔ CHARTER — WHAT THE AGENT BUILDING SUITE IS FOR (immutable)

**The point of this suite is TOOLING THAT LETS AN LLM AGENT — a model in a harness, with tools it
can call at will — BUILD A MAP BY HAND.** Not a script that emits a map. Not a generator with
knobs. A model that studies a song, forms a plan, and then places and edits notes itself, calling
as many tools as it needs, for as long as it takes.

> **Kyle, 2026-09-17:** *"If I wanted to autogenerate a map I would use the machine learning model
> we have been training and tuning for months. The purpose and goal of the agent building suite was
> to do exactly what I just laid out."*
>
> **And what he laid out:** *"you spending the time to (A) create a plan/vision for the map. Is it a
> power metal song, a groovy dance song, a fast tempo camelia song? Does the user ask for
> easy/med/hard/exp/exp+ with fast tempo or do you want the difficulty to come from hard swings.
> Then you reading the notesheet to understand the song as I would listen to it and create a plan
> before making it. Then think of a flow that would be fun to play and then slowly note by note /
> segment by segment — create the map."*
>
> **Kyle, 2026-09-02:** *"It does not need to one-shot build a map, or build it quickly. I want to
> give it the tools so it can achieve a great map in the end… should be able to call more tools and
> know it can, and recognise which parts need more tooling and attention to detail."*

### The loop every map build MUST follow
1. **PLAN / VISION first, written down, before any note.** What kind of song is this (power metal,
   groovy dance, fast-tempo Camellia…)? What difficulty was asked for, and **where does the
   difficulty come from** — tempo, density, or hard swings? What should the map's character be?
2. **READ THE SONG as a listener would** — `agent_mapper/score.py` (kick/snare/hat, bass and lead
   pitch, vocal pitch + lyric syllable, section, energy, onsets) and the top human map via
   `tutor.py`. Name what each section is doing and what the map should follow there.
3. **DESIGN THE FLOW** — how the hands move through each section, what makes it fun to play, where
   it breathes, where it hits hard.
4. **BUILD SEGMENT BY SEGMENT, NOTE BY NOTE**, editing with `mapedit.py`, reading the score back
   after each segment. Zoom where it needs attention; coarse where it does not.
5. **The queries / verdict / judge are a BACKSTOP, never the definition of good.** A clean page
   means *no defect anyone wrote a query for*. It is not evidence the map is good, and it must
   never be handed to Kyle as if it were.

### Standing prohibitions
- 🔴**Never hand Kyle a one-shot `autobuild` map and call it done.** `autobuild` is a *starting
  sketch* for the agent to edit, or a control arm for a measurement — not a deliverable.
- 🔴**Never quote a verdict page as quality.** Read the map yourself and say what you heard/read.
- 🔴**Adding another generator knob is not progress.** If a session's output is a new flag on
  `autobuild`, it has drifted. Build tooling that lets the AGENT decide and act instead.
- 🔴**The ML pipeline is the autogenerator.** This suite exists because that is not what Kyle wants
  from it. (Drift record: `PROGRESS.md` 2026-09-17r.)

<!-- ════════════════════════ END IMMUTABLE CHARTER ════════════════════════ -->

## 🎯 ACTIVE WORK ITEM — hand-build "SO TIRED ROCK" (NUEKI), by the charter loop

**Audio**: `data/eval_songset/SO TIRED ROCK - NUEKI.mp3` (also in `data/test_songs/`).
**Kyle's brief (2026-09-17)**: *"Map so tired rock. I want it to be a medium tempo expert level map
with ~6 nps and to be simple flows with the hype fast drop sections to still be hard and fast
tempo."*

| ask | target |
|---|---|
| difficulty | **Expert** |
| overall density | **~6 nps** (⚠️ 6.18 is the number Kyle once called unplayable — hold ~6, never above) |
| body of the song | **medium tempo, SIMPLE FLOWS** — readable, comfortable hand paths |
| drop / hype sections | **hard and fast** — the difficulty lives here, not spread evenly |

**How to run it** (charter loop, not autobuild-and-ship):
1. Write the PLAN first — song character, section map, where the difficulty comes from, what each
   section follows (this is a rock song: the kick/snare backbone and the guitar/bass riff matter
   more than the vocal in the heavy sections). Show Kyle the plan before building.
2. Read the song with `score.py --sections` then `--bars a-b` on each section; there is no human map
   of this song (`--vs` will be ⚪), so the song itself and the plan are the reference.
3. Build segment by segment; `autobuild` may seed a section's rhythm, but every segment is read and
   edited by hand before moving on.
4. Keep a running note of decisions per section, so the finished map has a rationale Kyle can argue
   with.
**DoD**: Kyle plays it and wants to keep playing. The verdict page is run at the end as a backstop
(and must show 0 note collisions — see below), never as the reason to ship.

### 🎮 FOR KYLE — to do when you have time (left 2026-09-18)

1. **Play it — v3 first:** `outputs/for_kyle_2026-10-04/SO_TIRED_ROCK_Expert_v3.zip` (sub-name "v3").
   v2 = v1 with the 60 chorus **inward up-diagonals** (red UR / blue UL, ~0 % in human maps) cut straight
   up; v3 = v2 with 24 doubles de-mirrored (96 % of v2's doubles were exact mirror images — "both hands
   do the same thing"). Both found by the new blindspot sweep (`outlier_sweep.py`). v1 is still at
   `outputs/for_kyle_2026-09-18/`. If v3 feels worse than v1 anywhere, that is the most useful answer.
2. **Skim the why:** 📖`agent_mapper/sessions/sotired/LOG.md` (one table + the four decisions made
   without you).
3. **Tell the next session, in any words — these are the decisions I took on your behalf:**
   - **Density:** 5.0 nps played (4.8 whole file) instead of ~6. Too easy in the verses/choruses?
   - **Drops:** are bars **33–40 (1:02)** and **65–72 (2:04)** the "hype fast" sections you meant?
   - **Hype 2:** two unbroken ~8 s 16th streams. Fun or tiring?
   - **Anything that felt wrong, with a rough time** (the log maps bars ↔ times).
4. Your words go into LOG.md verbatim. They are the DoD, and the only real evidence about quality.

### ▶️ NEXT SESSION STARTS HERE (2026-10-05)

✅**Blindspot sweep shipped** (PROGRESS 2026-10-04b, 2026-10-05c): `scripts/outlier_sweep.py map|local`
(unnamed defects vs human maps of the same density), on the verdict page as 🟡 UNNAMED. So Tired Rock
**v2/v3** built from its findings — Kyle plays v3 (FOR KYLE above). **The DoD is still his play.**

⬜**W7 — HEAVY METAL LOVE (2f1e5), the BLIND hand-build** — the rigorous test of the agent mapper.
State: song read ✅, lanes ✅ (`agent_mapper/sessions/hml/lanes.txt`), lyrics ✅, **PLAN written**
(`sessions/hml/PLAN.md`, force-add: `sessions/` is ignored). No note placed yet. Two human maps exist
(`2f1e5`, `2ef6a`) and have NOT been opened — keep it that way until the map is finished.
Tasks: (1) W6 below; (2) spec section by section (verse → chorus → bridge → chorus 2), running
`outlier_sweep.py map` + `local` after EACH section; (3) verdict + UNNAMED clean; (4) ONLY THEN open
the humans: verdict `--vs`, fingerprint distance ours↔human vs human↔human (2f1e5↔2ef6a is the
yardstick), songprint. **DoD**: on the fingerprint, ours sits within the human↔human distance on
most features, with every exception named and argued; then Kyle plays it.

⬜**W6 — promote `sessions/sotired/compose.py` → `agent_mapper/compose.py`**, figures and DOUBLE-PAIRS
declared in the spec (`@figdef`, `@dbldef`), so doubles stop being mirrors by construction and the
"even singles between hits" rule becomes optional (humans write 41 % mixed doubles). DoD: rebuilding
So Tired Rock from its spec + figdefs reproduces v3 note-for-note.

⬜**W8 — sweep follow-ups** (evidence: PROGRESS 2026-10-05c): (a) no feature has a human control
showing it COSTS play — the cheapest is Kyle's v1/v3 verdict; (b) doubles +25 % is only 23 % detected,
dead 8-bar sections 42 % map-wide / 0 % local; (c) re-read the 09-17 autobuild vision-block finding
on a fresh build before anyone uses autobuild as a sketch again.

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
clean one, read the charter at the top.

---

## 📍 CURRENT STATE — goal and audit framing (2026-09-02); live status is in the ▶️ block above

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

**Build (best known, 2026-09-17, after the parity-slip fix):** `autobuild <audio> --pulse --lead-bias 0.2
--lead-in --drop-orphan --carrier-bias 2.0 --hand-run-p 0.06` → `repeat.py` → `answer.py` (each
`<zip> --out <zip> --song <id>`). Measured: eval 20 → 23 of 69 ships, **held-out 16 → 20 of 36 with
nothing added** (re-measured on real audio energy, 09-17l); replicated on a 2nd held-out set, 21 → 30 ships
over 3 seeds (09-17o). `--taper 0.5` is a STYLE lever ("breathe before the drop"): it lifts the eval set 20 → 28 but not
the held-out set (16 → 17, adds D3) — use it on the songset (1f767 ships with it).
**Build (shipped defaults):** `python agent_mapper/autobuild.py <audio> --pulse --lead-bias 0.2` ([FULL] walls/arcs/
chains and phase-calibrate are the defaults since P0; `--notes-only`, `--no-phase-calibrate`), then
`agent_mapper/repeat.py <zip> --out <zip> --song <id>` then `answer.py` (manual chain steps) — or the
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

### 📦 Builder-era state + levers — moved to PROGRESS.md 2026-10-05c (charter: autobuild is a sketch, not the path)

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

⬜**The judge fails a HUMAN for being too precise** (09-17): 1fb71's own human map FAILs p 0.083 on
`offset_mad_ms` (5.1 ms, 0.7th pct) + `onset_precision` 0.991 (96th) — a song with very clean onsets
makes every good map atypical. Two-sided tails were a measured choice (see `mapjudge.CANDIDATES`
comment), so not changed; an alignment typicality read relative to the SAME song's human would fix it.

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

## 🟡 P6 — STYLE REQUESTS: "make it more X" as a lever table + presets
`docs/style_levers.md` — one row per request (*faster · harder · more diagonals · more doubles ·
one hand leads · follow the piano · breathe before the drop · more walls*) with the lever, its safe
range, and the score column that shows it moved; `mapctl auto --style {flow,tech,dance}`;
`verdict.py --style` judges against the preset where P0.1 uses the nps request.
**DoD**: three presets build clean on the songset with the named column moved. ⚠️Levers stay
monotone and default-off — they ship in a UI (`feedback-levers-are-user-facing`).

---

## 🔵 CARRIED FORWARD — still live, lower than P0–P6

### P1.0 — `1f9a0` fails `onset_precision` 0.48: a PHASE outlier first, selection second (09-16g)
Its best global shift is a constant −70 ms (0.556 → 0.758); raw phase (`--no-phase-calibrate`) is
right for it (0.654, residual = the −30/−40 ms detector bias) but wrong for the cohort (1f333
0.940 → 0.869, 1fa32 0.881 → 0.772). The +0.053-beat calibration is a cohort constant fit on
phase-0-by-convention songs.
⬜**Per-song phase**: an estimator that keeps the raw fit when it is trustworthy. 🔴A `grid_r` gate
is NOT SUPPORTED (09-16i: raw vs calibrated is 9 wins / 11 losses, uncorrelated with `grid_r`);
the next candidate needs a signal that is not the grid-fit strength. **DoD**: over the 23 songs, applied precision ≥ calibrated on
≥ 21 and 1f9a0 ≥ 0.65, the best-shift residual within ±20 ms of the human's on every song.
✅**`--phase-outlier` built 09-17a** (OFF): fires on 1f9a0 only of 23 (precision 0.556 → 0.698, human
notes covered 0.303 → 0.823), 22 byte-identical; OFF because a corpus proxy says it would mis-fire on
~4 % of songs. ⬜**DoD to flip**: a second signal separating true from false fires (≥ 3 true, 0 false).
⬜Selection: at its right phase 1f9a0 is ~0.70 against a 0.822 floor.
✅**The judge now sees phase** (09-16j): the game applies `_songTimeOffset` (IL-traced, 1.45.0 and
modded 1.40.8), and `mapjudge`/`scorecard` now do too. No bake needed.

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
