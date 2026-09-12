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
items grouped by tool)**. ⚠️Roughly a third of what is left is the permanent **REFERENCE** tail —
landmines that each cost a session. Curate the WORK half; leave those.

📖**A read of any map is one command:** `python scripts/verdict.py <map.zip>`. Before believing a
clean one, read `START THE NEXT SESSION HERE` below.

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
▶️**2026-09-10: read `PROGRESS.md 2026-09-10` FIRST — the bench's whole clean side was scored AGAINST ITSELF.**
Every CLEAN row is a human map with `--vs auto`, which resolves to the same zip, so a query shaped *"ours differs
from his"* — which is every query in `queries.py` — was zero there **by construction**. The rows that can actually
fail are the new `humanplus-*`: the same mapper's ExpertPlus of the same song read against his Expert
(`scripts/make_bench_fixtures.py`). ★**Run any reference-relative query on those before believing its clean side.**
It refuted the first `q_scatter` inside an hour, and it shows `q_events` firing D6 on a top human's harder map.
Together with 2026-09-03b: **a clean page is evidence about the queries; a clean bench row is evidence only if the
row could have failed.**

**P1 ✅ · P1b ✅ · P0 ✅ · P2 ✅ · P2b ✅ · P3 ✅ · P4 ✅ · P4b ✅ · P5b ✅ · P5c ✅ — all DoDs MET.**
🔴🔴**ALL FOUR staged pairs now fail their own gate**: 1f333 + 1f913 on SCATTER, 1f8d6 on ABSENCE, and 1f767 on
ELEMENTS once walls were read by coverage instead of by presence. **They stay staged, blinded and with their notes
UNCHANGED** — decided-and-logged — and the per-song prediction is in `.key.json`, written *before* he plays, one of
them now marked `superseded` rather than rewritten.
★★**2026-09-12 — the ELEMENTS red on three of those four was ONE BUG in the builder, now fixed**: walls were drawn
from the corpus's **pooled** duration marginal, so we emitted only its middle mode and covered a constant 23.8-28.9
beats on every song against humans at 12.1-203.7. Corridors are now placed by the song (onsets thinning out) and
**`W__1f767` is the first fully clean page this project has produced** (`outputs/walls_2026-09-12/`, the staged
zips untouched). ⇒**SCATTER is now the last red on the songset** — 1f333 and 1f913, both map-wide, both "nothing
comes back to lock into". 🔴And the new bound: **wall coverage has R² = 0.089 against the song** (600 maps), so it
is mapper style, not song content — see the item under P5b before touching that threshold.
❓**THE ONE DECISION THAT IS KYLE'S**: P4b's rule says a red map is not staged, because *"losing with a known red
teaches nothing the page did not say"*. That rule was written when a red meant a cheap fix. Today's reds come from
codes invented **after** these maps were built, with a written prediction riding on each — so playing them tests
the codes, and rebuilding first tests the builder. **Both are defensible; the pairs are staged either way.** Say
which and it takes one command. → **P5 is still the only thing that moves the headline: Kyle plays ONE pair**
(`compete.py verdict <sid> X|Y|tie --because …`) → then **P6**. ✅P5c closed the same day: the gate no longer calls
a human's own ExpertPlus over-dense (the map DECLARES its difficulty and `q_events` reads it), which also surfaced
that **1f913's only human map is an ExpertPlus while every build we make is an Expert** — that pair, and every
density read on that song, has been cross-difficulty all along.
🔴**DECIDE-AND-LOG.** Nothing below may block on Kyle.

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

## ✅→🔵 P5b + P5c — THE ABSENCE HUNT AND THE DIFFICULTY CONTROL, DoD MET 2026-09-10
`PROGRESS.md 2026-09-10 · b · c · d · e`. Shipped: **`q_scatter`/SCATTER** · **ABSENCE on the
verdict page** · the **`humanplus-*` / `humanexp-*` controls** + `make_bench_fixtures.py` ·
`score_row`'s `allows` (used by no row, which is the point) · **`MapData.difficulty`** through to
`queries.cross_difficulty` · the TUTOR line uncoloured.

**Still open:**
- 🔴**We can make NO density claim about `1f913`** — its only human map is an ExpertPlus, every
  build we make declares Expert, so its page prints ⚪ EMPTY · ⚪ D1 · ⚪ D4. Its **blind pair
  therefore asks Kyle to compare two difficulties**; it stays staged (`.key.json` carries the
  caveat) but a loss there is not purely a quality loss. ⬜Fix by building 1f913 at his
  difficulty, or by finding an Expert human map of it.
- ⬜**NEXT 2026-09-12b — SCATTER is the last red on the songset, and it now has a candidate locator.**
  At **n=400** human Experts, block echo tracks the **song's own** block echo (median r **0.502**, 83 % of maps
  > 0.3) and the human distribution is **tight**: median 0.602, p10 0.505, **sd 0.075**. Our four maps sit at
  **0.388–0.423** — below the human p10 on every song. ⚠️Unlike wall coverage this is an axis where humans agree
  with each other, so the level gap is a defect, not a style difference. ⇒**Build a per-block SCATTER that flags
  only the blocks where the SONG came back and the map did not** — a different question from the naive per-block
  version that was rejected on 2026-09-10 (that one asked every block). 🔴**DoD before it may ship**: silent on
  `humanplus-1f333` / `humanplus-1f8d6` / `humanexp-*`, and firing on 1f333 + 1f913 where the map-wide read
  already does. 🔴**BUILT AND REFUTED THE SAME SESSION** — every threshold that fires on 1f333 + 1f913 also
  fires on `humanplus-1f8d6`, and the song-echo conditioning was *anti*-enriched on two of our four maps while
  **concentrating the control's false fires in the half meant to carry the signal**. ⛔Do not retry it.
  **The map-wide read is the only SCATTER there is.** ✅By-product: a mapper's own two difficulties differ in
  map-wide echo by **0.064–0.084**, so `q_scatter`'s 0.15 margin sits ~2× above the difficulty effect — the
  first real validation of that threshold.
  ★★**AND THE METHOD NOTE THAT COST TWO REVERSALS IN ONE DAY: stop drawing mechanism from the songset.** The
  same four rows said "instants sit at high onset density" (n=3, refuted at n=484) and "song repetition does not
  drive human echo" (n=4, reversed at n=400). Four rows can refute a norm; they cannot establish one.
  ★★**AND: A CORRELATION IS NOT A LOCATOR.** r = 0.502 between human block echo and song echo, on 400 maps,
  is real and still decides no individual block. Second instance of `AUC is not an operating point` (REFERENCE).
- 🔴**THE SCATTER FIX IS BUILT AND BLOCKED ON RESET REPAIR** (`agent_mapper/repeat.py`, 2026-09-12c;
  `PROGRESS.md 2026-09-12c`). When the song returns to a section, the hands return to the shape they played on
  it — the song's **own section labels** decide, no threshold (a per-block fingerprint was tried and maxes out
  at 0.47 *between two passes of the same section*, so any threshold under that is choosing how many blocks
  fire). ✅**The mechanism works**: `--allow-resets` clears SCATTER on **all four** maps and makes 1f333 + 1f913
  read SHIP? YES. 🔴**And it costs resets: 0 → 15–24** against humans at 0–4. Violations stay 0, so nothing is
  unplayable, but flow is not tradeable for echo. **DoD NOT MET.**
  ✅**THE RESET BLOCKER IS SOLVED** (2026-09-12d). ★★**A reset is a parity PHASE problem, not a bad note** —
  parity alternates, so flipping one note inverts every swing after it, which is why single-note repair failed
  both ways and is what the landmine *"flipping the second note cascades"* has been saying since August. The
  repair inverts a **run**. On 1f913's 14 blocks, all at zero reset cost: one-note **0**, whole-hand 6,
  suffix 9, **interval 14 of 14 in 5 s**. ⇒Every planned block now applies with **resets and violations at 0**.
  ✅**1f767 and 1f8d6 now echo at or above their own human** (0.463 vs his 0.418; 0.527 vs 0.574) and SCATTER
  is clear on both.
  🔴**1f333 (gap 0.226) and 1f913 (0.164, misses by 0.014) still fail, and the cause is exact**: the repair
  costs echo because the cut DIRECTION is part of the figure `_echo` counts. Unrepaired the copies gain ~0.15;
  repairing gives ~0.10 back. 🔴**The metric was NOT changed** — dropping direction from `figures()` clears
  both maps instantly and is the forbidden move.
  ✅**2026-09-12e — SCATTER CLEARS ON 3 OF 4, at 0 resets and 0 violations.** The source is now *chosen*: each
  block is offered every earlier occurrence of its section and keeps the one that survives with the most of its
  figure intact (fewest notes flipped by the repair). Neither "always first" nor "always most recent" dominates,
  so neither is hard-coded. **`R__1f913` reads `SHIP? YES — nothing located`: zero reds AND zero yellows, the
  cleanest page this project has produced.** 1f767 ships with one yellow; 1f8d6's reds are ELEMENTS + lead-hand.
  🔴**1f333 alone still fails** (gap 0.230) — and the reason is structural, see below.
  ★★**AND THE READ THAT NAMES THE REST OF THE WORK.** Split every block by whether its section had an earlier
  occurrence: **with** a source we now score 0.543–0.656 against humans at 0.435–0.766 (we *beat* him on 1f767
  and 1f8d6); **without** one we score 0.231–0.446 against **his 0.359–0.674**. ⇒**The human is not repeating
  the song's structure, he is repeating his own VOCABULARY** — a small set of shapes reused throughout, over
  music he has never played before. That is the entire remaining gap, and it is why 1f333 is the last red:
  **64 % of its bars are a first occurrence.**
  ✅**2026-09-12f — the vocabulary is PRICED, and it needs a PALETTE, not a knob** (`PROGRESS.md 2026-09-12f`).
  CONFIRMED n=400: distinct shapes per hand human **28.5** (p10 18, p90 43) vs **ours 34–60**; top-10 share
  human **0.82** (p10 0.70) vs **ours 0.53–0.65**; correlation with echo **r = +0.654**. ★The songset ranks by
  it exactly — 1f767 is the one map whose vocabulary is tighter than its human's and the one whose echo *beats*
  his. ⚠️Not the refuted read: vocabulary breadth was rejected as a **defect query**; this is the same quantity
  as a **cohort-level builder target**.
  🔴**Three routes measured, none works**: post-hoc **snapping** is REFUTED (human concentration costs 26 parity
  violations + 179 resets); **`REPEAT_P`/`REPEAT_WINDOW`** NOT PROVEN (six settings span 0.473–0.504 unordered,
  vocabulary unmoved — and its window is 6 NOTES while SCATTER is 4 BARS); **`VOCAB_DEPTH`** NOT RESOLVABLE
  (depth 200's own seed spread 0.043 ≥ the whole between-depth range at n=2).
  ★★**The structural reason**: at every depth 200→2000 the per-map vocabulary stayed 38–55 per hand. **`idiomize`
  samples a fresh idiom PER NOTE, so the vocabulary tracks the NOTE COUNT, not the pool depth.** A human commits
  to a **palette** and places from it all map.
  ✅**2026-09-12g — BUILT, and it is a real lever**: `idiomize --palette N` commits the map to N landing shapes
  per hand (two passes; the first says what this song reaches, the top N become the palette, the second replays
  inside it, falling back whenever no palette move is legal). **Default 0 = off.**
  **1f333 at 3 seeds**: echo **0.490 ± 0.022 → 0.543 ± 0.016** at palette 20 (**+0.053, ~3 sd**), vocabulary
  **46 → 33 per hand** (human 18–43, median 28.5), top-10 share **62–65 % → 71–76 %** (human p10 0.70), with
  **0 violations, 0 resets, 0 fallbacks**. The first thing all session to move the vocabulary at all.
  ★Asking for 20 *realises* ~33 shapes — the request is not the outcome, and 28 scores worse than 20.
  🔴**1f333 still does not clear** (gap 0.167 vs 0.150; 1 of 3 seeds). His echo is **0.710, above the corpus
  p90 of 0.694** — an outlier repeater. **DoD NOT MET.** No regression elsewhere; bench not refuted.
  ⬜**TO FLIP THE DEFAULT ON**: sweep `palette ∈ {16, 20, 24}` over **≥10 corpus songs at ≥3 seeds**, confirm
  the gain holds and that `idiom_local` does not drop below the human floor (the *"more human than human"* tell
  this file already records twice), then default 20 in `autobuild` and re-run the songset end to end.
- 🔴🔴**FIXED 2026-09-12g — `idiomize --travel-target` WAS A DEAD CLI FLAG**: parsed, documented in `--help`,
  never passed to `idiomize_zip`. **Any sweep of it before today swept nothing.** ★**Third time this exact bug
  has shipped in that one file** (`width` 2026-08-21, `travel_target` inside `idiomize_zip`, now the CLI).
  ⇒The standing rule gains a line: **a knob whose arms are identical to 3 decimals is UNWIRED, not weak — and
  check the CLI as well as the function.**
  ⚠️`repeat.py` is still **NOT wired into `autobuild`** — wire it the session 1f333 clears too, after
  re-running the songset end to end.
- 🔴**NEW 2026-09-12 — wall coverage is barely a function of the song, and that bounds `q_elements`.**
  Regressing log10 total wall beats on song duration, onset rate, note count and note density over **600
  human maps** gives **R² = 0.089** (residual sd 0.515 against a total sd of 0.539 — a ~3.5× swing either
  side). ⇒Coverage is mostly a **mapper's style choice**. Two things follow. (a) ⛔**Do not chase the
  remaining per-song spread** — ours is still ~4× narrower than the humans' (log10 sd 0.126 vs 0.548) and
  the correlation only moved 0.152 → 0.230, but there is almost nothing there to predict. (b) ⚠️**the 0.5×
  coverage red asks a map to match an unpredictable quantity.** The *systematic* version (under-walled on
  every song) was real and is fixed; a **per-song** red at 2× on an axis with a 3.5× human spread is weakly
  grounded. 🔴**The threshold was deliberately NOT loosened** — loosening a gate to pass our own map is the
  one move that must never be made. **DoD for reopening it**: a human-vs-human negative — two different
  mappers' Expert maps of the same song — showing how often 0.5× fires between two people who both did it
  right. Until that exists the red stands as written. Centring the level also bought the opposite failure:
  **over-walled (>2× his) on 24 % of maps, up from 10 %**.
- 🔴**The doubles gap has no locator.** `LOOP__1f913` plays **4.2 % doubles against his 32.9 %**
  and every read says ✅ (ABSENCE reds only at *effectively zero*; D6 fires only the other way).
  ⛔The entry-accent locator is REFUTED (see the refuted list above). ⚠️Whatever comes next must
  not be "distance to the human's number" — that is the `h_dist` failure.
- ✅**The untouched third is done** (2026-09-10g). **Sustained/held sections: NOT REPRODUCED** — our rate inside
  the top sustain quartile, relative to our own overall rate, is 0.94-1.03× against the humans' 0.86-1.03×.
  **Walls: a real absence, now queried** — our builds cover **131-146 slots on every song** (the `--walls 89`
  default) where humans cover 83-667 and vary with the music; `q_elements` grew a graded branch (< 0.5× his
  coverage) and **1f767 went red, so all four staged maps now fail their own gate**.
  ✅**FIXED 2026-09-12 — it was one bug, and it was the DURATION, not the count** (`PROGRESS.md 2026-09-12`).
  `plan_walls` drew every wall from the corpus's **pooled** duration marginal, which has one mode and can emit
  neither the human's instant marker nor his corridor, so we sat in the rarest 8.5 % cohort on every song.
  It now emits all three modes and places **corridors in the note-free outer-lane gaps where the onsets thin
  out** — the one song-side signal that survived 567 corpus maps (instants-at-high-density did NOT: 0.98×,
  n=484). Control on 300 human maps re-walled from their own notes at the production budget: median ratio to
  that human **0.40× → 0.98×**, ELEMENTS would fire on **59 % → 21 %**. **1f767 is now a fully clean page**;
  1f8d6 stays red at exactly 0.50× against a p90-waller human. Zero notes inside walls; bench not refuted.
- ⬜`audit_map.py`'s ABSENCE reference (250 corpus maps) does not separate difficulties; the
  per-song human is preferred on the page, so this bites only on a song with no human map.


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

### Doc debt from the audit
`CLAUDE.md` V6-era, no `agent_mapper/`; `buildmap/SKILL.md` contradicts itself on doubles.
✅`/todo` Step 4 rewritten 2026-09-10 around the agent path — and the note it was checking, the
**late-song collapse, is NOT REPRODUCED**: all four current builds hold their event ratio to the
end (0.85-1.24 across every quintile), and the sag is visible only on the August maps, on top of a
map-wide deficit of 0.65-0.77 that is EMPTY/D1's job. `PROGRESS.md 2026-09-10d` has the table.

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
