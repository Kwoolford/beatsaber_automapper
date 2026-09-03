# Audit 2026-09-02 — does `/buildmap` + its tooling achieve the goal?

**The goal, in Kyle's words (2026-08-24):** *"complete when you call the skill and are confident in
the map you build and feel you can evaluate the map correctly and don't need to rely on me to
audit."* And 2026-09-02: *"the eval suite still to this day requires my approval and oversight on
when maps are good or bad… it's tedious and the errors are pretty obvious from my perspective."*

**Verdict: the BUILD half is in good shape; the JUDGE half is structurally unable to do what he
asks, and the process around it keeps him in the loop by design.** Details and the fix plan
below; the plan itself is `TODO.md` P0–P6 (P1, the score, is the centre — see F6).

---

## 1. What is working — keep it

| piece | state | evidence |
|---|---|---|
| Perception (`events.py`, `structure`, `percussion`, `melody`, `lyrics`, `brief`, `notesheet`) | ✅ mature, each with a control and a self-verdict | `agent_mapper/PROGRESS.md` 2026-08-16; the dogfood build |
| Per-section build (`mapctl init/auto/clear/reuse/export`) | ✅ works cold-cache | `24e6c` DoD run: 599 notes, breathing density, 0 violations |
| Reading tools (`map_view --align --elements --stems --sections --vs`) | ✅ the agent can see what a player perceives | PROGRESS 2026-08-24f |
| `audit_map.py` one-screen assembly | ✅ | every channel, with caveats printed where they are read |
| Levers (`--follow --pulse --lead-bias --target-notes --doubles --width --travel-target --walls/arcs/chains`) | ✅ each measured, clean ranges | matches [[feedback-levers-are-user-facing]] |
| `READING.md` discipline (page proposes, cohort disposes; look for ABSENCE) | ✅ | it caught the doubles defect the audit missed |

## 2. Why Kyle is still the gate — five findings

### F1. The judge answers "is this TYPICAL?"; Kyle asks "is this WRONG?"
Every scoring channel is a percentile against a human corpus and its own docs say what that
means: *"a PASS = NOT DEFECTIVE, not GOOD; a FAIL can mean NOT TYPICAL, not bad."* A typicality
test cannot produce the verdict he gives, which is **a named defect at a place**. Worse, under
his standing *"my target is the best mappers"*, the corpus median is a floor — so four of his
seven named defects were closed with *"humans do it too"* (D2 refuted as placement, D3's fix
refuted, D5 *"the human bursts more"*, pendulum-lock *"same rate as humans"*), which
[[feedback-target-is-best-mappers]] says is not a valid no-defect verdict for an aspirational axis.
**The suite is calibrated to decline to find exactly the errors he calls obvious.**

### F2. His verdicts are consumed as decisions, never kept as labels
`docs/eval_references/preference_verdicts.json` holds **2 preference verdicts and 7 defect codes**
after five months of listening sessions. Every review batch became a TODO item, not a labelled
example. No detector in the repo has ever been tested on *"does it fire on the map he called bad
and stay quiet on the one he called A+?"* — `preference_screen.py` tried once, at n=1, and
stopped. ⇒ Nothing can currently claim *"I would have caught that"*, so nothing can replace him.

**Labelled maps that already exist on disk** (this is the bench, nobody has used it):

| map | Kyle's label | pair / contrast |
|---|---|---|
| `outputs/kyle_review_2026-08-03/1f333_AFTER2_reach.zip` | **A+**, "promote it" | vs `1f8d6_AFTER2_reach.zip` (cross-song) |
| `outputs/kyle_review_2026-08-03/1f8d6_AFTER2_reach.zip` | **"feels really empty"** | vs 1f333 A+ |
| `outputs/reviewed/C_agent_built/Hunger_AGENT.zip` | **"notes flow in a really odd way"** | vs `for_review/A_structure_crossover/Hunger_BEFORE.zip` (same song, BEFORE preferred) |
| all 32 maps in `for_review/A_*` and `B_*` | D1–D6 *"for all of the songs"* (weak, set-level) | human maps of the same songs (assumed clean) |
| protected positives (his words) | hand-role division · breathing pacing · *"notes on beat that play part of the song"* | must NOT be flagged |

Same-song pairs are the strong labels: **1f333 has three** — A+ (ML path), AGENT (odd flow),
BEFORE (preferred over AGENT).

### F3. The "needs Kyle's ear" queue mixes decisions he has effectively already made with taste A/Bs
Seven blockers on the board: `[V2]/[FULL]`, `[W3]/[W5]`, `[PBASE]/[PCAL]`, `[DOD]`,
`[CROSSOVER]`, P0.1, P0.2. Applying his stated preferences:
- **P0.2** (alignment floor): he named *"slightly off beat"* as a defect and his target is the best
  mappers ⇒ rejecting 4 % more median-ish human maps to catch 92 % of off-beat maps is his stated
  trade. Decidable.
- **P0.1** (deliberate difficulty): *"map whatever difficulty we want"* ⇒ judge density against the
  request when one is made, corpus otherwise — which is what the item's own DoD says. Decidable.
- **`[PCAL]`**: held-out validated, moves the D2 axis on every song, corpus-independent. Decidable
  (ON), reversible by his ear later.
- **`[W3]/[W5]`, `[CROSSOVER]`, `[V2]/[FULL]`**: these are *taste* A/Bs. Per
  [[feedback-levers-are-user-facing]] they are knobs to ship, not gates to hold — pick a default,
  keep both reachable.
Holding these for weeks is [[feedback-never-block-kyle-overnight]] in slow motion.

### F4. The agent never gives a VERDICT — it gives a report and hands over
`audit_map.py` ends with *"WHAT THIS CANNOT TELL YOU — read the map"* and four prompts. There is
no place where the agent writes *"D5 at bars 44–47, D3 at 1:31, otherwise clean"* and no score of
how often that reproduces Kyle's answer. The DoD run concluded *"everything except is it fun"* —
but he says the errors are **obvious**, i.e. not taste. The missing artefact is a structured verdict
in **his** vocabulary, benchmarked on F2's labelled maps.

### F5. Per-defect tooling maturity is uneven, and none of it is validated on his labels

| his defect | locator that exists | validated on a Kyle-labelled map? | gap |
|---|---|---|---|
| D2 slightly off beat | `map_view --align` ✗-runs; `--phase-calibrate` | ✗ | threshold set from corpus median, not best mappers; PCAL off |
| D3 drops at the wrong time | none (`structure.py` refuted as a *locator*; "climax is taste") | ✗ | ★conflated with climax placement. A drop is an **audio energy step**; the question is whether note density steps *with* it (lag), not *where* the peak is |
| D4 not following main vocals / D6 nps wasted on non-main | `overlay.py` HIT/MISSED/WASTED | ✗ | never scored at the ENTRY of an instrument (K5 refinement); not in `audit_map` |
| D5 random fast non-flowy bursts | `flowview.py` (motivation, harsh, resets) | ✗ | built, never run on set A vs human |
| D1 very slow | `nps` percentile | ✗ | wrong reference — should be the request, or the same song's human |
| FLOW "odd way" (Hunger AGENT) | `effort.py`, `swing_sim`, legibility/recurrence reads | ✗ | one clean same-song label exists and was never used |
| EMPTY (Fallen Kingdom) | `emptiness.py` (refuted note coverage), `wall_duty` 16th pct | ✗ | still open; `[V2]/[FULL]` was the only probe |
| walls/arcs/chains | `elements.py` typicality | — | no playability read beyond trapped notes |

### F6 (Kyle's correction, same day — the finding the plan is built on). The model cannot SEE the map against the song
Kyle: *"The model doesn't have the visibility that I do when evaluating a map… convert the map
to text or code or a numpy array where the rows are possible note placements and the columns are
the notes, matched with another text or number array of the song in note-sheet format with lyrics
and all. This granular visibility with deep timings is what the model does not have. This would
catch the obvious errors more than a metric. This is the eval suite."*

Verified by running the model-facing view on a bench map
(`map_view Hunger_AGENT.zip --audio 1f333.ogg --bars 33-36 --stems --align`):

```
 bar    beat │ L          │ R          │ dr ba ot vo gu pi
  33  128.50 │ 1,0 ↓ F    │            │ ▅ · · · · ▅
  33  129.00 │ 1,2 ↑ B    │ 3,1 ↗ B    │ ▄ · · · · ▅
  33  130.00 │            │            │ · · · · · ·
```

- **Sparse rows** — only slots where something happens; the empty 1/4 and 1/8 slots between are
  not drawn, so "nothing here while the vocal is singing" is invisible.
- **Song side is six loudness blocks.** No kick-vs-snare, no bass/lead pitch, no vocal pitch,
  no lyric syllable, no section role, no energy level, no reference onset. The rich score that
  has all of that — `notesheet.py` (VOX + lyric, LEAD, BASS, KIT tab, section banners) — renders
  **HTML for Kyle's eye only**; nothing emits it as text or arrays for the model.
- **Two bugs**: the `--audio` lanes crash (`'numpy.ndarray' object has no attribute 'float'`),
  and `--stems` keys the event cache on the map's *filename*, so `Hunger_AGENT.zip` finds nothing.
- **No array form** — there is no `song[T,F]` / `map[T,C]` on a shared lattice, so every
  question ("which vocal onsets have no note?") has to be a hand-written metric instead of a
  one-line query.

Every perception cache the score needs already exists
(`outputs/{event,percussion,melody,lyrics,structure,chords,onset}_cache/`). The missing piece is
one join. ⇒ **P1 in `TODO.md` is now `agent_mapper/score.py`** — song and map on one lattice, text
and `.npz` — and its DoD (name Kyle's defects on the bench maps *by reading*, no metric) is the
acceptance test of the whole plan. The locators of F5 become numpy queries over its arrays.

## 3. Smaller defects found while auditing

- `TODO.md` lines ~210–227: the *"📦 AWAITING KYLE'S EAR"* section was overwritten mid-list by a
  pasted copy of the D1–D6 table (the table appears twice, the bullet list is truncated).
- ✅ items (P0.7, P0.4, LEG 3) are still in `TODO.md`, against its own forward-only rule.
- `buildmap/SKILL.md` says *"Still open: `double_share` 0.000"* in §Difficulty while step 3 says
  `--doubles` fixes it — internal contradiction.
- `CLAUDE.md` is the V6 ML-era document: no mention of `agent_mapper/`, `/buildmap`, `audit_map`,
  or the verdict ledger. An agent reading only CLAUDE.md builds the wrong thing.
- `/todo` skill Step 4 still checks the ML-era key notes (13–15 s silent drop, `eval_density_corr`).
  It is not the driver for agent-path work; `/buildmap` + the new plan are.
- `for_review/` holds 32 maps he was asked to play, 20 of which are now superseded
  (`LISTENING.md` cleanup never got its OK — another decision parked on him).

## 4. What "solving it" means — the DoD for the whole plan

> A map built with `/buildmap` gets a verdict from `scripts/verdict.py` in Kyle's defect
> vocabulary, with timestamps. That verdict **reproduces his recorded verdicts on the labelled
> bench** (flags every map he called defective, on the defect he named; stays quiet on the maps
> and properties he praised). A map ships when the verdict is clean. Kyle's role becomes
> spot-checking, and every disagreement he reports is appended to the bench as a new label.

The plan is `TODO.md` **P1 (the score — song and map on one lattice, text + arrays; F6)** with P0
(decide the parked decisions) alongside → P2 (read the bench with the score) → P3 (queries over
the arrays, one per named defect) → P4 (the verdict + the skill gate) → P5 (the label channel) →
P6 (style requests). No GPU. Everything is CPU and can run tonight.
