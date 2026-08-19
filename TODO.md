# Beat Saber Automapper — what we are working on next

**This file is forward-looking only.** What was done, and how it worked out, lives in
[`PROGRESS.md`](PROGRESS.md); the agent-authoring trail is in
[`agent_mapper/PROGRESS.md`](agent_mapper/PROGRESS.md). Evaluation-suite rationale is in
[`docs/eval_suite_v2.md`](docs/eval_suite_v2.md).

**Rule:** when an item finishes, its *outcome and what it taught* moves to PROGRESS.md and the
item is **deleted** from here. A completed item is history, not work. Curated 2026-08-02 (from
4,076 lines) and again 2026-08-14 (from 652).

---

## 📍 CURRENT STATE (2026-08-18)

★★**Kyle has judged the maps and the answer is a DEFECT LIST, not a preference.** He
reviewed the agent map, set A and set B and reported, across *every* song:

> *"it varys from very slow, slightly off beat, doing drops at the wrong time, not
> following the main vocals or having random bursts of really fast non flowy notes…
> **I think the nps is generally wasted on every few non main notes**… there is a good
> deal of notes that are on beat and I can tell play part of the song, but **they aren't
> hitting that main flow that mappers can generally see**."*

And the directive that follows from it:

> *"I think we need to make the visibility suite the top priority… so visibly great that
> you can go back through and evaluate through me instead of making another evaluation
> metric."*

🔴**Two things with strong numeric evidence have now failed to reach his ear**
(`BEAT_GRID_PHASE`, 74 better/0 worse — he still hears *"slightly off beat"*; and the
agent map at human `ebpm_burst`, human nps median, zero parity violations). ⇒**A passed DoD is evidence about the METRIC, not
about the map.** The answer is not a seventh axis — it is to make the map legible enough
that *he* is the evaluator. That is what P0 now is.

⚠️**And the review framing was wrong.** The A/B pipeline collects *preferences between
arms*; he produces *defects located in songs*. Preference is second, defects are first.

---

### ▶️ START THE NEXT SESSION HERE
✅**V1 (notesheet) + V2 (overlay) BUILT, and Kyle ENDORSED V1's read of the song** — *"It looks
correct. Playing instruments until the main words come in."* See PROGRESS.md for all numbers.

1. 🔴🔴**FIRST: THE PLAYER HAS NEVER RUN IN A BROWSER.** The page now embeds the song and draws a
   moving playhead, but there is no browser on this box and the republish was declined, so the
   transport / playhead / click-to-seek / space-to-play are **unverified code**. Open
   `outputs/notesheets/fallen_kingdom_overlay.html` locally, or republish, **before building
   anything else on top of it**.
2. 🔴**THE PUBLISHED PAGES ARE STALE** — both live artifacts carry the OLD lyrics and no audio.
   Republish (needs Kyle's approval): overlay
   `claude.ai/code/artifact/34bf3922-080b-4c9b-bcd6-90792bb1a6b9`, score
   `.../f47350fb-9592-42db-a992-8c9e5b85b015`.
3. ★**THE GATE IS STILL KYLE'S CORRECTION OF "MAIN"** — nothing substitutes for it. **Ask which
   events he would have called main**, then change `overlay.MAIN_DEFAULT`, not the map.
4. ✅**DONE 2026-08-18b** — `scripts/lyric_ablation.py`. VAD confirmed at both model sizes; the
   model upgrade is not a coverage lever; and the two transcripts **disagree on ~1 line in 4 with
   errors on both sides**, which sung-coverage cannot see. ⇒**The "bigger ASR" route is closed;
   forced alignment against supplied lyrics is the only route to word accuracy.** See PROGRESS.
   ⬜**Ask Kyle**: are roughly-right landmark words enough? If not, he can paste the real lyrics
   for the standing songs and we get a true WER for free.
5. ✅**V3 BUILT 2026-08-18c** (`agent_mapper/flowview.py` + the FLOW lane). Bursts are now
   located (`bar N · m:ss · motivation · travel`). 🔴**And D5's "random bursts" reading is NOT
   REPRODUCED — it is backwards**: the human bursts MORE than us on 3 of 4 songs and more
   *unmotivated* on all 4. **Do not build a burst-suppressor.** ★What is consistent is
   **doubling (36-40 % of our swings vs a human 7-26 %)** — that is what makes our bursts run at
   8-12 nps against their 5-7. See PROGRESS.
6. `python scripts/review.py list` — 32 maps staged; **`[CROSSOVER]` is unjudged and is the
   strongest candidate on the board.**

⚠️**Nothing is running.** No GPU job, no autonomous loop — `/todo` must be re-run to restart
research.

---

## 🔴🔴🔴 P0 — THE VISIBILITY SUITE ★KYLE'S PRIORITY, 2026-08-17

**The goal, in his words: "so visibly great that you can evaluate through me."** Every
item below is judged on whether it lets him point at a moment and name what is wrong —
not on whether it produces a number.

**What already exists to build on**: `melody.py` (pitch per onset), `percussion.py`
(kit labels), `structure.py` (sections + which repeat, CONFIRMED held-out p=0.019),
`brief.py` (per-bar stem grid + lyrics), `map_view.py` (map as text), ArcViewer (play
preview). ⚠️**None of it is time-aligned into one picture**, which is the whole gap.

✅**V1 + V2 SHIPPED 2026-08-18** — `agent_mapper/notesheet.py` (the score) and
`agent_mapper/overlay.py` (HIT/MISSED/WASTED), both published as pages. **Their DoD is not
met until he looks**, which is item 1 above. Still open inside them:
- ✅**DONE 2026-08-18e — `agent_mapper/chords.py` + `notesheet --chords`.** Polyphony is real and
  everywhere (median 2.0; **56-70 % of every song has 2+ notes sounding**), basic-pitch runs in
  **1.6-2.4 s a song** on ONNX, and it is **adopted per song by an in-key gate: 2 of 4 songs yes,
  Hunger and Digital Life Hacker no**. ⚠️Hunger's refusal is ambiguous (metal ⇒ the diatonic proxy
  cannot separate "wrong" from "chromatic"); the gate only ever *refuses*, which is the safe side.
- ⬜Only `1f8d6` is published. The other three standing songs render but are not up.
- 🔴**Word ACCURACY is unaddressed.** The VAD fix made the words *present* (sung-coverage
  0.927→0.967 on 1f8d6); Kyle's other complaint is that they are *wrong*, and coverage cannot
  see that. The ground-truth-free proxy (same section letter should transcribe alike) gave
  0.187 vs 0.198 over 3 pairs = **not resolvable**. ⇒If exact words matter, the route is
  **forced alignment against supplied lyrics**, not a bigger ASR model. **Ask him whether
  roughly-right landmark words are enough** before building that.

### V3 ✅BUILT — the FLOW view: `agent_mapper/flowview.py`
Hand paths, crossover marks and located, shaded bursts. **DoD met on the located half** (every
burst prints its bar and timestamp); **unmet on the half that needs his eye** — no browser here,
so the lane is verified only by element count. ⚠️Open inside it:
- 🔴**`harsh` is a DEAD metric** (0.00 in every burst of every map, ours and human) ⇒ *"non
  flowy" is not wrist rotation*. `travel` (ours 6.4-6.5 vs human 3.3-5.3 cells/s) is the live
  half. Do not report `angle_harsh_frac` as if it distinguished anything.
- ⬜The RANDOM rule uses **event density**. His "random" may mean *the wrong events* instead —
  which is D6, and needs his definition of main. **Ask him at a shaded bar: "is this one of the
  random bursts?"** That question is now askable, which it was not yesterday.

### V4 — One page per song
V1+V2+V3 on a single scrollable page, published so he can open it on any device and
scrub to a timestamp. This is the surface that replaces metric invention.
**DoD**: he reviews a map from the page + ArcViewer and never needs a scorecard.

### V5 ✅BUILT — defect capture (`review.py defect` / `review.py defects`)
Records his words first, then prints what we believed at that instant (bar, section, nearest
drop, local HIT/WASTED/MISSED, the burst there). **The mechanism is done; the ledger holds
zero located defects because that half is his.** ⇒★**Next time he plays anything, capture with
`review.py defect` rather than prose** — and the six standing complaints (2026-08-17) are still
unlocated, which `review.py defects` now prints as a backlog.

---

## 🟢 P0.6 — ★A POSITIVE LEVER AWAITING VALIDATION: `--beat-threshold 0.25` (2026-08-18r)
On production v4, threshold 0.40 → **0.25** lifts **vocal coverage 0.420 → 0.454** (paired
+0.0158, **20/23 songs, p=8e-05**) via **+88 notes at ZERO precision cost**, with **0/23 songs
above the 6.18 nps Kyle called unplayable** (median 4.06) and no `map_metrics` axis degrading.
✅**FINAL AT 3 SEEDS (2026-08-18u): +0.029 vocal coverage at 8.6× sd, for +0.122 playfeel at
4.2× sd (still inside its 1.00 bar). EVERYTHING ELSE — including alignment — IS SEED NOISE.**
🔴Both my single-seed claims (first "free", then "costs alignment") were wrong in opposite
directions. ⬜**Next: install for his ear.**
Earlier reads 2026-08-18t: coverage **+0.031 = 13× the seed sd** (prod 0.423±0.0024 →
0.454±0.0008); **rhythm better 3.9× sd**; **playfeel worse 3.9× sd**; alignment worse 1.8× sd
(needs seed 3); **flow/idiom/handrole: no effect** — 🔴**my earlier "flow FAIL→PASS" was SEED
NOISE** (prod flow ranges 0.296–0.588 across seeds).
🔴Earlier single-seed read 2026-08-18s — my "zero precision cost" was a coarse
proxy. **flow 0.588 FAIL → 0.408 PASS** and rhythm 0.425 → 0.385 (⇒C3's density-costs-rhythm fear
NOT realised), but **alignment 0.263 PASS → 0.515 FAIL** and handrole/playfeel spreads collapse.
★**0.25 is the operating point** — 0.15 buys no more notes and costs precision. ⏳Seeds 1-2
generating; read the alignment move against its seed spread (already 1.14-1.46) before believing
it. ⬜**Then his ear** — the suite has been wrong about "ready" twice.
Maps: `outputs/trackb/v4t0.25__*.zip`.

## 🔴🔴 P0.5 — THE SIX DEFECTS HE NAMED (2026-08-17)

★**Unifying hypothesis, and the most valuable thing he has said in weeks:**
> **Our nps problem is an ALLOCATION problem, not a budget problem.**

*"The nps is generally wasted on every few non main notes"* + *"not following the main
vocals"* + *"I'd like the general beat parts to be faster and play more main notes"* are
plausibly **one defect**: we do not distinguish the MAIN musical line from incidental
onsets, so the note budget is spent on filler and the map simultaneously feels **slow**
(the notes you want are absent) and **busy** (notes you do not want are present).
🔴🔴**MEASURED 2026-08-18, AND IT SPLITS IN TWO** (V2 overlay, 4 standing songs, ours vs
human under one rule):
- **As "our notes land off the music": NOT SUPPORTED.** Our precision 78–85% vs the human's
  75–90%, **mixed sign across songs** (+0.034, +0.028, −0.038, −0.047). We are not spending
  the budget on non-events more than a human does.
- **As "our notes don't buy new musical events": SUPPORTED 4/4.** Distinct main events
  covered per note: ours **0.464–0.537** vs human **0.565–0.647**; notes per event covered
  ours **1.86–2.16** vs human **1.54–1.77**. ⇒**both halves of a double land on the SAME
  event and both count as HIT.** 🔴This is **C5 (doubles) from the musical side, not a new
  finding** — but it prices C5 in his terms for the first time.
- ★**The dominant gap is MISSED, not WASTED**: we play **32–42%** of the main line where the
  human plays **39–83%**.
⚠️**PARTLY CONFIRMED, and the rule is a guess until he corrects it.**
⚠️⚠️**DO NOT STEER ON THESE NUMBERS — they FAIL the degenerate control.** A metronome on
1f8d6 scores precision 78.7% / recall 48.5% at 1/4, i.e. **better recall than our own map**.
Only `on-nothing` separates it (13.3% vs human 3.8%, ours 6.3%). The overlay is a **picture**;
the tally underneath it is a caption, never an axis.

| # | defect, his words | first read | DoD |
|---|---|---|---|
| **D1** | *"very slow"* | ours ~4.96 nps vs human 8.35 on Hunger — but see the allocation hypothesis before simply raising density | he stops calling it slow **without** a raw nps increase |
| **D2** | *"slightly off beat"* | 🔴🔴**REFUTED as note placement 2026-08-18g** (`eval_beat_phase_agreement.py`): on all four maps he played the bpm is **exactly** the human's, offset 0, and our note times match the human's **better than two humans match each other** (0.87/0.92/0.73–0.82/0.67–0.71 vs human-human **0.676**). Cohort-wide we are **more** on-beat (0.580 vs 0.515) and place **fewer** 16ths (p=0.0006). ⇒**Stop pointing D2 at tempo for these songs.** ⚠️Live but untested: we may be *too* quantised for a song with groove. | he stops calling it off-beat on a song whose tempo we call correct |
| **D3** | *"doing drops at the wrong time"* | ★★**CONFIRMED 2026-08-18f at n=144** (`eval_drop_agreement.py`): our biggest density moves coincide with the human's **0.347** [0.302, 0.392] against a null of **0.140** and a human-human band of **0.49**; **43 of 144 songs agree on nothing**. 🔴Both cheaper causes REFUTED — we lift density at drops exactly as much as the human does (1.09/1.21/1.08/0.95 vs 1.11/1.21/1.09/0.99). | agreement reaches the human-human band (0.49) at n≥100, **and** he stops saying it |
| **D4** | *"not following the main vocals"* | ★★**CONFIRMED 2026-08-18j at n=144, and it is the BIGGEST measured defect** (`eval_vocal_coverage.py`): we play **0.385** of the sung notes, the human **0.743**, lower on **141/144** songs (p=2.6e-25) — against a human-human spread of only **0.132**. ⚠️**Track B is necessary and NOT sufficient**: we reach only **58 %** of the human even where a drum marks the vocal (vs 46 % where none does). | vocal-onset coverage reaches the human band (≈0.74 ± 0.13) **and** he agrees the vocal line is being played |
| **D5** | *"random bursts of really fast non flowy notes"* | 🔴**MEASURED 2026-08-18c and the density reading is BACKWARDS** — the human bursts more, and more unmotivated, than we do on every song. What separates ours is **doubling** (36-40 % vs 7-26 %) and **travel**, not burst frequency; **`harsh` is dead**. | he cannot find a burst he calls random |
| **D6** | ★*"nps wasted on non main notes"* | ⚠️**re-read above** — measured, and it is a *doubles* defect, not an off-music one. Needs the definition of **main** that he endorses | main-line recall up with nps flat, and he agrees |

⚠️**D6 is the one that must not become another invented axis.** Define "main" in V2,
show it to him, and let him correct the definition. A metric he has endorsed by looking
at it is a different object from one built from first principles — building those from
first principles is exactly what produced the anti-correlation.

---

## 📦 AWAITING KYLE'S EAR — sets A and B, now answered globally

`for_review/` holds 32 maps; `python scripts/review.py next` is the shortlist and
`review.py list` the full pending list. **He has now answered both sets at the level of
defects rather than arms**, so:
- ★⚠️**`[CROSSOVER]` HAS NEVER BEEN PLAYED.** He reviewed *"the before and after as well
  as the before vs phase"* — crossover was not among them. It remains the **strongest
  unjudged candidate** (crossover 0.000 → 0.112 against a human 0.183 with 0 of 150 human
  maps at zero; flow 0.37 → **0.23**, i.e. it improves the exact axis he complains about).
  🔴Do not let it get swept into "numeric wins that failed his ear" — it has not been tried.
- ⬜**`[AFTER]` vs `[BEFORE]` (set A) is still unresolved per-arm** — the intentional/lazy
  question was never answered, and `mapctl reuse --vary 0.15` is still a guess.
- ⚠️**`[PHASE]` did not remove *"slightly off beat"*** ⇒ do NOT flip
  `BEAT_GRID_PHASE=search` on this evidence. Either the lever is inaudible or the
  off-beat feel has a different cause (tempo, D2).

**Also still open from 2026-08-04**: is Fallen Kingdom empty vs what our model *used to
do*, or vs what the *song wants*?

---

## 🔵 P1 — AGENT MAP AUTHORING (`agent_mapper/`) — demoted 2026-08-17

Was P0; **visibility is now above it**, because the agent map's verdict was
*"expecting… much better"* + *"the notes flow in a really odd way"* and we cannot fix
flow we cannot see. The perception tools built for it (`melody`/`percussion`/`structure`)
are exactly what the visibility suite is built from, so this is a demotion in priority,
not a retreat.
🔴**The judged map predates all of them** — it was planned off `brief.py`'s 8-bar onset
densities alone and knew no pitch, no sections and no kit. It is the BEFORE for the
perception work, not a test of it.
- Rebuild a map using melody + structure + percussion, on **Fallen Kingdom, not Hunger**
  (Hunger's vocals are genuinely unpitched — pYIN voiced-on-loud 0.19 vs 0.91–0.99).
- 🔴`travel` is a **sequence** property — per-note pitch placement is REFUTED (made it
  worse, 4.77 → 3.56). Do not try another per-note rule.
- Wire `percussion.py` into doubles (crashes/snare accents, not "strong beat + 2 stems").

---

## 🔵 P1 — TEMPO, the largest quantified upstream defect

**Our tempo is right on 70.5 % of songs** vs the mapper's declared bpm (n=149) — the old "30 %
wrong" confirmed on a 6× larger cohort. ✅Half-tempo is **handled** (`BEAT_SUBDIV_AUTO`; the `half`
group is now the *best-performing* group at 0.2× the base alignment failure rate).
🔴**Open: the 2:3 / odd-ratio misreads** — `three_halves` fails alignment at **4.6×** the base rate
and `other` at **5.5×**, together 48 % of the remaining failures from 11 % of the cohort. A
half-tempo error gives a coarse but *aligned* grid; a 2:3 misread puts beats in the **wrong
places** and drifts, so no global shift helps.
⚠️**Every cheap route is closed**: onset-energy balance (2026-07-27, made it worse), onset-gap
density, ACF periodicity, a cross-validated tempogram classifier, and the `fit_tempo` tie-break
(+1 song of 149). ⇒**What remains is a real tempo model**, with free supervision from the declared
bpm in 5,373 corpus maps.

⚠️**2026-08-18h/i — the subdivision ceiling is REAL but SMALL, and the build is NOT recommended.**
`BEAT_SUBDIV = 4` makes our maps the 1/4-beat grid by construction and **we place zero finer notes
on 0 of 144 songs**, while **131 of 144 human maps use them** — but only for a **median 3.9 %** of
their notes, and **96.9 % of a human's notes already fit our grid**. Sized properly (2026-08-18i):
of the 49 songs whose human map misfits, **subdiv 8 rescues 12, of which 8 are the half-tempo
group `BEAT_SUBDIV_AUTO` already handles ⇒ a marginal prize of ~4 songs in 144.** 🔴**Do not build
per-song subdiv-8 selection.** The residue is the tempo model plus **11 songs that fit NO grid
even at the right tempo and 1/16** (BPM changes or unquantised placement).

## 🔴 P0.7 — TRACK B IS ALREADY TRAINED (2026-08-18m) — evaluate, do not build
★★`logs/beat_classifier/version_7` and **`version_8`** were trained with `instr_dim=10`
(`instr_beat_features`: Demucs → basic-pitch on **vocals**/bass/lead, **cached on all 5,320
songs**). **Production `version_4` has only `drum_proj` + `mix_proj`.** ⇒The melodic-blindness
remedy exists on disk and was never evaluated at inference — plausibly because `val_f1_avg_tol`
rates it 0.58-0.60 vs v4's 0.603, **and that is the metric the landmine list says not to trust.**
🔴🔴**SETTLED 2026-08-18p — PARITY, NOT IMPROVEMENT. Do not promote v7/v8 for D4.** At a
near-matched budget (thr 0.12, 736 notes vs prod 752) v8 scores **0.417 vs 0.420, p=0.665** —
a coin flip. Its allocation is better (0.444 vs 0.410) and it doubles less, but **none of it
reaches coverage**. ★★**The sweep found the real lever instead: the human plays 1088 notes to our
750.** ⇒**D4 is BUDGET-dominated** — and D1 *"very slow"* + D4 + his *"play more main notes"* are
**one lever: more notes, spent on the vocal line** (⚠️not more notes everywhere — C3, and he
called 6.18 nps unplayable).
🔴Earlier detail 2026-08-18n: vocal coverage v4prod **0.420** vs
v7 0.362 (p=0.00013) and v8 0.373 (p=0.0019) on 23 songs. ★**But the features are not inert** —
v7 lifts *allocation* (+0.020, 18/23) and v8 lifts *efficiency* and cuts doubles (+0.018, 17/23);
both lose it to a **9-12 % smaller note budget**. ⬜**Budget-matched arms running** (v8 at
`--beat-threshold` 0.34/0.30). ⚠️**Not a clean ablation** — v7/v8 also carry `struct_proj`; a true
instr-only ablation is a v4-recipe retrain.

## 🔵 P1 — TRACK B: three defects are one defect

**W1** (can't find the tempo-carrying instrument), **W4** (phrases abandoned mid-vocal, 2.75× the
human at n=123) and **`follow_vocals`** (0.020 vs 0.149, **7×**) are three views of one cause:
**Stage-1 carries only `drum_proj` + `mix_proj`** and cannot hear the melodic instruments.
★The clincher: redistributing the note budget toward an abandoned phrase **does not fill it**
(γ 2.5 → 1.0 closes ~17 %, note count unchanged) ⇒**you cannot select what the model does not
propose.**
⚠️**BUT SIZED 2026-08-18j: Track B cannot close D4 on its own.** Split by whether a drum marks the
vocal onset, we reach **0.581** of the human where one does and **0.456** where none does
(p=0.00034) — so the melodic-blindness story is real, **but the drum-backed 58 % is the larger
half of the gap and Track B does not explain it.** ⚠️Does *not* revive v8 as built (its `follow_vocals` gain died at n=149) — it says the
**target** is right. Use W1 + W4 + `follow_vocals` as **one** acceptance target, not three.

---

## 🎯 STANDING CONSTRAINTS + the three older objections D1–D6 does NOT cover

★**His standing instruction, still binding**: *"tread carefully, make isolated and tactical
changes, and document like crazy."* One lever at a time, ≥3 seeds, nothing promoted without his ear.
⚠️He declined to name exemplary mappers *yet* — the best-mapper cohort is blocked **by his choice**.
*(The W1–W7 table and its mapping onto D1–D6 is history and lives in PROGRESS.md. Work D1–D6.)*

**Still open and NOT covered by D1–D6:**
- **W2** — Fallen Kingdom *"really empty"*. 🔴Cause unidentified; five instruments have failed.
  ⚠️**Ask him**: empty vs what our model *used to do*, or vs what the *song wants*?
- **W6** — multi-note swings (sliders/chains) are a **missing capability**, the right answer for
  grand low-density drops. Untouched, and a good `agent_mapper` target.
- **W5** — dot blocks as decoration: ⏸️**he deferred this himself.** Do not revive unprompted.

⚠️**Protect these — he named them by ear**: A6 hand-role division, and the density pacing
(*"when there is a slow spot we let the player breathe"*). ★And new on 2026-08-17: *"there is a
good deal of notes that are on beat and I can tell play part of the song"* — **that half works.
Do not regress it while chasing D1–D6.**

---

## 🔵 P2 — CARRIED FORWARD

### C1 — Precision sits at the greedy optimum; gains need better probabilities, not better picking
Three decode levers moved onset precision off ~0.90 by nothing (density, γ allocation, a
probability floor); the IOI prior moved it *down* to 0.769. **Stop hunting decode knobs.**
✅**2026-08-14 sharpened it**: the ~10 correct-tempo alignment failures are a **pure selection
defect**, established by elimination — not tempo, not phase, **not onset supply** (4.5 onsets
available per note we emit), and **not difficulty** (the human scores 0.943 on exactly those songs
vs 0.934 on the ones we handle fine).

### C2 — Grid PHASE ⚠️resolved ON THE METRIC ONLY — reopened 2026-08-17
`BEAT_GRID_PHASE=search` fixed ~18 of 39 failing songs **by the alignment axis**.
🔴**Kyle still reports *"slightly off beat"* across every song after playing `[PHASE]`**, so
either the lever is inaudible or the off-beat feel is not phase at all — see **D2**, where tempo
(right on only 70.5 % of songs) is the better suspect. Do not flip the default on the axis alone;
the axis is the thing the lever optimises. Remaining phase work is inside the tempo item above. ⚠️Never apply a blanket global shift — on `1f767` the human wants the same shift we do,
so that part is an **onset-detector offset**, and "fixing" it is the `h_dist` failure.

### C3 — Density/rhythm tension: you cannot thin your way to human density
Re-tuning toward the human note rate costs rhythm (`pulse_stability` −0.06 → −1.11). Humans at
3.9 nps have a pulse; we at 3.9 (thinned from 4.4) do not. `BEAT_IOI_PRIOR` made it worse. Needs a
different idea, not another sweep.

### C4 — Every beat-domain result predates the tempo fix
A2 rhythm, A6 handrole and the hand-offset work were tuned against a bpm that was wrong on 20 of
21 songs. Re-derive before building further on them.

### C5 — Doubles: root cause found; 🔴decode fix FAILED — but ★now PRICED against D4
★★**2026-08-18k gives C5 a payoff in the units of the biggest defect**: **39.6 % of the notes we
spend on the vocal line are doubles** onto an onset the other hand already covered (human 20.7 %),
so **fixing doubling alone lifts vocal coverage 0.385 → ≈0.487 — a quarter of the D4 gap, with no
extra notes.** ⚠️Still not reachable by decode (`BEAT_HAND_DEAL` failed); this justifies costing a
**representational** fix, not reviving the old one.
Not too many notes — **too few distinct times** (467 vs the human's 626). Stage-1's two hand
channels correlate **0.985–0.993**, so both hands get the same information and pick the same slots;
66 % doubles is structurally guaranteed. `BEAT_HAND_DEAL` hit every structural target and degraded
rhythm 6× — **C5 is not reachable by decode.**
★**2026-08-14 adds the flip side**: doubles are how a human buys density *without* speeding either
hand up (the human plays 8.35 nps at the same `ebpm_burst` as our 4.00).

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
