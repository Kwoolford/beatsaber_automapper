# `agent_mapper/` — what the first build taught

## 2026-08-14 — the toolkit exists, and building one map with it found four bugs

Built `brief.py` (perceive), `lyrics.py` (timestamped words from the vocals stem) and
`mapctl.py` (a session you author into), then used them to build a complete 1 081-note
map of **Hunger** from a **19-line plan**. Every number below is on the same ruler the
ML maps are scored with — no special-casing.

| | notes | nps | precision | ebpm_burst | travel | doubles | parity |
|---|---|---|---|---|---|---|---|
| **AGENT (hand-built)** | 1081 | **4.00** | 0.983 | **376** | 4.42 | 0.000 | 0 |
| ML baseline | 1328 | 4.96 | 0.960 | 376 | 5.94 | 0.391 | 0 |
| HUMAN | 2254 | 8.35 | 0.956 | 376 | 12.53 | 0.146 | 0 |

⚠️**The precision is CIRCULAR and must not be quoted as a win.** `auto` places notes
*on* the onsets the metric then scores against, so 0.983 is guaranteed by construction
— the same circularity flagged for `BEAT_GRID_PHASE=search`. What is *not* circular:
nps lands on the human Expert median (3.91), burst rate matches the human exactly, and
parity is clean.

★**Two real gaps remain, both nameable**: travel **4.42 vs the human's 12.53** (our
hands barely move — `auto` only uses two columns and two rows per hand) and
**doubles 0.000 vs 0.146** (we never play both hands at one instant).

## The four bugs, each found by measuring rather than by reading

1. **`_mmss` turned a −21 ms downbeat into "−1:59.98"** — a two-minute error on the
   first row of every brief, from a sign nobody had tested.
2. **Layering two `auto` passes doubled the burst rate** (376 → 752 against a human
   376). Each pass tracked its own hand and parity state, so the second pass landed
   notes *between* the first's and handed one hand two fast swings.
3. ★**The per-hand floor, and the first fix that looked applied and did nothing.**
   Measured across **31 723 human gaps in 40 songs**: a human hand almost never swings
   twice inside ~**150 ms** (cohort p5 148 ms; Hunger's human map floors at 160 ms).
   Ours allowed **80 ms**. The first guard checked only the *previous* note of that
   hand — but a later pass inserts notes *between* fixed ones, so 70 violations
   survived and the metric did not move at all. Checking **both neighbours** fixed it:
   ebpm 752 → **376**, min gap 159 ms, zero violations.
4. `DifficultyBeatmap` needs an explicit `version`.

## A hypothesis of mine, refuted by its own sweep
I expected **hand runs** to look more human than strict alternation, since
`role_asymmetry` is human 0.115 against our 0.026. For burst speed it is the
**opposite**: `runs=1` gives exactly the human's 376 and `runs≥2` gives 752, because
`ebpm_burst` is a **per-hand** rate and alternating is precisely what keeps each hand
slow. `--runs` survives as a stylistic knob; its default is 1 for a measured reason.

★**And the number that reframed it**: the human's *average* per-hand rate is **3.99**,
essentially identical to our 3.96 — while their burst rate is half. **The defect was
never the average, it was the fast tail.**

## What this says about the ML track
Nothing yet — the map has not been played. That is the whole point of building it:
`docs/eval_suite_v2.md`'s axes have been measured to not track Kyle's ear (M-F ranks
the map he called *"really empty"* second-best), so a hand-built map that scores well
proves only that it scores well. **Installed as `AUTO Hunger [AGENT]` for his verdict.**
