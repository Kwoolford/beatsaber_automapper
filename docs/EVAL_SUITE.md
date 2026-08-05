# The eval suite — how to see a map the way Kyle hears it

Built 2026-08-04 after Kyle made it P0: *"I can only communicate the problem but you can see the
correlation to the config of the model… Create a way for you to see the song and map in a way that
gives you my vision."*

## Start here

    python scripts/suite_report.py --song 1f8d6          # one song: metrics + timestamps + picture
    python scripts/suite_report.py --all --no-png        # cohort summary over the songset

`suite_report` prints the **grid confidence first**. A coverage number computed on a LOW-confidence
grid is not comparable to one computed on a high-confidence grid, and on `1fa48` / `1f9a0` coverage
reads ~0.00 for a real but specific reason (see S1 in TODO.md) rather than "we play nothing".

## The second half — the "masterpiece" axes (M1–M4, built 2026-08-04 night)

Kyle: *"We created a model to create a playable map but now need a model to start
producing masterpieces which we are far off from… syncing to rhythm more and making
significantly more intelligent and intentional placements of notes."*

Everything above scores a note against the audio **at its own instant** — is it on an
onset, on the main beat, in a busy window. A map can pass all of it and still be
lifeless, because intent is not a property of an instant. It is a property of a
**relation**: what the map does when the music does the same thing twice.

    python scripts/masterpiece_report.py --arm tf_trim_ev03_rc05      # all axes, cohort
    python scripts/masterpiece_report.py --arm mbb025 --vs tf_trim_ev03_rc05
    python scripts/view_structure.py --song 1f8d6                     # ★the picture
    python scripts/audit_masterpiece.py --n 13                        # may these steer?

| axis | question | verdict |
|---|---|---|
| **M1** `eval_motif_rhyme.py` | when the music comes back, does the pattern come back? | `*_rhythm` and `*_place` **may steer** |
| **M2** `eval_rhythm_fidelity.py` | is the map playing *this bar's* figure, and whose? | `follow_mean` / `follow_best` / `follow_vocals` **may steer** |
| **M3** `eval_accent.py` | is emphasis spent where the music emphasises? | `hands_x_downbeat` **may steer**; the rest diagnostic |
| **M4** `eval_arrangement.py` | does the map turn when the song turns? | ⚠️**not usable** (v1 and v2 both fail) |
| **M5** `eval_hand_intent.py` | do the two hands do two different jobs? | ⚠️**not measurable** — the human mean is 0 |
| **M6** `eval_anticipation.py` | does the map build *into* a drop? | ⚠️**null** — neither cohort does |

**The roster, by what a lever may be steered by:**

- ✅**MAY STEER (passes on both cohorts)** — `rhy_rhythm`, `harm_rhythm`, `timb_rhythm`,
  `harm_place`, `timb_place`, `rhy_place`, `follow_mean`, `follow_best`, `follow_vocals`,
  `hands_x_downbeat`.
- ⚠️**PROVISIONAL (verdict flips between cohorts — a flipping verdict is not a verdict)** —
  `follow_drums`, `hands_x_strength`, `arrange_ami`.
- ❌**DIAGNOSTIC ONLY** — `travel_*`, `turn_*`, `hand_stem*`, `arrange`, `hands_x_coincid`
  (resolvable at n=13, did **not** replicate at n=42), `double_share`, `corr_at_0`.

### ★ Why these are the first steer-safe axes here

Every metric this project built that scored a **level** was metronome-gameable
(`halfbeat_rate` 0.036 vs a human 0.084; `share_over_1s` 0.200 vs 0.250). These score
a **contrast**:

    what the map does where the music says X  −  what it does where the music says not-X

A metronome is identical everywhere, so it cannot correlate with a song that is not:
it scores **0 by construction**, and so do random note times, a bar-rotated map, and
another song's map. Measured, not assumed — `audit_masterpiece.py`.

### The two kinds of control, which are not the same test

- **Degenerate** (metronome, random times, bar-rotated, wrong song): must stay under
  **50 %** of the human value. A metric they can reach is a metric a lever can reach
  the cheap way.
- **Degradation** (a human map jittered ±60 ms, or thinned by 30 %): **not** pass/fail
  — a slightly damaged human map *should* land between ours and human. It fails only
  by scoring **above** the human, which means the metric rewards the damage.

Conflating them marked six good axes diagnostic-only on the first run, because a
30 %-thinned human map scored 0.86× on `follow_mean` — while our own maps score 0.30×.

### Two cohorts, and they are not interchangeable

- **The eval songset** (24 songs, 13 with a strict Expert human map) is the **fixed ruler** every
  historical arm was scored against. Do not change it.
- **The wide cohort** (`scripts/build_wide_cohort.py`, corpus songs **disjoint** from the songset, each
  with a strict Expert map and a seeded stem cache) exists for one purpose: **does a finding survive a
  bigger n?** Score it with `masterpiece_report.py --wide` / `audit_masterpiece.py --wide`.
  ⚠️Never pool the two into one median.

Every headline finding below replicated at n=42 on the wide cohort; `harm_rhythm` and `timb_rhythm`
became resolvable there and were not on 13 songs; **`hands_x_coincid` did not replicate** (it was
already diagnostic-only). `follow_drums` and `hands_x_stre**ngth** flip verdict between the two cohorts
and are therefore **PROVISIONAL** — a verdict that flips with the sample is not a verdict.

### Where the cohort stands (paired, 13 songs with a human Expert map)

| metric | ours | human | paired Δ | resolvable |
|---|---|---|---|---|
| `rhy_rhythm` | +0.060 | +0.148 | −0.116 | **yes** |
| `harm_place` (movement reuse) | +0.002 | +0.016 | −0.020 | **yes** (~9×, the largest ratio) |
| `follow_mean` | +0.033 | +0.107 | −0.089 | **yes** |
| `follow_vocals` | +0.020 | +0.149 | −0.129 | **yes** |
| `follow_best` | +0.074 | +0.218 | −0.142 | **yes** |
| `hands_x_downbeat` | +0.036 | +0.182 | −0.387 | **yes** |
| `lead_persistence` | 0.292 | 0.387 | −0.111 | **yes** |

Read together: we reproduce a bar's figure about a third as faithfully as a human, we
follow the **vocal** line 7× less, we change which instrument we are following far more
often, and we do not mark the downbeat. None of it is visible to any earlier axis.

### Rules these axes added (learned the same night)

8. ★**A correlation between two signals that each have slow structure is not evidence
   of correspondence.** M3's first version scored a bar-rotated map at 1.54× the human
   and *another song's map* at 0.77×, because emphasis and loudness both vary slowly.
   Difference **locally** — inside blocks, inside lag strata — or measure nothing.
9. **Similarity must be chance-corrected.** With cosine, our maps out-scored the humans
   on Hunger: `DENSITY_SELECT` makes note count track loudness, so similar-sounding bars
   hold similar note counts and overlap by chance. Cohen's **kappa** removes exactly
   that term.
10. **A control is only a test if it perturbs the domain the metric reads.**
    `shuffled_attrs` ties a time-domain metric exactly; a whole-bar rotation cannot move
    a metrical-position metric. Tie-to-three-decimals means "by construction", not "fail".
11. **The grid must be a property of the song, not of the map being graded**
    (`ss.song_end`), and every cohort row must come from the same subset of songs — the
    battery itself shipped a 0.2994-vs-0.1817 disagreement caused by breaking that.

## The pieces

| script | answers |
|---|---|
| `main_beat.py` | **which pulse is this song built on?** Scores ½×/1×/2× the fitted beat against drums∪bass on two opposing measures (support × capture). |
| `view_main_beat.py` | **the picture.** Stem lanes, Stage-1's probability, the MAIN BEAT lane, our notes and the human's. |
| `review_map.py` | **where do I listen?** Ranked timestamps: STARVED / MISSED_HIT / OFFBEAT / PHRASE_HOLE / MAPPING_SILENCE / ENDING. |
| `view_ab_diff.py` | **what did this lever change?** Only the differing notes, bucketed by coincidence order. |
| `view_song_strip.py` | **whole-song shape.** nps vs human, intensity, k≥3 response, offbeat rate. |
| `eval_coincidence.py`, `eval_beat_phase.py`, `eval_phrase_abandon.py`, `eval_pulse_consistency.py`, `eval_intensity_alloc.py` | the individual cohort metrics |
| `audit_phase_metrics.py` | **may this metric steer a lever?** (usually: no) |

## Reading `view_main_beat.py`

    ●  played            we put a note on this main beat
    ○  red  = SKIPPED    nothing within half a period — we ignored it
    ○  orange = AROUND   a note nearby but off-grid — we played beside it
    faint notes          not on the main beat
    Stage-1 p lane       the model's raw probability, with a red guide at each main beat

Red vs orange is the whole diagnosis: **skipping** and **playing beside** are different defects with
different fixes. A field of orange means a phase problem; a field of red means the beat lost its
window to something louder.

## Rules learned the hard way

1. **PNG is the primary artifact.** An agent can only look at an image by rendering it to a file and
   reading it back. A beautiful interactive page is built for the wrong reader.
2. **Tolerance must scale with the period.** A flat 70 ms tolerance makes `capture` 1.000 by
   construction for fine grids — it picked a 16th grid on 20 of 24 songs before this was fixed.
3. **Calibrate against humans, never against our own output.** The grid's 18 ms detector bias was
   calibrated on the human corpus for exactly this reason; the alternative is `h_dist` circularity.
4. **Every metric that rewards REGULARITY is metronome-gameable.** `halfbeat_rate` and
   `share_over_1s` both failed the control battery because a metronome beats a human on them. They
   diagnose; they must never steer a lever.
5. ★**A property of the probability field is not a property of the map.** Selection, thresholds and
   density sit in between. This project has been caught three times: the 2026-08-02 probs-replay, the
   adaptive main-beat lift, and S7.
6. **A cohort median cannot see a subset-of-songs defect, and a song median cannot see a
   subset-of-windows defect.** Bucket and name the songs.
7. **Look at the picture, not only the aggregate.** The main-beat bonus raises Fallen Kingdom's
   song-level coverage 67.8 % → 77.1 % while its worst stretch moves only 35 % → 40 %. The aggregate
   hid that; the render did not.
