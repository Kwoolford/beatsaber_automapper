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
