# `agent_mapper/` — tools for an agent to build a Beat Saber map by hand

**The thesis.** Our ML pipeline decides one slot at a time from a 16-note window. An agent
has the opposite shape: it can hold the *whole song* at once — structure, lyrics, where the
drop lands, what happened 90 seconds ago — but it cannot emit 1300 notes reliably by hand.
This folder gives the agent the half it is missing so its strength can be used.

★**Why this is not a detour from the ML work.** The single biggest finding of 2026-08-14 is
that **W1, W4 and `follow_vocals` are one defect: Stage-1's representation does not carry the
melodic instruments.** `version_4` has `drum_proj` + `mix_proj` and nothing else — it
literally cannot hear the guitar or the vocal line, which is why we abandon sung phrases
(2.75× the human rate) and play the vocal figure 7× less than a human does.

An agent reading **timestamped lyrics** and a per-stem timeline does not have that defect at
all. So this folder is a way to answer a question the ML track cannot currently answer:

> If the model *could* hear the vocal line and see the whole song, would the maps be good?

If a hand-built agent map is good, the representation is the problem and Track B is
justified by demonstration rather than by argument. If it is *still* bad, the problem is
somewhere else entirely and that is worth more than another lever.

## The three surfaces

| | | |
|---|---|---|
| **PERCEIVE** | `brief.py` | the song as a longitudinal text score — sections, per-stem rhythm per bar, energy, and **lyrics with timestamps** |
| **ACT** | `mapctl.py` | a stateful workspace: place / clear / audition / export, section by section |
| **CHECK** | `mapctl.py check` | parity, reachability, doubles, and the eval suite — before export, not after |

Existing tools this deliberately does **not** duplicate:
`scripts/map_view.py` (read a finished map as a score), `scripts/map_write.py` (write one),
`scripts/suite_report.py` (evaluate). The `/map` skill covers those. What was missing is
everything on the **song** side, and a way to build up a map incrementally without holding
all 1300 notes in context at once.

## Design rules, each from something this project already learned

1. **Compact beats complete.** A 4-minute song is ~2 000 sixteenth-notes; a per-bar summary
   is ~130 rows. The brief must fit in context *alongside* the reasoning, so it summarises
   per bar and expands only where asked.
2. **Times in seconds AND bars, always both.** 30 % of our maps are at the wrong tempo, so
   bar indices are not comparable across maps of the same song. Seconds are the ground truth.
3. **Never silently drop a song.** Missing stems, missing lyrics and missing onsets are
   reported, not skipped — `alignment` was silently absent from every scorecard for two
   nights because a cache miss looked like a clean run.
4. **Validate before export, not after.** A map that fails parity is not a draft, it is
   unplayable; `check` runs the same swing simulator the eval suite uses.
5. **The agent's map gets scored on the same ruler as the model's.** No special-casing.
