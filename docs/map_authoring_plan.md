# Direct map reading & authoring — investigation + plan

**Written:** 2026-07-27
**Question:** is it worth building a representation where Claude can read maps note-by-note in
musical context, cross-compare sections, and eventually *author* a map by hand?

**Answer: yes, and the evidence arrived within an hour of prototyping.** Detail below.

---

## 1. Why this has merit (evidence, not argument)

Every channel we currently have onto a map is either an **aggregate** (the eval suite) or an
**image** (`render_map.py`). Both have now demonstrably failed:

* Aggregates lie. `h_dist` ranked our maps as **more human than human** for weeks. The v2 suite
  fixed that specific failure, but the general risk is structural — a metric is a lossy summary,
  and we only find out it is the wrong summary when something independent contradicts it.
* Images can be looked at but not *queried, diffed, or edited*. `render_map.py` produces PNGs;
  there is no way to ask "show me every place the model did X", and no way to write back.

I built a text score view (`scripts/map_view.py`, prototype, working) and pointed it at one of
our maps next to the human map for the same song. Two findings in the first two comparisons:

### Finding 1 — hands have *roles* in human maps, and none in ours
Reading bars 33–36 side by side: the human map alternates hands at 1/16 offsets (128.50 L →
128.75 R; 140.00 R → 140.25 L → 140.50 R → 140.75 L), and within a passage **one hand carries a
sustained run while the other punctuates sparsely**. Ours fires both hands together at identical
density throughout, with no role division at all.

The lockstep half of that is A2's finding, already known. **The role-division half is new** — no
axis in the suite measures whether the two hands are doing *different musical jobs*. It was
invisible statistically and obvious on sight.

### Finding 2 — 22% of our maps are generated at the WRONG TEMPO, and the metrics reward it
The score header prints BPM, so putting the two maps side by side made it immediate: human
1f333 is 188 BPM, ours is **94**. Checked across the eval set:

| ratio (human ÷ ours) | songs | meaning |
|---|---|---|
| ≈ 1.00 | 17 / 23 | correct |
| ≈ 1.50 | 3 / 23 | 2:3 metrical misread |
| ≈ 2.00 | 2 / 23 | **exactly half tempo** |

At half tempo our finest grid slot (`BEAT_SUBDIV=4`, quarter-beat) is **twice as coarse in real
time**, so we structurally cannot place the fast notes the human map uses. And the kicker:

```
correct tempo (n=17): flow 0.93 | rhythm 2.54 | idiom 1.91
WRONG   tempo (n= 6): flow 0.73 | rhythm 1.96 | idiom 1.36
```

**The mis-tempo maps score BETTER on every axis.** A2 measures inter-onset intervals in the
*beat* domain, so halving the tempo stretches those intervals and manufactures apparent rhythmic
variety while the real-time grid gets coarser. The suite does not merely miss this bug — it
actively rewards it. That is precisely the "call out the model when the metrics are lying"
capability Kyle asked for, and it paid for the tooling immediately.

**Immediate consequences (both now on the stack):**
1. Add a wall-clock guard to A2 — intervals must also be checked in seconds, and a tempo-sanity
   check added against the source map / detected tempo.
2. Investigate BPM detection: 5/23 wrong is a large error rate for something everything
   downstream depends on.

---

## 2. What exists already

| piece | state |
|---|---|
| `scripts/render_map.py` | lattice panels + density strip + swing trace, PNG, read-only |
| `evaluation/swing_sim.py` | per-swing parity, resets, violations — the semantic layer |
| `data/instrument_features.py` | per-stem kick/snare/hat/bass/vocals/lead **per slot** — the "Demucs-style lanes", already computed and cached for training |
| `data/audio.py::detect_sections` | section boundaries |
| `evaluation/idiom.py` | the 2,510-idiom vocabulary — a *name* for each pattern |
| `generation/export.py` | map → playable zip (the write path already exists) |
| **`scripts/map_view.py`** | **NEW prototype: text score, hands side by side, stem lanes, section overview, side-by-side compare** |

The framework really is mostly in place. What is missing is a **single text-native
representation that is queryable, diffable, and round-trippable.**

---

## 3. The format

A tracker/score: time runs down, one row per grid slot, hands side by side, audio in its own
lanes. This is the prototype's current output, working today:

```
 bar    beat │ L          │ R          │ K S H │ bass lead
  34  132.00 │ 0,2 ↑ B    │ 2,0 ↙ F    │ █ · · │ 0.42 0.00
  34  133.00 │ 0,1 ↓ F    │ 2,0 ↗ B    │ · ▅ ▃ │ 0.42 0.61
```

`col,row` + cut-direction arrow + parity (F/B, `!` on a simulator violation). Reading **down** a
column is one hand's flow; reading **across** is what both hands and the music do at that
instant; the rows around it are the context. Audio lanes come from the same per-stem
transcription the model trains on, so what I see is what the model saw — which is the whole
point when asking "should a note be here at all?"

**Design requirements:**
1. **Lossless** — every field of a `ColorNote` recoverable, so it round-trips.
2. **Grid-aligned** — rhythm must be visually apparent, which is why rows are slots not notes.
3. **Diffable** — plain text, so two sections can be compared with ordinary tools.
4. **Addressable** — every row has a stable `bar/beat` address to talk about.

---

## 4. Plan

### Phase 1 — reading (prototype done, needs finishing)
- [x] score view, hands side by side, parity tags
- [x] section overview, side-by-side compare
- [x] per-stem audio lanes
- [ ] annotate each transition with its **idiom id + human corpus frequency** — lets me see "this
      is idiom #3, used in 4% of human transitions" or "this is out-of-vocabulary"
- [ ] mark swing-simulator violations and flow outliers inline
- [ ] `--find` queries: every occurrence of an idiom / a pattern / a violation, with context
- [ ] cache stem features per song so lanes are instant

### Phase 2 — auditing (the point of Phase 1)
- [ ] `--vs <other map>` aligned by time, not bar, so ours and the human map for the same song
      can be read together (the tempo bug means bar numbers do **not** align)
- [ ] section-cluster view: group sections by similar audio, print their maps together — "the two
      choruses got these two treatments"
- [ ] a standing **metric-audit routine**: sample N passages the suite scores well and N it
      scores badly, read them, and record where reading and metric disagree. Every disagreement
      is either a metric bug or a genuine insight; both are valuable.

### Phase 3 — authoring (the reverse direction)
- [ ] **parser**: score text → notes → `export.py` → playable zip. The write path already exists,
      so this is mostly parsing.
- [ ] compose at the **idiom/phrase level**, not note-by-note. A 3-minute map is 1300+ notes;
      hand-placing them is not realistic, and it is not how human mappers work either. The
      vocabulary gives named blocks to place, with the parser expanding them.
- [ ] `/map` skill driving the loop: pick a song → show sections + stem lanes → I place patterns
      per section → render → score → iterate.
- [ ] **DoD, and the real test of the whole thesis:** a map I author by hand should (a) score
      human-range on the v2 suite and (b) be judged good by Kyle on play. If it scores well and
      plays badly, the *suite* is still wrong and I have found the next blind spot. If it scores
      badly and plays well, likewise. Either outcome is a win.

### Phase 4 — close the loop back onto the model
- [ ] use hand-authored maps as reference targets for specific passages
- [ ] when the suite and reading disagree, the disagreement becomes a new axis candidate (the
      hand-role finding above is already one)

---

## 5. Honest limitations

* **I do not hear audio.** I read it as onsets, per-stem energy, and transcribed pitch. That is
  enough to judge "is there a musical event here and how strong is it", and it is exactly what
  the model gets — but it is not the same as listening, and Kyle remains the only one who can
  judge feel. This tooling narrows that gap; it does not close it.
* **Hand-authoring does not scale** past a few phrases at note level, hence the idiom-block
  approach in Phase 3.
* **Reading is sampling.** I will read tens of passages, not thousands. It is a check on the
  aggregates, not a replacement — the two are complementary, which is the entire point.
* **Risk of tool-building for its own sake.** Mitigated by Phase 1 already having produced two
  real findings before the plan was written; each later phase should be justified the same way.
