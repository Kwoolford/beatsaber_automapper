# Calibration reference snapshot

**These are a backup, not the live path.** (Kept under `docs/` because `data/` is gitignored too.) The suite still reads its references from `outputs/`,
which is entirely gitignored — `git ls-files outputs/` returns zero. Every bar in the evaluation
suite is defined by these seven small JSONs (28 KB total), and A8 *fails closed* when its reference
is missing, so losing them would look like a regression rather than a missing file.

Snapshotted 2026-08-03 because the hazard stopped being hypothetical: `flow_human_reference.json`
was regenerated that night after a loader fix (`BPMInfo.dat` being read instead of `Info.dat`), which
meant the repo's scoring behaviour had changed with nothing in version control to show it.

Deliberately a copy rather than a move: relocating the live path changes a project convention and
several scripts' assumptions, which wants a decision rather than a unilateral commit (TODO item C6).

**If you change a reference, re-copy it here in the same commit**, or this snapshot silently drifts
and becomes worse than useless.

| file | what it calibrates |
|---|---|
| `alignment_human_reference.json` | A8 audio alignment — precision 0.930, scatter 10.35 ms |
| `flow_human_reference.json` | A1 flow/ergonomics — regenerated 2026-08-03 |
| `rhythm_human_reference.json` | A2 rhythm grid |
| `idiom_human_reference.json` | A3 direction idiom |
| `handrole_human_reference.json` | A6 hand role |
| `playfeel_human_reference.json` | A7 difficulty/playfeel — nps 3.909, peak_nps 5.5 |
| `ioi_human_model.json` | interval bigram used by the (retired) `BEAT_IOI_PRIOR` lever |
