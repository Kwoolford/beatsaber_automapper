# Evaluation harness — the overnight research loop

The point of this harness is to test many generation theories per night and judge
them at a glance, against **human baselines**, without hand-running one config at a
time. A "theory" is one line in an `ARMS` table; a run produces a leaderboard + a
self-contained `report.md` with metric tables and before/after renders.

## Pieces

| file | role |
|---|---|
| `scripts/eval_sweep.py` | the driver: song set, arms, scoring, leaderboard, report |
| `scripts/map_metrics.py` | shared **map-only** metrics (no audio) — one source of truth |
| `scripts/eval_density_corr.py` | the density DoD (Spearman of gen vs onset density) |
| `scripts/eval_alignment.py` | stem separation + librosa onsets (reference plumbing) |
| `scripts/eval_layout_ckpt.py` | eval one layout checkpoint across the song set |
| `scripts/render_map.py` | the perception channel (density strip + lattice + swing trace) |

## Quick start

```bash
# one-time: build a cached full-length song set + Demucs reference onsets
python scripts/eval_sweep.py build-songset --n 6

# one-time (or refresh): human metric baselines from data/raw
python scripts/eval_sweep.py human-baseline --n 40

# run all arms (or a subset) → leaderboard.json + report.md + renders/
python scripts/eval_sweep.py sweep
python scripts/eval_sweep.py sweep --arms control,dsel_g2.5
```

Outputs land in `outputs/eval_sweep_cache/`: generated maps are cached per
`(arm, song)` (re-runs skip generation), plus `leaderboard.json`, `report.md`,
`renders/`, `human_baseline.json`, and per-song `*.ref.npz` (cached references).

## Adding a theory

Add one entry to `ARMS` in `eval_sweep.py`:

```python
ARMS = {
    "control":    ({}, []),                                   # env overrides, extra CLI flags
    "dsel_g2.5":  ({"DENSITY_SELECT": "1", "DENSITY_SELECT_GAMMA": "2.5"}, []),
    "mytheory":   ({"SOME_ENV": "1"}, ["--top-p", "0.97"]),   # extra flags override harness defaults
}
```

`env` sets environment variables for that generation (most experimental levers are
env-gated so the default pipeline is untouched); extra flags are appended to the
`generate.py` command (argparse last-wins, so they override harness defaults like
`--temperature`).

## Metrics (and where "good" comes from)

Audio-coupled (need the cached reference):
- **density_corr** (`spearman`) — Spearman of generated note density vs reference
  onset density over 2 s windows. **The DoD: ≥ 0.41.** Rank-based on purpose —
  catches "tracks the song's structure" rather than a scale match.
- **onset_hit** — fraction of notes within 50 ms of a real onset (placement proxy).

Map-only (`map_metrics.py`, no audio — cheap), each shown against the human mean:
- **row_conc / col_conc** — max fraction of notes in one row / column (collapse
  detector; human ≈ 0.49 / 0.29). Pre-fix v10 was 0.94 / 0.48 (the for-sport
  bottom-row/2-column stream).
- **grid_coverage** — fraction of the 12 grid cells used (human ≈ 0.96).
- **dir_entropy** — normalised entropy of cut directions (human ≈ 0.80).
- **monotony / pattern_repeat** — `best_of_n_poc` monotony proxies (human ≈ 0.43 / 0.00).
- **nps / n_notes** — density level.
- **viol** — swing-sim parity violations (playability; should stay 0).

Human baselines are computed by `human-baseline` over `data/raw` and cached; the
sweep loads them so every metric prints with its human target. Defaults in
`map_metrics.HUMAN_TARGET` are the 2026-06-30 40-map sample.

## Known-good levers (validated 2026-06-30)

- **Density** that tracks the song: `DENSITY_SELECT=1 DENSITY_SELECT_GAMMA≈2.5`
  (re-allocates the note budget per 2 s window ∝ (window-mean onset prob)^γ).
  density_corr ≈ 0.53 (5/6 songs ≥ 0.41) vs 0.26 control.
- **Layout variety**: the row/column collapse was a 1-line inference bug in
  `LayoutPhraseModel.generate_phrase` (returned `toks[1:]`, leaving the ctx_len
  context prefix in front of the event stream → off-by-ctx_n parse → every note
  clamped to row0). Fixed to `toks[ctx_n + 1:]` → row_conc 0.94 → 0.48 (human 0.49).

## Conventions / gotchas

- Generation is cheap (~25 s/song); the slow part is one-time Demucs reference
  caching. The 11-minute song in the set dominates each arm's wall time.
- `generate.py` takes `audio` **positionally**; load beat ckpts with `strict=False`.
- Don't pick layout/beat checkpoints by `val_token_acc` / `val_f1` — they
  anti-correlate with structure quality. Use density_corr / alignment / row_conc.
- Production inference defaults: layout `version_10` (ctx16 + song-memory), beat
  `version_4`, `section_gate="loud_only"`.
- Diagnostic env flags (gated, default off): `LAYOUT_DIAG=1` logs the model's
  per-step argmax-prob and sampled row/col histograms; `BEAT_PROBS_DUMP=path`
  dumps raw Stage-1 onset probs; `BS_PREPOST_OUT=path` dumps the pre-postprocess map.
