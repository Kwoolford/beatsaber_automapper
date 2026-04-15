# Auto-Researcher

## Concept

One spec → one short training run → one generated test map → one leaderboard row.
Target 10-20 runs/day so we can actually iterate on architecture choices.

## Directory layout

```
experiments/
├── queue/              # YAML batches of experiment specs
│   └── initial.yaml
├── runs/               # One subdir per run (keyed by experiment_id)
│   └── a1b2c3d4e5f6/
│       ├── spec.yaml
│       ├── train.log
│       ├── generate.log
│       ├── checkpoints/
│       ├── generated/test_map.zip
│       ├── metrics.json
│       └── status.json
└── leaderboard.jsonl   # Append-only ranked results
```

## Spec

See `src/beatsaber_automapper/research/spec.py:ExperimentSpec`. Minimum:

```yaml
name: joetastic_baseline
cohort: joetastic        # OR bucket: anime_jpop_flow
max_epochs: 8
max_wall_clock_min: 45
```

Experiment ID is the SHA-256 hash of the spec (minus name/notes). Same spec → same id → dedup on resume.

## Running

```bash
# One queue
python scripts/auto_research.py experiments/queue/initial.yaml

# Resume (skip already-done)
python scripts/auto_research.py experiments/queue/initial.yaml --resume

# View
python scripts/leaderboard.py
python scripts/leaderboard.py --cohort joetastic --sort-by parity_rate --asc
```

## Metrics

**Playability** (computed on generated map):
- `n_notes` — did it actually emit notes?
- `parity_rate` — per-color swing alternation violations
- `collision_rate` — multiple notes at same (x,y,b)
- `n_walls`, `n_arcs`, `n_chains`, `n_bombs`
- `notes_per_sec`
- `direction_histogram` — distribution over 0..8

**Style-closeness** (vs cohort reference, when `cohort` is set):
- `direction_kl` — KL(generated || cohort)
- `nps_gap`
- `parity_rate_gap`
- `color_balance_gap`

**Composite**: weighted combo — parity (0.4) + collision (0.2) + notes-emitted (0.1) + style (0.3).

## Extending

- New metrics → `research/metrics.py`
- New conditioning axis → add field to `ExperimentSpec`, extend `_build_train_cmd` in `runner.py`
- New model preset → `configs/model/<name>.yaml`, reference in spec's `model_preset`

## What's NOT wired yet (known TODOs)

- `data.cohort=<slug>` / `data.bucket=<id>` Hydra overrides rely on dataset.py changes (Phase V5-0, step 0.5)
- Cohort reference stats (`data/cohorts/<slug>/reference.json`) are computed by a separate script that doesn't exist yet (Phase V5-0, step 0.6)
- Generation currently uses default onset + fresh seq ckpt; may need `--min-length` tuning per cohort
