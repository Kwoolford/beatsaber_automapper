# Style levers — "make it more X", measured

**P6.** One row per request a player might make, with the build lever that answers it, the range
that stays clean, and the column that shows it moved. Every row was measured on the **real
best-build chain as of 2026-09-13** (`autobuild --pulse --lead-bias 0.2 --lead-in --drop-orphan --carrier-bias 2.0`
→ `repeat.py` → `answer.py`, seed 0 — the base arm reproduces `outputs/best_2026-09-13b/`
byte-for-byte) on the four songset maps, and the page cost is read off `verdict.py` against that
base. Script: `scripts/p6_levers.py` (`--report` reprints the numbers). Data:
`outputs/p6_levers_2026-09-17/`. Levers stay **monotone and default-off** because they are meant to
become player-facing knobs (`feedback-levers-are-user-facing`).

⚠️**Four songs.** A lever that moves its column on 4/4 with no page cost is *well-behaved on the
songset*, not proven general. Default flips need the 23-song check (see `--taper` below).
⚠️"Page cost" counts **every** red the SHIP line counts — query codes, ABSENCE rows and a judge
FAIL. (The first report read only query codes and missed an ABSENCE red; fixed in the script.)

Base page: `1f333` 2 red (BREATHING · SCATTER) · `1f767` 1 red (D3) · `1f8d6` SHIP · `1f913` SHIP.

## ✅ Levers that work — the column moves on 4/4

| request | lever | measured range | column (lo → hi, median of 4) | page cost at hi | judge p (Δ median) |
|---|---|---|---|---|---|
| **more doubles** | `--doubles-rate` | 0.1 – 0.6 | double share 0.07 → 0.38 (human median 0.14) | 1f8d6 loses its last lead-hand passage at 0.6 | +0.13 |
| **one hand leads** | `--lead-bias` | 0.0 – 0.5 | role asymmetry 0.08 → 0.16 | none | +0.03 |
| **more walls** | `--walls` | 40 – 160 | wall count exactly as asked | none (no metric reads walls) | 0.00 |
| **more variety / diagonals** | `--width` | 1 – 8 | local variety 0.75 → 0.94; diagonal share 0.05 → 0.32 | none | width 1: **−0.48** (vertical share 0.95) |
| **faster / denser** | `--nps` | 3.0 – 5.0 | nps **2.9–3.2 → 4.4–5.4** (honoured since 2026-09-17e; 1f8d6 tops out at 4.36, the song's supply) | at 5.0: +D3 on 1f333, +D6 and lead-hand ABSENCE on 1f767 ⇒ keep ≤ ~4.5; 6.18 is the density Kyle called unplayable | — |
| **figures come back** | `--palette` | 0 – 20 | SCATTER room 0.40 → 0.68 on 1f333 (line 1.00) | 1f8d6 lead-hand passage at 20 | −0.15 |
| **figures come back** | `--map-memory` | 0 – 4 | SCATTER room 0.40 → 0.64 on 1f333 | none | −0.20 |

✅**Held-out replication (2026-09-17j, 12 songs never used for tuning, seed 0, post-fix)** — every
lever moves its column on **12 / 12**:

| lever | column, median lo → hi | reds added at hi vs base | judge p Δ |
|---|---|---|---|
| `--doubles-rate` 0.1 → 0.6 | double share 0.06 → 0.29 | lead-hand ABSENCE ×2 | −0.08 |
| `--lead-bias` 0 → 0.5 | role asymmetry 0.07 → 0.13 | lead-hand ×1 | +0.01 |
| `--walls` 40 → 160 | walls as asked | none | 0.00 |
| `--width` 1 → 8 | local variety 0.73 → 0.91 | lead-hand ×3, SCATTER ×1 | +0.05 |
| `--palette` 0 → 20 | SCATTER room 1.48 → 1.83 | lead-hand ×2 | −0.11 |
| `--map-memory` 0 → 4 | SCATTER room 1.48 → 1.75 | lead-hand ×2 | −0.16 |
| `--nps` 3 → 5 | nps 3.08 → 5.00 | **D6 ×3**, lead-hand ×2 | −0.02 |

The lead-hand reds are the fragile ABSENCE row this table's base does not protect (it predates
`--hand-run-p` joining the best build); D6 at 5.0 confirms the ≤ ~4.5 range.

Notes:
- `--width 1` is legal but extreme: nearly every swing vertical, judge p collapses. Offer 2–8.
- Neither vocabulary lever clears 1f333's SCATTER (room 0.23), and both cost local variety. But as a
  **triage** step they work: on the 8 songs × 3 seeds where SCATTER fires under the best build,
  `--palette 20` clears it on **10 of 17** with no new red. **Held-out check** (9 reds, general
  build): the pre-registered "≥ 0.6 clears ≥ 4 of 5" FAILED (3 of 5). Pooled: room **≥ 0.7 → 11 of 11**,
  0.6–0.7 → 2 of 5, below 0.6 → 0. `verdict.py` prints the ≥ 0.7 rule on the SCATTER line.

## 🔴 Levers that do NOT answer their request

| request | lever tried | result |
|---|---|---|
| **follow the vocals** | `--carrier-bias` 1.0 → 4.0 | D4 hits 2→3 and 0→1 — **moves the wrong way or not at all**. A vocal-following lever still does not exist. |
| **breathe / calm** | `--style calm` | hits 7/20 of its percentile targets; `ebpm_burst` calm < dense on 0/4; adds D3 on 1f333 |

## ⏱ `--taper` — "breathe before the drop"

Moves budget from the bars before an energy rise to the bars after (Kyle's own P6 request, and the
mechanism behind D3). **23-song check, taper 0.5 vs base** (`outputs/taper23_2026-09-17/`):
ships **5 → 7**, reds **−7 / +3**, median judge p Δ 0.000, notes −0.5 to −4 % (moved, not spent).
Removed: D3 (1f767), D1 + SCATTER (1f3d7), SCATTER (1fbda), JUDGE (1fb71), lead-hand ABSENCE (1fa48,
1fa93). **All three added reds are the lead-hand ABSENCE row** (1f335, 1f8d6, 1fb2a) on maps that
had one passage or none to spare — the taper thins the run that was holding the row green.
On the songset, adding `--hand-run-p 0.06` (held one-hand runs, inside the human band) keeps
1f8d6 shipping: **taper + runs ships 3/4 with no new red** (base 2/4). The 23-song check of that
pair:

| arm (23 songs, seed 0) | ships | reds removed / added | median judge p Δ | `idiom_coverage` Δ |
|---|---|---|---|---|
| base | 5 | — | — | — |
| `--taper 0.5` | 7 | −7 / +3 (all lead-hand ABSENCE) | +0.000 | −0.011 |
| `--hand-run-p 0.06` | 8 | −8 / +1 (SCATTER on 1fa50) | +0.059 | +0.001 |
| **`--taper 0.5 --hand-run-p 0.06`** | **11** | **−10 / +2** (SCATTER on 1f335, 1fa50) | **+0.115** | −0.004 |

No shipping map is lost by any arm at seed 0.
✅**REPLICATED at seeds 1 and 2** — base vs taper+runs, 23 songs each:

| seed | ships | total reds | median judge p Δ |
|---|---|---|---|
| 0 | 5 → 11 | 27 → 19 | +0.115 |
| 1 | 5 → 9 | 26 → 18 | +0.064 |
| 2 | 8 → 9 | 25 → 18 | +0.068 |

Over the 69 builds: removed lead-hand ABSENCE ×16, SCATTER ×3, D1 ×3, JUDGE ×3, D3 ×3; added
SCATTER ×3, lead-hand ABSENCE ×2. `idiom_coverage` Δ median +0.001 (p10 −0.089 — the tail cost
`--hand-run-p` was known for). Per song, taper+runs ships at least as often as base on **all 23**;
the single-seed losses (1f65d, 1f8a3, 1fb44) are seed noise.
🔴**Held-out check (12 corpus songs never used to tune anything, 3 seeds)**: ships **6 → 7 of 36**,
reds 60 → 55 — lead-hand ABSENCE 12 → 2 (the runs generalise) but **D3 15 → 18 and JUDGE 9 → 12**
(the taper does not). The 23-song gain is partly selection. Single-lever split pending.
✅**Resolved after the parity-slip fix (2026-09-17i, 3 seeds each)**:

| arm vs fixed base | eval (69) ships | held-out (36) ships | held-out reds added |
|---|---|---|---|
| `--hand-run-p 0.06` | 20 → 23 | 11 → **14** | **none** |
| `--taper 0.5 --hand-run-p 0.06` | 20 → **28** | 11 → 11 | D3 ×3 |

⇒**`--hand-run-p 0.06` joins the general best build** (it generalises); **`--taper` stays a style
lever** — its eval gain is specific to the songs it was developed on. The songset best build keeps
it because it clears 1f767's D3. It stays OFF in `autobuild`'s
defaults, like every P6 lever: the best build is a documented flag set, not a default.

## Presets (`autobuild --style NAME`, `agent_mapper/style.py`)

Targets are human percentiles (a style is a position on a continuum — the corpus has no clusters).
Hit rate = share of a preset's targets landed within ±20 percentile points, over the 4 songs.

| preset | targets hit | nps landed | page vs base |
|---|---|---|---|
| human | **16/20** | 4.2–4.4 | +D3,D4 on 1f333 · +lead-hand ABSENCE on 1f767 |
| calm | **12/20** | 3.7–3.9 | +D1,D3 on 1f333 · 1f767 still red |
| dense | **15/20** | 4.4–5.6 | +D6 on 1f767 |
| flowing | **14/24** | 4.2–4.4 | +D3 on 1f333 · **1f767 ships** |
| technical | **16/24** | 4.2–4.7 | +D3 on 1f333 · **1f767 ships** |

⚠️Re-measured 2026-09-17e after `--nps` became a closed loop — the first table (7-13 hits) was built
on a density request the builder never honoured; the old builds are kept in
`outputs/p6_levers_2026-09-17/stale_predensityfix/`.

Orderings the presets claim (per song): nps calm < dense **4/4** · peak_nps 3/4 · crossover
flowing < technical **4/4** · travel calm < technical 3/4 · angle_change flowing < technical 2/4 ·
ebpm_burst calm < dense **0/4**.
⇒**P6's DoD ("three presets build clean with the named column moved") is NOT met**: every preset
adds a red on some songset map (mostly D3 on 1f333), though hit rates are now 58-80 %. The single-lever rows above are the usable UI
today; presets need their targets mapped to the levers that demonstrably move them.
