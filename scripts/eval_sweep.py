#!/usr/bin/env python3
"""Multi-song / multi-arm evaluation sweep harness (2026-06-30).

Purpose: test many generation theories per night and get a leaderboard, instead
of hand-running one config at a time. Each ARM is a named set of env vars + CLI
flags for scripts/generate.py; each SONG is a cached full-length audio file with
a PRECOMPUTED reference onset density (the expensive Demucs step, cached once).
For every (arm, song) it generates a map (cached on disk) and scores it with the
DoD density-corr (+ note count, CV); prints an arm×song Spearman matrix with
mean and pass-count, and writes a JSON leaderboard.

Subcommands
-----------
  build-songset --n N         extract N full-length (>=MIN_DUR s) songs from
                              data/raw into data/eval_songset/ and cache refs.
  sweep [--arms a,b,...]       run the arms (default: all defined) over the songset.
  list-arms                    print the registered arms.

Add a theory = add one entry to ARMS below (name -> env dict + extra flags).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
SONGSET = REPO / "data" / "eval_songset"
CACHE = REPO / "outputs" / "eval_sweep_cache"
MIN_DUR = 150.0          # seconds; below this, density-corr Spearman is too noisy
WIN_SEC = 2.0
SR = 44100

BEAT_CKPT = "logs/beat_classifier/version_4/checkpoints/beat-epoch=11-val_f1_avg_tol=0.603.ckpt"
LAYOUT_CKPT = "logs/layout_phrase/version_10/checkpoints/layout-epoch=09-val_token_acc=0.865.ckpt"

# ---- ARMS: name -> (env overrides, extra generate flags). Add theories here. ----
_DS25 = {"DENSITY_SELECT": "1", "DENSITY_SELECT_GAMMA": "2.5"}
# A theory = one entry: name -> (env overrides, extra generate.py flags).
# History (2026-06-30): the density-select gamma sweep found g2.5 best (5/6 pass);
# a Stage-2 temperature sweep was a dead end (layout collapse was a decode bug, now
# fixed in generate_phrase). Keep the live comparison set lean.
# 2026-06-30 (PM-3): the hardened scorecard exposed two layout-quality gaps that
# survived the decode-bug fix — grid_coverage ~0.65 vs human 0.96 and dir_entropy
# ~0.60 vs 0.80: the model under-uses the 12 grid cells and the 9 cut directions.
# The sweep decodes layout GREEDILY (temp 0.0 → nucleus collapses to argmax), so
# those numbers are the model's *argmax* diversity. Two no-retrain levers, both on
# the production density config (dsel_g2.5): (a) stochastic decode (raise temp+top_p
# lets the tail through); (b) the env-gated frequency penalty (deterministic
# anti-repeat), now extended to the DIR role via LAYOUT_DIV_D so it can move
# dir_entropy, not just grid_coverage.
_DIV = {"LAYOUT_DIVERSITY": "1"}
# 2026-07-23: grid_cov/dir_entropy gaps CLOSED (temp 0.9/top_p 0.97 promoted to
# generate.py prod defaults; composite monotony 0.44 ≈ human 0.43). Remaining
# layout-quality residual is the HIDDEN sub-signal `pattern_repeat` (adjacent-
# identical (x,y,dir) tuples; human ≈ 0.002) now surfaced as its own column.
# The cumulative LAYOUT_DIV_* penalty over-flattens the whole distribution
# (div10 → grid 1.0 / rows 0.35, past human) so it's the wrong tool for adjacency.
# NEW lever = windowed ADJACENCY anti-repeat (LAYOUT_ANTIREPEAT window +
# LAYOUT_AR_STRENGTH): penalize only tokens seen in the last-W emissions per role,
# breaking back-to-back loops WITHOUT touching global cell/dir spread.
# 2026-07-23 PROMOTED: the sweep winner ar_w1_s2 (W=1/S=2.0) is now the baked-in
# LAYOUT_ANTIREPEAT/LAYOUT_AR_STRENGTH default in layout_model.py, so `prod` (the
# density-select-only config) now inherits anti-repeat and IS the new production
# control. `noar` (LAYOUT_ANTIREPEAT=0) preserves the pre-promotion baseline for
# regression; keep g2.5_div10 as the over-flatten failure-mode reference.
def _ar(w: str, s: str) -> dict[str, str]:
    return {**_DS25, "LAYOUT_ANTIREPEAT": w, "LAYOUT_AR_STRENGTH": s}
ARMS: dict[str, tuple[dict[str, str], list[str]]] = {
    "prod":        (_DS25, []),                                             # control = NEW PRODUCTION (W1/S2 baked default + temp 0.9/top_p 0.97)
    "noar":        ({**_DS25, "LAYOUT_ANTIREPEAT": "0"}, []),               # pre-promotion baseline (anti-repeat OFF) — regression reference
    "ar_w1_s2":    (_ar("1", "2.0"), []),                                   # promoted config, explicit (== prod default now)
    "ar_w2_s2":    (_ar("2", "2.0"), []),                                   # 2-step window, moderate
    "ar_w3_s3":    (_ar("3", "3.0"), []),                                   # 3-step window, stronger loop-break
    "g2.5_div10":  ({**_DS25, **_DIV, "LAYOUT_DIV_X": "1.0", "LAYOUT_DIV_Y": "1.0", "LAYOUT_DIV_D": "1.0"}, []),  # over-flatten reference
    # --- eval-suite v2 axis A1 (flow/ergonomics) levers, added 2026-07-27 ---
    # travel: our hands move ~50% further per second than human hands (flow
    # `travel` shift +2.48 human-MADs). Penalize long jumps in short windows.
    "tp1":         ({**_DS25, "LAYOUT_TRAVEL_PENALTY": "1.0"}, []),
    "tp2":         ({**_DS25, "LAYOUT_TRAVEL_PENALTY": "2.0"}, []),
    "tp4":         ({**_DS25, "LAYOUT_TRAVEL_PENALTY": "4.0"}, []),
    # crossover: enforce_color_separation moves EVERY wrong-side note, so we
    # measure crossover 0.000 vs a human median of 0.218. "extreme" keeps the
    # mild one-column crossovers the model chose; "off" is the ablation.
    "xsep_ext":    ({**_DS25, "COLOR_SEP_MODE": "extreme"}, []),
    "xsep_off":    ({**_DS25, "COLOR_SEP_MODE": "off"}, []),
    # do the two levers compose, or fight?
    "tp2_xsep":    ({**_DS25, "LAYOUT_TRAVEL_PENALTY": "2.0", "COLOR_SEP_MODE": "extreme"}, []),
    # --- eval-suite v2 axis A2 (rhythm) lever, added 2026-07-27 ---
    # Our hands fire simultaneously on 85.6% of beats (human 17.5%), which is what
    # makes the union rhythm metronomic. Soft-penalise the right hand on slots the
    # left hand already took. Single-song probe: il0.5 -> simultaneity 0.12 (human
    # 0.175) but little rhythm gain; il0.9 -> simultaneity 0.02 (too far) but
    # cond-entropy 0.49 and switch-rate 12.2, both near human (0.54 / 13.7). The
    # sweet spot is somewhere between, hence three strengths.
    "il5":         ({**_DS25, "BEAT_HAND_INTERLEAVE": "0.5"}, []),
    "il7":         ({**_DS25, "BEAT_HAND_INTERLEAVE": "0.7"}, []),
    "il9":         ({**_DS25, "BEAT_HAND_INTERLEAVE": "0.9"}, []),
    # --- eval-suite v2 axis A3 (idiom) lever, added 2026-07-27 ---
    # Boost cut directions that COMPLETE a known human idiom given this hand's
    # previous note. Single-song probe at strength 2.0: coverage 0.759 -> 0.946
    # (human 0.919), top50 0.337 -> 0.398 (human 0.386), viol still 0, travel
    # untouched. Slight overshoot past human suggests ~1.0-1.5 is the sweet spot.
    "ib1":         ({**_DS25, "LAYOUT_IDIOM_BONUS": "1.0"}, []),
    "ib2":         ({**_DS25, "LAYOUT_IDIOM_BONUS": "2.0"}, []),
    "ib3":         ({**_DS25, "LAYOUT_IDIOM_BONUS": "3.0"}, []),
    # rhythm lever + the best-guess flow levers, to check the axes do not fight
    "il7_tp1_xsep": ({**_DS25, "BEAT_HAND_INTERLEAVE": "0.7",
                      "LAYOUT_TRAVEL_PENALTY": "1.0", "COLOR_SEP_MODE": "extreme"}, []),
    # everything that looked good, together — the candidate next production config
    "combo":       ({**_DS25, "BEAT_HAND_INTERLEAVE": "0.7",
                     "LAYOUT_TRAVEL_PENALTY": "1.0", "COLOR_SEP_MODE": "extreme",
                     "LAYOUT_IDIOM_BONUS": "1.5"}, []),
}

sys.path.insert(0, str(REPO / "scripts"))
from eval_alignment import _separate_stems, _detect_onsets_librosa, _load_generated_beatmap, _beat_to_seconds  # noqa: E402
from eval_density_corr import _bin_counts, _spearman, _pearson  # noqa: E402

# Map-only quality axes (no Demucs): row/col spread, grid coverage, dir variety,
# monotony (the original complaint) + playability. Shared with the human-baseline
# command via scripts/map_metrics.py so every metric is computed identically.
try:
    from map_metrics import map_metrics, HUMAN_TARGET, BETTER  # noqa: E402
    from best_of_n_poc import swing_violations  # noqa: E402
    _HAVE_MAP = True
except Exception as _e:  # noqa: BLE001
    print(f"(map-quality axes unavailable: {_e})")
    _HAVE_MAP = False
    HUMAN_TARGET, BETTER = {}, {}


def _list_songs() -> list[pathlib.Path]:
    return sorted(p for p in SONGSET.glob("*") if p.suffix.lower() in (".ogg", ".mp3"))


def _ref_npz(song: pathlib.Path) -> pathlib.Path:
    return song.with_suffix(".ref.npz")


def _get_ref(song: pathlib.Path) -> tuple[np.ndarray, float]:
    """Reference onset times (drums∪other librosa) + duration; cached per song."""
    cache = _ref_npz(song)
    if cache.exists():
        d = np.load(cache)
        return d["ref_times"], float(d["duration"])
    import librosa
    dur = float(librosa.get_duration(path=str(song)))
    stems = _separate_stems(song, SR)
    drum_on = _detect_onsets_librosa(stems.get("drums", np.zeros(1)), SR)
    other_on = _detect_onsets_librosa(stems.get("other", np.zeros(1)), SR)
    ref_times = np.union1d(drum_on, other_on)
    np.savez(cache, ref_times=ref_times, duration=dur)
    return ref_times, dur


def build_songset(n: int) -> None:
    SONGSET.mkdir(parents=True, exist_ok=True)
    import librosa
    have = _list_songs()
    print(f"songset has {len(have)} songs; target {n}")
    raw = sorted((REPO / "data" / "raw").glob("*.zip"))
    for zp in raw:
        if len(_list_songs()) >= n:
            break
        name = zp.stem
        dst = SONGSET / f"{name}.ogg"
        if dst.exists():
            continue
        try:
            with zipfile.ZipFile(zp) as zf:
                egg = next((m for m in zf.namelist() if m.lower().endswith((".egg", ".ogg"))), None)
                if not egg:
                    continue
                data = zf.read(egg)
            dst.write_bytes(data)
            dur = float(librosa.get_duration(path=str(dst)))
            if dur < MIN_DUR:
                dst.unlink()
                continue
            print(f"  + {name}  dur={dur:.0f}s  — computing ref onsets …")
            _get_ref(dst)
        except Exception as e:
            print(f"  ! {name}: {e}")
            if dst.exists():
                dst.unlink()
    final = _list_songs()
    print(f"songset now {len(final)} songs: {[s.stem for s in final]}")


def _true_bpm(song: pathlib.Path) -> float | None:
    """BPM declared in the human map for this song, if we have it.

    Tempo detection is wrong on 30% of the eval set (7/23 songs; see
    scripts/bpm_octave_probe.py), including two at exactly half tempo, where the
    beat grid is twice as coarse in real time and the fast notes cannot be
    represented at all. Worse, the mis-tempo maps score BETTER on the beat-domain
    rhythm axis, so the confound actively distorts our measurements.

    Passing the human-declared BPM removes tempo detection as a confound from
    evaluation. This is an EVALUATION-ONLY fix — production has no human map to
    read a BPM from, and the detector itself still needs real work.
    """
    src = REPO / "data" / "raw" / f"{song.stem}.zip"
    if not src.exists():
        return None
    try:
        from feel_disc_poc import _zip_bpm
        b = _zip_bpm(str(src))
        return float(b) if b else None
    except Exception:  # noqa: BLE001
        return None


def _gen(arm: str, song: pathlib.Path, force: bool,
         true_bpm: bool = False) -> pathlib.Path | None:
    env_over, extra = ARMS[arm]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{arm}__{song.stem}.zip"
    if out.exists() and not force:
        return out
    if true_bpm:
        b = _true_bpm(song)
        if b:
            extra = [*extra, "--bpm", str(b)]
    env = dict(os.environ)
    env.update(env_over)
    cmd = [
        sys.executable, "scripts/generate.py", str(song), "--v7", "--difficulty", "Expert",
        "--beat-ckpt", BEAT_CKPT, "--layout-ckpt", LAYOUT_CKPT,
        # Decode at production generate.py defaults. Promoted 2026-07-23 to
        # temp 0.9/top_p 0.97 (closes grid_cov/dir_entropy vs human at h_dist 0.05).
        # Arms can still override via extra flags.
        "--section-gate", "loud_only", "--temperature", "0.9", "--top-p", "0.97",
        "--output", str(out), *extra,
    ]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print(f"  ! gen failed {arm}/{song.stem}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'rc='+str(r.returncode)}")
        return None
    return out


def _score(zip_path: pathlib.Path, ref_times: np.ndarray, duration: float) -> dict:
    notes, bpm = _load_generated_beatmap(zip_path, "Expert")
    gen_times = np.array(sorted(_beat_to_seconds(b, bpm) for b, _x, _c in notes), dtype=np.float64)
    dur = float(max(duration, gen_times.max() if len(gen_times) else 0.0))
    gen_d = _bin_counts(gen_times, dur, WIN_SEC)
    ref_d = _bin_counts(ref_times, dur, WIN_SEC)
    n = min(len(gen_d), len(ref_d))
    gen_d, ref_d = gen_d[:n], ref_d[:n]
    # onset-alignment proxy: fraction of generated notes within 50 ms of a real
    # reference onset (are notes placed on actual musical events?).
    onset_hit = None
    if len(ref_times) and len(gen_times):
        ref_sorted = np.sort(ref_times)
        idx = np.searchsorted(ref_sorted, gen_times).clip(1, len(ref_sorted) - 1)
        dl = np.abs(gen_times - ref_sorted[idx - 1])
        dr = np.abs(gen_times - ref_sorted[idx])
        onset_hit = float((np.minimum(dl, dr) <= 0.05).mean())
    rec = {
        "spearman": _spearman(gen_d, ref_d),
        "pearson": _pearson(gen_d, ref_d),
        "gen_cv": float(gen_d.std() / gen_d.mean()) if gen_d.mean() else 0.0,
        "n_windows": int(n),
        "onset_hit": onset_hit,
        "monotony": None, "row_conc": None, "viol": None,
    }
    if _HAVE_MAP:
        try:
            rec.update(map_metrics(zip_path, "Expert"))  # row_conc, col_conc, grid_coverage, dir_entropy, monotony, pattern_repeat, nps, n_notes
        except Exception:  # noqa: BLE001
            pass
        try:
            rec["viol"] = swing_violations(zip_path, "Expert")
        except Exception:  # noqa: BLE001
            pass
    # eval-suite v2 axis A1 — flow/ergonomics (sequence-aware). Reported per map
    # here; the ARM is ranked by the COHORT statistic (flow_gap), never by a mean
    # of per-map distances — see docs/eval_suite_v2.md §A1 lesson 1.
    try:
        rec.update(_flow_metrics_for(zip_path))
    except Exception:  # noqa: BLE001
        pass
    return rec


def _flow_metrics_for(zip_path: pathlib.Path) -> dict:
    from beatsaber_automapper.data.beatmap import ColorNote
    from beatsaber_automapper.evaluation import flow as _fl
    from eval_contour_follow import _load_notes_with_direction
    from feel_disc_poc import _zip_bpm

    recs = _load_notes_with_direction(zip_path, "Expert")
    if not recs:
        return {}
    notes = [ColorNote(beat=b, x=int(x), y=int(y), color=int(c), direction=int(d))
             for (b, x, y, c, d) in recs]

    class _BM:
        color_notes = notes
        bomb_notes: list = []

    return dict(_fl.flow_metrics(_BM(), bpm=float(_zip_bpm(str(zip_path)) or 120.0)).metrics)


def _load_human_baseline() -> None:
    """Refresh HUMAN_TARGET from a cached human-baseline run, if present."""
    f = CACHE / "human_baseline.json"
    if _HAVE_MAP and f.exists():
        try:
            for k, v in json.loads(f.read_text()).items():
                if isinstance(v, dict) and "mean" in v:
                    HUMAN_TARGET[k] = round(v["mean"], 3)
        except Exception:  # noqa: BLE001
            pass


def sweep(arms: list[str], force: bool, true_bpm: bool = False) -> None:
    songs = _list_songs()
    if not songs:
        print("no songs — run: eval_sweep.py build-songset --n 6")
        return
    _load_human_baseline()
    print(f"sweep: {len(arms)} arms × {len(songs)} songs\n")
    refs = {s: _get_ref(s) for s in songs}
    results: dict[str, dict[str, dict]] = {}
    import time as _time
    for ai, arm in enumerate(arms, 1):
        results[arm] = {}
        for si, s in enumerate(songs, 1):
            t0 = _time.time()
            zp = _gen(arm, s, force, true_bpm)
            if zp is None:
                continue
            try:
                rec = _score(zp, *refs[s])
                results[arm][s.stem] = rec
                print(f"  [{ai}/{len(arms)} {arm}] [{si}/{len(songs)} {s.stem[:14]}] "
                      f"row_conc={rec.get('row_conc')} spear={rec.get('spearman'):+.2f} "
                      f"viol={rec.get('viol')} ({_time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  ! score failed {arm}/{s.stem}: {e}")
        done = results[arm]
        sp = [v["spearman"] for v in done.values()]
        print(f"  [{arm}] scored {len(done)}/{len(songs)}  mean Spearman={np.mean(sp):+.3f}" if sp else f"  [{arm}] none scored")

    song_names = [s.stem for s in songs]
    print("\n=== density_corr Spearman (DoD >= 0.41) ===")
    hdr = "arm".ljust(12) + "".join(s[:10].rjust(11) for s in song_names) + "      mean   #pass"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for arm in arms:
        row = results[arm]
        cells, sp, npass = [], [], 0
        for s in song_names:
            if s in row and row[s].get("spearman") is not None:
                v = row[s]["spearman"]; sp.append(v); npass += int(v >= 0.41)
                cells.append(f"{v:+.3f}".rjust(11))
            else:
                cells.append("    --     ")
        mean = float(np.mean(sp)) if sp else float("nan")
        summary[arm] = {
            "mean_spearman": mean, "n_pass": npass, "n_scored": len(sp),
            "per_song": {s: results[arm].get(s) for s in song_names},
        }
        # aggregate every numeric map/audio metric as a mean over songs
        for k in ("row_conc", "col_conc", "grid_coverage", "dir_entropy", "monotony",
                  "pattern_repeat", "onset_hit", "gen_cv", "nps", "n_notes"):
            vals = [r[k] for r in row.values() if r.get(k) is not None]
            summary[arm][f"mean_{k}"] = float(np.mean(vals)) if vals else None
        viol = [r["viol"] for r in row.values() if r.get("viol") is not None]
        summary[arm]["total_viol"] = int(np.sum(viol)) if viol else None
        # composite human-distance: mean |arm - human| / human over the map-shape
        # metrics that have a human target. Lower = more human-like layout.
        dists = []
        for k in ("row_conc", "col_conc", "grid_coverage", "dir_entropy", "monotony"):
            mv, hv = summary[arm].get(f"mean_{k}"), HUMAN_TARGET.get(k)
            if mv is not None and hv:
                dists.append(abs(mv - hv) / abs(hv))
        summary[arm]["human_dist"] = round(float(np.mean(dists)), 3) if dists else None
        print("".join([arm.ljust(12)] + cells) + f"   {mean:+.3f}    {npass}/{len(sp)}")

    # quality-vs-human table — every metric with its human target + arrow
    cols = [  # (summary key, header, human-target key in HUMAN_TARGET)
        ("mean_row_conc", "row_conc", "row_conc"), ("mean_col_conc", "col_conc", "col_conc"),
        ("mean_grid_coverage", "grid_cov", "grid_coverage"), ("mean_dir_entropy", "dir_ent", "dir_entropy"),
        ("mean_monotony", "monoton", "monotony"), ("mean_pattern_repeat", "prep", "pattern_repeat"),
        ("mean_onset_hit", "onset_hit", None),
        ("mean_gen_cv", "gen_cv", None), ("mean_nps", "nps", None), ("total_viol", "viol", None),
        ("human_dist", "h_dist↓", None),  # composite layout distance to human (lower=better)
    ]
    def arrow(htk):
        return {"low": "↓", "high": "↑"}.get(BETTER.get(htk), "") if htk else ""
    print("\n=== quality vs human (mean over songs) ===")
    print("arm".ljust(12) + "".join(f"{h}{arrow(tk)}".rjust(10) for _k, h, tk in cols))
    tgt = "HUMAN".ljust(12) + "".join(
        (f"{HUMAN_TARGET[tk]:.2f}" if tk and HUMAN_TARGET.get(tk) is not None else "·").rjust(10)
        for _k, _h, tk in cols)
    print(tgt); print("-" * (12 + 10 * len(cols)))
    for arm in arms:
        s = summary[arm]
        def _f(k, fmt="{:.3f}"):
            return (fmt.format(s[k]) if s.get(k) is not None else "--").rjust(10)
        print(arm.ljust(12) + "".join(
            _f(k, "{:.0f}" if k in ("total_viol",) else "{:.2f}") for k, _h, _tk in cols))

    # ---- v2 axis A1: flow/ergonomics, ranked by the COHORT statistic ----
    try:
        from beatsaber_automapper.evaluation import flow as _fl
        fcols = _fl.SEQUENCE_KEYS
        print("\n=== flow / ergonomics (v2 axis A1) — shift = median offset in human MADs ===")
        print("rank arms by flow_gap (mean |shift|); spread <1 = under-dispersed vs human")
        print("arm".ljust(12) + "".join(f"{k:>20s}" for k in fcols)
              + "crossover".rjust(11) + "flow_gap".rjust(10) + "min_spr".rjust(9))
        for arm in arms:
            rows = [r for r in results[arm].values() if r]
            cc = _fl.cohort_comparison(rows)
            if "_summary" not in cc:
                continue
            cells = "".join(
                f"{cc[k]['shift']:+9.2f}/{cc[k]['spread']:<10.2f}" if k in cc
                else f"{'--':>20s}" for k in fcols)
            xo = cc.get("crossover", {}).get("median")
            s = cc["_summary"]
            print(arm.ljust(12) + cells
                  + (f"{xo:11.3f}" if xo is not None else f"{'--':>11s}")
                  + f"{s['flow_gap']:10.2f}{s['min_spread']:9.2f}")
        print(f"{'HUMAN':12s}" + "".join(f"{'+0.00/1.00':>20s}" for _ in fcols)
              + f"{0.218:11.3f}{0.0:10.2f}{1.0:9.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"(flow axis unavailable: {e})")

    # ---- v2 axis A2: rhythm, also ranked by the cohort statistic ----
    try:
        from beatsaber_automapper.evaluation import rhythm as _rh
        print("\n=== rhythm (v2 axis A2) — our largest measured gap ===")
        print("arm".ljust(12) + "".join(f"{k:>20s}" for k in _rh.SEQUENCE_KEYS)
              + "dom_share".rjust(11) + "rhy_gap".rjust(9) + "min_spr".rjust(9))
        for arm in arms:
            rows = [r for r in results[arm].values() if r]
            cc = _rh.cohort_comparison(rows)
            if "_summary" not in cc:
                continue
            cells = "".join(
                f"{cc[k]['shift']:+9.2f}/{cc[k]['spread']:<10.2f}" if k in cc
                else f"{'--':>20s}" for k in _rh.SEQUENCE_KEYS)
            ds = cc.get("dominant_share", {}).get("median")
            s = cc["_summary"]
            print(arm.ljust(12) + cells
                  + (f"{ds:11.3f}" if ds is not None else f"{'--':>11s}")
                  + f"{s['rhythm_gap']:9.2f}{s['min_spread']:9.2f}")
        print(f"{'HUMAN':12s}" + "".join(f"{'+0.00/1.00':>20s}" for _ in _rh.SEQUENCE_KEYS)
              + f"{0.509:11.3f}{0.0:9.2f}{1.0:9.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"(rhythm axis unavailable: {e})")

    out = CACHE / "leaderboard.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    _write_report(summary, results, song_names, arms, cols)
    return summary, results, song_names


def _render(arm: str, song: str) -> str | None:
    """Render an arm's cached map for `song`; return a repo-relative png path."""
    zp = CACHE / f"{arm}__{song}.zip"
    if not zp.exists():
        return None
    rdir = CACHE / "renders"; rdir.mkdir(exist_ok=True)
    png = rdir / f"{arm}__{song}.png"
    if not png.exists():
        r = subprocess.run(
            [sys.executable, "scripts/render_map.py", str(zp), "--difficulty", "Expert",
             "--out", str(png), "--no-audio"], cwd=REPO, capture_output=True, text=True)
        if not png.exists():
            return None
    return str(png.relative_to(CACHE))


def _write_report(summary, results, song_names, arms, cols) -> None:
    """Emit a single self-contained report.md: tables vs human + embedded renders
    of the headline arm vs the control, so a sweep is judged at a glance."""
    import datetime
    lines = [f"# Eval sweep report — {datetime.datetime.now():%Y-%m-%d %H:%M}",
             f"\n{len(arms)} arms × {len(song_names)} songs. DoD: density_corr Spearman ≥ 0.41.\n"]
    # density_corr table
    lines.append("## density_corr (Spearman, DoD ≥ 0.41)\n")
    lines.append("| arm | " + " | ".join(s[:10] for s in song_names) + " | mean | #pass |")
    lines.append("|" + "---|" * (len(song_names) + 3))
    for arm in arms:
        per = summary[arm]["per_song"]
        cellvals = [(f"{per[s]['spearman']:+.2f}" if per.get(s) and per[s].get("spearman") is not None else "—") for s in song_names]
        lines.append(f"| {arm} | " + " | ".join(cellvals)
                     + f" | **{summary[arm]['mean_spearman']:+.3f}** | {summary[arm]['n_pass']}/{summary[arm]['n_scored']} |")
    # quality vs human table
    lines.append("\n## quality vs human (mean over songs)\n")
    hdr = [h for _k, h, _tk in cols]
    lines.append("| arm | " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    humanrow = [(f"{HUMAN_TARGET[tk]:.2f}" if tk and HUMAN_TARGET.get(tk) is not None else "·") for _k, _h, tk in cols]
    lines.append("| **HUMAN** | " + " | ".join(humanrow) + " |")
    for arm in arms:
        s = summary[arm]
        vals = [(f"{s[k]:.0f}" if k == "total_viol" and s.get(k) is not None
                 else f"{s[k]:.2f}" if s.get(k) is not None else "—") for k, _h, _tk in cols]
        lines.append(f"| {arm} | " + " | ".join(vals) + " |")
    # headline arm (best mean_spearman) vs first/control arm, rendered
    headline = max(arms, key=lambda a: summary[a]["mean_spearman"] if not np.isnan(summary[a]["mean_spearman"]) else -9)
    ctrl = "control" if "control" in arms else arms[0]
    song0 = song_names[0]
    lines.append(f"\n## renders — {song0}\n")
    for label, ra in (("control", ctrl), ("headline (best density_corr)", headline)):
        p = _render(ra, song0)
        lines.append(f"**{label}** (`{ra}`)\n" + (f"\n![{ra}]({p})\n" if p else "\n_(render unavailable)_\n"))
    (CACHE / "report.md").write_text("\n".join(lines))
    print(f"wrote {CACHE / 'report.md'}")


def human_baseline(n: int) -> dict:
    """Compute map-only metric distributions over n human maps from data/raw.

    Writes outputs/eval_sweep_cache/human_baseline.json and prints mean/p10/p90 so
    every metric in the leaderboard has a real human reference (not a hard-coded
    guess). Refreshes map_metrics.HUMAN_TARGET for this process.
    """
    if not _HAVE_MAP:
        print("map_metrics unavailable"); return {}
    raw = sorted((REPO / "data" / "raw").glob("*.zip"))
    rows: list[dict] = []
    for zp in raw:
        if len(rows) >= n:
            break
        for diff in ("Expert", "ExpertPlus", "Hard"):
            try:
                m = map_metrics(zp, diff)
                if m.get("n_notes", 0) > 20:
                    rows.append(m); break
            except Exception:  # noqa: BLE001
                continue
    if not rows:
        print("no human maps scored"); return {}
    keys = ("row_conc", "col_conc", "grid_coverage", "dir_entropy", "monotony",
            "pattern_repeat", "nps", "n_notes")
    base = {}
    print(f"\n=== HUMAN baseline (n={len(rows)} maps) ===")
    print("metric".ljust(16) + "mean".rjust(9) + "p10".rjust(9) + "p90".rjust(9))
    for k in keys:
        vals = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        if not len(vals):
            continue
        base[k] = {"mean": float(vals.mean()), "p10": float(np.percentile(vals, 10)),
                   "p90": float(np.percentile(vals, 90))}
        HUMAN_TARGET[k] = round(float(vals.mean()), 3)
        print(f"{k:16s}{vals.mean():9.3f}{np.percentile(vals,10):9.3f}{np.percentile(vals,90):9.3f}")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "human_baseline.json").write_text(json.dumps(base, indent=2))
    print(f"\nwrote {CACHE / 'human_baseline.json'}")
    return base


def main() -> None:
    try:  # line-buffer stdout so nohup/background sweeps show live progress
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-songset"); b.add_argument("--n", type=int, default=6)
    sw = sub.add_parser("sweep")
    sw.add_argument("--arms", default=None, help="comma list; default all")
    sw.add_argument("--force", action="store_true", help="regenerate even if cached")
    sw.add_argument("--true-bpm", action="store_true",
                    help="generate with the human map's declared BPM. Tempo detection is "
                         "wrong on 30%% of the eval set and the beat-domain rhythm axis "
                         "REWARDS the error, so this removes a real confound. Evaluation "
                         "only -- production has no human BPM to read.")
    hb = sub.add_parser("human-baseline"); hb.add_argument("--n", type=int, default=40)
    sub.add_parser("list-arms")
    a = ap.parse_args()
    if a.cmd == "build-songset":
        build_songset(a.n)
    elif a.cmd == "human-baseline":
        human_baseline(a.n)
    elif a.cmd == "list-arms":
        for k, (e, x) in ARMS.items():
            print(f"  {k:14s} env={e} flags={x}")
    elif a.cmd == "sweep":
        arms = a.arms.split(",") if a.arms else list(ARMS)
        bad = [x for x in arms if x not in ARMS]
        if bad:
            sys.exit(f"unknown arms: {bad}")
        sweep(arms, a.force)


if __name__ == "__main__":
    main()
