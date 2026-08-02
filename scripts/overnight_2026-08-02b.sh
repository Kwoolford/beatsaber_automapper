#!/usr/bin/env bash
# IS THE OFF-BEAT DEFECT JUST A WRONG TEMPO? (single-song evidence says yes)
#
# Chain of findings, 2026-08-01 -> 2026-08-02:
#   1. Kyle played the first two maps to pass all five v2 axes: "painfully obvious
#      the notes are off beat".
#   2. Axis A8 (new) measured notes against the AUDIO for the first time. Our arms
#      score alignment_gap 5.0-7.2 against a bar of 0.39 and a held-out human
#      cohort at 0.20 -- WORSE than a metronome (2.37), and `b1_e17_ds055` (7.21)
#      scores worse than a map whose note times were replaced with RANDOM ones
#      (6.74).
#   3. Our detected bpm is exact on 1 of 21 eval songs. Median error 0.74%; four
#      songs land at 2/3 of the true tempo.
#   4. Human maps sit on the SAME 1/4-beat slot grid we do (557 of 561 notes on
#      1f767). The grid is not too coarse -- it is in the wrong PLACE, and with a
#      0.94% tempo error it slides through every phase as the song plays.
#
# SINGLE-SONG PROBE (1f767, already run) -- hand the generator the true bpm:
#       map                     bpm    prec   mad_ms   lag_ms
#       HUMAN                 160.0   0.968     8.7    +10.5
#       ds055 (detected)      161.5   0.803    11.7     +0.0
#       ds055 + ORACLE bpm    160.0   0.899     8.5     +9.9
# A 1.5 bpm error was costing ~10 points of onset precision. With the true tempo
# our SCATTER beats the human's and our LAG matches it.
#
# ★ THE 1f333 TRAP APPLIES HERE: that single-song probe is exactly the kind of
# evidence this project has been burned by before (TODO: "the 1f333 SINGLE-SONG
# PROBE TRAP"). Hence this sweep across all 24 songs before anything is believed.
#
# ARMS (oracle bpm from the human map's Info.dat, via BEAT_BPM_ORACLE)
#   obpm_ds055        vs cached control ds055
#   obpm_hl014_ds055  vs cached control hl014_ds055
#   obpm_prod         vs cached control prod
#
# VERDICT LOGIC
#   precision rises toward 0.90+ across the cohort -> THE DEFECT IS TEMPO
#       ESTIMATION. Not a modelling problem: a beat tracker that returns phase, or
#       fitting the grid to the onsets we ALREADY compute for A8, would close it.
#       Next build item is a real tempo+phase estimator, and every axis measured
#       before this becomes suspect (a wrong grid moves rhythm and handrole too).
#   precision rises only on the songs whose detected tempo was already close ->
#       partial: tempo is necessary but not sufficient; grid PHASE is the next
#       suspect (detect_bpm discards librosa's beat positions and the grid is
#       anchored at t=0).
#   precision flat -> the single-song probe was the trap again. Say so plainly,
#       and move alignment work to Stage-1 slot selection.
#
# NOTE: BEAT_BPM_ORACLE IS A DIAGNOSTIC AND CANNOT SHIP -- production has no human
# map to read a tempo from. It is here to settle attribution, nothing else.

set -uo pipefail
cd /home/kyle/repos/beatsaber_automapper
source .venv/bin/activate

LOG=logs/overnight/oraclebpm_2026-08-02.log
mkdir -p logs/overnight
exec > >(tee -a "$LOG") 2>&1

echo "=== ORACLE-BPM SWEEP START $(date -Is) ==="

ARMS="obpm_ds055,obpm_hl014_ds055,obpm_prod"
python scripts/eval_sweep.py sweep --arms "$ARMS"
echo "=== SWEEP DONE $(date -Is) ==="

echo "=== SIX-AXIS SCORECARDS (A8 included) ==="
for arm in ds055 hl014_ds055 prod $(echo "$ARMS" | tr ',' ' '); do
  zips=(outputs/eval_sweep_cache/${arm}__*.zip)
  if [ ! -e "${zips[0]}" ]; then echo "-- $arm: NO CACHED MAPS, skipping"; continue; fi
  echo "-- $arm (${#zips[@]} maps)"
  python -m beatsaber_automapper.evaluation.scorecard "${zips[@]}" --label "$arm" || true
done

echo "=== ORACLE-BPM VERDICT $(date -Is) ==="
python - <<'PY'
import json, pathlib, statistics, sys
REPO = pathlib.Path(".").resolve(); sys.path.insert(0, str(REPO / "src"))
from beatsaber_automapper.evaluation import alignment, scorecard
CACHE = REPO / "outputs" / "eval_sweep_cache"
TRUE_BPM = json.loads((REPO / "outputs" / "true_bpm_eval_songset.json").read_text())
HUMAN_PREC, HUMAN_MAD = 0.930, 10.35
PAIRS = [("ds055", "obpm_ds055"), ("hl014_ds055", "obpm_hl014_ds055"),
         ("prod", "obpm_prod")]

def cohort(arm):
    zips = sorted(CACHE.glob(f"{arm}__*.zip"))
    out = {}
    for p in zips:
        try:
            r = scorecard._load_any(p)
        except Exception:
            r = None
        if not r or r[2] is None:
            continue
        bm, bpm, on = r
        out[p.stem.split("__")[-1]] = (alignment.alignment_metrics(bm, bpm=bpm, onsets=on).metrics, bpm)
    return out

def axes(arm):
    zips = sorted(CACHE.glob(f"{arm}__*.zip"))
    loaded = []
    for p in zips:
        try:
            r = scorecard._load_any(p)
        except Exception:
            r = None
        if r:
            loaded.append(r)
    if len(loaded) < 20:
        return None
    res = scorecard.score_cohort(loaded, arm)
    return {a.name: (a.gap, a.min_spread, a.passed) for a in res["axes"]}, res["total_viol"]

print(f"\n{'arm':22s}{'prec':>8s}{'mad_ms':>9s}{'align':>8s}{'rhythm':>8s}"
      f"{'flow':>7s}{'idiom':>7s}{'hrole':>7s}{'pfeel':>7s}{'pass':>7s}")
print("-" * 96)
rows = {}
for base, orc in PAIRS:
    for arm in (base, orc):
        if arm in rows:
            continue
        c = cohort(arm)
        ax = axes(arm)
        if not c or not ax:
            print(f"{arm:22s}  not enough maps"); continue
        a, viol = ax
        prec = statistics.median(m["onset_precision"] for m, _ in c.values())
        mad = statistics.median(m["offset_mad_ms"] for m, _ in c.values())
        rows[arm] = {"prec": prec, "mad": mad, "axes": a, "viol": viol, "cohort": c}
        g = lambda k: a[k][0] if k in a else float("nan")
        npass = sum(1 for v in a.values() if v[2])
        print(f"{arm:22s}{prec:8.3f}{mad:9.1f}{g('alignment'):8.2f}{g('rhythm'):8.2f}"
              f"{g('flow'):7.2f}{g('idiom'):7.2f}{g('handrole'):7.2f}{g('playfeel'):7.2f}"
              f"{npass:6d}/6")
print(f"\nhuman reference: precision {HUMAN_PREC:.3f}, offset_mad {HUMAN_MAD:.1f}ms")

print("\n--- PER-SONG, DOES THE GAIN TRACK THE TEMPO ERROR? ---")
print("(if it does, the mechanism is confirmed and not a coincidence)")
print(f"{'song':10s}{'true':>8s}{'ours':>8s}{'err%':>8s}{'prec_det':>10s}"
      f"{'prec_orc':>10s}{'gain':>8s}")
gains = []
base, orc = "ds055", "obpm_ds055"
if base in rows and orc in rows:
    for sid in sorted(rows[base]["cohort"]):
        if sid not in rows[orc]["cohort"]:
            continue
        (md, bd), (mo, bo) = rows[base]["cohort"][sid], rows[orc]["cohort"][sid]
        tb = TRUE_BPM.get(sid)
        err = (bd - tb) / tb * 100 if tb else float("nan")
        gain = mo["onset_precision"] - md["onset_precision"]
        gains.append((abs(err) if err == err else 0.0, gain))
        print(f"{sid[:9]:10s}{tb or 0:8.1f}{bd:8.1f}{err:+8.2f}"
              f"{md['onset_precision']:10.3f}{mo['onset_precision']:10.3f}{gain:+8.3f}")
    if len(gains) >= 8:
        big = [g for e, g in gains if e >= 0.5]
        small = [g for e, g in gains if e < 0.5]
        print(f"\n  songs with tempo error >=0.5%: n={len(big)} mean precision gain "
              f"{statistics.fmean(big):+.3f}" if big else "\n  no songs with error >=0.5%")
        if small:
            print(f"  songs with tempo error <0.5% : n={len(small)} mean precision gain "
                  f"{statistics.fmean(small):+.3f}")
        print("  => the gain should be LARGE where the tempo was wrong and ~0 where it")
        print("     was already right. If it is uniform, something else changed too.")

print("\n--- VERDICT ---")
if base in rows and orc in rows:
    d = rows[orc]["prec"] - rows[base]["prec"]
    dm = rows[base]["mad"] - rows[orc]["mad"]
    print(f"  precision {rows[base]['prec']:.3f} -> {rows[orc]['prec']:.3f} ({d:+.3f})")
    print(f"  scatter   {rows[base]['mad']:.1f}ms -> {rows[orc]['mad']:.1f}ms ({dm:+.1f}ms better)")
    if d >= 0.08:
        print("  => THE DEFECT IS TEMPO ESTIMATION. Build a real tempo+phase estimator;")
        print("     it is the highest-value item in the project. Note this also means")
        print("     every beat-domain axis measured to date was scored on a wrong grid.")
    elif d >= 0.03:
        print("  => tempo is PART of it. Necessary, not sufficient — grid phase next.")
    else:
        print("  => the single-song probe was the 1f333 trap again. Tempo is not the")
        print("     binding constraint; move to Stage-1 slot selection.")
PY

echo "=== COMPLETE $(date -Is) ==="
