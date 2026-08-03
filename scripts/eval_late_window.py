#!/usr/bin/env python
"""Late-song / final-chorus collapse diagnostic.

The last untouched original complaint (after drop-@-13s fixed via section_gate and
flat-density fixed via density-select): generated maps thin out / die in the final
section while the song (final chorus) is still energetic — the "late-song collapse".

There was no metric for it. This one is deliberately simple and human-relative:
per song we compare the GENERATED note-time distribution against the HUMAN reference
onset distribution over the LATE tail of the song. A model that collapses late puts a
smaller share of its notes in the tail than the human map does.

Metrics (per song, then mean over songs):
  ref_late_frac  = fraction of reference onsets in the final `--tail` of the song
  gen_late_frac  = fraction of generated notes  in the final `--tail`
  late_gap       = ref_late_frac - gen_late_frac   (POSITIVE => gen under-produces
                   the tail relative to human = collapse; ~0 => tracks human)
  late_corr      = Spearman(gen_density, ref_density) computed ONLY over tail windows
                   (is late density still tracking the song, or flat/dead?)

DoD proposal for the fix: mean late_gap <= 0.03 (gen tail-share within 3 pts of human)
AND late_corr >= 0.30, while whole-song density_corr and parity/monotony hold.

Usage:
  python scripts/eval_late_window.py --arm prod            # score cached maps for an arm
  python scripts/eval_late_window.py --map path/to.zip --ref data/eval_songset/1f333.ref.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import shutil
import sys
import tempfile
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from eval_alignment import _load_generated_beatmap, _beat_to_seconds  # noqa: E402
from eval_density_corr import _bin_counts, _spearman  # noqa: E402

CACHE = REPO / "outputs" / "eval_sweep_cache"
SONGSET = REPO / "data" / "eval_songset"
RAW = REPO / "data" / "raw"
WIN_SEC = 2.0


def _gen_times(zip_path: str) -> np.ndarray:
    notes, bpm = _load_generated_beatmap(pathlib.Path(zip_path), "Expert")
    return np.array(sorted(_beat_to_seconds(b, bpm) for b, _x, _c in notes), dtype=np.float64)


def _pick_human_diff(names: list[str], difficulty: str) -> tuple[str | None, str]:
    """Pick a Standard-characteristic difficulty .dat, preferring `difficulty`.

    Standard only — Lawless/OneSaber/90-360 variants have different note
    conventions (same lesson the swing simulator learned). Many maps ship only
    ExpertPlus, so we fall back to it: `late_frac` is a *normalized share*, so the
    tail proportion is comparable across difficulties even though absolute
    density is not. The chosen difficulty is returned so the report can say so.
    """
    std = [n for n in names if n.lower().split("/")[-1].endswith("standard.dat")]
    for cand in (difficulty, "ExpertPlus", "Expert", "Hard"):
        c = cand.lower()
        for n in std:
            b = n.lower().split("/")[-1]
            if b.startswith(c) and "plus" not in b.replace(c, "", 1):
                return n, cand
    return None, ""


def _human_times(stem: str, difficulty: str = "Expert") -> tuple[np.ndarray | None, str]:
    """Note times (s) of the HUMAN map for this song, from data/raw/<stem>.zip.

    The audio-onset reference tells us what the *song* does; this tells us what a
    human mapper actually did in the tail — the direct comparison the collapse
    complaint is about. Only the two small .dat files are extracted (never the
    audio) so this stays cheap over a large songset.
    """
    src = RAW / f"{stem}.zip"
    if not src.exists():
        return None, ""
    from beatsaber_automapper.data.beatmap import parse_info_dat, parse_difficulty_dat

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="late_window_human_"))
    try:
        with zipfile.ZipFile(src) as zf:
            names = zf.namelist()
            # EXACT basename: "BPMInfo.dat" also ends with "info.dat", and 73 of 300
            # corpus zips list it FIRST -- picking it makes parse_info_dat find no
            # bpm and silently fall back to 120, which stretches every note time.
            info = next((n for n in names
                         if n.split("/")[-1].lower() == "info.dat"), None)
            diff, used = _pick_human_diff(names, difficulty)
            if info is None or diff is None:
                return None, ""
            for n in (info, diff):
                (tmp / pathlib.Path(n).name).write_bytes(zf.read(n))
        meta = parse_info_dat(tmp / pathlib.Path(info).name)
        beatmap = parse_difficulty_dat(tmp / pathlib.Path(diff).name)
        if meta is None or beatmap is None:
            return None, ""
        bpm = float(meta.bpm)
        times = np.array(sorted(_beat_to_seconds(n.beat, bpm) for n in beatmap.color_notes),
                         dtype=np.float64)
        return times, used
    except Exception:  # noqa: BLE001
        return None, ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _late_stats(gen_times: np.ndarray, ref_times: np.ndarray, duration: float,
                tail: float, human_times: np.ndarray | None = None) -> dict:
    dur = float(max(duration, gen_times.max() if len(gen_times) else 0.0,
                    ref_times.max() if len(ref_times) else 0.0))
    cut = (1.0 - tail) * dur
    ref_late = float((ref_times >= cut).mean()) if len(ref_times) else 0.0
    gen_late = float((gen_times >= cut).mean()) if len(gen_times) else 0.0
    # tail-only density correlation
    gen_d = _bin_counts(gen_times, dur, WIN_SEC)
    ref_d = _bin_counts(ref_times, dur, WIN_SEC)
    n = min(len(gen_d), len(ref_d))
    gen_d, ref_d = gen_d[:n], ref_d[:n]
    tail_start = int((1.0 - tail) * n)
    late_corr = _spearman(gen_d[tail_start:], ref_d[tail_start:]) if n - tail_start >= 3 else float("nan")
    out = {
        "ref_late_frac": ref_late,
        "gen_late_frac": gen_late,
        "late_gap": ref_late - gen_late,
        "late_corr": late_corr,
        "dur": dur,
    }
    if human_times is not None and len(human_times):
        hum_late = float((human_times >= cut).mean())
        out["human_late_frac"] = hum_late
        # The complaint's direct form: did the HUMAN mapper keep the tail busy
        # while we thinned out? POSITIVE => we under-produce vs the human map.
        out["human_gap"] = hum_late - gen_late
    return out


def _ref_for(stem: str) -> pathlib.Path | None:
    f = SONGSET / f"{stem}.ref.npz"
    return f if f.exists() else None


def score_arm(arm: str, tail: float, worst: int = 5) -> None:
    maps = sorted(glob.glob(str(CACHE / f"{arm}__*.zip")))
    if not maps:
        print(f"no cached maps for arm '{arm}' under {CACHE}")
        return
    rows = []
    print(f"=== late-window collapse: arm '{arm}', tail=final {tail:.0%} ===")
    print(f"{'song':26s} {'ref_late':>8s} {'gen_late':>8s} {'late_gap':>8s} "
          f"{'hum_late':>8s} {'hum_gap':>8s} {'late_corr':>9s}")
    for m in maps:
        stem = os.path.basename(m)[len(arm) + 2:-4]  # strip 'arm__' ... '.zip'
        ref = _ref_for(stem)
        if ref is None:
            continue
        d = np.load(ref)
        hum, hum_diff = _human_times(stem)
        s = _late_stats(_gen_times(m), d["ref_times"], float(d["duration"]), tail,
                        human_times=hum)
        s["song"] = stem
        s["human_diff"] = hum_diff
        rows.append(s)
        lc = f"{s['late_corr']:+.2f}" if s["late_corr"] == s["late_corr"] else "  n/a"
        hl = f"{s['human_late_frac']:8.3f}" if "human_late_frac" in s else f"{'—':>8s}"
        hg = f"{s['human_gap']:+8.3f}" if "human_gap" in s else f"{'—':>8s}"
        print(f"{stem:26s} {s['ref_late_frac']:8.3f} {s['gen_late_frac']:8.3f} "
              f"{s['late_gap']:+8.3f} {hl} {hg} {lc:>9s}")
    if not rows:
        return
    mg = float(np.mean([r["late_gap"] for r in rows]))
    hgaps = [r["human_gap"] for r in rows if "human_gap" in r]
    mh = float(np.mean(hgaps)) if hgaps else float("nan")
    corrs = [r["late_corr"] for r in rows if r["late_corr"] == r["late_corr"]]
    mc = float(np.mean(corrs)) if corrs else float("nan")
    mhs = f"{mh:+8.3f}" if hgaps else f"{'—':>8s}"
    print("-" * 80)
    print(f"{'MEAN (n=' + str(len(rows)) + ')':26s} {'':8s} {'':8s} {mg:+8.3f} "
          f"{'':8s} {mhs} {mc:+9.2f}")

    # Population view: a mean can hide a minority of songs that DO collapse —
    # which is exactly the open caveat (does the set contain a collapse song?).
    bad = sorted((r for r in rows if r["late_gap"] > 0.10), key=lambda r: -r["late_gap"])
    bad_h = sorted((r for r in rows if r.get("human_gap", 0.0) > 0.10),
                   key=lambda r: -r["human_gap"])
    frac_bad = len(bad) / len(rows)
    print(f"\nsongs with late_gap > 0.10 (audio-ref): {len(bad)}/{len(rows)} ({frac_bad:.0%})")
    for r in bad[:worst]:
        print(f"    {r['song']:26s} late_gap {r['late_gap']:+.3f}")
    if hgaps:
        print(f"songs with human_gap > 0.10 (vs human map): {len(bad_h)}/{len(hgaps)} "
              f"({len(bad_h) / len(hgaps):.0%})")
        for r in bad_h[:worst]:
            print(f"    {r['song']:26s} human_gap {r['human_gap']:+.3f}")
        diffs = sorted({r["human_diff"] for r in rows if r.get("human_diff")})
        print(f"    (human reference difficulties used: {', '.join(diffs)}; "
              f"tail share is normalized so cross-difficulty is comparable)")

    print(f"\nverdict: mean late_gap {mg:+.3f} "
          f"({'COLLAPSE (gen under-produces tail)' if mg > 0.03 else 'tracks song'})")
    if hgaps:
        print(f"         mean human_gap {mh:+.3f} "
              f"({'COLLAPSE vs human map' if mh > 0.03 else 'tracks human map'})")
    print(f"         mean late_corr {mc:+.2f} "
          f"({'tail density tracks song' if mc >= 0.30 else 'tail density weak/flat'})")
    closed = mg <= 0.03 and (not hgaps or mh <= 0.03) and frac_bad <= 0.10
    print(f"         population DoD (mean gaps <= 0.03 AND <=10% of songs > 0.10): "
          f"{'MET — late-collapse CLOSED' if closed else 'NOT met — see outlier songs above'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Late-song collapse diagnostic.")
    ap.add_argument("--arm", help="score all cached maps for this eval_sweep arm")
    ap.add_argument("--map", help="single generated map zip")
    ap.add_argument("--ref", help="ref .npz for --map")
    ap.add_argument("--tail", type=float, default=0.20, help="tail fraction of the song (default 0.20)")
    ap.add_argument("--worst", type=int, default=5, help="how many outlier songs to list")
    ap.add_argument("--human-stem", help="with --map: song stem in data/raw for the human comparison")
    a = ap.parse_args()
    if a.arm:
        score_arm(a.arm, a.tail, a.worst)
    elif a.map and a.ref:
        d = np.load(a.ref)
        hum = _human_times(a.human_stem)[0] if a.human_stem else None
        s = _late_stats(_gen_times(a.map), d["ref_times"], float(d["duration"]), a.tail,
                        human_times=hum)
        for k, v in s.items():
            print(f"{k:14s} {v:+.4f}")
    else:
        ap.error("give --arm, or --map with --ref")


if __name__ == "__main__":
    main()
