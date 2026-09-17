#!/usr/bin/env python
"""P6 — every style preset and request lever, measured on the REAL best-build chain (2026-09-17).

`sweep_style.py` predates the best build (lead-bias 0.3, no lead-in / drop-orphan / carrier-bias,
no repeat.py / answer.py, no verdict page). This builds each arm exactly as a shipped map is built:

    autobuild <audio> --pulse --lead-bias 0.2 --lead-in --drop-orphan --carrier-bias 2.0 [ARM]
    → repeat.py → answer.py → verdict.py --json + mapjudge

and records, per song × arm: the judge's 23 metrics (value + human pct), the verdict page (reds,
SHIP, which codes), SCATTER room, double share and wall count. `--report` prints, per lever, the
column it claims to move (low vs high, per song) and what it cost on the page.

⚠️An arm's flags REPLACE the base flag of the same name (e.g. `--lead-bias`), never duplicate it.

Run:  python scripts/p6_levers.py            # build + measure (resumable)
      python scripts/p6_levers.py --report
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
AM = REPO / "agent_mapper"
OUT = REPO / "outputs" / "p6_levers_2026-09-17"
SONGS = ("1f333", "1f767", "1f8d6", "1f913")
BASE = ["--pulse", "--lead-bias", "0.2", "--lead-in", "--drop-orphan", "--carrier-bias", "2.0"]
AUDIO_DIR = REPO / "data" / "eval_songset"
SEED = "0"   # BEST2 (outputs/best_2026-09-13b) is seed 0 — verified byte-identical 2026-09-17

# arm name -> extra flags. Pairs named <lever>_lo / <lever>_hi are read as a lever.
ARMS: dict[str, list[str]] = {
    "base": [],
    "style_calm": ["--style", "calm"],
    "style_human": ["--style", "human"],
    "style_dense": ["--style", "dense"],
    "style_flowing": ["--style", "flowing"],
    "style_technical": ["--style", "technical"],
    "doubles_lo": ["--doubles-rate", "0.1"],
    "doubles_hi": ["--doubles-rate", "0.6"],
    "lead_lo": ["--lead-bias", "0.0"],
    "lead_hi": ["--lead-bias", "0.5"],
    "taper_lo": ["--taper", "0.0"],
    "taper_hi": ["--taper", "0.5"],
    "walls_lo": ["--walls", "40"],
    "walls_hi": ["--walls", "160"],
    "width_lo": ["--width", "1"],
    "width_hi": ["--width", "8"],
    "palette_lo": ["--palette", "0"],
    "palette_hi": ["--palette", "20"],
    "memory_lo": ["--map-memory", "0"],
    "memory_hi": ["--map-memory", "4"],
    "vocals_lo": ["--carrier-bias", "1.0"],
    "vocals_hi": ["--carrier-bias", "4.0"],
    "nps_lo": ["--nps", "3.0"],
    "nps_hi": ["--nps", "5.0"],
    # 2026-09-17: the taper costs 1f8d6 its LAST lead-hand passage (ABS:lead_runs, a zero-margin
    # pass since 09-13). Give the map runs of its own and ask whether the taper is then free.
    "runs": ["--hand-run-p", "0.06"],
    "taper_runs": ["--taper", "0.5", "--hand-run-p", "0.06"],
    # 2026-09-17c: SCATTER is the commonest red left under the new best build (8 songs). Is a
    # vocabulary lever a usable TRIAGE tool when the page says SCATTER?
    "tr_memory": ["--taper", "0.5", "--hand-run-p", "0.06", "--map-memory", "4"],
    "tr_palette": ["--taper", "0.5", "--hand-run-p", "0.06", "--palette", "20"],
    # 2026-09-17d: the D3 and D6 reds left under the best build
    "tr_taper08": ["--taper", "0.8", "--hand-run-p", "0.06"],
    "tr_dbl015": ["--taper", "0.5", "--hand-run-p", "0.06", "--doubles-rate", "0.15"],
    # D6 on 1f9a0/1fbfb is the OVER-DENSE branch (2.1-2.5x his events), not doubles: ask for the
    # human map's own density, which /buildmap's study step has in hand.
    "tr_nps188": ["--taper", "0.5", "--hand-run-p", "0.06", "--nps", "1.88"],
    "tr_nps389": ["--taper", "0.5", "--hand-run-p", "0.06", "--nps", "3.89"],
    # 2026-09-17g: the taper at energy jumps INSIDE sections (where the leftover D3s sit)
    "tr_intra": ["--taper", "0.5", "--hand-run-p", "0.06", "--taper-intra", "0.5"],
}

# lever -> (what the request says, the column that should move, direction lo→hi)
LEVERS = {
    "doubles": ("more doubles", "double_share", +1),
    "lead": ("one hand leads", "role_asymmetry", +1),
    "taper": ("breathe before the drop", "D3_hits", -1),
    "walls": ("more walls", "walls", +1),
    "width": ("more variety / diagonals", "idiom_local", +1),
    "palette": ("figures come back (narrow vocabulary)", "scatter_room", +1),
    "memory": ("figures come back (map memory)", "scatter_room", +1),
    "vocals": ("follow the vocals", "D4_hits", -1),
    "nps": ("faster / denser", "nps", +1),
}


def merged_flags(extra: list[str]) -> list[str]:
    base = list(BASE)
    i = 0
    while i < len(extra):
        f = extra[i]
        has_val = i + 1 < len(extra) and not extra[i + 1].startswith("--")
        if f in base:
            j = base.index(f)
            if has_val and j + 1 < len(base) and not base[j + 1].startswith("--"):
                del base[j:j + 2]
            else:
                del base[j]
        i += 2 if has_val else 1
    return base + extra


def build(sid: str, arm: str) -> pathlib.Path | None:
    z = OUT / f"{arm}__{sid}.zip"
    if z.exists():
        return z
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{arm}__{sid}.tmp.zip"
    cmds = [
        [sys.executable, str(AM / "autobuild.py"), str(AUDIO_DIR / f"{sid}.ogg"),
         *merged_flags(ARMS[arm]), "--seed", SEED, "--name", f"p6 {arm} {sid}", "--out", str(tmp)],
        [sys.executable, str(AM / "repeat.py"), str(tmp), "--out", str(tmp), "--song", sid],
        [sys.executable, str(AM / "answer.py"), str(tmp), "--out", str(tmp), "--song", sid],
    ]
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True, cwd=REPO)
        if not tmp.exists():
            print(f"  FAILED {arm} {sid}: {c[1]}\n{r.stderr[-400:]}", flush=True)
            return None
    tmp.rename(z)
    return z


def measure(z: pathlib.Path, sid: str) -> dict:
    js = z.with_suffix(".json")
    if js.exists():
        return json.loads(js.read_text())
    sys.path.insert(0, str(REPO / "scripts"))
    import queries as Q
    from verdict import load_arrays
    from beatsaber_automapper.evaluation import mapjudge as mj
    import numpy as np

    rec: dict = {"sid": sid, "arm": z.stem.split("__")[0]}
    on_f = REPO / "outputs" / "onset_cache" / f"{sid}.npz"
    on = np.load(on_f)["onsets"] if on_f.exists() else None
    res = mj.judge_zip(z, onsets=on, reference=mj.load_reference())
    rec["p"] = res.p_value
    rec["judge"] = res.verdict()
    for m in res.metrics:
        rec[m.name] = m.value
        rec[m.name + "_pct"] = m.pct
    vj = z.with_suffix(".verdict.json")
    subprocess.run([sys.executable, "scripts/verdict.py", str(z), "--song", sid, "--no-bench",
                    "--json", str(vj)], capture_output=True, cwd=REPO)
    v = json.loads(vj.read_text())
    rec["reds"] = v["reds"]
    rec["yellows"] = v["yellows"]
    rec["ship"] = v["ship"]
    rec["red_codes"] = sorted({ln["code"] for ln in v["lines"] if ln.get("state") == "🔴"})
    arrs, _, _ = load_arrays(z, sid, "auto")
    for q, codes in ((Q.q_drops, ("D3",)), (Q.q_vocals, ("D4",))):
        hits = q(arrs)
        for c in codes:
            rec[f"{c}_hits"] = sum(1 for h in hits if h[0] == c)
    rep: dict = {}
    Q.q_scatter(arrs, report=rep)
    rec["scatter_room"] = rep.get("SCATTER", ("", float("nan")))[1]
    with zipfile.ZipFile(z) as zf:
        n = next(x for x in zf.namelist() if x.lower().endswith("standard.dat"))
        d = json.loads(zf.read(n))
    beats = [round(float(x["b"]), 4) for x in d.get("colorNotes", [])]
    from collections import Counter
    c = Counter(beats)
    rec["double_share"] = sum(1 for b, k in c.items() if k > 1) / max(len(c), 1)
    rec["walls"] = len(d.get("obstacles", []))
    rec["notes"] = len(beats)
    js.write_text(json.dumps(rec, indent=1))
    return rec


def all_reds(vj: pathlib.Path) -> list[str]:
    """EVERY red on the page: query codes, ABSENCE rows (`ABS:<key>`) and a judge FAIL.

    ⚠️The first version read only the query codes, so a map whose one red was an ABSENCE row
    (1f8d6 losing its last lead-hand passage) showed `reds 1` with an empty code list and read as
    "no page cost". A page-cost column must read the same rows the SHIP line counts.
    """
    v = json.loads(vj.read_text())
    out = {ln["code"] for ln in v["lines"] if ln.get("state") == "🔴"}
    out |= {"ABS:" + a["key"] for a in v.get("absence", []) if a.get("state") == "🔴"}
    if (v.get("judge") or {}).get("verdict") == "FAIL":
        out.add("JUDGE")
    return sorted(out)


def report() -> int:
    rows = {}
    for js in OUT.glob("*__*.json"):
        if js.name.endswith(".verdict.json"):
            continue
        r = json.loads(js.read_text())
        r["red_codes"] = all_reds(js.with_suffix(".verdict.json"))
        rows[(r["arm"], r["sid"])] = r
    base = {s: rows.get(("base", s)) for s in SONGS}
    print("== baseline")
    for s in SONGS:
        b = base[s]
        if b:
            print(f"  {s}: SHIP {b['ship']:<4} reds {b['reds']} {b['red_codes']}  p {b['p']:.3f}  "
                  f"notes {b['notes']}")
    print("\n== request levers: the named column, lo → hi per song, and the page cost at hi")
    for lev, (ask, col, sign) in LEVERS.items():
        cells, moved, cost = [], 0, []
        for s in SONGS:
            lo, hi = rows.get((f"{lev}_lo", s)), rows.get((f"{lev}_hi", s))
            if not lo or not hi:
                continue
            a, b = lo.get(col), hi.get(col)
            ok = a is not None and b is not None and (b - a) * sign > 0
            moved += ok
            cells.append(f"{s} {a:.3g}→{b:.3g}{'' if ok else '✗'}")
            bb = base[s]
            if bb:
                new = sorted(set(hi["red_codes"]) - set(bb["red_codes"]))
                gone = sorted(set(bb["red_codes"]) - set(hi["red_codes"]))
                cost.append(f"{s}:{'+' + ','.join(new) if new else ''}{'-' + ','.join(gone) if gone else ''}"
                            f"{'' if new or gone else '='}")
        print(f"  {lev:<8} “{ask}” [{col}] moved on {moved}/{len(cells)}: {' | '.join(cells)}")
        print(f"           page at hi vs base: {' '.join(cost)}")
    print("\n== style presets: targets hit (±20 pct) and the page")
    sys.path.insert(0, str(AM))
    import style as ST
    for st in ("human", "calm", "dense", "flowing", "technical"):
        spec = ST.PRESETS[st]
        hits = tot = 0
        pages = []
        for s in SONGS:
            r = rows.get((f"style_{st}", s))
            if not r:
                continue
            for m, want in spec.items():
                got = r.get(m + "_pct")
                if got is None:
                    continue
                tot += 1
                hits += abs(got - want) <= 0.20
            bb = base[s]
            new = sorted(set(r["red_codes"]) - set(bb["red_codes"])) if bb else []
            pages.append(f"{s}:{r['ship']}{'+' + ','.join(new) if new else ''}")
        print(f"  {st:<10} targets {hits}/{tot}   {' '.join(pages)}")
    print("\n== style orderings (sweep_style's claims), per song")
    for metric, lo_s, hi_s in (("nps", "calm", "dense"), ("peak_nps", "calm", "dense"),
                               ("angle_change", "flowing", "technical"),
                               ("crossover", "flowing", "technical"),
                               ("travel", "calm", "technical"), ("ebpm_burst", "calm", "dense")):
        held = []
        for s in SONGS:
            a, b = rows.get((f"style_{lo_s}", s)), rows.get((f"style_{hi_s}", s))
            if a and b and a.get(metric) is not None and b.get(metric) is not None:
                held.append(b[metric] > a[metric])
        print(f"  {metric:<13} {lo_s} < {hi_s}: {sum(held)}/{len(held)}")
    return 0


def main() -> int:
    global OUT, SONGS, SEED, AUDIO_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--songs", nargs="*", default=None, help="default: the four songset maps")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="default: " + str(OUT))
    ap.add_argument("--seed", default=None, help="default: " + SEED)
    ap.add_argument("--audio-dir", type=pathlib.Path, default=None,
                    help="where <sid>.ogg lives (default data/eval_songset; held-out: data/heldout)")
    a = ap.parse_args()
    if a.out:
        OUT = a.out
    if a.songs:
        SONGS = tuple(a.songs)
    if a.seed is not None:
        SEED = str(a.seed)
    if a.audio_dir is not None:
        AUDIO_DIR = a.audio_dir
    if a.report:
        return report()
    for arm in (a.arms or ARMS):
        for sid in SONGS:
            z = build(sid, arm)
            if z is not None:
                r = measure(z, sid)
                print(f"{arm:<16} {sid}  SHIP {r['ship']:<4} reds {r['reds']}  p {r['p']:.3f}", flush=True)
    print("P6_LEVERS_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
