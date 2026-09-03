#!/usr/bin/env python
"""THE VERDICT — one page on a map, in Kyle's vocabulary, every red an address and a tool.

★**Why (TODO P4, 2026-09-02).** Three instruments now read a map and they disagree on
purpose: the **queries** (`scripts/queries.py`) measure *wrong* — his defect codes, each
hit a bar; the **tutor** (`scripts/tutor.py`) measures *like him* — did we answer the
song's situations the way the top human map of this song did; the **judge** (`mapjudge`)
measures *typical* — a corpus percentile that a bland map passes. `TUTOR__1f8d6` was
0 hits / 4 of 15 / PASS; the pulse build of the same song was 4 hits / 1 of 15 / a higher
p. Nobody should have to run three commands and hold the three frames in their head to
decide whether a map ships. This prints them on one page and says SHIP? — and the bench
line underneath is COMPUTED every time (`bench.run_score`), so a reader that drifts from
his verdicts announces it on every verdict it gives.

    verdict.py <map.zip>                  # song id from the file stem, --vs auto
    verdict.py <map.zip> --song 1f8d6 --no-bench --json out.json

Colours. **🔴** = a code whose hits cover ≥ 10 % of the bars (his "the VAST MAJORITY is A+"
was a map with FLOW on 2 %), or ANY hit of D1 / D3 / ELEMENTS (one missed drop is a
wrong drop). **🟡** = hits under 10 % — ship, but read them. **✅** = asked, nothing.
**⚪** = could not be asked (no human map of this song). SHIP? is NO on any red, and
every red line ends with the bars to open in the score and the tool that fixes them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q = _load("queries")
bench = _load("bench")
tutor = _load("tutor")

# What each code is called at the table, and what fixes it. The tool column is the
# point of the page: a red that names no tool is a complaint, not a task.
CODES = [
    ("EMPTY", "empty / not playing the song", "mapctl auto --bars a-b --target-notes N, or copy the tutor's rows (tutor.py --bars, mapedit.py from)"),
    ("D1", "very slow overall", "raise --nps / --target-notes per section; read q_events ratios"),
    ("D6", "nps wasted (doubles / over-dense)", "mapctl clear --bars a-b then auto with --doubles-rate lower or a smaller --target-notes"),
    ("FLOW", "does not flow (jitter off the grid)", "mapedit.py: move the odd-16th notes onto the 8th grid, or clear+auto without --pulse"),
    ("D2", "slightly off beat (shifted grid)", "mapedit.py shift --bars a-b, or re-init with --phase-shift; check ON/±ms in the score"),
    ("D4", "not following the main vocals", "mapctl auto --bars a-b --follow vocals; the why names the unanswered words"),
    ("D3", "drop at the wrong time", "mapedit.py: first note on the bar line, density step like the tutor's (tutor.py --bars)"),
    ("ELEMENTS", "walls / arcs / chains", "autobuild --walls N (default on); mapctl walls --bars a-b"),
]
ALWAYS_RED = {"D1", "D3", "ELEMENTS"}
RED_SHARE = 0.10


def load_arrays(src: pathlib.Path, song: str | None, vs: str):
    """Arrays for a zip (built with the human reference) or a score.py --npz file."""
    if src.suffix == ".npz":
        z = np.load(src, allow_pickle=True)
        return {k: z[k] for k in z.files}, None, None
    from agent_mapper import score as S
    sid = song or next(iter(re.findall(r"[0-9a-f]{4,6}", src.stem)), None)
    m, song_obj, how, vsm, lat, sc, mc, hc = S.build(pathlib.Path(src), sid, 4, vs, False)
    arrs = S.to_arrays(m, sc, mc, lat, hc)
    header = S.header_lines(m, song_obj, sc, mc, lat, how, vsm)
    return arrs, dict(sid=sid, m=m, mc=mc, hc=hc, vs=vsm, header=header), song_obj


def _span(h: tuple) -> tuple[int, int]:
    b0 = int(h[2]); b1 = int(h[4]) if len(h) > 4 and h[4] else b0
    return b0, b1


def _spans_text(hits: list[tuple], limit: int = 4) -> str:
    spans = [_span(h) for h in hits]
    txt = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in spans[:limit])
    if len(spans) > limit:
        txt += f" … ×{len(spans)}"
    return txt


def judge(src: pathlib.Path, sid: str | None) -> dict | None:
    """The typicality gate, as a caveat line: PASS/FAIL, p, and why."""
    try:
        from beatsaber_automapper.evaluation import mapjudge as mj
        sweep = _load("sweep_snap")
        on = sweep.onsets_for(sid) if sid else None
        res = mj.judge_zip(src, onsets=on, reference=mj.load_reference())
        return dict(verdict=res.verdict(), p=float(res.p_value), viol=int(res.viol or 0),
                    why=res.why_fail(), onset=float(res.align_value or 0),
                    deaf=on is None)
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e))


def tutor_line(sid: str | None, arrs: dict) -> tuple[str, list[dict]]:
    """`N/M situations his way` and the ones that differ, with their bars."""
    if not sid or not Q.has_human(arrs):
        return "⚪ no tutor (no human map of this song)", []
    sits = tutor.find_situations(arrs)
    bar = arrs["bar"]; sub = int(arrs["sub"])
    diffs = []; same = 0
    for s in sits:
        t = tutor.pattern(arrs["human"], bar, s["bar"], sub)
        o = tutor.pattern(arrs["map"], bar, s["bar"], sub)
        if tutor.same_way(t, o):
            same += 1
        else:
            diffs.append(dict(bar=s["bar"], kind=s["kind"], song=s["song"],
                              tutor=tutor._fmt(t), ours=tutor._fmt(o)))
    word = f"{same}/{len(sits)} situations his way"
    return word, diffs


def verdict(src: pathlib.Path, song: str | None = None, vs: str = "auto",
            with_bench: bool = True) -> dict:
    arrs, built, song_obj = load_arrays(src, song, vs)
    sid = built["sid"] if built else song
    n_bars = int(np.asarray(arrs["bar"]).max())
    hits = Q.q_all(arrs)
    by: dict[str, list[tuple]] = defaultdict(list)
    for h in hits:
        by[h[0]].append(h)
    human = Q.has_human(arrs)
    lines: list[dict] = []
    reds = yellows = 0
    for code, name, tool in CODES:
        hs = by.get(code, [])
        askable = human or code in ("FLOW", "D2")
        if not askable:
            state = "⚪"
        elif not hs:
            state = "✅"
        else:
            share = bench.coverage(hs, {code}, n_bars)
            state = "🔴" if (code in ALWAYS_RED or share >= RED_SHARE) else "🟡"
        reds += state == "🔴"; yellows += state == "🟡"
        lines.append(dict(code=code, name=name, state=state, n=len(hs),
                          share=bench.coverage(hs, {code}, n_bars) if hs else 0.0,
                          spans=_spans_text(hs), first=(hs[0][3] if hs else ""),
                          tool=tool, hits=[dict(t=float(h[1]), bar=int(h[2]),
                                                end=int(h[4]) if len(h) > 4 else int(h[2]),
                                                why=h[3]) for h in hs]))
    play = None
    if built:
        mc = built["mc"]
        hc = built.get("hc")
        play = dict(violations=int(mc.violations), resets=int(mc.resets),
                    human_resets=(int(hc.resets) if hc is not None else None))
        # ★Resets are legal (a stop-and-reverse the player can make in time) but each is
        # a re-cock; the 1f767 loop thinned a section by deleting the alternating notes
        # and went 0 -> 27 while this line stayed green (his map: 4). Yellow past twice
        # his count + 2 (no human: past 12) — read `mapedit.py resets`, never ship blind.
        limit = 2 * play["human_resets"] + 2 if play["human_resets"] is not None else 12
        play["reset_warn"] = play["resets"] > limit
        if play["violations"]:
            reds += 1
        elif play["reset_warn"]:
            yellows += 1
    tut_word, tut_diffs = tutor_line(sid, arrs)
    jd = judge(src, sid) if src.suffix == ".zip" else None
    # P0's gate stands: parity -> alignment floor -> requested density -> typicality.
    # A FAIL there is a red here; the page says which gate and never ranks by p.
    if jd and jd.get("verdict") not in (None, "PASS"):
        reds += 1
    bench_res = None
    if with_bench:
        bench_res = bench.run_score(Q.q_all, "queries:q_all", echo=lambda *a, **k: None)
    ship = "NO" if reds else "YES"
    return dict(map=str(src), song=sid, n_bars=n_bars, human=human, lines=lines, reds=reds,
                yellows=yellows, ship=ship, playability=play, tutor=tut_word,
                tutor_diffs=tut_diffs, judge=jd, header=(built["header"] if built else []),
                bench=(None if bench_res is None else
                       dict(line=bench_res["line"], bad=bench_res["bad"],
                            hits=bench_res["hits"], n_rows=bench_res["n_rows"])))


def render(v: dict) -> str:
    L = []
    src = pathlib.Path(v["map"]).name
    hdr = f"# VERDICT  {src}  —  song {v['song']} · {v['n_bars']} bars"
    if v["header"]:
        hdr += "  ·  " + v["header"][0].split("—", 1)[1].strip()
    L.append(hdr)
    if v["bench"]:
        b = v["bench"]
        L.append(f"# reader: {b['line']}  ({b['hits']} of {b['n_rows']} bench rows hit — "
                 f"computed now, never typed)")
        if b["bad"]:
            L.append("# 🔴 THE READER IS REFUTED ON HIS VERDICTS — fix queries.py before trusting this page")
    if not v["human"]:
        L.append("# ⚪ no human map of this song: EMPTY / D1 / D4 / D6 / ELEMENTS could not be asked; "
                 "only FLOW / D2 read against the song's onsets")
    L.append("")
    for ln in v["lines"]:
        head = f"{ln['state']} {ln['code']:<8s} {ln['name']:<38s}"
        if ln["n"] == 0:
            L.append(head)
            continue
        L.append(f"{head} {ln['n']} hit(s), {ln['share']:.0%} of bars — bars {ln['spans']}")
        L.append(f"{'':>13s}first: {ln['first']}")
        if ln["state"] == "🔴":
            first = ln["hits"][0]
            span = f"{first['bar']}-{first['end']}" if first["end"] != first["bar"] else str(first["bar"])
            L.append(f"{'':>13s}read:  score.py <map> --song {v['song']} --vs auto --bars {span}")
            L.append(f"{'':>13s}fix:   {ln['tool']}")
    L.append("")
    p = v["playability"]
    if p:
        st = "🔴" if p["violations"] else ("🟡" if p.get("reset_warn") else "✅")
        hr = p.get("human_resets")
        L.append(f"{st} PLAYABILITY  parity violations {p['violations']} · resets {p['resets']}"
                 + (f" (human {hr})" if hr is not None else "")
                 + "  (unplayable is non-negotiable)")
        if p.get("reset_warn"):
            L.append(f"{'':>13s}read:  mapedit.py <map> resets   — reconcile with ONE note per hand "
                     "(place / delete / flip … X), not a chain of flips")
    L.append(f"{'✅' if v['tutor'].startswith(('⚪',)) or _tutor_ok(v['tutor']) else '🟡'} TUTOR"
             f"        {v['tutor']}"
             + ("" if not v["tutor_diffs"] else
                "  — differs at bars " + ", ".join(str(d["bar"]) for d in v["tutor_diffs"][:8])
                + (" …" if len(v["tutor_diffs"]) > 8 else "")
                + "  (tutor.py <song> --map <map>; --bars b-b+1 to copy)"))
    j = v["judge"]
    if j:
        if "error" in j:
            L.append(f"⚪ JUDGE        unavailable: {j['error'][:80]}")
        else:
            L.append(f"{'✅' if j['verdict'] == 'PASS' else '🔴'} JUDGE        {j['verdict']} p={j['p']:.3f}"
                     f" onset={j['onset']:.3f}"
                     + (f" — {'; '.join(j['why'])}" if j["why"] else "")
                     + ("  ⚠️deaf: no onset cache" if j["deaf"] else "")
                     + "  (typicality, not quality — a floor, never a rank)")
            if j["verdict"] != "PASS" and v["reds"] == 1:
                L.append("             the only red is the judge's gate: read its why in the score "
                         "(ON / ±ms columns) — his verdicts outrank the corpus median, the "
                         "alignment floor does not")
    L.append("")
    fix = [ln for ln in v["lines"] if ln["state"] == "🔴"]
    if v["ship"] == "NO":
        order = ", ".join(f"{ln['code']} (bars {ln['spans'].split(' …')[0]})" for ln in fix)
        if p and p["violations"]:
            order = f"PLAYABILITY ({p['violations']} violations)" + (", " + order if order else "")
        j = v["judge"]
        if j and j.get("verdict") not in (None, "PASS"):
            order += (", " if order else "") + f"JUDGE ({'; '.join(j['why']) or 'p < 0.10'})"
        L.append(f"SHIP? NO — {v['reds']} red. Fix in this order: {order}. Then run verdict.py again.")
    else:
        note = (f" {v['yellows']} yellow — read them before shipping." if v["yellows"] else
                " nothing located.")
        L.append(f"SHIP? YES —{note} A clean page means no LOCATED defect; the tutor line and a "
                 f"listen are what is left.")
    return "\n".join(L)


def _tutor_ok(word: str) -> bool:
    m = re.match(r"(\d+)/(\d+)", word)
    return bool(m) and int(m.group(2)) > 0 and int(m.group(1)) / int(m.group(2)) >= 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=pathlib.Path, help="map zip (or a score.py --npz file)")
    ap.add_argument("--song", help="song id (default: the 4-6 hex id in the file name)")
    ap.add_argument("--vs", default="auto", help="reference human map: zip, corpus id, or 'auto'")
    ap.add_argument("--no-bench", action="store_true",
                    help="skip the reader-agreement line (saves ~5 s)")
    ap.add_argument("--json", type=pathlib.Path, help="also write the verdict as JSON")
    a = ap.parse_args()
    v = verdict(a.src, a.song, a.vs, with_bench=not a.no_bench)
    print(render(v))
    if a.json:
        a.json.write_text(json.dumps(v, indent=1, default=str))
    return 1 if v["ship"] == "NO" else 0


if __name__ == "__main__":
    sys.exit(main())
