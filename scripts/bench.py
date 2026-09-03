#!/usr/bin/env python
"""THE BENCH -- Kyle's verdicts as rows a locator can be scored against (P2, 2026-09-02).

    python scripts/bench.py list                      the rows, paths checked, reads counted
    python scripts/bench.py read 1f333-agent-flow     open the score where the label points
    python scripts/bench.py read 1f8d6-empty --bars 9-16 --vs
    python scripts/bench.py note 1f8d6-empty --bars 9-24 --codes EMPTY --text "..."
    python scripts/bench.py stats                     event-level facts per row
    python scripts/bench.py score agent_mapper.queries:flow_fold
    python scripts/bench.py score smoke:density        (the built-in smoke query)

Why this exists (docs/audit_2026-09-02_buildmap.md): the judge measures typicality and
Kyle asks "wrong, and WHERE?". His verdicts were never kept as labels, so no locator
could ever be checked against him. `docs/eval_references/labelled_maps.json` keeps the
labels; this script (1) opens the score at the labelled bars so the agent can write a
read, (2) stores that read next to the label, (3) scores any query against every row.

**A query** is any callable `q(arrs: dict) -> list[hit]` where `arrs` is
`agent_mapper.score.to_arrays()` (song[T,F], map[T,C], human[T,C] when the human map
is loaded, t_sec, beat, bar, section, lyric, ...) and a hit is `(code, t_sec, bar, why)`.
`code` is one of the defect codes in the label file. That is the whole protocol; P3
queries live in `scripts/queries.py` and are addressed as `module:function`.

**Scoring is counts, never rates.** n is under twenty. A DEFECT row is a HIT when the
query fires its code there (AT the labelled bars if the label has any, else anywhere);
a CLEAN/GOOD/PREFERRED row is a FALSE FIRE when the query fires at all, and a
VIOLATION when it fires a code in `must_not_flag` on more of the map than the row's
`tolerance` (0 unless Kyle said "vast majority"). A DEFECT row with `worse_than` must
draw more of its code than the map he preferred over it. A weak row can refute, not
confirm. Queries may declare `q.codes`; rows wanting none of them are n/a for that query.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

LABELS = REPO / "docs" / "eval_references" / "labelled_maps.json"
READS = REPO / "docs" / "eval_references" / "bench_reads.json"
CACHE = REPO / "outputs" / "bench_cache"
BEATS_PER_BAR = 4

NEGATIVE = ("CLEAN", "GOOD", "PREFERRED")


# ----------------------------------------------------------------------------- labels
def load_rows() -> list[dict]:
    return json.loads(LABELS.read_text())["rows"]


def row_by_id(rid: str) -> dict:
    for r in load_rows():
        if r["id"] == rid:
            return r
    raise SystemExit(f"no bench row '{rid}' -- see `bench.py list`")


def load_reads() -> dict:
    return json.loads(READS.read_text()) if READS.exists() else {}


def _bars(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    a, _, b = spec.partition("-")
    return int(a), int(b or a)


# ----------------------------------------------------------------------------- arrays
def arrays_for(row: dict, rebuild: bool = False) -> dict | None:
    """`score.to_arrays()` for a row, cached in outputs/bench_cache/<id>.npz.

    The human map is always loaded (`--vs auto`) so an EMPTY/D6 query can compare
    against it; for the human rows themselves that is the same map twice.
    """
    if not row.get("readable", True):
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{row['id']}.npz"
    if f.exists() and not rebuild:
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    from agent_mapper import score as S
    m, song, _how, _vsm, lat, sc, mc, hc = S.build(REPO / row["map"], row["song"], 4, "auto")
    arrs = S.to_arrays(m, sc, mc, lat, hc)
    np.savez(f, **arrs)
    return arrs


# ----------------------------------------------------------------------------- list
def cmd_list(a) -> int:
    rows, reads = load_rows(), load_reads()
    print(f"{'id':<18} {'label':<10} {'str':<6} {'codes':<22} {'bars':<7} {'ok':<3} "
          f"{'rd':<3} map")
    n_ok = 0
    for r in rows:
        ok = (REPO / r["map"]).exists() and all((REPO / p).exists() for p in r.get("also", []))
        n_ok += ok
        codes = ",".join(r["codes"]) or ("¬" + ",".join(r["must_not_flag"][:3]) + "…"
                                         if r["must_not_flag"] else "-")
        n_reads = len(reads.get(r["id"], []))
        print(f"{r['id']:<18} {r['label']:<10} {r['strength']:<6} {codes[:22]:<22} "
              f"{(r['bars'] or '-'):<7} {'✓' if ok else '✗':<3} "
              f"{(str(n_reads) if n_reads else '·'):<3} "
              f"{r['map']}{'' if r.get('readable', True) else '  (unreadable)'}")
    strong = [r for r in rows if r["strength"] == "strong"]
    unread = [r["id"] for r in strong if not reads.get(r["id"])]
    print(f"\n{len(rows)} rows, {n_ok} with every path present; {len(strong)} strong, "
          f"{sum(1 for r in rows if r['strength']=='weak')} weak, "
          f"{sum(1 for r in rows if not r.get('readable', True))} unreadable.")
    if unread:
        print(f"strong rows WITHOUT a written read: {', '.join(unread)}   "
              f"(P2 DoD: every strong row needs one -- `bench.py read <id>` then `note`)")
    else:
        print("every strong row has a written read.")
    return 0


# ----------------------------------------------------------------------------- read
def cmd_read(a) -> int:
    from agent_mapper import score as S
    r = row_by_id(a.id)
    if not r.get("readable", True):
        print(f"{r['id']} is unreadable: {r.get('note', '')}")
        return 1
    print(f"# {r['id']}  {r['label']}  codes={r['codes'] or '-'}  strength={r['strength']}")
    print(f"# quote: {r['quote']}")
    if r.get("bars"):
        print(f"# labelled bars {r['bars']} (from: {r['bars_from']})")
    vs = "auto" if (a.vs or r["label"] != "CLEAN") else None
    m, song, how, vsm, lat, sc, mc, hc = S.build(REPO / r["map"], r["song"], a.sub, vs)
    print("\n".join(S.header_lines(m, song, sc, mc, lat, how, vsm)))
    bars = a.bars or r.get("bars")
    if a.sections or not bars:
        print()
        print("\n".join(S.render_sections(m, sc, mc, lat, hc)))
        n_bars = int(math.ceil(lat.n / (lat.sub * BEATS_PER_BAR)))
        print(f"\n# {n_bars} bars. Zoom with --bars a-b.")
    if bars:
        s0, s1 = S._bar_range(bars, lat)
        print()
        print("\n".join(S.render_rows(m, sc, mc, lat, s0, s1, hc)))
    prev = load_reads().get(r["id"], [])
    if prev:
        print(f"\n# {len(prev)} stored read(s):")
        for p in prev:
            print(f"#  [{p['date']}] bars {p.get('bars') or '-'} codes {p.get('codes') or '-'}: "
                  f"{p['text']}")
    return 0


# ----------------------------------------------------------------------------- note
def cmd_note(a) -> int:
    r = row_by_id(a.id)
    reads = load_reads()
    entry = {"date": _dt.date.today().isoformat(), "bars": a.bars,
             "codes": [c.upper() for c in (a.codes or "").split(",") if c],
             "text": a.text}
    reads.setdefault(r["id"], []).append(entry)
    READS.write_text(json.dumps(reads, indent=1, ensure_ascii=False) + "\n")
    agree = set(entry["codes"]) & set(r["codes"])
    bad = set(entry["codes"]) & set(r["must_not_flag"])
    print(f"stored read #{len(reads[r['id']])} for {r['id']}")
    if r["codes"]:
        print(f"  label says {r['codes']}; read names {entry['codes'] or 'nothing'} -> "
              f"{'names the labelled defect' if agree else 'DOES NOT name the labelled defect'}")
    if bad:
        print(f"  ⚠️ read names {sorted(bad)} on a row whose label forbids it")
    return 0


# ----------------------------------------------------------------------------- stats
def event_stats(map_path: pathlib.Path) -> dict:
    """Event-level facts the P2 reads kept reaching for. A 'row' is one lattice
    16th with any note on it = one thing the player does; doubles inflate the note
    count while halving the row count, which is how 1f8d6-empty has MORE notes than
    its human and fewer events."""
    import collections
    from agent_mapper import score as S
    m = S.load_map(map_path)
    by: dict[int, set] = collections.defaultdict(set)
    for n in m.notes:
        by[round(n.beat * 4)].add(n.color)
    rows = sorted(by)
    if len(rows) < 2:
        return {"notes": len(m.notes), "rows": len(rows)}
    dbl = sum(1 for r in rows if len(by[r]) == 2)
    gaps = np.diff(rows) / 4 * 60 / m.bpm
    pos = collections.Counter((n.color, round((n.beat * 4) % 4)) for n in m.notes)
    on = [pos[(c, 0)] for c in (0, 1)]
    off16 = [pos[(c, 1)] + pos[(c, 3)] for c in (0, 1)]
    tot = [max(1, sum(pos[(c, k)] for k in range(4))) for c in (0, 1)]
    seq = [next(iter(by[r])) if len(by[r]) == 1 else "D" for r in rows]
    runs, cur = [], 1
    for x, y in zip(seq, seq[1:]):
        if x == y and x != "D":
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return {
        "notes": len(m.notes), "rows": len(rows), "doubles_pct": 100 * dbl / len(rows),
        "gap_med_ms": float(np.median(gaps)) * 1000, "gaps_over_1s": int((gaps > 1).sum()),
        "onbeat_pct": [100 * on[c] / tot[c] for c in (0, 1)],
        "off16_pct": [100 * off16[c] / tot[c] for c in (0, 1)],
        "runs4": sum(1 for r in runs if r >= 4),
    }


def cmd_stats(a) -> int:
    print(f"{'id':<18} {'label':<10} {'notes':>5} {'rows':>5} {'dbl%':>5} {'gap':>5} {'>1s':>4} "
          f"{'onbeat% L/R':>12} {'16th% L/R':>10} {'runs4':>5}")
    for r in load_rows():
        if a.ids and r["id"] not in a.ids:
            continue
        s = event_stats(REPO / r["map"])
        if "doubles_pct" not in s:
            print(f"{r['id']:<18} {r['label']:<10} {s['notes']:>5} {s['rows']:>5}")
            continue
        print(f"{r['id']:<18} {r['label']:<10} {s['notes']:>5} {s['rows']:>5} "
              f"{s['doubles_pct']:>5.0f} {s['gap_med_ms']:>5.0f} {s['gaps_over_1s']:>4} "
              f"{s['onbeat_pct'][0]:>5.0f}/{s['onbeat_pct'][1]:<6.0f} "
              f"{s['off16_pct'][0]:>4.0f}/{s['off16_pct'][1]:<5.0f} {s['runs4']:>5}")
    print("\nrows = 16ths with any note (player events); dbl% = rows that are two-hand doubles;\n"
          "gap = median event gap ms; onbeat%/16th% per hand = share of that hand's notes on\n"
          "the beat / on 16th-offbeats; runs4 = same-hand runs of 4+ events (streams).")
    return 0


# ----------------------------------------------------------------------------- score
def notes_per_row(arr: np.ndarray) -> np.ndarray:
    """Notes on each lattice row: the 12 `c<k>_color` columns (0 = no note) of a
    `score.to_arrays()` map/human array."""
    return (arr[:, 0:24:2] > 0).sum(axis=1)


def _smoke_density(arrs: dict) -> list[tuple]:
    """Built-in smoke query, NOT a locator: bars where the map has under half the
    human's notes. It exists so `score` can be exercised before P3 -- and because
    'empty is not note coverage' it should get 1f8d6-empty WRONG."""
    if "human" not in arrs or arrs["human"].size == 0:
        return []
    bar = arrs["bar"]
    m_n, h_n = notes_per_row(arrs["map"]), notes_per_row(arrs["human"])
    hits = []
    for b in np.unique(bar):
        sel = bar == b
        hm, hh = int(m_n[sel].sum()), int(h_n[sel].sum())
        if hh >= 8 and hm < hh / 2:
            hits.append(("EMPTY", float(arrs["t_sec"][sel][0]), int(b),
                         f"{hm} map notes vs {hh} human"))
    return hits


SMOKE = {"density": _smoke_density}


def resolve_query(spec: str):
    if spec.startswith("smoke:"):
        return SMOKE[spec.split(":", 1)[1]]
    mod, _, fn = spec.partition(":")
    if not fn:
        raise SystemExit("query must be module:function (or smoke:<name>)")
    return getattr(importlib.import_module(mod), fn)


def coverage(hits: list[tuple], codes: set, n_bars: int) -> float:
    """Share of the map's bars inside fires of `codes`. A hit may carry an end bar as h[4]
    (queries that merge windows do); otherwise it covers its one bar."""
    covered: set[int] = set()
    for h in hits:
        if h[0] in codes:
            covered.update(range(int(h[2]), int(h[4] if len(h) > 4 else h[2]) + 1))
    return len(covered) / max(n_bars, 1)


def score_row(row: dict, hits: list[tuple], claims: set | None = None,
              n_bars: int = 0) -> tuple[str, str]:
    """(result, detail). Results: HIT, HIT-elsewhere, MISS, FALSE, VIOLATION, CLEAN, n/a.
    `claims` = the codes the query says it can emit (its `.codes` attribute): a DEFECT row
    wanting none of them is n/a for this query, not a MISS -- q_all scores the whole set.
    A negative row may carry `tolerance` (share of bars, 0-1): Kyle's "the VAST MAJORITY is
    A+" forbids a code on 90 % of the map, not on every bar -- a forbidden code covering
    up to that share is 'tolerated', more is a VIOLATION."""
    codes = {h[0] for h in hits}
    if row["label"] == "UNLABELLED":
        return "n/a", f"{len(hits)} fire(s) -- unlabelled, neither hit nor false"
    if claims is not None and row["label"] not in NEGATIVE and not (claims & set(row["codes"])):
        return "n/a", f"query claims {sorted(claims)}, label wants {row['codes']}"
    if row["label"] in NEGATIVE:
        viol = codes & set(row["must_not_flag"])
        if viol:
            tol = float(row.get("tolerance", 0.0))
            cov = coverage(hits, viol, n_bars)
            if cov > tol or n_bars == 0:
                return "VIOLATION", (f"fired {sorted(viol)} which the label forbids "
                                     f"({cov:.0%} of bars, tolerance {tol:.0%})")
            return "tolerated", (f"fired {sorted(viol)} on {cov:.0%} of bars -- label "
                                 f"tolerates {tol:.0%} ('vast majority')")
        if hits and row["label"] == "CLEAN":
            return "FALSE", f"{len(hits)} fire(s) {sorted(codes)} on a clean map"
        if hits:
            return "fires", f"{len(hits)} fire(s) {sorted(codes)} (label does not forbid)"
        return "CLEAN", "silent"
    want = codes & set(row["codes"])
    if not want:
        return "MISS", (f"fired {sorted(codes)}, label wants {row['codes']}" if hits
                        else f"silent, label wants {row['codes']}")
    rng = _bars(row.get("bars"))
    if rng:
        at = [h for h in hits if h[0] in want and rng[0] <= h[2] <= rng[1]]
        if at:
            return "HIT", f"{sorted(want)} at labelled bars {row['bars']} ({len(at)} fire(s))"
        return "HIT-elsewhere", (f"{sorted(want)} fires but not in bars {row['bars']}; "
                                 f"bars {sorted({h[2] for h in hits if h[0] in want})[:8]}")
    return "HIT", f"{sorted(want)} at bars {sorted({h[2] for h in hits if h[0] in want})[:8]}"


def run_score(q, query_name: str = "", rebuild: bool = False, verbose: int = 0,
              echo=print) -> dict:
    """Score `q` on every bench row. Returns the tally plus the one-line word; `echo`
    receives the table (pass a no-op to run silently -- verdict.py does)."""
    rows = load_rows()
    tally: dict[str, int] = {}
    per: dict[str, list[tuple]] = {}
    n_bars: dict[str, int] = {}
    echo(f"query {query_name}\n")
    echo(f"{'id':<18} {'label':<10} {'str':<6} {'result':<13} detail")
    for r in rows:
        if not r.get("readable", True):
            echo(f"{r['id']:<18} {r['label']:<10} {r['strength']:<6} {'skip':<13} unreadable")
            continue
        try:
            arrs = arrays_for(r, rebuild=rebuild)
            hits = list(q(arrs))
        except Exception as e:  # noqa: BLE001
            echo(f"{r['id']:<18} {r['label']:<10} {r['strength']:<6} {'ERROR':<13} {e}")
            tally["ERROR"] = tally.get("ERROR", 0) + 1
            continue
        per[r["id"]] = hits
        n_bars[r["id"]] = int(np.asarray(arrs["bar"]).max())
        res, detail = score_row(r, hits, getattr(q, "codes", None), n_bars[r["id"]])
        key = f"{res}/{r['strength']}" if res in ("HIT", "HIT-elsewhere", "MISS") else res
        tally[key] = tally.get(key, 0) + 1
        echo(f"{r['id']:<18} {r['label']:<10} {r['strength']:<6} {res:<13} {detail}")
        if verbose:
            for h in hits[:verbose]:
                echo(f"{'':<18} {'':<10} {'':<6} {'':<13}   {h[0]} bar {h[2]} "
                     f"t={h[1]:.1f}s: {h[3]}")
    # The pair that must agree: same notes, different offset.
    if "1f333-aplus" in per and "1f333-before" in per:
        ca = sorted({h[0] for h in per["1f333-aplus"]})
        cb = sorted({h[0] for h in per["1f333-before"]})
        same = ca == cb and len(per["1f333-aplus"]) == len(per["1f333-before"])
        echo(f"\nsame-notes pair 1f333-aplus / 1f333-before: "
             f"{'agree' if same else 'DISAGREE'} ({len(per['1f333-aplus'])} vs "
             f"{len(per['1f333-before'])} fires)")
    # Ordering pairs: a DEFECT row with `worse_than` must draw MORE of its code than the
    # map Kyle preferred over it (same song) -- the comparison he actually made.
    claims = getattr(q, "codes", None)
    for r in rows:
        other = r.get("worse_than")
        if other and r["id"] in per and other in per:
            want = set(r["codes"])
            if claims is not None and not (claims & want):
                continue
            mine = coverage(per[r["id"]], want, n_bars[r["id"]])
            theirs = coverage(per[other], want, n_bars[other])
            ok = mine > theirs
            echo(f"\norder {r['id']} > {other} on {sorted(want)}: "
                 f"{'holds' if ok else 'ORDER VIOLATION'} ({mine:.0%} vs {theirs:.0%} of bars)")
            if not ok:
                tally["ORDER"] = tally.get("ORDER", 0) + 1
    echo("\ncounts (n is tiny -- these refute, they do not validate):")
    for k in sorted(tally):
        echo(f"  {k:<20} {tally[k]}")
    bad = sum(v for k, v in tally.items() if k in ("FALSE", "VIOLATION", "ERROR", "ORDER"))
    strong_hits = tally.get("HIT/strong", 0)
    # fires on the maps he liked: only codes the label FORBIDS count (1f333-aplus forbids
    # FLOW and its note says EMPTY/D1/D6 are true of those notes)
    good_fires = sum(1 for r in rows if r["label"] in ("GOOD", "PREFERRED") and r["id"] in per
                     for h in per[r["id"]] if h[0] in r["must_not_flag"])
    strong_hit_fires = sum(len(per[r["id"]]) for r in rows
                           if r["strength"] == "strong" and r["label"] == "DEFECT"
                           and r["id"] in per
                           and score_row(r, per[r["id"]], None, n_bars[r["id"]])[0] == "HIT")
    if bad:
        word = "REFUTED"
    elif strong_hits and good_fires > strong_hit_fires:
        word = (f"not refuted, but fires forbidden codes MORE on the maps he liked ({good_fires}) "
                f"than on the defects it hits ({strong_hit_fires}) -- it is not measuring the label")
    elif strong_hits:
        word = "not refuted"
    else:
        word = "SILENT on every strong row -- says nothing either way"
    line = (f"{word}: {strong_hits} strong hit(s), {tally.get('FALSE', 0)} false fire(s), "
            f"{tally.get('VIOLATION', 0)} violation(s)")
    echo(f"\n{line}")
    hits_total = sum(v for k, v in tally.items() if k.startswith("HIT"))
    return dict(tally=tally, word=word, line=line, bad=bad, strong_hits=strong_hits,
                hits=hits_total, n_rows=len(per), per=per, n_bars=n_bars)


def cmd_score(a) -> int:
    q = resolve_query(a.query)
    return 1 if run_score(q, a.query, a.rebuild, a.verbose)["bad"] else 0


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="rows, paths checked, reads counted")
    p = sub.add_parser("read", help="open the score where the label points")
    p.add_argument("id")
    p.add_argument("--bars", help="override the labelled bars, e.g. 9-16")
    p.add_argument("--sections", action="store_true", help="force the one-row-per-bar overview")
    p.add_argument("--vs", action="store_true", help="show the human map alongside (default for non-CLEAN rows)")
    p.add_argument("--sub", type=int, default=4)
    p = sub.add_parser("note", help="store a written read next to the label")
    p.add_argument("id")
    p.add_argument("--text", required=True, help="the read: what the score shows, at which bars")
    p.add_argument("--bars", help="bars the read is about")
    p.add_argument("--codes", help="comma-separated defect codes the read names (empty = clean)")
    p = sub.add_parser("stats", help="event-level facts per row (doubles, hand roles, streams)")
    p.add_argument("ids", nargs="*", help="restrict to these row ids")
    p = sub.add_parser("score", help="score a query module:function against every row")
    p.add_argument("query")
    p.add_argument("--rebuild", action="store_true", help="ignore outputs/bench_cache")
    p.add_argument("-v", "--verbose", type=int, default=0, metavar="N", help="print up to N hits per row")
    a = ap.parse_args()
    if a.cmd is None:
        a.cmd = "list"
    return {"list": cmd_list, "read": cmd_read, "note": cmd_note, "stats": cmd_stats,
            "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
