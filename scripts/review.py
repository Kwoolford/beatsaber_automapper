#!/usr/bin/env python
"""The review pipeline: maps wait in `for_review/`, verdicts land in the ledger,
reviewed sets are cataloged into `outputs/reviewed/`.

**Why this exists.** Kyle's ear is the only ground truth in this project and the
scarcest resource in it — the suite has been measured *not* to track it. Until now the
maps awaiting his judgement were scattered through `outputs/` among ~8 000 other zips,
indistinguishable from cohort runs, and the only way to know what was still pending was
to read three separate review docs. Staging them in one top-level folder makes "what
needs my ear" a directory listing.

**The pipeline, and why it is one-directional**

    outputs/kyle_review_*   --stage-->   for_review/<set>/   --done-->   outputs/reviewed/<set>/
                                              ^                              ^
                                         PENDING: he has                CATALOGED: verdict
                                         not played these               is in the ledger

A set may only be marked `done` once at least one verdict for it exists in
`docs/eval_references/preference_verdicts.json`. That rule is the whole point: it makes
it impossible to quietly file away a set nobody actually judged, which is exactly how
the three current sets accumulated to 33 maps.

⚠️`for_review/` and `outputs/` are both **gitignored**. The verdicts are not — they live
in `docs/`, because losing them means asking him to listen to everything again.

Usage:
    python scripts/review.py list                       # what is pending, and the question
    python scripts/review.py open Hunger AGENT          # launch ArcViewer on matching maps
    python scripts/review.py verdict --set C --song 1f333 --name Hunger \
        --better AGENT --worse BEFORE --quote "his words"
    python scripts/review.py defect --song 1f8d6 --at 2:10 --kind drop_timing \
        --quote "the drop lands late" --map for_review/A_.../FallenKingdom_BEFORE.zip
    python scripts/review.py defects                    # ★the located defect ledger
    python scripts/review.py done C                     # catalog the set into outputs/
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
STAGE = REPO / "for_review"
CATALOG = REPO / "outputs" / "reviewed"
LEDGER = REPO / "docs" / "eval_references" / "preference_verdicts.json"
ARCVIEWER = pathlib.Path.home() / ".local" / "bin" / "arcviewer"

# The three sets currently awaiting his ear. `question` is the thing to actually ask
# him — a set with no question attached is a pile of files, not a review.
SETS = {
    "A": {
        "dir": "A_structure_crossover",
        "src": "outputs/kyle_review_2026-08-11",
        "doc": "docs/review_2026-08-11.md",
        "title": "structure reuse + crossover",
        "question": "Play [BEFORE] vs [BOTH]. Does the repetition read INTENTIONAL or "
                    "LAZY? And do the crossovers play better?",
        "pairs": [("BEFORE", "BOTH")],
    },
    "B": {
        "dir": "B_grid_phase",
        "src": "outputs/kyle_review_2026-08-14",
        "doc": "docs/review_2026-08-14.md",
        "title": "beat grid phase",
        "question": "Lead with BEcause. Does [PHASE] sit on the beat better than "
                    "[BEFORE]? \"Can't tell\" is a real answer and worth recording.",
        "pairs": [("BEFORE", "PHASE")],
    },
    "C": {
        "dir": "C_agent_built",
        "src": "outputs/kyle_review_agent",
        "doc": "agent_mapper/PROGRESS.md",
        "title": "agent-built map (hand-authored)",
        "question": "Is AUTO Hunger [AGENT] better than the generator's Hunger "
                    "([BEFORE], in set A)? Note Hunger's vocals are unpitched, so this "
                    "map was built on rhythm alone.",
        "pairs": [("BEFORE", "AGENT")],
    },
}


# ★THE SHORTLIST. 33 maps is not a review, it is a chore nobody does — and three sets
# sat untouched long enough to prove that. Each entry below is ONE comparison that
# unblocks ONE decision that is currently blocked. Ordered by value per minute of his
# time, which is the scarcest resource this project has.
#
# Song choice is deliberate: three of the four are Hunger, which he already knows well
# enough to have graded parts of it A+, so he learns two songs instead of ten. BEcause
# is unavoidable for the grid-phase question — the lever left his four standing songs
# BYTE-IDENTICAL, so there is literally nothing to hear on them.
SHORTLIST = [
    {
        "n": 1,
        "set": "A",
        "play": ["Hunger_BEFORE.zip", "Hunger_CROSSOVER.zip"],
        "decides": "Flip COLOR_SEP_MODE=extreme (crossover) ON by default?",
        "why": "The cleanest numbers on this project. We cross hands over on 0.000 of "
               "notes; 150 human maps have a median of 0.183 and NOT ONE of them has "
               "zero. Flow improves 0.37 -> 0.23 and two reachability measures land "
               "exactly on the human value. It changes every song, not just repetitive "
               "ones. If you play one thing, play this.",
        "ask": "Do the crossovers feel natural, or do they feel like the map is "
               "fighting you?",
    },
    {
        "n": 2,
        "set": "C",
        "play": ["Hunger_AGENT.zip"],
        "against": "Hunger_BEFORE.zip (set A) — the same song you just played",
        "decides": "Is agent-authored mapping worth keeping as P0?",
        "why": "A whole research direction rests on this one map. Every suite number on "
               "it is either circular (it places notes on the onsets the metric scores) "
               "or known not to track your ear, so nothing but you can answer it.",
        "ask": "Better or worse than the generator's Hunger? Blunt is fine.",
    },
    {
        "n": 3,
        "set": "B",
        "play": ["BEcause_BEFORE.zip", "BEcause_PHASE.zip"],
        "decides": "Flip BEAT_GRID_PHASE=search ON by default?",
        "why": "It passed its DoD at n=149 and made the alignment axis pass for the "
               "FIRST TIME EVER (0.62 FAIL -> 0.35 PASS), 74 songs better and 0 worse. "
               "It is still default-OFF for one reason: it optimises the same onsets "
               "the axis scores, so the axis cannot be trusted to judge it. Your ear is "
               "the only thing outside that circle.",
        "ask": "Does [PHASE] sit ON the beat better? \"Can't tell\" is a real answer "
               "and worth recording — it would mean the axis is measuring something "
               "inaudible.",
    },
    {
        "n": 4,
        "set": "A",
        "optional": True,
        "play": ["Hunger_BEFORE.zip", "Hunger_BOTH.zip"],
        "decides": "Does deliberate repetition read INTENTIONAL or LAZY?",
        "why": "Only if you still have patience. Hunger's chorus is genuinely the same "
               "music three times (bars 55/113/194, confirmed) so it is the sharpest "
               "test. It matters because mapctl's new `reuse` defaults to varying 15 % "
               "of a repeated section — if repetition reads LAZY that default is wrong. "
               "⚠️[BOTH] stacks crossover AND structure reuse, so judge it only after "
               "comparison 1 has told you what crossover alone feels like.",
        "ask": "Does the repeat feel like a callback, or like I got lazy?",
    },
]


def cmd_next(a) -> int:
    """The shortlist: a few comparisons, each with the decision it unblocks."""
    print("★ SHORTLIST — 4 comparisons, 6 distinct maps, 2 songs.\n"
          "  (33 maps are staged; these are the ones whose answers change what I build.)\n")
    for c in SHORTLIST:
        if c.get("optional") and not a.all:
            continue
        tag = "  [optional]" if c.get("optional") else ""
        print(f"{c['n']}. set {c['set']}{tag}  —  {c['decides']}")
        for f in c["play"]:
            hit = next((z for z in STAGE.rglob(f)), None)
            mark = " " if hit else " ⚠️MISSING "
            print(f"     play{mark}{f}")
        if c.get("against"):
            print(f"     against  {c['against']}")
        print(f"     why      {c['why']}")
        print(f"     ★ask     {c['ask']}")
        print(f"     open     python scripts/review.py open "
              f"{' '.join(c['play'][0].replace('.zip', '').split('_'))}")
        print()
    if not a.all:
        print("A 4th optional comparison exists: python scripts/review.py next --all")
    print("\nRecord an answer:\n"
          "  python scripts/review.py verdict --set A --song 1f333 --name Hunger \\\n"
          "      --better CROSSOVER --worse BEFORE --quote \"your words\"")
    return 0


def _ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"_README": [], "verdicts": []}


def _verdicts_for(set_id: str) -> list[dict]:
    return [v for v in _ledger().get("verdicts", [])
            if str(v.get("set", "")).upper() == set_id.upper()]


def cmd_stage(a) -> int:
    """Move each pending set out of `outputs/` into the staging folder."""
    STAGE.mkdir(exist_ok=True)
    for sid, meta in SETS.items():
        src = REPO / meta["src"]
        dst = STAGE / meta["dir"]
        if dst.exists():
            print(f"{sid}: already staged at {dst.relative_to(REPO)}")
            continue
        if not src.exists():
            print(f"{sid}: source {meta['src']} is gone — nothing to stage")
            continue
        dst.mkdir(parents=True)
        n = 0
        for z in sorted(src.glob("*.zip")):
            shutil.move(str(z), str(dst / z.name))
            n += 1
        # Everything else moves too — logs, READMEs and supporting subfolders explain
        # the arms, and a set split across two places is a set nobody can review.
        # ⚠️Subdirectories are included on purpose: set A keeps its structure/ folder
        # of PNGs plus a deliberate over-repetition reference map, and an earlier
        # file-only version left all of it orphaned back in outputs/.
        for extra in sorted(src.iterdir()):
            shutil.move(str(extra), str(dst / extra.name))
        if not any(src.iterdir()):
            src.rmdir()
        print(f"{sid}: staged {n} maps -> {dst.relative_to(REPO)}")
    _write_readme()
    print(f"\nStaged into {STAGE.relative_to(REPO)}/ (gitignored). "
          "Run `python scripts/review.py list`.")
    return 0


def _write_readme() -> None:
    lines = ["# Maps waiting on Kyle's ear", "",
             "Open one with ArcViewer:", "",
             "```bash",
             "python scripts/review.py next          # ★ the 3 that matter, start here",
             "python scripts/review.py open Hunger CROSSOVER",
             "```", "",
             "**33 maps are staged. You do not need to play 33 maps** — "
             "`review.py next` lists the handful whose answers change what gets built.",
             ""]
    for sid, meta in SETS.items():
        d = STAGE / meta["dir"]
        if not d.exists():
            continue
        zips = sorted(d.glob("*.zip"))
        lines += [f"## Set {sid} — {meta['title']}  ({len(zips)} maps)", "",
                  f"**{meta['question']}**", "",
                  f"Detail: `{meta['doc']}`", ""]
        for z in zips:
            lines.append(f"- `{z.name}`")
        lines.append("")
    lines += ["---", "",
              "When a set is judged, record it and file it away:", "",
              "```bash",
              "python scripts/review.py verdict --set A --song 1f333 --name Hunger \\",
              "    --better BOTH --worse BEFORE --quote \"his exact words\"",
              "python scripts/review.py done A",
              "```", ""]
    (STAGE / "README.md").write_text("\n".join(lines))


def cmd_list(a) -> int:
    if not STAGE.exists():
        print("nothing staged. Run: python scripts/review.py stage")
        return 0
    print(f"PENDING REVIEW  ({STAGE.relative_to(REPO)}/)\n")
    total = 0
    for sid, meta in SETS.items():
        d = STAGE / meta["dir"]
        if not d.exists():
            continue
        zips = sorted(d.glob("*.zip"))
        total += len(zips)
        vs = _verdicts_for(sid)
        print(f"  set {sid}  {meta['title']:<34} {len(zips):>3} maps   "
              f"{len(vs)} verdict(s) recorded")
        print(f"          ★ {meta['question']}")
        if a.long:
            for z in zips:
                print(f"            {z.name}")
        print()
    print(f"{total} maps pending.")
    done = sorted(CATALOG.glob("*")) if CATALOG.exists() else []
    if done:
        print(f"\nCATALOGED ({CATALOG.relative_to(REPO)}/): "
              + ", ".join(p.name for p in done))
    return 0


def cmd_open(a) -> int:
    """Launch ArcViewer on the staged maps matching every given term."""
    if not ARCVIEWER.exists():
        print(f"ArcViewer not found at {ARCVIEWER}", file=sys.stderr)
        return 2
    hits = [z for z in sorted(STAGE.rglob("*.zip"))
            if all(t.lower() in z.name.lower() for t in a.terms)]
    if not hits:
        print(f"no staged map matches {a.terms}. Try: python scripts/review.py list --long")
        return 1
    if len(hits) > 1 and not a.all:
        print("matches more than one map — narrow it, or pass --all:")
        for z in hits:
            print(f"  {z.relative_to(STAGE)}")
        return 1
    for z in hits:
        print(f"opening {z.relative_to(STAGE)}")
        subprocess.Popen([str(ARCVIEWER), str(z)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def cmd_verdict(a) -> int:
    """Record a verdict, tagged with its set so `done` can find it."""
    d = _ledger()
    if a.tie:
        entry = {"kind": "tie", "arms": list(a.tie)}
    elif a.better and a.worse:
        entry = {"kind": "same_song_ab", "better": a.better, "worse": a.worse}
    else:
        print("give --better/--worse or --tie A B", file=sys.stderr)
        return 2
    entry.update({
        "id": f"{_dt.date.today().isoformat()}/{a.song}-{a.set.upper()}",
        "date": _dt.date.today().isoformat(),
        "set": a.set.upper(),
        "song": a.song,
        "name": a.name or a.song,
        "quote": a.quote,
        "note": a.note,
    })
    d.setdefault("verdicts", []).append(entry)
    LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    print(f"recorded: {entry['id']}  {entry.get('better', '')}"
          f"{' > ' + entry['worse'] if entry.get('worse') else ' (tie)'}")
    print(f"  \"{a.quote}\"" if a.quote else
          "  ⚠️no quote recorded — his exact words are the part that survives")
    print(f"\nledger: {LEDGER.relative_to(REPO)} (tracked in git)")
    return 0


# ★V5 — DEFECT CAPTURE. The A/B pipeline above collects *preferences between arms*;
# Kyle produces *defects located in songs* ("the drop is late at 2:10"). On 2026-08-17
# he reviewed three sets and answered none of them per-arm — he answered with a defect
# list spanning every song. ⇒**Preference is the second-class record and defects are
# the first.** A preference says which of two guesses he minded less; a defect says
# what is wrong, and where.
#
# The kinds are the six defects he named, so a captured defect lands on a work item
# rather than in a free-text pile. `other` is deliberately available — a vocabulary
# that cannot express his complaint would quietly discard it, which is the failure
# mode this whole file exists to prevent.
DEFECT_KINDS = {
    "slow": "D1 — very slow",
    "offbeat": "D2 — slightly off beat",
    "drop_timing": "D3 — drops at the wrong time",
    "vocals": "D4 — not following the main vocals",
    "burst": "D5 — random bursts of really fast non flowy notes",
    "wasted": "D6 — nps wasted on non main notes",
    "empty": "W2 — this part is really empty",
    "other": "not one of the named six",
}


def _at_seconds(v: str) -> float:
    """`2:10`, `2:10.5` or plain seconds -> seconds. His timestamps come as mm:ss."""
    v = v.strip()
    if ":" in v:
        m, sec = v.rsplit(":", 1)
        return int(m) * 60 + float(sec)
    return float(v)


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def _context_at(song: str, t: float, map_zip: pathlib.Path | None) -> dict:
    """What our tools claim is happening at the moment he is complaining about.

    ★This is the point of the command. A defect recorded as prose is a note to
    self; a defect recorded **beside what the pipeline believed at that instant** is
    evidence. If he says the drop is late at 2:10 and `structure.py` puts the DROP at
    2:04, that is a measured disagreement the same day he reports it.

    ⚠️Best-effort by design: if the perception cache is cold or the map will not load,
    the defect is still recorded. Losing his words to a traceback is not acceptable.
    """
    out: dict = {}
    audio = REPO / "data" / "eval_songset" / f"{song}.ogg"
    if not audio.exists():
        return {"error": f"no audio at {audio.relative_to(REPO)}"}
    sys.path.insert(0, str(REPO / "agent_mapper"))
    try:
        import notesheet as _ns
        d = _ns.collect(audio, map_zip=map_zip)
    except Exception as e:  # noqa: BLE001
        return {"error": f"perception failed: {e}"}

    g = d["grid"]
    try:
        import brief as _brief
        out["bar"] = int((t - _brief.bar_time(g, 0)) // g["bar_s"])
    except Exception:  # noqa: BLE001
        pass
    out["bpm"] = round(g["bpm"], 2)
    for sec in d["sections"]:
        if sec["t0"] <= t <= sec["t1"]:
            out["section"] = f'{sec["label"]} · {sec["role"]}'
            out["section_span"] = f'{_mmss(sec["t0"])}-{_mmss(sec["t1"])}'
    # the nearest DROP either side, because "the drop is at the wrong time" is a claim
    # about a distance, not about a moment
    drops = [s_ for s_ in d["sections"] if s_["role"] in ("DROP", "peak")]
    if drops:
        near = min(drops, key=lambda s_: abs(s_["t0"] - t))
        out["nearest_drop"] = f'{near["label"]} at {_mmss(near["t0"])} ' \
                              f'({near["t0"] - t:+.1f}s from you)'
    ov = d.get("overlay")
    if ov:
        w = [v for v in ov["verdicts"] if abs(v["t"] - t) <= 2.0]
        miss = [m for m in ov["missed"] if abs(m["t"] - t) <= 2.0]
        hit = sum(1 for v in w if v["v"] == "hit")
        out["within_2s"] = (f'{len(w)} of our notes ({hit} hit, {len(w) - hit} wasted), '
                            f'{len(miss)} main events we missed')
    fl = d.get("flow") or {}
    for b in fl.get("bursts", []):
        if b["t0"] - 1.0 <= t <= b["t1"] + 1.0:
            out["burst_here"] = (f'{b["n"]} notes at {b["nps"]:.1f} nps, music '
                                 f'{b["motivation"]:.2f}x its median rate '
                                 f'({b["verdict"]}), travel {b["travel"]:.1f} cells/s')
    if fl.get("rows") and "burst_here" not in out:
        out["burst_here"] = "no burst at this moment"
    return out


def cmd_defect(a) -> int:
    """Record one located defect, with what the pipeline believed at that instant."""
    if a.kind not in DEFECT_KINDS:
        print(f"unknown kind {a.kind!r}; known: {', '.join(DEFECT_KINDS)}",
              file=sys.stderr)
        return 2
    t = _at_seconds(a.at)
    entry = {
        "id": f"{_dt.date.today().isoformat()}/{a.song}-{a.kind}-{_mmss(t)}",
        "date": _dt.date.today().isoformat(),
        "song": a.song, "name": a.name or a.song,
        "at": round(t, 2), "at_mmss": _mmss(t),
        "kind": a.kind, "means": DEFECT_KINDS[a.kind],
        "map": a.map.name if a.map else None,
        "quote": a.quote, "note": a.note,
    }
    # record FIRST, analyse second: his words must survive a broken cache
    d = _ledger()
    d.setdefault("defects", []).append(entry)
    LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    print(f"recorded: {entry['id']}")
    if not a.quote:
        print("  ⚠️no quote — his exact words are the part that survives")

    if a.no_context:
        return 0
    print("\n  what we believed was happening there:")
    ctx = _context_at(a.song, t, a.map)
    for k, v in ctx.items():
        print(f"    {k:<14} {v}")
    entry["context"] = ctx
    LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    print(f"\nledger: {LEDGER.relative_to(REPO)} (tracked in git)")
    return 0


def cmd_defects(a) -> int:
    """The defect ledger: the six standing complaints, and every LOCATED instance.

    ⚠️Two schemas live here on purpose. The 2026-08-17 entries are Kyle's original six,
    recorded **unlocated** ("ALL songs he played") because that is how he gave them —
    they carry `code`/`phrase` and no timestamp. Located instances carry `song`/`at`.
    ★**Converting the first kind into the second is the whole job of this command**, so
    it prints the standing six as an explicit backlog rather than dropping them for
    lacking a field.
    """
    ds = _ledger().get("defects", [])
    standing = [x for x in ds if "at" not in x]
    located = [x for x in ds if "at" in x]
    if a.song:
        located = [x for x in located
                   if x["song"] == a.song or x.get("name") == a.song]

    if standing and not a.song:
        # P1 is a POSITIVE ("that half works"), recorded deliberately so it is not
        # regressed while chasing the six. Counting it as a complaint would misreport
        # the one thing he said was right.
        n_bad = sum(1 for x in standing if not str(x.get("code", "")).startswith("P"))
        print(f"■ STANDING, UNLOCATED — his {n_bad} original complaints "
              f"(+{len(standing) - n_bad} thing he said WORKS, {standing[0]['date']})\n")
        for x in sorted(standing, key=lambda r: r.get("code", "")):
            print(f"   {x.get('code', '?'):<4} {x.get('phrase', x.get('kind', ''))}")
        print("\n   ★These have no timestamp. A located instance of any of them is worth "
              "more\n   than another metric — `review.py defect --song X --at m:ss "
              "--kind ...`\n")

    if not located:
        print("No LOCATED defects yet. That is the gap this command exists to close.")
        return 0

    print(f"■ LOCATED — {len(located)} instances\n")
    for kind, means in DEFECT_KINDS.items():
        rows = [x for x in located if x.get("kind") == kind]
        if not rows:
            continue
        print(f"  {means}   ({len(rows)})")
        for x in sorted(rows, key=lambda r: (r.get("song", ""), r["at"])):
            bar = (x.get("context") or {}).get("bar")
            print(f"   {x.get('name', x.get('song', '?')):<20} {x['at_mmss']:>8}"
                  f"{f'  bar {bar}' if bar is not None else '':<9}  {x['date']}")
            if x.get("quote"):
                print(f"      \u201c{x['quote']}\u201d")
        print()
    return 0


def cmd_done(a) -> int:
    """Catalog a reviewed set: move it out of staging into `outputs/reviewed/`."""
    sid = a.set.upper()
    meta = SETS.get(sid)
    if not meta:
        print(f"unknown set {sid}; known: {', '.join(SETS)}", file=sys.stderr)
        return 2
    src = STAGE / meta["dir"]
    if not src.exists():
        print(f"set {sid} is not staged")
        return 1
    vs = _verdicts_for(sid)
    if not vs and not a.force:
        print(f"⚠️set {sid} has NO verdict in the ledger — refusing to file away a set "
              "nobody judged.\n   Record one first:\n"
              f"     python scripts/review.py verdict --set {sid} --song <id> "
              "--better X --worse Y --quote \"...\"\n"
              "   or pass --force if it is genuinely being abandoned.")
        return 1

    CATALOG.mkdir(parents=True, exist_ok=True)
    dst = CATALOG / meta["dir"]
    if dst.exists():
        dst = CATALOG / f"{meta['dir']}_{_dt.date.today().isoformat()}"
    shutil.move(str(src), str(dst))

    lines = [f"# Set {sid} — {meta['title']}", "",
             f"Cataloged {_dt.date.today().isoformat()}. "
             f"Question asked: {meta['question']}", "",
             f"## Verdicts ({len(vs)})", ""]
    for v in vs:
        if v.get("kind") == "tie":
            lines.append(f"- **{v.get('name')}**: tie between "
                         f"{' and '.join(v.get('arms', []))} — \"{v.get('quote','')}\"")
        else:
            lines.append(f"- **{v.get('name')}**: {v.get('better')} > {v.get('worse')}"
                         f" — \"{v.get('quote','')}\"")
    if not vs:
        lines.append("- ⚠️none — this set was filed away with --force, unjudged.")
    lines += ["", f"Source detail: `{meta['doc']}`",
              "", "The authoritative record is "
              "`docs/eval_references/preference_verdicts.json` (tracked in git); "
              "this file is a local convenience copy."]
    (dst / "CATALOG.md").write_text("\n".join(lines) + "\n")
    _write_readme()
    print(f"set {sid} cataloged -> {dst.relative_to(REPO)}  ({len(vs)} verdict(s))")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stage", help="move pending sets out of outputs/ into for_review/")
    p.set_defaults(fn=cmd_stage)

    p = sub.add_parser("next", help="★the shortlist — a few comparisons that matter")
    p.add_argument("--all", action="store_true", help="include the optional one")
    p.set_defaults(fn=cmd_next)

    p = sub.add_parser("list", help="what is pending, and the question each set asks")
    p.add_argument("--long", action="store_true", help="list every map file")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("open", help="launch ArcViewer on a staged map")
    p.add_argument("terms", nargs="+", help="substrings that must all match, e.g. Hunger AGENT")
    p.add_argument("--all", action="store_true", help="open every match")
    p.set_defaults(fn=cmd_open)

    p = sub.add_parser("verdict", help="record a judgement into the tracked ledger")
    p.add_argument("--set", required=True)
    p.add_argument("--song", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--better")
    p.add_argument("--worse")
    p.add_argument("--tie", nargs=2, metavar=("ARM_A", "ARM_B"))
    p.add_argument("--quote", default="")
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_verdict)

    p = sub.add_parser("defect", help="★record a LOCATED defect in his words")
    p.add_argument("--song", required=True, help="corpus id, e.g. 1f333")
    p.add_argument("--at", required=True, help="when, as mm:ss or seconds")
    p.add_argument("--kind", required=True,
                   help="one of: " + ", ".join(DEFECT_KINDS))
    p.add_argument("--quote", default="", help="★his exact words")
    p.add_argument("--name", default=None)
    p.add_argument("--note", default="")
    p.add_argument("--map", type=pathlib.Path, default=None,
                   help="the map he was playing, so the context includes our notes")
    p.add_argument("--no-context", action="store_true",
                   help="skip the analysis and just record it")
    p.set_defaults(fn=cmd_defect)

    p = sub.add_parser("defects", help="the defect ledger, grouped by kind")
    p.add_argument("--song", default=None)
    p.set_defaults(fn=cmd_defects)

    p = sub.add_parser("done", help="catalog a reviewed set into outputs/reviewed/")
    p.add_argument("set")
    p.add_argument("--force", action="store_true",
                   help="file it away even with no verdict (abandoning it)")
    p.set_defaults(fn=cmd_done)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
