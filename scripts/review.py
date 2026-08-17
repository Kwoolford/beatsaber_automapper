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

    p = sub.add_parser("done", help="catalog a reviewed set into outputs/reviewed/")
    p.add_argument("set")
    p.add_argument("--force", action="store_true",
                   help="file it away even with no verdict (abandoning it)")
    p.set_defaults(fn=cmd_done)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
