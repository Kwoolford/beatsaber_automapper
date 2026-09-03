#!/usr/bin/env python
"""THE COMPETE TEST — our best map of a song against the top human map of it, blind.

★**Why (TODO P4b, 2026-09-02).** The bench (P2) answers *"did we catch what Kyle caught"*
and the verdict (P4) answers *"is anything located wrong"*. Neither answers the question
the project is for — *"is it as good as the best map of this song"* — and only a player
can. `preference_screen.py` was the n=1 version of this and stalled; `23/23 PASS` was the
headline it left behind, and a PASS is typicality. **The headline is now a win rate**, and
a loss with a reason is worth more than a win: it becomes a bench row (P2) the same day,
and the query that would have found it is the next thing to build (P3).

    compete.py stage 1f767                 # blind pair -> for_review/compete/{X,Y}__1f767.zip
    compete.py stage --songset             # the four standing songs (skips any without a
                                           #   SHIP? YES map; --force stages them anyway)
    compete.py list                        # what is staged, and the one question to ask
    compete.py verdict 1f767 Y --because "X drops the vocal at 1:02" --code D4 --bars 51-52
    compete.py verdict 1f767 tie
    compete.py table                       # ★win rate, n, and every loss with its reason

**Blind means blind.** Both zips are rewritten to the same skeleton: `Info.dat` with the
song name `X <sid>` / `Y <sid>`, the same author strings, no cover, no preview point, no
`_customData` (editor names, bookmarks, difficulty labels), the audio as `song.ogg`, ONE
difficulty each. What stays is the map: notes, walls, arcs, chains, NJS/offset (they are
part of how a map plays). The key lives in `for_review/compete/.key.json` — a dotfile so a
directory listing does not spoil it — and `verdict` is the only reader of it.

**Which human map.** `data/raw/<sid>.zip`, the corpus's one rating-sorted map of the song
(see `tutor.py`), and the SAME difficulty the tutor and queries read: `ExpertStandard`
first, else `ExpertPlus` — so a loss is against the map every red on the page was judged
against. **Which of ours.** `--ours`, else the first that exists in `BEST`: the map that
went through the loop, then today's default build. A map whose verdict page is red is not
staged without `--force`: losing with a known red teaches nothing the page did not say.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import pathlib
import random
import shutil
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(REPO))

STAGE = REPO / "for_review" / "compete"
KEY = STAGE / ".key.json"
CATALOG = REPO / "outputs" / "reviewed" / "compete"
LEDGER = REPO / "docs" / "eval_references" / "preference_verdicts.json"
BENCH = REPO / "docs" / "eval_references" / "labelled_maps.json"
HUMAN = REPO / "data" / "raw"

SONGSET = ["1f767", "1f8d6", "1f913", "1f333"]
NAMES = {"1f333": "Hunger", "1f8d6": "Fallen Kingdom",
         "1f913": "Digital Life Hacker", "1f767": "AliceBlue"}
# Our best map of a song, in order of preference: the loop's output beats the coarse build.
BEST = ["outputs/p4_loop/LOOP__{sid}.zip",
        "outputs/p4/NOPULSE__{sid}.zip",
        "outputs/p0_songset_2026-09-02/NEW__{sid}.zip"]
CODES = ("D1", "D2", "D3", "D4", "D5", "D6", "EMPTY", "FLOW", "ELEMENTS")
QUESTION = ("Play X, then Y. Which is the better map of this song? If one is worse, say "
            "what it did WRONG and where (mm:ss or a bar) — that sentence is the next "
            "thing I build. \"Can't tell\" is a real answer.")

# ------------------------------------------------------------------------- blinding


def _rel(p: pathlib.Path) -> str:
    """Repo-relative when inside the repo (the ledger keeps portable paths), else as is."""
    p = pathlib.Path(p)
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def _read_zip(path: pathlib.Path) -> tuple[dict, dict, bytes, str, str]:
    """(info, beatmap, audio bytes, difficulty name, difficulty file) — the Standard set,
    Expert first, else ExpertPlus, else the last listed (the hardest), like score.load_map."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        base = {n.split("/")[-1].lower(): n for n in names}
        info_n = base.get("info.dat")
        if info_n is None:
            raise ValueError(f"{path}: no Info.dat")
        info = json.loads(zf.read(info_n).decode("utf-8-sig"))
        sets = info.get("_difficultyBeatmapSets") or []
        std = next((s for s in sets if s.get("_beatmapCharacteristicName") == "Standard"),
                   sets[0] if sets else None)
        if std is None or not std.get("_difficultyBeatmaps"):
            raise ValueError(f"{path}: no difficulty set in Info.dat")
        diffs = std["_difficultyBeatmaps"]
        pick = (next((d for d in diffs if d["_difficulty"] == "Expert"), None)
                or next((d for d in diffs if d["_difficulty"] == "ExpertPlus"), None)
                or diffs[-1])
        bm_n = base.get(pick["_beatmapFilename"].lower())
        if bm_n is None:
            raise ValueError(f"{path}: {pick['_beatmapFilename']} missing from the zip")
        bm = json.loads(zf.read(bm_n).decode("utf-8-sig"))
        audio_n = base.get(str(info.get("_songFilename", "")).lower())
        if audio_n is None:
            audio_n = next((n for n in names if n.lower().endswith((".ogg", ".egg"))), None)
        if audio_n is None:
            raise ValueError(f"{path}: no audio")
        audio = zf.read(audio_n)
    return info, bm, audio, pick["_difficulty"], pick


def blind_zip(src: pathlib.Path, letter: str, sid: str, out: pathlib.Path) -> dict:
    """Write `out` = `src` with every tell removed. Returns what was kept (for the key)."""
    info, bm, audio, diff_name, diff = _read_zip(src)
    keep_cd = {k: v for k, v in (diff.get("_customData") or {}).items()
               if k in ("_requirements", "_suggestions")}
    entry = {"_difficulty": diff_name,
             "_difficultyRank": int(diff.get("_difficultyRank", 7)),
             "_beatmapFilename": f"{diff_name}Standard.dat",
             "_noteJumpMovementSpeed": float(diff.get("_noteJumpMovementSpeed", 16.0)),
             "_noteJumpStartBeatOffset": float(diff.get("_noteJumpStartBeatOffset", 0.0))}
    if keep_cd:
        entry["_customData"] = keep_cd
    new_info = {
        "_version": "2.1.0",
        "_songName": f"{letter} {sid}", "_songSubName": "", "_songAuthorName": "compete",
        "_levelAuthorName": "compete",
        "_beatsPerMinute": float(info.get("_beatsPerMinute", 120.0)),
        "_shuffle": 0, "_shufflePeriod": 0.5,
        "_previewStartTime": 30.0, "_previewDuration": 10.0,
        "_songFilename": "song.ogg", "_coverImageFilename": "",
        "_environmentName": "DefaultEnvironment",
        "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
        "_songTimeOffset": float(info.get("_songTimeOffset", 0.0)),
        "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard",
                                    "_difficultyBeatmaps": [entry]}],
    }
    # editor bookmarks / BPM-change lists / `_time` live at the top of the beatmap; note-
    # level customData (Chroma colours, NE) is left alone — it is part of the map.
    bm = {k: v for k, v in bm.items() if k not in ("_customData", "customData")}
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in (("Info.dat", json.dumps(new_info, indent=1).encode()),
                           (entry["_beatmapFilename"], json.dumps(bm).encode()),
                           ("song.ogg", audio)):
            zi = zipfile.ZipInfo(name, date_time=stamp)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, data)
    n_notes = len(bm.get("colorNotes", bm.get("_notes", [])))
    return {"src": _rel(src),
            "difficulty": diff_name, "njs": entry["_noteJumpMovementSpeed"],
            "notes": n_notes, "kept_customData": sorted(keep_cd)}


# ------------------------------------------------------------------------- the key


def _key() -> dict:
    return json.loads(KEY.read_text()) if KEY.exists() else {}


def _save_key(k: dict) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    KEY.write_text(json.dumps(k, indent=1, ensure_ascii=False) + "\n")


def _ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"_README": [], "verdicts": []}


def _save_ledger(d: dict) -> None:
    LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")


def _bench() -> dict:
    return json.loads(BENCH.read_text()) if BENCH.exists() else {"_README": [], "rows": []}


def _save_bench(d: dict) -> None:
    BENCH.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")


def compete_verdicts(d: dict | None = None) -> list[dict]:
    return [v for v in (d or _ledger()).get("verdicts", []) if v.get("kind") == "compete"]


# ------------------------------------------------------------------------- stage


def best_of_ours(sid: str) -> pathlib.Path | None:
    for pat in BEST:
        p = REPO / pat.format(sid=sid)
        if p.exists():
            return p
    return None


def _page(ours: pathlib.Path) -> dict | None:
    """The verdict page of our map, so the key records what the reader already knew."""
    try:
        spec = importlib.util.spec_from_file_location("verdict", HERE / "verdict.py")
        V = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(V)
        v = V.verdict(ours, with_bench=False)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"ship": v["ship"], "reds": v["reds"], "yellows": v["yellows"],
            "red_codes": [ln["code"] for ln in v["lines"] if ln["state"] == "🔴"],
            "tutor": v["tutor"],
            "judge": (v["judge"] or {}).get("verdict") if v.get("judge") else None}


def stage_one(sid: str, ours: pathlib.Path | None, force: bool, seed: int | None,
              with_page: bool = True, restage: bool = False) -> int:
    key = _key()
    if sid in key and key[sid].get("status") == "staged":
        if not restage:
            print(f"{sid}: already staged as {_rel(STAGE)}/{{X,Y}}__{sid}.zip "
                  f"(judge it: compete.py verdict {sid} X|Y|tie)"
                  f"\n  our map changed since? re-blind it with --restage")
            return 0
        # ★A staged pair goes STALE the moment our map is edited (2026-09-03a: 1f333 was
        # staged, then BREATHING was found in it and fixed). Spending a listening session on a
        # map we already know is defective is the one thing this test cannot afford, so
        # re-blinding is a first-class operation instead of hand-editing .key.json.
        # Only a STAGED pair is stale-able. Once judged, the entry's status is the verdict and
        # this branch is not reached at all -- staging that song again is a NEW comparison of a
        # newer build, which is allowed and leaves the ledger's verdict untouched.
        for letter in ("X", "Y"):
            (STAGE / f"{letter}__{sid}.zip").unlink(missing_ok=True)
        print(f"{sid}: re-blinding (the previous pair was never judged)")
        key.pop(sid, None)
        _save_key(key)
    human = HUMAN / f"{sid}.zip"
    if not human.exists():
        print(f"{sid}: no human map at {_rel(human)} — nothing to compete with")
        return 1
    ours = ours or best_of_ours(sid)
    if ours is None:
        print(f"{sid}: none of ours exists ({', '.join(BEST)}) — build one (/buildmap)")
        return 1
    page = _page(ours) if with_page else None
    if page and page.get("ship") == "NO" and not force:
        print(f"{sid}: {_rel(ours)} is SHIP? NO ({page['reds']} red: "
              f"{', '.join(page['red_codes'])}) — run the loop first "
              f"(scripts/verdict.py), or --force to stage it anyway")
        return 1
    rng = random.Random(seed)
    roles = ["OURS", "HUMAN"]
    rng.shuffle(roles)
    srcs = {"OURS": ours, "HUMAN": human}
    rec = {"status": "staged", "staged": _dt.date.today().isoformat(), "blind": {},
           "page": page}
    for letter, role in zip("XY", roles):
        out = STAGE / f"{letter}__{sid}.zip"
        kept = blind_zip(srcs[role], letter, sid, out)
        rec["blind"][letter] = {"role": role, **kept}
    key[sid] = rec
    _save_key(key)
    _write_readme(key)
    # ⚠️Nothing per-letter is printed here — a note count beside X or Y would unblind
    # the pair for anyone who knows how many notes our map has (the key holds it).
    diffs = sorted({rec["blind"][L]["difficulty"] for L in "XY"})
    print(f"{sid} ({NAMES.get(sid, sid)}): staged X/Y -> {_rel(STAGE)}/  [{'/'.join(diffs)}]"
          + (f"  ours: {page['ship']}, {page['reds']} red, tutor {page['tutor']}"
             if page and "ship" in page else ""))
    if rec["blind"]["X"]["difficulty"] != rec["blind"]["Y"]["difficulty"]:
        print(f"   ⚠️different difficulties — the human map has no Expert; the pair is "
              f"what the tutor and queries compared against")
    return 0


def _write_readme(key: dict) -> None:
    lines = ["# COMPETE — our map vs the top human map, blind", "",
             f"**{QUESTION}**", "",
             "Open one: `python scripts/review.py open X__1f767` (ArcViewer). Answer with",
             "`python scripts/compete.py verdict <sid> X|Y|tie --because \"...\"`.", ""]
    for sid, rec in key.items():
        if rec.get("status") != "staged":
            continue
        lines.append(f"- **{NAMES.get(sid, sid)}** (`{sid}`): `X__{sid}.zip` · `Y__{sid}.zip`"
                     f"  — staged {rec['staged']}")
    lines.append("")
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "README.md").write_text("\n".join(lines))


def cmd_stage(a) -> int:
    sids = SONGSET if a.songset else [a.song]
    if not sids or sids == [None]:
        print("give a song id or --songset", file=sys.stderr)
        return 2
    rc = 0
    for sid in sids:
        ours = pathlib.Path(a.ours) if a.ours and not a.songset else None
        rc |= stage_one(sid, ours, a.force, a.seed, with_page=not a.no_page,
                        restage=a.restage)
    return rc


# ------------------------------------------------------------------------- list / verdict


def cmd_list(a) -> int:
    key = _key()
    staged = {s: r for s, r in key.items() if r.get("status") == "staged"}
    if not staged:
        print("nothing staged. Run: python scripts/compete.py stage --songset")
    else:
        print(f"COMPETE — {len(staged)} blind pair(s) in {_rel(STAGE)}/\n")
        for sid, rec in staged.items():
            print(f"  {NAMES.get(sid, sid):<20} {sid}   X__{sid}.zip  Y__{sid}.zip   "
                  f"(staged {rec['staged']})")
        print(f"\n★ {QUESTION}")
        print("\nanswer: python scripts/compete.py verdict <sid> X|Y|tie "
              "--because \"his words\" [--code D4] [--bars a-b]")
    return 0


def cmd_verdict(a) -> int:
    key = _key()
    rec = key.get(a.song)
    if not rec or rec.get("status") != "staged":
        print(f"{a.song} is not staged (compete.py list)", file=sys.stderr)
        return 2
    pick = a.pick.upper()
    if pick not in ("X", "Y", "TIE"):
        print("pick is X, Y or tie", file=sys.stderr)
        return 2
    if a.code and a.code.upper() not in CODES:
        print(f"--code must be one of {', '.join(CODES)}", file=sys.stderr)
        return 2
    blind = rec["blind"]
    roles = {L: blind[L]["role"] for L in "XY"}
    maps = {blind[L]["role"]: blind[L]["src"] for L in "XY"}
    today = _dt.date.today().isoformat()
    if pick == "TIE":
        result = "tie"
        better = worse = None
    else:
        better = roles[pick]
        worse = roles["Y" if pick == "X" else "X"]
        result = "win" if better == "OURS" else "loss"
    entry = {
        "kind": "compete", "id": f"{today}/{a.song}-COMPETE", "date": today,
        "set": "COMPETE", "song": a.song, "name": NAMES.get(a.song, a.song),
        "pick": pick, "result": result, "better": better, "worse": worse,
        "blind": roles, "maps": maps,
        "ours_page": rec.get("page"),
        "quote": a.because, "code": (a.code.upper() if a.code else None),
        "bars": a.bars, "note": a.note,
    }
    d = _ledger()
    d.setdefault("verdicts", []).append(entry)
    _save_ledger(d)
    ours_letter = next(L for L in "XY" if roles[L] == "OURS")
    print(f"recorded {entry['id']}: {result.upper()}  (ours was {ours_letter}; "
          f"X = {roles['X']}, Y = {roles['Y']})")
    if a.because:
        print(f"  \"{a.because}\"")
    else:
        print("  ⚠️no --because — his words are the part that survives")

    # ★A loss with a reason is a bench row the same day. Codes [] scores as a MISS on
    # every query, which is the honest reading: no locator names it yet (the P3 task).
    if result == "loss" and a.because:
        b = _bench()
        row = {
            "id": f"{a.song}-compete-{today}", "map": maps["OURS"], "song": a.song,
            "name": NAMES.get(a.song, a.song), "label": "DEFECT",
            "codes": [a.code.upper()] if a.code else [], "must_not_flag": [],
            "strength": "strong", "bars": a.bars, "bars_from": ("kyle" if a.bars else None),
            "readable": True, "date": today, "verdict_id": entry["id"],
            "quote": a.because,
            "note": ("COMPETE loss vs the human map, blind X/Y. Our page at staging: "
                     f"{json.dumps(rec.get('page'))}. "
                     + ("No code given: no query claims this defect -- naming it is the "
                        "P3 task." if not a.code else
                        "If the code's query is silent on these bars, that is the P3 task.")),
        }
        b.setdefault("rows", []).append(row)
        _save_bench(b)
        print(f"  bench row {row['id']} -> {_rel(BENCH)}  "
              f"(scripts/bench.py score queries:q_all — does any query fire on it?)")
    elif result == "loss":
        print("  a loss WITHOUT a reason is not a bench row — ask what was wrong, and where")

    # unblind on disk: the pair moves out of staging under its real names
    CATALOG.mkdir(parents=True, exist_ok=True)
    dst = CATALOG / f"{a.song}_{today}"
    dst.mkdir(exist_ok=True)
    for L in "XY":
        src = STAGE / f"{L}__{a.song}.zip"
        if src.exists():
            shutil.move(str(src), str(dst / f"{L}_{roles[L]}__{a.song}.zip"))
    (dst / "VERDICT.json").write_text(json.dumps(entry, indent=1, ensure_ascii=False) + "\n")
    rec["status"] = result
    rec["verdict_id"] = entry["id"]
    _save_key(key)
    _write_readme(key)
    print(f"  pair filed -> {_rel(dst)}/   ledger: {_rel(LEDGER)}")
    return 0


# ------------------------------------------------------------------------- table


def table(verdicts: list[dict], staged: dict | None = None) -> str:
    wins = [v for v in verdicts if v["result"] == "win"]
    losses = [v for v in verdicts if v["result"] == "loss"]
    ties = [v for v in verdicts if v["result"] == "tie"]
    n = len(verdicts)
    L = ["# COMPETE — blind A/B, our best map vs the top human map of the same song"]
    if n:
        L.append(f"★ WIN RATE {len(wins)}/{n} ({100 * len(wins) / n:.0f} %)  ·  "
                 f"{len(wins)} win · {len(losses)} loss · {len(ties)} tie   "
                 f"(n={n}; a rate at this n is a count wearing a hat)")
    else:
        L.append("★ WIN RATE —/0  no pair judged yet")
    L.append("")
    L.append(f"{'song':<20} {'date':<11} {'result':<7} {'ours':<36} reason")
    L.append("-" * 100)
    for v in verdicts:
        ours = v.get("maps", {}).get("OURS", "?")
        why = v.get("quote") or ""
        tag = ""
        if v.get("code"):
            tag += f"[{v['code']}] "
        if v.get("bars"):
            tag += f"bars {v['bars']} "
        L.append(f"{v.get('name', v['song']):<20} {v['date']:<11} {v['result'].upper():<7} "
                 f"{pathlib.Path(ours).name:<36} {tag}{why}")
    if losses:
        L += ["", "LOSSES name the next thing to build:"]
        for v in losses:
            L.append(f"  {v.get('name', v['song'])}: {v.get('quote') or '(no reason recorded)'}"
                     + (f"  -> bench row {v['song']}-compete-{v['date']}" if v.get("quote") else ""))
    if staged:
        L += ["", f"staged, unjudged: {', '.join(f'{NAMES.get(s, s)} ({s})' for s in staged)}"
              "  — compete.py list"]
    return "\n".join(L)


def cmd_table(a) -> int:
    key = _key()
    staged = {s: r for s, r in key.items() if r.get("status") == "staged"}
    print(table(compete_verdicts(), staged))
    return 0


# ------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stage", help="blind a pair into for_review/compete/")
    p.add_argument("song", nargs="?", help="song id, e.g. 1f767")
    p.add_argument("--songset", action="store_true", help="stage the four standing songs")
    p.add_argument("--ours", default=None, help="our map (default: first of BEST that exists)")
    p.add_argument("--force", action="store_true", help="stage even when our page is SHIP? NO")
    p.add_argument("--restage", action="store_true",
                   help="re-blind a pair that is already staged but not yet judged "
                        "(use when our map changed after staging)")
    p.add_argument("--seed", type=int, default=None, help="shuffle seed (tests)")
    p.add_argument("--no-page", action="store_true", help="skip the verdict page (faster)")
    p.set_defaults(fn=cmd_stage)

    p = sub.add_parser("list", help="what is staged, and the question")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("verdict", help="record his pick, unblind, file the pair")
    p.add_argument("song")
    p.add_argument("pick", help="X, Y or tie")
    p.add_argument("--because", default="", help="★his words: what was wrong, and where")
    p.add_argument("--code", default=None, help="defect code if he named one: " + " ".join(CODES))
    p.add_argument("--bars", default=None, help="a-b if he pointed at a place")
    p.add_argument("--note", default="")
    p.set_defaults(fn=cmd_verdict)

    p = sub.add_parser("table", help="★the headline: win rate and every loss with its reason")
    p.set_defaults(fn=cmd_table)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
