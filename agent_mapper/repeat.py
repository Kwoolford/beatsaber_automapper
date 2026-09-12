#!/usr/bin/env python
"""REPEAT — bring a figure back, because a map you cannot lock into is not fun.

★`READING.md`'s FIRST rule for finding an unfun map: *"Does a cell COME BACK? A scatter has
nothing to lock into."* `q_scatter` has read that since 2026-09-10 and it is the **last red on
the songset**. This is the builder's answer to it.

## What it is fixing, and why it is the same bug as the walls

Measured 2026-09-12 over **400 human Experts**, a human map's 4-bar block echo — how much of each
block is a figure the map has **already played** — sits at median **0.602**, p10 0.505, p90 0.694,
**sd 0.075**. Humans agree with each other closely here. Our four builds score **0.388–0.423**:
below the human p10 *on every song*, with almost no variance between songs.

⇒**Another builder constant that ignores the music**, exactly like the wall duration was.
`idiomize.py` re-places every block's cells from the human vocabulary **without ever asking
whether this block's music has been heard before**, so each block is drawn fresh and nothing
comes back. A mapper does the opposite: when the song returns to a phrase, his hands return to
the shape they played on it.

## What it does

**The song's own section analysis decides when the song has come back** — not a similarity
threshold. `outputs/structure_cache/<sid>.json` already labels the repeats (1f333 is
`A B C D B C D E F G D`, 1f913 is `A B A B A C A`), so a block in the second `D` is matched to the
block at **the same offset inside the first `D`**, which keeps the phrase position. The later
block's notes then **keep their own times and their own hands** and take the earlier block's
**cells and cut directions**, in order, cycling if the counts differ. The figure comes back; the
rhythm stays the song's.

🔴**A per-block song fingerprint was tried first and abandoned** (2026-09-12c): kit-pattern +
onset per slot, scored like `_echo`. It **maxes out at 0.47** on 1f333 even between two passes of
the same labelled section, because keeping the slot index in the key makes any percussion-detection
jitter break the match — and picking a threshold under that ceiling is choosing how many blocks
fire, not detecting that the song repeated. The structure cache needs **no threshold at all**.

⚠️**The mechanism is the objective; the echo number is only the validation.** Rewriting cells to
raise a number that counts cells would be the `h_dist` failure — the reason this is written as
*"return to the shape you played when the song last did this"* and not as *"raise echo to 0.60"*
is that only the first is a thing a mapper does.

🔴**MUST RUN AFTER `idiomize`** — for the same reason `walls.py` must: `idiomize` redraws every
note's column, so a figure copied before it would be overwritten. And walls must come after
BOTH, since a wall is chosen against the final note columns.

Usage:
    python agent_mapper/repeat.py in.zip --out out.zip --song 1f333
    python agent_mapper/repeat.py in.zip --out out.zip --song 1f333 --report
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import zipfile
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

B = 4                      # block size in bars — q_scatter's own
BEATS_PER_BAR = 4
MIN_NOTES = 6              # a block with fewer notes has no figure to speak of
COUNT_TOL = 0.40           # the two blocks' per-hand note counts must be within this ratio


def section_repeats(sid: str) -> list[tuple[int, list[int]]]:
    """`[(bar, [every earlier bar playing the same thing])]` from the song's own section analysis.

    A section labelled `D` at bar 113 is the same music as the `D` at bar 55, so bar 113+k
    answers bar 55+k. ★No threshold: the structure cache has already decided the song repeated,
    and asking it is the whole point — see the docstring on why a fingerprint could not.

    ★**Every earlier occurrence is offered, not just the first**, because which one to bring back
    is a real choice and it matters. Measured 2026-09-12e on the two songs that still failed
    SCATTER: always copying the FIRST occurrence gives 1f913 echo 0.462, always copying the MOST
    RECENT gives 0.488 — enough to clear it — while on 1f333 most-recent is slightly *worse*
    (0.474 vs 0.484). ⇒Neither rule dominates, so the caller picks per block by what actually
    swings (`apply_repeats`). A mapper makes the same choice: he reuses whichever pass of the
    phrase fits the hand he is on.
    """
    p = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "structure_cache" / f"{sid}.json"
    if not p.exists():
        return []
    secs = json.loads(p.read_text())["sections"]
    seen: dict[str, list[tuple[int, int]]] = {}
    out = []
    for s in secs:
        lab, bar0, n = str(s["label"]), int(s["bar0"]), int(s["bars"])
        for f0, fn in seen.get(lab, []):
            for k in range(min(n, fn)):
                out.append((bar0 + k, f0 + k))
        seen.setdefault(lab, []).append((bar0, n))
    # collapse to one row per bar, keeping every earlier answer, most recent first
    by_bar: dict[int, list[int]] = {}
    for bar, src in out:
        by_bar.setdefault(bar, []).append(src)
    return [(b, sorted(set(v), reverse=True)) for b, v in sorted(by_bar.items())]


def plan_repeats(notes: list[dict],
                 reps: list[tuple[int, list[int]]]) -> list[tuple[int, list[int], str]]:
    """`[(block, [earlier blocks it could echo], why)]`, most recent candidate first.

    ⚠️A candidate is only offered when BOTH blocks have enough notes and their per-hand counts
    are within `COUNT_TOL` — cycling a 3-note figure onto 20 notes is not a figure coming back,
    it is a stutter. That guard is what stops this from being a cell-rewriter that raises a number.
    """
    by_block: dict[int, dict[int, list[int]]] = {}
    for i, n in enumerate(notes):
        bar = int(float(n.get("b", 0.0)) // BEATS_PER_BAR) + 1
        b0 = ((bar - 1) // B) * B + 1
        by_block.setdefault(b0, {}).setdefault(int(n.get("c", 0)), []).append(i)

    def fits(mine: dict, his: dict) -> bool:
        for h in (0, 1):
            nm, nh = len(mine.get(h, [])), len(his.get(h, []))
            if nm == 0 and nh == 0:
                continue
            if nm == 0 or nh == 0 or abs(nm - nh) / max(nm, nh) > COUNT_TOL:
                return False
        return True

    seen, out = set(), []
    for bar, src_bars in reps:
        b0 = ((bar - 1) // B) * B + 1
        if b0 in seen or b0 not in by_block:
            continue
        if sum(len(v) for v in by_block[b0].values()) < MIN_NOTES:
            continue
        cands = []
        for sb in src_bars:
            s0 = ((sb - 1) // B) * B + 1
            if s0 >= b0 or s0 not in by_block or s0 in cands:
                continue
            if fits(by_block[b0], by_block[s0]):
                cands.append(s0)
        if not cands:
            continue
        seen.add(b0)
        out.append((b0, cands,
                    f"bars {b0}-{b0 + B - 1} are the section that played at "
                    + " / ".join(str(c) for c in cands)))
    return out


def _swing_cost(notes: list[dict], bpm: float) -> tuple[int, int]:
    """`(parity violations, resets)` for these notes, via the same simulator the score uses."""
    try:
        from beatsaber_automapper.evaluation import swing_sim as ss
    except Exception:  # noqa: BLE001
        return 0, 0

    class _N:
        __slots__ = ("beat", "x", "y", "color", "direction")

        def __init__(self, n):
            self.beat = float(n.get("b", 0.0)); self.x = int(n.get("x", 0))
            self.y = int(n.get("y", 0)); self.color = int(n.get("c", 0))
            self.direction = int(n.get("d", 0))

    class _BM:
        def __init__(self, ns):
            self.color_notes = [_N(n) for n in ns]; self.bomb_notes = []
    card = ss.simulate(_BM(sorted(notes, key=lambda n: float(n.get("b", 0.0)))), bpm=bpm)
    return int(card.violations), int(card.resets)


OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5, 8: 8}


def _repair(notes: list[dict], touched: dict[int, list[int]], bpm: float,
            base: tuple[int, int], passes: int = 4) -> int:
    """Bring the swing cost back to `base` by inverting a RUN of one hand's new notes at a time.

    Returns the **number of notes flipped**, or **-1** if the cost could not be brought back.

    ★★**A reset is a parity PHASE problem, not a bad note.** Parity alternates, so flipping one
    note inverts every swing after it. That is why single-note repair is refuted in *both*
    directions (2026-09-12c: neither reverting a note nor flipping one moved the count on any
    failing block) and why `TODO`'s landmine says *flipping the second note cascades*. The repair
    that matches the mechanism inverts a **run**: the copied figure swings the other way from the
    break until it re-syncs.

    Measured on 1f913's 14 planned blocks, all at zero reset cost:

    | repair | blocks freed |
    |---|---|
    | revert one note / flip one note | **0** |
    | invert a whole hand | 6 |
    | invert a hand's **suffix** | 9 |
    | invert an **interval** (this) | **14 of 14**, in 5 s |

    A suffix cannot fix a break that re-syncs before the block ends, which is most of them.
    Hands are searched one at a time and alternated, because the two hands' parity chains are
    independent until they share an instant.
    """
    cur, flipped = _swing_cost(notes, bpm), 0
    for _ in range(passes):
        improved = False
        for h in (0, 1):
            L = touched.get(h, [])
            best = None
            for k in range(len(L)):
                for j in range(k + 1, len(L) + 1):
                    for i in L[k:j]:
                        notes[i]["d"] = OPPOSITE.get(notes[i]["d"], notes[i]["d"])
                    c = _swing_cost(notes, bpm)
                    for i in L[k:j]:
                        notes[i]["d"] = OPPOSITE.get(notes[i]["d"], notes[i]["d"])
                    # ⚠️Shortest run wins at equal cost: every flipped note is a note whose
                    # DIRECTION no longer matches the figure being brought back, and `figures()`
                    # counts direction — so a bigger repair gives back more of the echo it was
                    # applied to win. Measured 2026-09-12d: repairing to 0 resets costs ~0.10 of
                    # the ~0.15 echo the copies gain.
                    if c < cur and (best is None or (c, j - k) < (best[0], best[2] - best[1])):
                        best = (c, k, j)
            if best is not None:
                c, k, j = best
                for i in L[k:j]:
                    notes[i]["d"] = OPPOSITE.get(notes[i]["d"], notes[i]["d"])
                cur, improved = c, True
                flipped += j - k
        if cur <= base or not improved:
            break
    return (flipped if cur <= base else -1)


def apply_repeats(notes: list[dict], plan: list[tuple[int, int, str]],
                  bpm: float = 0.0) -> tuple[int, int]:
    """Rewrite each planned block's cells + cut directions from its source block.

    Returns `(notes moved, blocks kept)`.

    The notes keep their **own beat and their own hand** — only `x`, `y` and `d` are taken from the
    figure being brought back. So the rhythm stays the song's and only the shape returns.

    🔴**Every block is then checked and REVERTED if it costs a swing** (`bpm=0` disables the
    check — that is what `--allow-resets` passes). Unguarded, this pass clears SCATTER on all four
    songset maps and pushes **resets from 0 to 15–24** (the humans of those songs: 0–4). Parity
    violations stay 0 so nothing is unplayable, but a reset is a break in the flow and trading
    flow for echo is not the deal.

    🔴🔴**AND THE REPAIR IS NOT A ONE-NOTE FIX — measured both ways, 2026-09-12c.** On the four
    failing blocks of 1f913 (1–3 new resets each): reverting **any single** moved note back to its
    old cell leaves the count **unchanged**, and flipping **any single** moved note's cut direction
    to its opposite leaves it unchanged too. The swing cost is live — 5 of 12 random direction
    flips elsewhere in the same map do move it — so this is a real negative, not a dead measurement.
    ⇒The reset is structural to the copied figure in its new context, which is the same thing
    `TODO`'s reset-reconciliation landmine already says: *flipping the second note cascades*.
    ✅**2026-09-12d — solved.** A reset is a parity PHASE problem, so the repair inverts a RUN
    (`_repair`), and every planned block then applies at zero reset cost.
    ✅**2026-09-12e — and the SOURCE is chosen, not assumed.** Each block is offered every earlier
    occurrence of its section, and the one kept is the one that survives with **the most of its
    figure intact** — fewest notes flipped by the repair. That matters because a flipped note is a
    note whose direction no longer matches the figure being brought back, and `figures()` counts
    direction. Neither "always the first occurrence" nor "always the most recent" dominates
    (1f913 prefers recent, 1f333 prefers first), so neither is hard-coded.
    """
    by_block: dict[int, dict[int, list[int]]] = {}
    for i, n in enumerate(notes):
        bar = int(float(n.get("b", 0.0)) // BEATS_PER_BAR) + 1
        b0 = ((bar - 1) // B) * B + 1
        by_block.setdefault(b0, {}).setdefault(int(n.get("c", 0)), []).append(i)
    for d in by_block.values():
        for h in d:
            d[h].sort(key=lambda i: float(notes[i].get("b", 0.0)))
    base = _swing_cost(notes, bpm) if bpm else (0, 0)

    def copy_from(b: int, src: int) -> tuple[list, dict]:
        undo, touched = [], {0: [], 1: []}
        for h, idxs in by_block.get(b, {}).items():
            srcs = by_block.get(src, {}).get(h, [])
            if not srcs:
                continue
            for j, i in enumerate(idxs):
                s = notes[srcs[j % len(srcs)]]
                was = (notes[i].get("x"), notes[i].get("y"), notes[i].get("d"))
                now = (s.get("x"), s.get("y"), s.get("d"))
                if was == now:
                    continue
                undo.append((i, was))
                notes[i]["x"], notes[i]["y"], notes[i]["d"] = now
                touched.setdefault(h, []).append(i)
        return undo, touched

    def put_back(undo: list) -> None:
        for i, (x, y, dd) in undo:
            notes[i]["x"], notes[i]["y"], notes[i]["d"] = x, y, dd

    moved = kept = 0
    for b, cands, _why in plan:
        best = None                    # (flipped, n_moved, src, the cells that won)
        for src in cands:
            undo, touched = copy_from(b, src)
            if not undo:
                put_back(undo)
                continue
            if not bpm:
                flipped = 0
            else:
                flipped = 0 if _swing_cost(notes, bpm) <= base else _repair(notes, touched, bpm, base)
            if flipped >= 0 and (best is None or (flipped, -len(undo)) < (best[0], -best[1])):
                cells = {i: (notes[i]["x"], notes[i]["y"], notes[i]["d"]) for i, _ in undo}
                best = (flipped, len(undo), src, cells)
            put_back(undo)
            if best is not None and best[0] == 0 and not bpm:
                break
        if best is None:
            continue
        for i, (x, y, dd) in best[3].items():
            notes[i]["x"], notes[i]["y"], notes[i]["d"] = x, y, dd
        moved += best[1]
        kept += 1
    return moved, kept


def repeat_zip(src: pathlib.Path, dst: pathlib.Path, song: str | None = None,
               report: bool = False, allow_resets: bool = False) -> tuple[int, int]:
    """Copy `src` to `dst` with figures brought back. Returns (blocks echoed, notes moved)."""
    import score as S
    sid, _audio, _how = S.resolve_song(song, src)
    reps = section_repeats(sid)
    if not reps:
        raise ValueError(f"no structure cache for {sid} — nothing says the song repeats")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="repeat_"))
    try:
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
            names = zf.namelist()
        cands = [n for n in names if n.lower().endswith(".dat") and "info" not in n.lower()]
        dat = next((n for n in cands if "expert" in n.lower()), cands[0] if cands else None)
        if dat is None:
            raise ValueError("no difficulty .dat in the zip")
        f = tmp / dat
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        if not str(d.get("version", "")).startswith("3"):
            raise ValueError("only v3 maps are supported (ours are 3.3.0)")
        notes = d.get("colorNotes") or []
        if len(notes) < 20:
            raise ValueError("too few notes to bring a figure back")
        plan = plan_repeats(notes, reps)
        moved, kept = apply_repeats(notes, plan,
                                    bpm=0.0 if allow_resets else S.load_map(src).bpm)
        if report:
            for b, cands, why in plan:
                print(f"  {why}")
            note = ("every planned block applied (--allow-resets)" if allow_resets else
                    "brought a figure back at zero reset cost; the rest could not be repaired")
            print(f"  {kept} of {len(plan)} planned blocks {note}")
        d["colorNotes"] = notes
        f.write_text(json.dumps(d), encoding="utf-8")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    zo.write(p, p.relative_to(tmp).as_posix())
        return kept, moved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--song", default=None, help="song id (default: resolved from the zip)")
    ap.add_argument("--report", action="store_true", help="print which block echoes which")
    ap.add_argument("--allow-resets", action="store_true",
                    help="apply every planned block even when it costs a swing -- this is the "
                         "arm that clears SCATTER on all four songset maps at 15-24 resets")
    a = ap.parse_args()
    nb, moved = repeat_zip(a.zip, a.out, a.song, a.report, a.allow_resets)
    print(f"{a.zip.name}: {nb} block(s) brought a figure back, {moved} note(s) re-cut -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
