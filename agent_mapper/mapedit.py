#!/usr/bin/env python
"""THE SCORE IS WRITABLE — slot-level edits on a finished map, in the score's own addresses.

★**Why (TODO P1b, 2026-09-02).** Until now the only fix for "bar 45 is wrong" was
`mapctl clear` + `auto` with different levers — a knob-turner with good eyesight. A top mapper
moves *that* note. This edits the **zip** rather than the session, because everything after
`mapctl export` (idiomize, walls, arcs, chains) is zip surgery: the zip is the map the score
reads and the player plays; the session is a draft of its notes only.

Addresses are the score's row headers: `bar.beat.sub` (bar from 1, beat 1–4, sub 0–3 at the
default 1/16 lattice; `--sub 8|12` for finer). Cells are `x,y` (x 0–3 left→right, y 0–2
bottom→top). Cuts are arrows `↑↓←→↖↗↙↘•` or names `U D L R UL UR DL DR X`. Hands `L`/`R`.

    mapedit.py <map.zip> place  45.2.1 R 2,1 ↙          # add a note
    mapedit.py <map.zip> move   45.2.1 R 45.3.0 [2,1] [↙] # move (optionally re-cell / re-cut)
    mapedit.py <map.zip> flip   45.2.1 R [↗]             # reverse the cut, or set it
    mapedit.py <map.zip> delete 45.2.1 [R]               # one hand, or the whole slot
    mapedit.py <map.zip> double 45.3.0                   # add the other hand, mirrored
    mapedit.py <map.zip> mirror 44-47                    # swap hands + x across a bar range
    mapedit.py <map.zip> wall   44.1.0 45.1.0 lane 0 [--width 1] [--crouch]
    mapedit.py <map.zip> bomb   45.2.1 1,0
    mapedit.py <map.zip> arc    45.1.0 R 45.3.0          # between two existing R notes
    mapedit.py <map.zip> chain  45.1.0 R 2,0 --slices 3  # from an existing note to a tail cell
    mapedit.py <map.zip> from   edits.txt                # a batch: one op per line, no zip
    mapedit.py <map.zip> undo                            # pop the last write
    mapedit.py <map.zip> log                             # every op ever applied
    mapedit.py <map.zip> resets                          # same-parity repeats, as addresses
                                                         #  (★fix with ONE note per hand, not
                                                         #   a chain of flips — see cmd_resets)
    scripts/tutor.py <sid> --map <map.zip> --copy 9-12   # ops that copy the human's bars
    scripts/tutor.py <sid> --map <map.zip> --thin 13-24  # ops that thin ours to his density

**Checks on write** (refuse with the reason unless `--force`): a NEW parity violation
(`swing_sim`, the same model the judge uses), a same-hand gap under 150 ms (human p5), a note
inside an active wall, two notes in one cell at one beat, out-of-range cells. Only problems the
edit *introduces* block it — a map with an old violation elsewhere can still be edited.

After every write the touched bars are re-printed from the score when `--song` is given.
History lives in `<dir>/.mapedit/<stem>/NNN.zip` (undo stack) and `edits.log` (replayable).
v3 maps only (ours are 3.3.0); a v2 human map is refused with the reason.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import shlex
import shutil
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

BEATS_PER_BAR = 4
MIN_GAP_MS = 150.0
DIRS = {"U": 0, "D": 1, "L": 2, "R": 3, "UL": 4, "UR": 5, "DL": 6, "DR": 7, "X": 8,
        "↑": 0, "↓": 1, "←": 2, "→": 3, "↖": 4, "↗": 5, "↙": 6, "↘": 7, "•": 8, "O": 8}
ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "↖", 5: "↗", 6: "↙", 7: "↘", 8: "•"}
REVERSE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4, 8: 8}
H_MIRROR = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}
HAND = {"L": 0, "R": 1}


class EditError(Exception):
    pass


# ----------------------------------------------------------------------------- addresses
def parse_addr(s: str, sub: int) -> float:
    """'bar.beat.sub' -> beat (float). Also accepts a bare beat number 'b12.75'."""
    if s.startswith("b"):
        return float(s[1:])
    p = s.split(".")
    if len(p) != 3:
        raise EditError(f"address must be bar.beat.sub (e.g. 45.2.1), got {s!r}")
    bar, beat, su = (int(x) for x in p)
    if bar < 1 or not 1 <= beat <= BEATS_PER_BAR or not 0 <= su < sub:
        raise EditError(f"address out of range: bar≥1, beat 1–{BEATS_PER_BAR}, sub 0–{sub - 1}: {s}")
    return (bar - 1) * BEATS_PER_BAR + (beat - 1) + su / sub


def fmt_addr(beat: float, sub: int) -> str:
    bar = int(beat // BEATS_PER_BAR) + 1
    b = int(beat % BEATS_PER_BAR) + 1
    su = int(round((beat % 1) * sub))
    return f"{bar}.{b}.{su}"


def parse_cell(s: str) -> tuple[int, int]:
    try:
        x, y = (int(v) for v in s.split(","))
    except ValueError:
        raise EditError(f"cell must be x,y (x 0–3, y 0–2), got {s!r}") from None
    if not (0 <= x <= 3 and 0 <= y <= 2):
        raise EditError(f"cell out of range (x 0–3, y 0–2): {s}")
    return x, y


def parse_cut(s: str) -> int:
    k = s.upper() if s.isascii() else s
    if k not in DIRS:
        raise EditError(f"cut must be one of {' '.join(ARROW.values())} or U D L R UL UR DL DR X, got {s!r}")
    return DIRS[k]


def parse_hand(s: str) -> int:
    if s.upper() not in HAND:
        raise EditError(f"hand must be L or R, got {s!r}")
    return HAND[s.upper()]


def _near(a: float, b: float) -> bool:
    return abs(a - b) < 1e-3


# ----------------------------------------------------------------------------- the map
class Map:
    def __init__(self, path: pathlib.Path):
        self.path = path
        with zipfile.ZipFile(path) as zf:
            self.names = zf.namelist()
            self.blobs = {n: zf.read(n) for n in self.names}
        self.info_name = next((n for n in self.names if n.split("/")[-1].lower() == "info.dat"), None)
        cands = [n for n in self.names if n.lower().endswith(".dat")
                 and n.split("/")[-1].lower() != "info.dat" and "bpminfo" not in n.lower()]
        pref = [n for n in cands if n.split("/")[-1].lower() == "expertstandard.dat"] or \
               [n for n in cands if n.split("/")[-1].lower() == "expertplusstandard.dat"] or cands
        if not pref or self.info_name is None:
            raise EditError(f"no difficulty in {path}")
        self.dat_name = pref[0]
        self.dat = json.loads(self.blobs[self.dat_name].decode("utf-8-sig"))
        if not str(self.dat.get("version", "")).startswith("3"):
            raise EditError("v3 maps only — this is a v2 map (a human raw map?). Convert first "
                            "(scripts have parse_difficulty_dat_json + beatmap_to_v3_dict).")
        info = json.loads(self.blobs[self.info_name].decode("utf-8-sig"))
        self.bpm = float(next(v for k, v in info.items() if "beatsperminute" in k.lower()))
        for key in ("colorNotes", "bombNotes", "obstacles", "sliders", "burstSliders"):
            self.dat.setdefault(key, [])

    @property
    def notes(self) -> list[dict]:
        return self.dat["colorNotes"]

    def find(self, beat: float, color: int) -> dict | None:
        return next((n for n in self.notes if _near(n["b"], beat) and n["c"] == color), None)

    def at(self, beat: float) -> list[dict]:
        return [n for n in self.notes if _near(n["b"], beat)]

    def write(self) -> None:
        self.notes.sort(key=lambda n: (n["b"], n["c"], n["x"]))
        self.dat["obstacles"].sort(key=lambda o: o["b"])
        self.dat["sliders"].sort(key=lambda o: o["b"])
        self.dat["burstSliders"].sort(key=lambda o: o["b"])
        self.blobs[self.dat_name] = json.dumps(self.dat).encode("utf-8")
        tmp = self.path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
            for n in self.names:
                zo.writestr(n, self.blobs[n])
        tmp.replace(self.path)


# ----------------------------------------------------------------------------- checks
def _problems(m: Map) -> dict[str, set]:
    """Every problem in the map, keyed by kind, as sets of (beat, color) or descriptions."""
    from beatsaber_automapper.evaluation import swing_sim as ss

    class _N:
        __slots__ = ("beat", "x", "y", "color", "direction")

        def __init__(self, n):
            self.beat, self.x, self.y, self.color, self.direction = (
                float(n["b"]), int(n["x"]), int(n["y"]), int(n["c"]), int(n["d"]))

    class _BM:
        def __init__(self, notes):
            self.color_notes = sorted((_N(n) for n in notes), key=lambda n: (n.beat, n.color))
            self.bomb_notes = []
    out: dict[str, set] = {"parity": set(), "gap": set(), "wall": set(), "cell": set()}
    card = ss.simulate(_BM(m.notes), bpm=m.bpm)
    for color, hand in card.per_hand.items():
        for sw in hand.swings:
            if sw.reset_kind == "violation":
                out["parity"].add((round(sw.beat, 3), color))
    spb = 60.0 / m.bpm
    for color in (0, 1):
        bs = sorted({round(n["b"], 4) for n in m.notes if n["c"] == color})
        for a, b in zip(bs, bs[1:]):
            if (b - a) * spb * 1000.0 < MIN_GAP_MS - 1e-6:
                out["gap"].add((round(b, 3), color))
    for o in m.dat["obstacles"]:
        b0, b1 = o["b"], o["b"] + o.get("d", 0)
        xs = range(int(o["x"]), int(o["x"]) + max(int(o.get("w", 1)), 1))
        ys = range(int(o.get("y", 0)), int(o.get("y", 0)) + max(int(o.get("h", 5)), 1))
        for n in m.notes:
            if b0 - 1e-3 <= n["b"] <= b1 + 1e-3 and n["x"] in xs and n["y"] in ys:
                out["wall"].add((round(n["b"], 3), n["c"]))
    seen: dict = {}
    for n in m.notes:
        k = (round(n["b"], 3), n["x"], n["y"])
        if k in seen:
            out["cell"].add((k[0], n["c"]))
        seen[k] = n
    return out


def _guard(m: Map, before: dict[str, set], force: bool, sub: int) -> None:
    after = _problems(m)
    new = {k: after[k] - before[k] for k in after}
    msgs = []
    for k, label in (("parity", "NEW parity violation"), ("gap", f"same-hand gap < {MIN_GAP_MS:.0f} ms"),
                     ("wall", "note inside an active wall"), ("cell", "two notes in one cell")):
        for beat, color in sorted(new[k]):
            msgs.append(f"  {label} at {fmt_addr(beat, sub)} {'LR'[color]}")
    if msgs and not force:
        raise EditError("refused — the edit introduces:\n" + "\n".join(msgs) +
                        "\n  (--force writes it anyway)")
    if msgs:
        print("⚠️ forced through:\n" + "\n".join(msgs))


# ----------------------------------------------------------------------------- ops
def op_place(m: Map, args: list[str], sub: int) -> str:
    if len(args) != 4:
        raise EditError("place <addr> <L|R> <x,y> <cut>")
    beat = parse_addr(args[0], sub); c = parse_hand(args[1]); x, y = parse_cell(args[2]); d = parse_cut(args[3])
    if m.find(beat, c):
        raise EditError(f"{'LR'[c]} already has a note at {args[0]} — move or delete it first")
    m.notes.append({"b": beat, "x": x, "y": y, "c": c, "d": d, "a": 0})
    return f"placed {'LR'[c]} {x},{y}{ARROW[d]} at {args[0]}"


def op_move(m: Map, args: list[str], sub: int) -> str:
    if len(args) < 3:
        raise EditError("move <addr> <L|R> <addr2> [x,y] [cut]")
    beat = parse_addr(args[0], sub); c = parse_hand(args[1]); beat2 = parse_addr(args[2], sub)
    n = m.find(beat, c)
    if n is None:
        raise EditError(f"no {'LR'[c]} note at {args[0]}")
    if not _near(beat, beat2) and m.find(beat2, c):
        raise EditError(f"{'LR'[c]} already has a note at {args[2]}")
    n["b"] = beat2
    for extra in args[3:]:
        if "," in extra:
            n["x"], n["y"] = parse_cell(extra)
        else:
            n["d"] = parse_cut(extra)
    return f"moved {'LR'[c]} {args[0]} → {args[2]} ({n['x']},{n['y']}{ARROW[n['d']]})"


def op_flip(m: Map, args: list[str], sub: int) -> str:
    if len(args) < 2:
        raise EditError("flip <addr> <L|R> [cut]")
    beat = parse_addr(args[0], sub); c = parse_hand(args[1])
    n = m.find(beat, c)
    if n is None:
        raise EditError(f"no {'LR'[c]} note at {args[0]}")
    old = n["d"]
    n["d"] = parse_cut(args[2]) if len(args) > 2 else REVERSE[old]
    return f"cut {'LR'[c]} {args[0]} {ARROW[old]} → {ARROW[n['d']]}"


def op_delete(m: Map, args: list[str], sub: int) -> str:
    if not args:
        raise EditError("delete <addr> [L|R]")
    beat = parse_addr(args[0], sub)
    victims = m.at(beat) if len(args) == 1 else [n for n in [m.find(beat, parse_hand(args[1]))] if n]
    if not victims:
        raise EditError(f"nothing to delete at {' '.join(args)}")
    for n in victims:
        m.notes.remove(n)
    return f"deleted {len(victims)} note(s) at {args[0]}"


def op_double(m: Map, args: list[str], sub: int) -> str:
    if len(args) != 1:
        raise EditError("double <addr>")
    beat = parse_addr(args[0], sub)
    have = m.at(beat)
    if len(have) != 1:
        raise EditError(f"double needs exactly one note at {args[0]} (found {len(have)})")
    n = have[0]
    m.notes.append({"b": beat, "x": 3 - n["x"], "y": n["y"], "c": 1 - n["c"],
                    "d": H_MIRROR[n["d"]], "a": 0})
    return f"doubled {args[0]}: added {'LR'[1 - n['c']]} {3 - n['x']},{n['y']}{ARROW[H_MIRROR[n['d']]]}"


def op_mirror(m: Map, args: list[str], sub: int) -> str:
    if len(args) != 1:
        raise EditError("mirror <bar-bar>")
    a, _, b = args[0].partition("-")
    b0, b1 = (int(a) - 1) * BEATS_PER_BAR, int(b or a) * BEATS_PER_BAR
    k = 0
    for n in m.notes:
        if b0 - 1e-3 <= n["b"] < b1 - 1e-3:
            n["x"], n["c"], n["d"] = 3 - n["x"], 1 - n["c"], H_MIRROR[n["d"]]
            k += 1
    for o in m.dat["obstacles"]:
        if b0 - 1e-3 <= o["b"] < b1 - 1e-3:
            o["x"] = 4 - o["x"] - max(o.get("w", 1), 1)
    for s in m.dat["sliders"]:
        if b0 - 1e-3 <= s["b"] < b1 - 1e-3:
            s["x"], s["tx"], s["c"] = 3 - s["x"], 3 - s["tx"], 1 - s["c"]
            s["d"], s["tc"] = H_MIRROR[s["d"]], H_MIRROR[s["tc"]]
    for s in m.dat["burstSliders"]:
        if b0 - 1e-3 <= s["b"] < b1 - 1e-3:
            s["x"], s["tx"], s["c"], s["d"] = 3 - s["x"], 3 - s["tx"], 1 - s["c"], H_MIRROR[s["d"]]
    return f"mirrored bars {args[0]}: {k} notes"


def op_wall(m: Map, args: list[str], sub: int, width: int = 1, crouch: bool = False) -> str:
    if len(args) != 4 or args[2] != "lane":
        raise EditError("wall <addr-from> <addr-to> lane <0-3> [--width N] [--crouch]")
    b0, b1 = parse_addr(args[0], sub), parse_addr(args[1], sub)
    lane = int(args[3])
    if b1 <= b0 or not 0 <= lane <= 3:
        raise EditError("wall needs to > from and lane 0–3")
    o = {"b": b0, "d": b1 - b0, "x": lane, "y": 2 if crouch else 0,
         "w": 4 if crouch else max(width, 1), "h": 3 if crouch else 5}
    if crouch:
        o["x"] = 0
    m.dat["obstacles"].append(o)
    kind = "crouch" if crouch else f"lane {lane} w{o['w']}"
    return f"wall {args[0]}→{args[1]} {kind}"


def op_bomb(m: Map, args: list[str], sub: int) -> str:
    if len(args) != 2:
        raise EditError("bomb <addr> <x,y>")
    beat = parse_addr(args[0], sub); x, y = parse_cell(args[1])
    m.dat["bombNotes"].append({"b": beat, "x": x, "y": y})
    return f"bomb {x},{y} at {args[0]}"


def op_arc(m: Map, args: list[str], sub: int) -> str:
    if len(args) != 3:
        raise EditError("arc <addr-head> <L|R> <addr-tail>  (both notes must exist)")
    b0 = parse_addr(args[0], sub); c = parse_hand(args[1]); b1 = parse_addr(args[2], sub)
    h, t = m.find(b0, c), m.find(b1, c)
    if h is None or t is None or b1 <= b0:
        raise EditError(f"arc needs an existing {'LR'[c]} note at both {args[0]} and {args[2]}, tail after head")
    m.dat["sliders"].append({"b": b0, "c": c, "x": h["x"], "y": h["y"], "d": h["d"], "mu": 1.0,
                             "tb": b1, "tx": t["x"], "ty": t["y"], "tc": t["d"], "tmu": 1.0, "m": 0})
    return f"arc {'LR'[c]} {args[0]} → {args[2]}"


def op_chain(m: Map, args: list[str], sub: int, slices: int = 3, squish: float = 1.0) -> str:
    if len(args) != 4:
        raise EditError("chain <addr-head> <L|R> <x,y-tail> <addr-tail> [--slices N]")
    b0 = parse_addr(args[0], sub); c = parse_hand(args[1]); tx, ty = parse_cell(args[2]); b1 = parse_addr(args[3], sub)
    h = m.find(b0, c)
    if h is None or b1 <= b0:
        raise EditError(f"chain needs an existing {'LR'[c]} note at {args[0]} and a tail after it")
    m.dat["burstSliders"].append({"b": b0, "x": h["x"], "y": h["y"], "c": c, "d": h["d"],
                                  "tb": b1, "tx": tx, "ty": ty, "sc": max(slices, 2), "s": squish})
    return f"chain {'LR'[c]} {args[0]} → {tx},{ty} @ {args[3]} ×{max(slices, 2)}"


OPS = {"place": op_place, "move": op_move, "flip": op_flip, "delete": op_delete,
       "double": op_double, "mirror": op_mirror, "wall": op_wall, "bomb": op_bomb,
       "arc": op_arc, "chain": op_chain}


# ----------------------------------------------------------------------------- history
def _hist_dir(path: pathlib.Path) -> pathlib.Path:
    d = path.parent / ".mapedit" / path.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _snapshot(path: pathlib.Path) -> pathlib.Path:
    d = _hist_dir(path)
    n = len(list(d.glob("*.zip")))
    dst = d / f"{n:03d}.zip"
    shutil.copy2(path, dst)
    return dst


def _log(path: pathlib.Path, line: str) -> None:
    with (_hist_dir(path) / "edits.log").open("a") as f:
        f.write(f"{_dt.datetime.now().isoformat(timespec='seconds')}\t{line}\n")


def reset_swings(m: "Map") -> list[tuple[float, int, int, str]]:
    """Every same-parity repeat (a stop-and-reverse the simulator calls a reset) as
    (beat, color, direction, kind). Violations are the unplayable subset; the rest are
    legal but each one is a swing the player must re-cock — misterlihao's 1f767 has 4,
    a section thinned by deleting the alternating notes had 27 (2026-09-02)."""
    from beatsaber_automapper.evaluation import swing_sim as ss

    class _N:
        __slots__ = ("beat", "x", "y", "color", "direction")

        def __init__(self, n):
            self.beat, self.x, self.y, self.color, self.direction = (
                float(n["b"]), int(n["x"]), int(n["y"]), int(n["c"]), int(n["d"]))

    class _BM:
        def __init__(self, notes):
            self.color_notes = sorted((_N(n) for n in notes), key=lambda n: (n.beat, n.color))
            self.bomb_notes = []
    card = ss.simulate(_BM(m.notes), bpm=m.bpm)
    out = [(sw.beat, color, sw.direction, sw.reset_kind)
           for color, hand in card.per_hand.items() for sw in hand.swings if sw.is_reset]
    return sorted(out)


def cmd_resets(path: pathlib.Path, sub: int) -> int:
    """List the resets as addresses. ★Fixing one by flipping the SECOND note of the pair
    walks the parity error forward through every note after it (a chain of flips through
    bars 65-66 on 1f767 before it was undone). Reconcile with ONE note per hand instead:
    add one (`place`), remove one (`delete`), or make one neutral (`flip <addr> <hand> X`)
    — a dot absorbs the flip, which is what the human did at 86.1.2."""
    m = Map(path)
    rs = reset_swings(m)
    for beat, color, d, kind in rs:
        s = round(beat * sub)
        bar, r = s // (BEATS_PER_BAR * sub) + 1, s % (BEATS_PER_BAR * sub)
        print(f"{bar}.{r // sub + 1}.{r % sub:<3d} {'LR'[color]} {ARROW[d]}  {kind}")
    print(f"# {len(rs)} reset(s) · {sum(1 for r in rs if r[3] == 'violation')} violation(s)  "
          f"— fix with ONE note per hand (place / delete / flip … X), not a chain of flips")
    return 0


def cmd_undo(path: pathlib.Path) -> int:
    d = _hist_dir(path)
    snaps = sorted(d.glob("*.zip"))
    if not snaps:
        print("nothing to undo")
        return 1
    shutil.copy2(snaps[-1], path)
    snaps[-1].unlink()
    _log(path, "undo")
    print(f"restored {path.name} from {snaps[-1].name}  ({len(snaps) - 1} snapshots remain)")
    return 0


# ----------------------------------------------------------------------------- driver
def _touched_bars(lines: list[list[str]], sub: int) -> set[int]:
    bars = set()
    for parts in lines:
        for tok in parts[1:]:
            try:
                if tok.count(".") == 2:
                    bars.add(int(parse_addr(tok, sub) // BEATS_PER_BAR) + 1)
                elif "-" in tok and tok.replace("-", "").isdigit():
                    a, _, b = tok.partition("-")
                    bars.update(range(int(a), int(b or a) + 1))
            except EditError:
                pass
    return bars


def apply(path: pathlib.Path, ops: list[list[str]], sub: int, force: bool, song: str | None,
          width: int = 1, crouch: bool = False, slices: int = 3) -> int:
    m = Map(path)
    before = _problems(m)
    done = []
    try:
        for parts in ops:
            name, args = parts[0], parts[1:]
            if name not in OPS:
                raise EditError(f"unknown op {name!r}; ops: {' '.join(OPS)}")
            if name == "wall":
                done.append(op_wall(m, args, sub, width=width, crouch=crouch))
            elif name == "chain":
                done.append(op_chain(m, args, sub, slices=slices))
            else:
                done.append(OPS[name](m, args, sub))
        _guard(m, before, force, sub)
    except EditError as e:
        print(f"✗ {e}\n  nothing was written.", file=sys.stderr)
        return 2
    _snapshot(path)
    m.write()
    for parts, msg in zip(ops, done):
        _log(path, shlex.join(parts) + ("  --force" if force else ""))
        print(f"✓ {msg}")
    after = _problems(m)
    print(f"  {len(m.notes)} notes · parity violations {len(after['parity'])} "
          f"(was {len(before['parity'])}) · walls {len(m.dat['obstacles'])} · "
          f"arcs {len(m.dat['sliders'])} · chains {len(m.dat['burstSliders'])}")
    bars = sorted(_touched_bars(ops, sub))
    if song and bars:
        import score as S
        mm, sg, how, vsm, lat, sc, mc, hc = S.build(path, song, sub, "auto")
        for b0 in bars:
            s0, s1 = S._bar_range(f"{b0}-{b0}", lat)
            print()
            print("\n".join(S.render_rows(mm, sc, mc, lat, s0, s1, hc)))
    elif bars:
        print(f"  (pass --song <id> to re-read bars {bars} from the score)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("map", type=pathlib.Path)
    ap.add_argument("op", nargs="+", help="op and its arguments; see the docstring")
    ap.add_argument("--sub", type=int, default=4, help="subs per beat in addresses (4 = 1/16)")
    ap.add_argument("--song", help="song id/audio, to re-print the touched bars from the score")
    ap.add_argument("--force", action="store_true", help="write even if a check fails")
    ap.add_argument("--width", type=int, default=1, help="wall width in lanes")
    ap.add_argument("--crouch", action="store_true", help="wall: full-width crouch wall")
    ap.add_argument("--slices", type=int, default=3, help="chain: number of links")
    a = ap.parse_args()

    if a.op[0] == "undo":
        return cmd_undo(a.map)
    if a.op[0] == "resets":
        return cmd_resets(a.map, a.sub)
    if a.op[0] == "log":
        f = _hist_dir(a.map) / "edits.log"
        print(f.read_text() if f.exists() else "(no edits)")
        return 0
    if a.op[0] == "from":
        if len(a.op) != 2:
            print("from <file>", file=sys.stderr)
            return 2
        ops = [shlex.split(ln) for ln in pathlib.Path(a.op[1]).read_text().splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    else:
        ops = [a.op]
    return apply(a.map, ops, a.sub, a.force, a.song, a.width, a.crouch, a.slices)


if __name__ == "__main__":
    sys.exit(main())
