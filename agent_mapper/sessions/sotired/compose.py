"""Compose So Tired Rock from a hand-written spec: rhythm per bar + a cell figure per section.

Spec lines:
    @fig NAME                      switch both hands to figure NAME (parity carried over)
    <bar>: <16 chars>              x = a free single, d = double, l / r = that hand's single,
                                   u = a double that is MEANT to swing up, . = rest.
    <bar>.<slot> <L|R> c r DIR     a literal note

Every choice of WHEN, and of WHERE (the figures), is in the spec. This file plans only WHICH
HAND takes each free single, phrase by phrase:

★A double is a hit, so both hands must arrive at it needing a DOWN swing. Between two doubles
each hand therefore needs an odd number of swings (if it left the last double needing up).
Plain alternation gets that only when the phrase's singles split odd/odd; otherwise one hand
takes two singles in a row, at the widest gap in the phrase (never under a quarter note).
A phrase with an odd number of singles cannot be fixed by hand choice at all and is REPORTED:
add or drop one note there.
"""
import sys, pathlib, collections, itertools

UPS = {"U", "UL", "UR"}
DOWNS = {"D", "DL", "DR"}
MIN_GAP = 2          # a hand never swings twice inside a 16th (122 ms at 123 bpm)
REPEAT_GAP = 4       # a hand may take two singles in a row only across a quarter note or more

FIG = {
    # A — verse: bottom row, one lift to the middle on the outside column.
    "A": {"L": [(1, 0, "D"), (1, 0, "U"), (0, 0, "D"), (0, 1, "U")],
          "R": [(2, 0, "D"), (2, 0, "U"), (3, 0, "D"), (3, 1, "U")]},
    # P — pre-chorus: ringing 8ths, a diagonal out and a top-row chop.
    "P": {"L": [(1, 0, "D"), (0, 1, "UL"), (1, 2, "D"), (1, 0, "U")],
          "R": [(2, 0, "D"), (3, 1, "UR"), (2, 2, "D"), (2, 0, "U")]},
    # C — chorus: taller, the up-swing reaches the top outside corner.
    "C": {"L": [(1, 0, "D"), (0, 2, "UL"), (0, 0, "D"), (1, 0, "UR")],
          "R": [(2, 0, "D"), (3, 2, "UR"), (3, 0, "D"), (2, 0, "UL")]},
    # H — hype stream: bottom-row core with reaches to the outside and the top.
    "H": {"L": [(1, 0, "D"), (0, 0, "UL"), (0, 1, "DR"), (1, 0, "U"), (0, 0, "D"), (0, 2, "U")],
          "R": [(2, 0, "D"), (3, 0, "UR"), (3, 1, "DL"), (2, 0, "U"), (3, 0, "D"), (3, 2, "U")]},
    # H2 — second hype: the top row leads.
    "H2": {"L": [(0, 0, "D"), (1, 0, "U"), (1, 2, "D"), (0, 1, "UL")],
           "R": [(3, 0, "D"), (2, 0, "U"), (2, 2, "D"), (3, 1, "UR")]},
}


def parity(d):
    return "down" if d in DOWNS else "up" if d in UPS else None


def parse(spec):
    """-> list of events (abs, bar, slot, kind, fig, literal)"""
    fig, ev = "A", []
    for raw in spec.splitlines():
        ln = raw.split("#")[0].strip()
        if not ln:
            continue
        if ln.startswith("@fig"):
            fig = ln.split()[1]
            ev.append((None, 0, 0, "fig", fig, None))
            continue
        if ":" in ln:
            b, pat = ln.split(":")
            bar, pat = int(b), pat.strip()
            assert len(pat) == 16, f"bar {bar}: {len(pat)} slots"
            for s, ch in enumerate(pat):
                if ch != ".":
                    ev.append(((bar - 1) * 16 + s, bar, s, ch, fig, None))
            continue
        pos, h, c, r, d = ln.split()
        bar, slot = map(int, pos.split("."))
        ev.append(((bar - 1) * 16 + slot, bar, slot, h.lower(), fig, (int(c), int(r), d)))
    return ev


def plan_hands(ev, report):
    """Assign L/R to every free single ('x'), phrase by phrase between doubles."""
    notes = [e for e in ev if e[3] != "fig"]
    hand = {}
    need = {"L": "down", "R": "down"}
    last = {"L": -99, "R": -99}
    toggle = "R"
    i = 0
    while i < len(notes):
        j = i
        while j < len(notes) and notes[j][3] not in ("d", "u"):
            j += 1
        seg = notes[i:j]
        target = notes[j] if j < len(notes) else None
        want = "up" if (target and target[3] == "u") else "down"
        best = None
        free = [k for k, e in enumerate(seg) if e[3] == "x"]
        # candidate assignments: alternation from either hand, then with one or two repeats
        def build(start, repeats):
            h, out = start, []
            for k, e in enumerate(seg):
                if e[3] in ("l", "r"):
                    out.append(e[3].upper()); h = "L" if e[3] == "r" else "R"; continue
                if k in repeats and out:
                    h = out[-1]
                out.append(h); h = "L" if h == "R" else "R"
            return out
        def score(asg, strict=True):
            ln, nd = dict(last), dict(need)
            for e, h in zip(seg, asg):
                if e[0] - ln[h] < MIN_GAP:
                    return None
                ln[h] = e[0]; nd[h] = "up" if nd[h] == "down" else "down"
            if target:
                for h in "LR":
                    if target[0] - ln[h] < MIN_GAP:
                        return None
                if nd["L"] != nd["R"] or (strict and nd["L"] != want):
                    return None
            return ln, nd
        cands = []
        for start in (toggle, "L" if toggle == "R" else "R"):
            cands.append((0, 0, build(start, set())))
        gaps = {k: seg[k][0] - seg[k - 1][0] for k in free if k > 0}
        for k in free:
            if k > 0 and gaps[k] >= REPEAT_GAP:
                for start in (toggle, "L" if toggle == "R" else "R"):
                    cands.append((1, -gaps[k], build(start, {k})))
        for k1, k2 in itertools.combinations([k for k in gaps if gaps[k] >= REPEAT_GAP], 2):
            for start in (toggle, "L" if toggle == "R" else "R"):
                cands.append((2, -(gaps[k1] + gaps[k2]), build(start, {k1, k2})))
        cands.sort(key=lambda c: (c[0], c[1]))
        # down on the hit if any plan allows it; else both hands at least in phase (an UP
        # double — natural after back-to-back hits, e.g. d...d...d)
        for strict in (True, False):
            for _n, _g, asg in cands:
                r = score(asg, strict)
                if r:
                    best = (asg, r); break
            if best:
                break
        if best is None:
            asg = cands[0][2]
            where = f"{seg[0][1]}.{seg[0][2]}–{target[1]}.{target[2]}" if seg and target else "end"
            if target:
                report.append(f"phrase {where}: {len(seg)} singles, no hand plan brings both hands "
                              f"in phase to the double — add or drop one note")
            ln, nd = dict(last), dict(need)
            for e, h in zip(seg, asg):
                ln[h] = e[0]; nd[h] = "up" if nd[h] == "down" else "down"
            best = (asg, (ln, nd))
        asg, (last, need) = best[0], best[1]
        last, need = dict(last), dict(need)
        for e, h in zip(seg, asg):
            hand[e[0]] = h
        if seg:
            toggle = "L" if asg[-1] == "R" else "R"
        if target:
            for h in "LR":
                last[h] = target[0]; need[h] = "up" if need[h] == "down" else "down"
        i = j + 1
    return hand


def compose(spec: str):
    ev = parse(spec)
    report = []
    hand = plan_hands(ev, report)
    idx = {"L": 0, "R": 0}
    nxt = {"L": "down", "R": "down"}
    last = {"L": None, "R": None}
    fig = "A"
    notes, problems = [], []
    per_bar = collections.Counter()

    def take(h):
        cyc = FIG[fig][h]
        for _ in range(len(cyc)):
            if parity(cyc[idx[h] % len(cyc)][2]) == nxt[h]:
                break
            idx[h] += 1
        c = cyc[idx[h] % len(cyc)]
        idx[h] += 1
        return c

    def emit(h, a, bar, slot, c, r, d):
        if last[h] is not None and a - last[h] < MIN_GAP:
            problems.append(f"{bar}.{slot} {h}: same hand {a - last[h]} slot after its last swing")
        if parity(d) and parity(d) != nxt[h]:
            problems.append(f"{bar}.{slot} {h}: {d} breaks parity (needs {nxt[h]})")
        last[h] = a
        if parity(d):
            nxt[h] = "up" if parity(d) == "down" else "down"
        notes.append((bar, slot, h, c, r, d))
        per_bar[bar] += 1

    for a, bar, slot, kind, f, lit in ev:
        if kind == "fig":
            fig = f; idx = {"L": 0, "R": 0}; continue
        if kind in ("d", "u"):
            for h in ("L", "R"):
                emit(h, a, bar, slot, *take(h))
        elif lit:
            emit(kind.upper(), a, bar, slot, *lit)
        else:
            h = hand[a]
            emit(h, a, bar, slot, *take(h))
    return notes, problems + report, per_bar


if __name__ == "__main__":
    spec = pathlib.Path(sys.argv[1]).read_text()
    notes, problems, per_bar = compose(spec)
    out = pathlib.Path(sys.argv[2])
    out.write_text("\n".join(f"{b}.{s} {h} {c} {r} {d}" for b, s, h, c, r, d in notes) + "\n")
    print(f"{len(notes)} notes over bars {min(per_bar)}-{max(per_bar)}")
    for p in problems:
        print("  ✗", p)
    at = collections.defaultdict(dict)
    for b, s, h, c, r, d in notes:
        at[(b, s)][h] = d
    mixed = [f"{b}.{s}" for (b, s), hs in sorted(at.items())
             if len(hs) == 2 and parity(hs["L"]) != parity(hs["R"])]
    up = [f"{b}.{s}" for (b, s), hs in sorted(at.items())
          if len(hs) == 2 and parity(hs["L"]) == parity(hs["R"]) == "up"]
    n2 = sum(1 for hs in at.values() if len(hs) == 2)
    print(f"doubles {n2}: mixed {len(mixed)} {' '.join(mixed)}")
    print(f"           up    {len(up)} {' '.join(up)}")
