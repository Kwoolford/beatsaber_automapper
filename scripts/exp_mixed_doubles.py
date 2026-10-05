"""W5 control: how often do human maps write mixed (one up, one down) or both-up doubles?"""
import json, pathlib, sys, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, "/home/kyle/repos/beatsaber_automapper")
from agent_mapper.score import load_map

UP, DOWN = {0, 4, 5}, {1, 6, 7}


def kind(d):
    return "u" if d in UP else "d" if d in DOWN else "o"


def one(p):
    try:
        m = load_map(p)
    except Exception:
        return None
    ns = m.notes
    if len(ns) < 100 or any(not (0 <= n.x <= 3 and 0 <= n.y <= 2 and 0 <= n.direction <= 8) for n in ns):
        return None  # modded / tiny
    at = collections.defaultdict(lambda: {0: [], 1: []})
    for n in ns:
        at[round(n.beat * 48)][n.color].append(n.direction)
    c = collections.Counter()
    for t, h in at.items():
        if len(h[0]) == 1 and len(h[1]) == 1:
            a, b = kind(h[0][0]), kind(h[1][0])
            if "o" in (a, b):
                c["other"] += 1
            elif a != b:
                c["mixed"] += 1
            elif a == "u":
                c["up"] += 1
            else:
                c["down"] += 1
    return p.stem, m.difficulty, len(ns), dict(c)


if __name__ == "__main__":
    zips = sorted(pathlib.Path("/home/kyle/repos/beatsaber_automapper/data/raw").glob("*.zip"))
    with ProcessPoolExecutor(16) as ex:
        rows = [r for r in ex.map(one, zips, chunksize=16) if r]
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "mixed_doubles.json")
    out.write_text(json.dumps(rows))
    print(len(rows), "maps")
