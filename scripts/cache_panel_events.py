#!/usr/bin/env python
"""Event caches for the human-vs-human PANEL (two mappers, one song) — the control songprint needs.

Each map is read against ITS OWN zip's audio (two uploads of one song can differ in lead silence).
    python scripts/cache_panel_events.py [max_pairs]
"""
import json, pathlib, sys, time, traceback
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_mapper.score import _audio_from_zip
from agent_mapper import events

P = json.loads((ROOT / "outputs/human_panel_2026-09-13.json").read_text())
pairs, seen = [], set()
for r in P:
    k = tuple(sorted((r["a"], r["b"])))
    if k not in seen and r.get("difficulty") in ("Expert", "ExpertPlus"):
        seen.add(k); pairs.append(k)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 70
ids = [i for p in pairs[:n] for i in p]
print(f"{len(ids)} maps from {min(n, len(pairs))} pairs", flush=True)
for i, sid in enumerate(ids):
    out = ROOT / "outputs/event_cache" / f"{sid}.6s.json"
    if out.exists():
        continue
    t = time.time()
    try:
        a = _audio_from_zip(ROOT / "data/raw" / f"{sid}.zip", sid)
        if a is None:
            print(sid, "no audio", flush=True); continue
        events.analyse(a, six=True)
        print(f"[{i+1}/{len(ids)}] {sid} {time.time()-t:.0f}s", flush=True)
    except Exception:
        print(sid, "FAILED", traceback.format_exc().splitlines()[-1], flush=True)
print("COMPLETE", flush=True)
