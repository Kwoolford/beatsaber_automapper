#!/usr/bin/env python
"""WALLS — the element we have never emitted, and 93 % of human maps have.

★**Why this exists.** Measured 2026-08-19j over 147 paired maps: **137 of 147 human maps contain
walls (median 86 per map) and we emit ZERO in every map we have ever produced.** On Fallen Kingdom
the human map has **124** obstacles and ours has **0** — and that is the song Kyle called *"really
empty"*, a complaint five separate instruments have failed to explain, every one of them by
looking at notes. A map missing an entire physical layer would feel empty at any note count.

This is deliberately a **post-processor on a finished zip**, not a generator change: it can be
given to his ear as a `[WALLS]` arm without touching a single note of the map it is compared to,
so the A/B isolates exactly one thing.

## The vocabulary it copies, measured from 135 human maps (16,504 vanilla walls)

⚠️**54 % of the corpus's walls are MODDED** (Mapping/Noodle Extensions repurpose the fields —
negative durations, lane −4750, width 1000) **and are discarded.** Reading them as vanilla gives a
median duration of *minus 2.5 beats*, which is how this was caught.

| property | human |
|---|---|
| walls per map | median **84** (p10 19, p90 222) |
| duration | median **0.12** beats (p90 1.25) |
| width | **90 % are 1 lane** |
| lane | **52 % at x=0, 41 % at x=3** — 93 % in an OUTER lane |
| height | 62 % crouch, 38 % full |
| notes inside a wall's own lanes while it is active | median **0.000**, any overlap only **8 %** |

⇒The idiom is a **short, one-lane wall hugging an outer edge, in a lane the hands are not using**.
That last line is the hard constraint: a wall where a note is is unplayable, and humans essentially
never do it.

## 🔴 2026-09-12 — the duration row above is a POOLED marginal, and sampling it was the bug

The table's `duration median 0.12 (p90 1.25)` is measured over **16,504 walls pooled across 135
maps**. `plan_walls` drew every wall independently from that pool, which reproduces the pooled
median and **destroys the per-map structure**. Measured on 400 corpus maps, a human map's walls
are not one population but **three, and the mix is a per-map choice**:

| mode | duration | share of a map's walls (p10 / med / p90) | maps where it is ≥70 % |
|---|---|---|---|
| **instant** | < 0.15 beats | 0.00 / **0.41** / 0.88 | 96 / 400 |
| mid | 0.15–0.75 | 0.00 / 0.18 / 0.67 | 34 / 400 |
| **corridor** | ≥ 0.75 beats | 0.03 / **0.27** / 0.81 | 55 / 400 |

Total wall beats per map: p10 **13.9**, median **60.3**, p90 **172.9**.

The log-uniform draw over (0.03, 1.25) can emit **only the middle mode** — no instant marker, and
no corridor, whose human maximum is 9.5 beats. So every map we have ever shipped lands in the
rarest cohort (34/400 = 8.5 %) *on every song*, with a total of **23.8–28.9 wall beats regardless
of the music** against humans at 12.1–203.7 on the same four songs. That constant is what
`q_elements`' coverage branch fires on.
★This is the `h_dist` failure in a new place: **matching a pooled marginal is not matching the
thing**. Draw the per-map MIX, then place each mode by what the song is doing.

## Where each mode goes — one signal survived, one did not (2026-09-12)

Tested on 567 corpus maps with cached onsets, as local onset rate over the wall's span ÷ the
song's own rate:

- ✅**CONFIRMED — corridors sit where the onsets thin out**: median **0.90×**, p75 1.00, and
  74 % of maps put them below their own song average. A long wall goes where the song stops
  chopping. This is the rule `plan_walls` now places corridors by.
- 🔴**NOT REPRODUCED — instants at high onset density**: median **0.98×** over 484 maps. The
  songset showed 1.2–2.0× but that is n=3. ⇒Instants keep the collision-constrained random
  placement; **we have no song-side rule for them and must not pretend otherwise.**

Usage:
    python agent_mapper/walls.py in.zip --out out.zip
    python agent_mapper/walls.py in.zip --out out.zip --per-map 84 --seed 0
    python agent_mapper/walls.py in.zip --out out.zip --song 1f767   # song-driven corridors
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
import zipfile

import numpy as np

# Measured human medians (vanilla only) — see the table above.
# ⚠️Sampled log-uniformly between the human p10 and p90 — NOT between the median and p90,
# which cannot reproduce the median (that first attempt gave 0.38 against a human 0.12).
# 🔴Retained only for `--legacy`: this is the pooled marginal whose sampling was the 2026-09-12 bug.
DUR_BEATS = (0.03, 1.25)      # (p10, p90); human median 0.12, this yields ≈0.19
OUTER_LANES = (0, 3)
SAFETY_BEATS = 0.30           # keep this clear of any note in the wall's lane, both sides
MIN_GAP_BEATS = 2.0           # do not stack walls on top of each other in one lane

# --- the three modes, measured over 400 corpus maps (see the docstring table) ---
INSTANT_BEATS = 0.04          # the human's lane marker: a wall with no length at all
MID_BEATS = (0.15, 0.75)      # log-uniform inside the mid band
CORRIDOR_MIN = 0.75           # a corridor is a wall you have to lean around
CORRIDOR_MAX = 9.50           # the longest vanilla wall on the songset's humans
# among a map's NON-corridor walls the corpus splits instant:mid ≈ 41:18
INSTANT_SHARE_OF_SHORT = 0.70
# a corridor is only placed where the local onset rate is under the song's own — the one
# song-side signal that survived the corpus test (median 0.90x, 74 % of maps). Not a
# threshold on an absolute rate: absolute norms have been refuted four times in this repo.
CORRIDOR_QUIET_FRAC = 0.60    # ≥60 % of the span must be below the song's median onset rate


def _difficulty_dat(names: list[str]) -> str | None:
    cands = [n for n in names if n.lower().endswith(".dat") and "info" not in n.lower()]
    return next((n for n in cands if "expert" in n.lower()), cands[0] if cands else None)


def _shape(rng: np.random.Generator) -> dict:
    """The 62 % crouch / 38 % full split, as the two height fields."""
    crouch = rng.random() < 0.62
    return {"w": 1, "y": 2 if crouch else 0, "h": 3 if crouch else 5}


def _free(note_beats: np.ndarray, note_x: np.ndarray, lane: int,
          start: float, dur: float) -> bool:
    """No note of this lane inside [start, start+dur] plus a SAFETY_BEATS margin either side.

    ⚠️The margin is the point: a wall that ends a hair before a note in the same lane is still a
    wall the player is inside when they have to swing there.
    """
    lo, hi = start - SAFETY_BEATS, start + dur + SAFETY_BEATS
    return not ((note_beats >= lo) & (note_beats <= hi) & (note_x == lane)).any()


def quiet_mask(onset_beats: np.ndarray, span: tuple[float, float],
               res: float = 0.5) -> tuple[np.ndarray, float]:
    """Where is this song below its OWN onset rate? Returns (mask over `res`-beat bins, res).

    The reference is the song's own median bin rate, never an absolute number — absolute norms
    have been refuted four times in this repo (see TODO's three rules).
    """
    b0, b1 = span
    nb = max(int((b1 - b0) / res), 4)
    rate, _ = np.histogram(onset_beats, bins=nb, range=(b0, b1))
    if not rate.any():
        return np.zeros(nb, bool), res
    return rate <= np.median(rate), res


def plan_corridors(note_beats: np.ndarray, note_x: np.ndarray, span: tuple[float, float],
                   quiet: np.ndarray, res: float, budget: int,
                   rng: np.random.Generator) -> list[dict]:
    """Long walls, in the note-free outer-lane gaps where the onsets have thinned out.

    A corridor's LENGTH is the gap the map leaves, not a draw from a distribution — which is why
    this varies per song where the pooled sampler could not. ★The quiet test is the one song-side
    signal that survived the 567-map corpus check; see the module docstring.
    """
    b0, b1 = span
    cands: list[tuple[float, float, float, int]] = []   # (quiet_frac, start, dur, lane)
    for lane in OUTER_LANES:
        lb = np.sort(note_beats[note_x == lane])
        # the gaps this lane leaves, bounded by the map's own span
        edges = np.concatenate(([b0], lb, [b1]))
        for i in range(len(edges) - 1):
            g0, g1 = edges[i] + SAFETY_BEATS, edges[i + 1] - SAFETY_BEATS
            while g1 - g0 >= CORRIDOR_MIN:
                dur = min(g1 - g0, CORRIDOR_MAX)
                i0 = int((g0 - b0) / res)
                i1 = max(i0 + 1, int((g0 + dur - b0) / res))
                q = float(quiet[i0:min(i1, len(quiet))].mean()) if i0 < len(quiet) else 0.0
                if q >= CORRIDOR_QUIET_FRAC:
                    cands.append((q, float(g0), float(dur), lane))
                g0 += dur + MIN_GAP_BEATS      # walk the gap, do not stack
    # quietest first; ties broken by the rng so two lanes do not always resolve the same way
    rng.shuffle(cands)
    cands.sort(key=lambda c: -c[0])
    out, placed = [], {x: [] for x in OUTER_LANES}
    for q, start, dur, lane in cands:
        if len(out) >= budget:
            break
        if any(abs(start - p) < MIN_GAP_BEATS for p in placed[lane]):
            continue
        placed[lane].append(start)
        out.append({"b": round(start, 3), "d": round(dur, 3), "x": lane, **_shape(rng)})
    return out


def plan_walls(note_beats: np.ndarray, note_x: np.ndarray, span: tuple[float, float],
               n_walls: int, rng: np.random.Generator,
               onset_beats: np.ndarray | None = None,
               legacy: bool = False) -> list[dict]:
    """Choose wall placements that no note collides with, in the human's THREE modes.

    Without `onset_beats` (or with `legacy=True`) this falls back to the pooled log-uniform draw
    that shipped until 2026-09-12 — one mode, song-blind. 🔴That fallback is the documented bug;
    it is kept only so a map with no onset cache still gets walls rather than none.
    """
    b0, b1 = span
    out: list[dict] = []
    placed: dict[int, list[float]] = {x: [] for x in OUTER_LANES}

    if onset_beats is not None and len(onset_beats) >= 20 and not legacy:
        quiet, res = quiet_mask(np.asarray(onset_beats, float), span)
        # the corridor budget is the corpus median share of a map's walls, and the SONG decides
        # how many of those it can actually seat — a song with no quiet gaps gets fewer.
        want_c = int(round(n_walls * 0.27))
        out = plan_corridors(note_beats, note_x, span, quiet, res, want_c, rng)
        for o in out:
            placed[o["x"]].append(o["b"])

    # the short modes fill the rest. ⚠️Their placement is random by DESIGN: "instants sit where
    # the onsets are dense" was NOT REPRODUCED on 484 corpus maps (median 0.98x), so there is no
    # song-side rule to place them by and inventing one would be a story, not a measurement.
    cands = np.arange(b0 + 4.0, max(b1 - 4.0, b0 + 4.0), 0.5)
    rng.shuffle(cands)
    for start in cands:
        if len(out) >= n_walls:
            break
        lane = int(OUTER_LANES[rng.integers(0, len(OUTER_LANES))])
        if legacy or onset_beats is None or len(onset_beats) < 20:
            dur = float(np.exp(rng.uniform(np.log(DUR_BEATS[0]), np.log(DUR_BEATS[1]))))
        elif rng.random() < INSTANT_SHARE_OF_SHORT:
            dur = INSTANT_BEATS
        else:
            dur = float(np.exp(rng.uniform(np.log(MID_BEATS[0]), np.log(MID_BEATS[1]))))
        if not _free(note_beats, note_x, lane, float(start), dur):
            continue
        if any(abs(start - p) < MIN_GAP_BEATS for p in placed[lane]):
            continue
        placed[lane].append(float(start))
        out.append({"b": round(float(start), 3), "d": round(dur, 3), "x": lane, **_shape(rng)})
    out.sort(key=lambda o: o["b"])
    return out


def song_onset_beats(src: pathlib.Path, song: str | None = None) -> np.ndarray | None:
    """The song's cached onsets, in the MAP's beat space. None when there is no cache.

    ⚠️Returning None is not a failure — it selects the legacy one-mode fallback, and the caller
    says so out loud. A silent downgrade is how the pooled-marginal bug survived three weeks.
    """
    try:
        import score as _S
        sid, _audio, _how = _S.resolve_song(song, src)
        cache = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "onset_cache" / f"{sid}.npz"
        if not cache.exists():
            return None
        t = np.asarray(np.load(cache, allow_pickle=True)["onsets"], dtype=float)
        m = _S.load_map(src)
        return np.array([m.beat_of(float(x)) for x in t], dtype=float)
    except Exception:
        return None


def add_walls(src: pathlib.Path, dst: pathlib.Path, per_map: int = 84,
              seed: int = 0, song: str | None = None, legacy: bool = False) -> int:
    """Copy `src` to `dst` with walls added to its Expert difficulty. Returns the count."""
    rng = np.random.default_rng(seed)
    onsets = None if legacy else song_onset_beats(src, song)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="walls_"))
    try:
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
            names = zf.namelist()
        dat = _difficulty_dat(names)
        if dat is None:
            raise ValueError("no difficulty .dat in the zip")
        f = tmp / dat
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        if not str(d.get("version", "")).startswith("3"):
            raise ValueError("only v3 maps are supported (ours are 3.3.0)")
        notes = d.get("colorNotes") or []
        if len(notes) < 20:
            raise ValueError("too few notes to place walls around")
        nb = np.array([n.get("b", 0.0) for n in notes], dtype=float)
        nx = np.array([n.get("x", 0) for n in notes], dtype=int)
        walls = plan_walls(nb, nx, (float(nb.min()), float(nb.max())), per_map, rng,
                           onset_beats=onsets, legacy=legacy)
        d["obstacles"] = walls
        f.write_text(json.dumps(d), encoding="utf-8")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zo:
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    zo.write(p, p.relative_to(tmp).as_posix())
        return len(walls)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--per-map", type=int, default=84, help="human median is 84")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--song", default=None,
                    help="song id for the onset cache (default: resolved from the zip). "
                         "Without a cache the placer falls back to the legacy one-mode draw.")
    ap.add_argument("--legacy", action="store_true",
                    help="the pre-2026-09-12 pooled log-uniform draw: one mode, song-blind")
    a = ap.parse_args()
    had = a.legacy or song_onset_beats(a.zip, a.song) is not None
    n = add_walls(a.zip, a.out, a.per_map, a.seed, song=a.song, legacy=a.legacy)
    mode = "legacy (one mode, song-blind)" if a.legacy else (
        "song-driven corridors + instants" if had else
        "LEGACY FALLBACK — no onset cache for this song, so corridors could not be placed")
    print(f"{a.zip.name}: added {n} walls -> {a.out}   [{mode}]")
    if n < a.per_map:
        print(f"  (asked for {a.per_map}; the rest had no note-free slot — that is the "
              f"constraint working, not a failure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
