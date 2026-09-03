#!/usr/bin/env python
"""THE SCORE — song and map on ONE time lattice, every slot, as text and as arrays.

★**Why this exists (Kyle, 2026-09-02).** *"The model doesn't have the visibility that I
do when evaluating a map. Convert the map to text or a numpy array where the rows are
possible note placements and the columns are the notes, matched with another array of
the song in note-sheet format with lyrics and all. This granular visibility with deep
timings is what the model does not have. This would catch the obvious errors more than
a metric. This is the eval suite."*

Before this, the model-facing view (`scripts/map_view.py`) printed only the slots where a
note fell, and the song beside it as six loudness blocks — no kick-vs-snare, no pitch, no
lyric, no section, no energy. The rich score (`notesheet.py`) drew all of that as HTML
for a human eye. This is that score for the model, on the map's own clock.

## The lattice
One row per slot, **every slot** — silence is information: "the vocal is singing and
nothing is here" is only visible when the empty rows are drawn. `--sub 4` (default) is a
1/16 note; `--sub 8` for 32nds, `--sub 12` for triplets. A note off the lattice lands on
its nearest row marked `~`. Row header: `bar.beat.sub · m:ss.mmm · section`.

## Song columns (left)         all from caches that already exist; blank-with-reason if not
  KIT   percussion_cache   K S H C at their positions; UPPER = velocity ≥ 0.5, lower = softer
  BASS  melody_cache       pitch name at the onset, `─` while it sustains, `·` silence
  LEAD  melody_cache       the `other` stem's top line (salience peak — see coverage line)
  VOX   melody_cache       vocal pitch, then the SYLLABLE (lyrics_cache) sung at this slot
  gt pn event_cache        guitar / piano loudness blocks
  E     audio RMS          energy 0–9 (98th pct = 9); a drop is a visible step
  ON    onset_cache        ● the judge's reference onset is here (±half a slot)
  MAIN  overlay rule       which main line sounds here: vox / kik / snr / led
## Map columns (right)
  L / R   cell x,y · cut glyph · parity F/B (`!` = simulator violation) · `~` off-lattice
  B W A C bombs · walls as the 4 lanes (`█` blocked) · arcs ⌐ head ─ flight ¬ tail · chains ╞n
  DBL     D when both hands strike this slot
  TRV     cells travelled from this hand's previous note (L/R)
  ROT     degrees rotated from this hand's previous cut (L/R)
  ALT     hand(s) striking; `!` when the same hand struck the previous note-slot
  ±ms     signed ms to the nearest reference onset: ● ≤50 · ○ ≤120 · ✗ nothing there
`--vs <human.zip|auto>` adds the human map of the same song as a third block, `HL HR`, on the
same rows — the answer key. `auto` = `data/raw/<song>.zip`.

## Arrays (`--npz out.npz`)
  t_sec[T] beat[T] bar[T] section[T]              the lattice
  song[T,F] + song_names                            kit×4, bass/lead/vox midi (0 = none),
                                                    bass/lead/vox sustain, gtr, pno, energy,
                                                    onset, main (0 none 1 vox 2 kik 3 snr 4 led)
  lyric[T]  (str)
  map[T,C] + map_names                              12 cells × (color+1, dir+1) → 24,
                                                    bombs, 4 wall lanes, arc L/R, chain L/R
  human[T,C] if --vs
A question like "every VOX onset with no note within ±1 slot" is one line of numpy.

## Overview (`--sections`)  — the TRIAGE page, one row per bar
  notes L/R · doubles · E · main onsets answered/total · ✗ notes · wall share · human notes · Δ
Read it first; zoom (`--bars`) where it says the map is weak. Clean bars stay coarse.

Usage:
    python agent_mapper/score.py <map.zip> --song 1f333 --bars 33-36
    python agent_mapper/score.py <map.zip> --song 1f333 --sections
    python agent_mapper/score.py <map.zip> --song 1f333 --vs auto --bars 60-63
    python agent_mapper/score.py <map.zip> --song data/eval_songset/1f333.ogg --npz s.npz
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
import zipfile
from dataclasses import dataclass, field

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "outputs"
SCORE_CACHE = OUT / "score_cache"
BEATS_PER_BAR = 4
ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "↖", 5: "↗", 6: "↙", 7: "↘", 8: "•"}
# unit vectors of the cut directions, for rotation between successive cuts of one hand
_DIR_VEC = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
            4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1)}
BLOCKS = "·▁▂▃▄▅▆▇█"
ALIGN_TOL = 0.070      # overlay.py's TOL: wide enough for a correct note, narrow for the next 16th
REST_BEATS = 2.0       # overlay.py: a vocal gap this long hands the main line to the lead
MAIN_CODE = {"": 0, "vox": 1, "kik": 2, "snr": 3, "led": 4}
_HAND = {0: "L", 1: "R"}


# ----------------------------------------------------------------------------- maps
@dataclass
class Note:
    beat: float
    x: int
    y: int
    color: int
    direction: int
    src_beat: float | None = None   # the note's beat on ITS OWN map's clock (for --vs retiming)


@dataclass
class MapData:
    path: pathlib.Path
    bpm: float
    offset: float
    notes: list[Note]
    bombs: list[dict]
    walls: list[dict]      # beat, dur, x, w
    arcs: list[dict]       # c, b, tb
    chains: list[dict]     # c, b, tb, sc

    def t(self, beat: float) -> float:
        """Seconds on the game's clock. Same convention as map_view: offset + beat·60/bpm."""
        return self.offset + beat * 60.0 / self.bpm

    def beat_of(self, t: float) -> float:
        return (t - self.offset) * self.bpm / 60.0


def load_map(path: pathlib.Path) -> MapData:
    """v2 or v3, Expert(+)Standard preferred. Bomb-only/Info-only zips raise ValueError."""
    from beatsaber_automapper.data.beatmap import parse_difficulty_dat_json

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        info = next((n for n in names if n.split("/")[-1].lower() == "info.dat"), None)

        def pick(pred):
            return next((n for n in names
                         if pred(n.split("/")[-1].lower()) and "bpminfo" not in n.lower()), None)
        diff = (pick(lambda b: b == "expertstandard.dat")
                or pick(lambda b: b == "expertplusstandard.dat")
                or pick(lambda b: b.endswith("standard.dat"))
                or pick(lambda b: b.endswith(".dat") and b != "info.dat"))
        if info is None or diff is None:
            raise ValueError(f"could not read a difficulty from {path}")
        meta = json.loads(zf.read(info).decode("utf-8-sig"))
        dat = json.loads(zf.read(diff).decode("utf-8-sig"))
    bpm = next((float(v) for k, v in meta.items() if "beatsperminute" in k.lower()), 120.0)
    offset = next((float(v) for k, v in meta.items()
                   if "songtimeoffset" in k.lower().replace("_", "")), 0.0)
    bm = parse_difficulty_dat_json(dat)
    if bm is None:
        raise ValueError(f"unrecognised difficulty format in {path}")
    notes = [Note(float(n.beat), int(n.x), int(n.y), int(n.color), int(n.direction))
             for n in bm.color_notes]
    notes.sort(key=lambda n: (n.beat, n.color))
    return MapData(
        path=path, bpm=bpm, offset=offset, notes=notes,
        bombs=[{"b": float(b.beat), "x": int(b.x), "y": int(b.y)} for b in bm.bomb_notes],
        walls=[{"b": float(o.beat), "d": float(o.duration), "x": int(o.x),
                "w": max(int(o.width), 1)} for o in bm.obstacles],
        arcs=[{"c": int(s.color), "b": float(s.beat), "tb": float(s.tail_beat)}
              for s in bm.sliders],
        chains=[{"c": int(bs.color), "b": float(bs.beat), "tb": float(bs.tail_beat),
                 "sc": int(bs.slice_count)} for bs in bm.burst_sliders],
    )


# ----------------------------------------------------------------------------- songs
@dataclass
class Song:
    sid: str
    audio: pathlib.Path | None
    events: dict | None = None
    perc: dict | None = None
    melody: dict | None = None
    lyrics: dict | None = None
    structure: dict | None = None
    onsets: np.ndarray | None = None
    rms_t: np.ndarray | None = None      # seconds
    rms: np.ndarray | None = None        # 0..1
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def resolve_song(spec: str | None, map_path: pathlib.Path) -> tuple[str, pathlib.Path | None, str]:
    """(song id, audio path or None, how it was resolved).

    ⚠️Never key on the map's filename silently: `Hunger_AGENT.zip` is song 1f333 and the
    old view found nothing for it. A filename guess is printed as a guess.
    """
    if spec:
        p = pathlib.Path(spec)
        if p.suffix.lower() in (".ogg", ".egg", ".mp3", ".wav", ".flac") and p.exists():
            return p.stem, p, f"--song audio {p}"
        sid = spec
        how = "--song id"
    else:
        sid = map_path.stem.split("__")[-1].split("_")[0]
        how = f"GUESSED from the map filename '{map_path.name}' — pass --song to be sure"
    for d in (REPO / "data" / "eval_songset", REPO / "data" / "test_songs"):
        for ext in (".ogg", ".egg", ".mp3", ".wav"):
            c = d / f"{sid}{ext}"
            if c.exists():
                return sid, c, how
    return sid, None, how + " (no audio found — E from events, no onset recompute)"


def _load_json(p: pathlib.Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def load_song(sid: str, audio: pathlib.Path | None, perceive: bool = False) -> Song:
    s = Song(sid=sid, audio=audio)
    s.events = _load_json(OUT / "event_cache" / f"{sid}.6s.json")
    s.perc = _load_json(OUT / "percussion_cache" / f"{sid}.json")
    s.melody = _load_json(OUT / "melody_cache" / f"{sid}.json")
    s.lyrics = _load_json(OUT / "lyrics_cache" / f"{sid}.json")
    s.structure = _load_json(OUT / "structure_cache" / f"{sid}.json")
    if perceive and audio is not None:
        # run whichever perception tool is missing; each caches itself
        if s.events is None:
            import events as _EV
            s.events = _EV.analyse(audio)
        if s.perc is None:
            import percussion as _PC
            s.perc = _PC.analyse(audio)
        if s.melody is None:
            import melody as _ML
            s.melody = _ML.analyse(audio)
        if s.structure is None:
            import structure as _ST
            s.structure = _ST.analyse(audio)
        if s.lyrics is None:
            import lyrics as _LY
            s.lyrics = _LY.transcribe(audio) if hasattr(_LY, "transcribe") else None
    for name, val in (("event_cache", s.events), ("percussion_cache", s.perc),
                      ("melody_cache", s.melody), ("lyrics_cache", s.lyrics),
                      ("structure_cache", s.structure)):
        if val is None:
            s.missing.append(name)
    try:
        import refonsets as _RO
        s.onsets = _RO.reference_onsets(sid, audio=audio, compute=False)
    except Exception as e:  # noqa: BLE001
        s.notes.append(f"onsets unavailable: {e}")
    if s.onsets is None:
        s.missing.append("onset_cache")
    s.rms_t, s.rms = _energy(sid, audio)
    if s.rms is None:
        s.missing.append("audio (E falls back to event loudness)")
    return s


def _energy(sid: str, audio: pathlib.Path | None):
    """Song RMS on a 20 ms hop, scaled so the 98th percentile is 1.0. Cached."""
    SCORE_CACHE.mkdir(parents=True, exist_ok=True)
    f = SCORE_CACHE / f"{sid}.rms.npz"
    if f.exists():
        z = np.load(f)
        return z["t"], z["rms"]
    if audio is None or not audio.exists():
        return None, None
    try:
        import librosa
        y, sr = librosa.load(str(audio), sr=22050, mono=True)
        hop = 441
        r = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        t = librosa.frames_to_time(np.arange(len(r)), sr=sr, hop_length=hop)
        r = r / max(float(np.percentile(r, 98)), 1e-9)
        np.savez(f, t=t, rms=np.clip(r, 0, 1.2))
        return t, np.clip(r, 0, 1.2)
    except Exception as e:  # noqa: BLE001
        print(f"(energy unavailable: {e})", file=sys.stderr)
        return None, None


# ----------------------------------------------------------------------------- lattice
@dataclass
class Lattice:
    sub: int
    n: int                  # slots
    t_sec: np.ndarray
    beat: np.ndarray
    dt: float               # seconds per slot

    def slot_of_t(self, t) -> np.ndarray:
        return np.rint((np.asarray(t, dtype=float) - self.t_sec[0]) / self.dt).astype(int)

    def slot_of_beat(self, b) -> np.ndarray:
        return np.rint(np.asarray(b, dtype=float) * self.sub).astype(int)


def make_lattice(m: MapData, sub: int, end_beat: float) -> Lattice:
    n = int(math.ceil(end_beat * sub)) + 1
    beat = np.arange(n) / sub
    return Lattice(sub=sub, n=n, t_sec=m.offset + beat * 60.0 / m.bpm, beat=beat,
                   dt=60.0 / m.bpm / sub)


def _end_beat(m: MapData, song: Song, vs: MapData | None) -> float:
    last = max([n.beat for n in m.notes] + [w["b"] + w["d"] for w in m.walls] + [0.0])
    if vs is not None:
        last = max(last, max([n.beat for n in vs.notes] + [0.0]))
    if song.structure:
        last = max(last, m.beat_of(float(song.structure["sections"][-1]["t1"])))
    elif song.rms_t is not None:
        last = max(last, m.beat_of(float(song.rms_t[-1])))
    return math.ceil(last / BEATS_PER_BAR) * BEATS_PER_BAR


# ----------------------------------------------------------------------------- song side
@dataclass
class SongCols:
    kit: np.ndarray          # [T,4] velocity of K S H C
    midi: dict               # stem -> [T] int (0 none) at ONSET
    sus: dict                # stem -> [T] bool sustaining (after onset, within dur)
    pname: dict              # stem -> [T] str
    gtr: np.ndarray          # [T] 0..1
    pno: np.ndarray
    energy: np.ndarray       # [T] 0..1
    onset: np.ndarray        # [T] bool
    main: np.ndarray         # [T] str
    lyric: np.ndarray        # [T] str  (word at onset; '─' continuing; '')
    section: np.ndarray      # [T] str
    energy_src: str


def song_columns(song: Song, lat: Lattice, m: MapData) -> SongCols:
    T = lat.n
    kit = np.zeros((T, 4))
    if song.perc:
        idx = {"kick": 0, "snare": 1, "hat": 2, "crash": 3}
        for h in song.perc["hits"]:
            s = int(lat.slot_of_t(h["t"]))
            if 0 <= s < T:
                kit[s, idx[h["piece"]]] = max(kit[s, idx[h["piece"]]], float(h["vel"]))
    midi, sus, pname = {}, {}, {}
    for stem in ("bass", "other", "vocals"):
        mi = np.zeros(T, dtype=int)
        su = np.zeros(T, dtype=bool)
        nm = np.full(T, "", dtype=object)
        if song.melody:
            for e in song.melody["stems"].get(stem, []):
                s0 = int(lat.slot_of_t(e["t"]))
                s1 = int(lat.slot_of_t(e["t"] + float(e.get("dur", 0.0))))
                if 0 <= s0 < T:
                    mi[s0] = int(e["midi"])
                    nm[s0] = e.get("name", "")
                    su[s0 + 1:min(s1 + 1, T)] = True
        midi[stem], sus[stem], pname[stem] = mi, su, nm
    gtr, pno = np.zeros(T), np.zeros(T)
    ev_loud = np.zeros(T)
    if song.events:
        for e in song.events["events"]:
            s = int(lat.slot_of_t(e["t"]))
            if not 0 <= s < T:
                continue
            v = min(max((float(e.get("loud", 0.0)) + 20.0) / 30.0, 0.0), 1.0)
            ev_loud[s] = max(ev_loud[s], v)
            if e["stem"] == "guitar":
                gtr[s] = max(gtr[s], v)
            elif e["stem"] == "piano":
                pno[s] = max(pno[s], v)
    if song.rms is not None:
        # mean RMS inside each slot's window
        edges = np.concatenate([lat.t_sec - lat.dt / 2, [lat.t_sec[-1] + lat.dt / 2]])
        idx = np.clip(np.searchsorted(edges, song.rms_t) - 1, 0, T - 1)
        sums = np.bincount(idx, weights=song.rms, minlength=T)
        cnt = np.bincount(idx, minlength=T)
        energy = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
        # slots shorter than the hop inherit their neighbour
        for i in range(1, T):
            if cnt[i] == 0:
                energy[i] = energy[i - 1]
        energy_src = "audio RMS"
    else:
        energy, energy_src = ev_loud, "event loudness (no audio)"
    onset = np.zeros(T, dtype=bool)
    if song.onsets is not None and len(song.onsets):
        s = lat.slot_of_t(song.onsets)
        s = s[(s >= 0) & (s < T)]
        onset[s] = True
    main = np.full(T, "", dtype=object)
    vt = np.array([e["t"] for e in (song.melody or {}).get("stems", {}).get("vocals", [])],
                  dtype=float)
    spb = 60.0 / m.bpm
    if song.melody:
        for e in song.melody["stems"].get("other", []):
            if len(vt) == 0 or float(np.min(np.abs(vt - e["t"]))) > REST_BEATS * spb:
                s = int(lat.slot_of_t(e["t"]))
                if 0 <= s < T:
                    main[s] = "led"
    for s in np.nonzero(kit[:, 1] > 0)[0]:
        main[s] = "snr"
    for s in np.nonzero(kit[:, 0] > 0)[0]:
        main[s] = "kik"
    for s in np.nonzero(midi["vocals"] > 0)[0]:
        main[s] = "vox"
    lyric = np.full(T, "", dtype=object)
    if song.lyrics:
        for w in song.lyrics.get("words", []):
            t0, t1 = float(w["t"]), float(w.get("end", w["t"]))
            if t1 <= t0 and t0 == 0.0:
                continue            # whisper's zero-length hallucination at t=0
            s0, s1 = int(lat.slot_of_t(t0)), int(lat.slot_of_t(max(t1, t0)))
            if 0 <= s0 < T:
                lyric[s0] = w["word"].strip()
                for s in range(s0 + 1, min(s1 + 1, T)):
                    if not lyric[s]:
                        lyric[s] = "─"
    section = np.full(T, "", dtype=object)
    if song.structure:
        for sec in song.structure["sections"]:
            s0 = max(int(lat.slot_of_t(sec["t0"])), 0)
            s1 = min(int(lat.slot_of_t(sec["t1"])), T)
            section[s0:s1] = sec["label"]
    return SongCols(kit=kit, midi=midi, sus=sus, pname=pname, gtr=gtr, pno=pno,
                    energy=energy, onset=onset, main=main, lyric=lyric, section=section,
                    energy_src=energy_src)


# ----------------------------------------------------------------------------- map side
@dataclass
class MapCols:
    cell: dict               # color -> [T] str ("x,y↓F", "" )
    offgrid: dict            # color -> [T] bool
    count: dict              # color -> [T] int notes in slot
    trv: dict                # color -> [T] float (nan = none)
    rot: dict                # color -> [T] float (nan)
    bombs: np.ndarray        # [T] int
    lanes: np.ndarray        # [T] str "██··"
    arc: dict                # color -> [T] str "⌐" "─" "¬" ""
    chain: dict              # color -> [T] str "╞3" ""
    align_ms: np.ndarray     # [T] float (nan = no note)
    align_mark: np.ndarray   # [T] str
    same_hand: np.ndarray    # [T] bool: same hand as previous note-slot
    grid: np.ndarray         # [T,12,2] (color+1, dir+1)
    violations: int
    resets: int


def _parity_tags(m: MapData) -> tuple[dict, int, int]:
    """(beat, color) -> 'F'/'B' (+'!' on a simulator violation), via swing_sim."""
    try:
        from beatsaber_automapper.evaluation import swing_sim as ss
    except Exception:  # noqa: BLE001
        return {}, 0, 0

    class _N:
        __slots__ = ("beat", "x", "y", "color", "direction")

        def __init__(self, n):
            self.beat, self.x, self.y, self.color, self.direction = (
                n.beat, n.x, n.y, n.color, n.direction)

    class _BM:
        def __init__(self, notes):
            self.color_notes = [_N(n) for n in notes]
            self.bomb_notes = []
    card = ss.simulate(_BM(m.notes), bpm=m.bpm)
    out = {}
    for color, hand in card.per_hand.items():
        for sw in hand.swings:
            tag = "F" if sw.parity is ss.Parity.FOREHAND else "B"
            if sw.reset_kind == "violation":
                tag += "!"
            out[(round(sw.beat, 3), color)] = tag
    return out, int(card.violations), int(card.resets)


def _rotation(d0: int, d1: int) -> float:
    if d0 not in _DIR_VEC or d1 not in _DIR_VEC:
        return float("nan")
    a = math.atan2(*_DIR_VEC[d0][::-1])
    b = math.atan2(*_DIR_VEC[d1][::-1])
    return abs(math.degrees(math.atan2(math.sin(b - a), math.cos(b - a))))


def map_columns(m: MapData, lat: Lattice, song: Song, with_parity: bool = True) -> MapCols:
    T = lat.n
    par, viol, resets = _parity_tags(m) if with_parity else ({}, 0, 0)
    cell = {c: np.full(T, "", dtype=object) for c in (0, 1)}
    offgrid = {c: np.zeros(T, dtype=bool) for c in (0, 1)}
    count = {c: np.zeros(T, dtype=int) for c in (0, 1)}
    trv = {c: np.full(T, np.nan) for c in (0, 1)}
    rot = {c: np.full(T, np.nan) for c in (0, 1)}
    grid = np.zeros((T, 12, 2), dtype=np.int8)
    prev = {0: None, 1: None}
    for n in m.notes:
        s = int(round(n.beat * lat.sub))
        if not 0 <= s < T:
            continue
        own = n.beat if n.src_beat is None else n.src_beat
        off = abs(own * lat.sub - round(own * lat.sub)) > 0.05   # off ITS OWN lattice
        count[n.color][s] += 1
        if count[n.color][s] == 1:
            tag = par.get((round(n.beat, 3), n.color), "")
            cell[n.color][s] = f"{n.x},{n.y}{ARROW.get(n.direction, '?')}{tag}"
            offgrid[n.color][s] = off
            p = prev[n.color]
            if p is not None:
                trv[n.color][s] = math.hypot(n.x - p.x, n.y - p.y)
                rot[n.color][s] = _rotation(p.direction, n.direction)
            prev[n.color] = n
        else:
            cell[n.color][s] += f"+{count[n.color][s] - 1}"
        grid[s, n.y * 4 + n.x] = (n.color + 1, n.direction + 1)
    bombs = np.zeros(T, dtype=int)
    for b in m.bombs:
        s = int(round(b["b"] * lat.sub))
        if 0 <= s < T:
            bombs[s] += 1
    lanes = np.full(T, "····", dtype=object)
    lane_bits = np.zeros((T, 4), dtype=bool)
    for w in m.walls:
        s0 = int(round(w["b"] * lat.sub))
        s1 = int(round((w["b"] + w["d"]) * lat.sub))
        for s in range(max(s0, 0), min(max(s1, s0 + 1), T)):
            for c in range(w["x"], min(w["x"] + w["w"], 4)):
                if c >= 0:
                    lane_bits[s, c] = True
    for s in np.nonzero(lane_bits.any(axis=1))[0]:
        lanes[s] = "".join("█" if b else "·" for b in lane_bits[s])
    arc = {c: np.full(T, "", dtype=object) for c in (0, 1)}
    for a in m.arcs:
        s0, s1 = int(round(a["b"] * lat.sub)), int(round(a["tb"] * lat.sub))
        c = a["c"] if a["c"] in (0, 1) else 0
        for s in range(max(s0, 0), min(s1 + 1, T)):
            arc[c][s] = "⌐" if s == s0 else ("¬" if s == s1 else "─")
    chain = {c: np.full(T, "", dtype=object) for c in (0, 1)}
    for ch in m.chains:
        s0, s1 = int(round(ch["b"] * lat.sub)), int(round(ch["tb"] * lat.sub))
        c = ch["c"] if ch["c"] in (0, 1) else 0
        for s in range(max(s0, 0), min(s1 + 1, T)):
            chain[c][s] = f"╞{ch['sc']}" if s == s0 else "╌"
    align_ms = np.full(T, np.nan)
    align_mark = np.full(T, "", dtype=object)
    if song.onsets is not None and len(song.onsets):
        o = np.sort(np.asarray(song.onsets, dtype=float))
        for n in m.notes:
            s = int(round(n.beat * lat.sub))
            if not 0 <= s < T:
                continue
            t = m.t(n.beat)
            i = int(np.clip(np.searchsorted(o, t), 1, len(o) - 1))
            near = o[i - 1] if abs(t - o[i - 1]) <= abs(t - o[i]) else o[i]
            d = (t - near) * 1000.0
            if np.isnan(align_ms[s]) or abs(d) > abs(align_ms[s]):
                align_ms[s] = d          # the worst note in the slot is the one you hear
        for s in np.nonzero(~np.isnan(align_ms))[0]:
            d = abs(align_ms[s])
            align_mark[s] = "●" if d <= 50 else ("○" if d <= 120 else "✗")
    same_hand = np.zeros(T, dtype=bool)
    last, last_s = None, -10 ** 9
    for s in range(T):
        hands = tuple(c for c in (0, 1) if count[c][s])
        if not hands:
            continue
        # a repeated hand only reads as "no alternation" inside a beat; across a rest it is a restart
        if last is not None and len(hands) == 1 and hands == last and s - last_s <= lat.sub:
            same_hand[s] = True
        last, last_s = hands, s
    return MapCols(cell=cell, offgrid=offgrid, count=count, trv=trv, rot=rot, bombs=bombs,
                   lanes=lanes, arc=arc, chain=chain, align_ms=align_ms,
                   align_mark=align_mark, same_hand=same_hand, grid=grid,
                   violations=viol, resets=resets)


# ----------------------------------------------------------------------------- text
def _mmss(t: float) -> str:
    t = max(t, 0.0)
    return f"{int(t // 60)}:{t % 60:06.3f}"


def _lyric_cell(word: str, w: int) -> str:
    """Pad by DISPLAY width — CJK syllables are two columns wide."""
    import unicodedata
    out, width = "", 0
    for ch in str(word):
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if width + cw > w:
            break
        out += ch
        width += cw
    return out + " " * (w - width)


def _kit_str(v: np.ndarray) -> str:
    out = []
    for ch, x in zip("KSHC", v):
        out.append("·" if x <= 0 else (ch if x >= 0.5 else ch.lower()))
    return "".join(out)


def _pitch_cell(sc: SongCols, stem: str, s: int, w: int) -> str:
    if sc.midi[stem][s] > 0:
        return str(sc.pname[stem][s])[:w].ljust(w)
    if sc.sus[stem][s]:
        return "─".ljust(w)
    return "·".ljust(w)


def _hand_cell(mc: MapCols, c: int, s: int, w: int) -> str:
    v = mc.cell[c][s]
    if v and mc.offgrid[c][s]:
        v += "~"
    return v[:w].ljust(w)


def header_lines(m: MapData, song: Song, sc: SongCols, mc: MapCols, lat: Lattice,
                 how: str, vs: MapData | None) -> list[str]:
    L = [f"# SCORE  {m.path.name}  —  {len(m.notes)} notes · {len(m.walls)} walls · "
         f"{len(m.arcs)} arcs · {len(m.chains)} chains · {m.bpm:g} bpm · offset {m.offset:+.3f}s"
         f" · 1/{lat.sub * 4} lattice · parity violations {mc.violations} · resets {mc.resets}",
         f"# song {song.sid}  ({how})"]
    if song.melody:
        meta = song.melody.get("meta", {})
        cov = " · ".join(f"{k} cov {v.get('coverage', 0):.2f}" for k, v in meta.items())
        L.append(f"# melody: {cov}   — a blank lane at low coverage is the SONG "
                 f"(untrackable pitch), not a bug")
    if song.lyrics:
        L.append(f"# lyrics: {song.lyrics.get('language')} p={song.lyrics.get('language_probability', 0):.2f}"
                 f" · {song.lyrics.get('n_words')} words — whisper guesses; screamed/EDM vocals hallucinate")
    L.append(f"# E = {sc.energy_src} · ON = reference onsets"
             f"{' (NONE — ±ms and ON blank)' if song.onsets is None else f' ({len(song.onsets)})'}"
             f" · MAIN = vox > kik > snr > led-in-rests")
    if song.structure:
        secs = []
        for sec in song.structure["sections"]:
            b0 = int(m.beat_of(float(sec["t0"])) // BEATS_PER_BAR) + 1
            b1 = int(math.ceil(m.beat_of(float(sec["t1"])) / BEATS_PER_BAR))
            secs.append(f"{sec['label']} {b0}–{b1}")
        L.append("# sections (bars): " + " · ".join(secs))
    if vs is not None:
        L.append(f"# HL/HR = {vs.path.name}  ({len(vs.notes)} notes @ {vs.bpm:g} bpm, offset "
                 f"{vs.offset:+.3f}s) — the human answer key; H±ms = its own alignment")
    if song.missing:
        L.append(f"# ⚠️ missing: {', '.join(song.missing)}  (run the tool, or --perceive)")
    for n in song.notes:
        L.append(f"# ⚠️ {n}")
    return L


def render_rows(m: MapData, sc: SongCols, mc: MapCols, lat: Lattice, s0: int, s1: int,
                hc: MapCols | None = None) -> list[str]:
    W = 8
    head = (f"{'bar.b.s':>8s} {'time':>9s} S│KIT │BASS│LEAD│VOX  lyric     │gt pn│E ON MAIN"
            f"│{'L':<{W}s}│{'R':<{W}s}│B W    A  C  │DBL TRV   ROT   ALT ±ms")
    if hc is not None:
        head += f"│{'HL':<{W}s}│{'HR':<{W}s} H±ms"
    lines = [head, "─" * len(head)]
    last_bar = None
    for s in range(max(s0, 0), min(s1, lat.n)):
        beat = lat.beat[s]
        bar = int(beat // BEATS_PER_BAR) + 1
        b_in = int(beat % BEATS_PER_BAR) + 1
        sub = int(round((beat % 1) * lat.sub))
        if last_bar is not None and bar != last_bar:
            lines.append("┈" * len(head))
        last_bar = bar
        row = (f"{bar:>4d}.{b_in}.{sub:<2d} {_mmss(lat.t_sec[s]):>9s} "
               f"{str(sc.section[s])[:1] or ' '}│{_kit_str(sc.kit[s])}│"
               f"{_pitch_cell(sc, 'bass', s, 4)}│{_pitch_cell(sc, 'other', s, 4)}│"
               f"{_pitch_cell(sc, 'vocals', s, 4)} {_lyric_cell(sc.lyric[s], 9)} │"
               f"{BLOCKS[min(int(sc.gtr[s] * 8), 8)]}  {BLOCKS[min(int(sc.pno[s] * 8), 8)]} │"
               f"{min(int(round(sc.energy[s] * 9)), 9)} {'●' if sc.onset[s] else ' '} "
               f"{str(sc.main[s]):<4s}│"
               f"{_hand_cell(mc, 0, s, W)}│{_hand_cell(mc, 1, s, W)}│"
               f"{mc.bombs[s] if mc.bombs[s] else '·'} {mc.lanes[s]} "
               f"{(mc.arc[0][s] or ' ')}{(mc.arc[1][s] or ' ')} "
               f"{(mc.chain[0][s] or mc.chain[1][s] or '  '):<2s} │")
        dbl = "D" if (mc.count[0][s] and mc.count[1][s]) else " "
        def _f(v, w=2):
            return ("·" if np.isnan(v) else f"{v:.0f}").rjust(w)
        trv = f"{_f(mc.trv[0][s])}/{_f(mc.trv[1][s])}"
        rot = f"{_f(mc.rot[0][s], 3)}/{_f(mc.rot[1][s], 3)}"
        hands = "".join(_HAND[c] for c in (0, 1) if mc.count[c][s])
        alt = (hands + ("!" if mc.same_hand[s] else "")).ljust(3)
        if not np.isnan(mc.align_ms[s]):
            al = f"{mc.align_mark[s]}{mc.align_ms[s]:+4.0f}"
        else:
            al = ""
        row += f" {dbl}  {trv} {rot} {alt} {al}"
        if hc is not None:
            hal = "" if np.isnan(hc.align_ms[s]) else f"{hc.align_mark[s]}{hc.align_ms[s]:+4.0f}"
            row += f"│{_hand_cell(hc, 0, s, W)}│{_hand_cell(hc, 1, s, W)} {hal}"
        lines.append(row)
    return lines


def render_sections(m: MapData, sc: SongCols, mc: MapCols, lat: Lattice,
                    hc: MapCols | None = None) -> list[str]:
    """One row per bar — the triage page."""
    n_bars = int(math.ceil(lat.n / (lat.sub * BEATS_PER_BAR)))
    head = (f"{'bar':>4s} {'time':>8s} S │ L  R  D │ E │ main hit/all│ ✗ ○ │ wall │ vox │ "
            f"burst │ TRV")
    if hc is not None:
        head += " │ HL HR   Δ"
    head += " │ flags"
    lines = [head, "─" * len(head)]
    per = lat.sub * BEATS_PER_BAR
    for b in range(n_bars):
        s0, s1 = b * per, min((b + 1) * per, lat.n)
        L = int(mc.count[0][s0:s1].sum())
        R = int(mc.count[1][s0:s1].sum())
        D = int(((mc.count[0][s0:s1] > 0) & (mc.count[1][s0:s1] > 0)).sum())
        E = int(round(float(sc.energy[s0:s1].mean()) * 9)) if s1 > s0 else 0
        main_slots = np.nonzero(sc.main[s0:s1] != "")[0] + s0
        struck = (mc.count[0] + mc.count[1]) > 0
        hit = sum(1 for s in main_slots if struck[max(s - 1, 0):s + 2].any())
        bad = int((mc.align_mark[s0:s1] == "✗").sum())
        near = int((mc.align_mark[s0:s1] == "○").sum())
        wall = float((mc.lanes[s0:s1] != "····").mean()) if s1 > s0 else 0.0
        vox = int((sc.midi["vocals"][s0:s1] > 0).sum())
        # burst: the longest run of consecutive struck slots at the lattice's own spacing
        run = best = 0
        for s in range(s0, s1):
            run = run + 1 if struck[s] else 0
            best = max(best, run)
        tr = np.concatenate([mc.trv[0][s0:s1], mc.trv[1][s0:s1]])
        tr = tr[~np.isnan(tr)]
        trv = float(tr.mean()) if len(tr) else 0.0
        sec = str(sc.section[s0])[:1] or " "
        row = (f"{b + 1:>4d} {_mmss(lat.t_sec[s0])[:-4]:>8s} {sec} │{L:>3d}{R:>3d}{D:>3d} │ {E} │"
               f"{hit:>5d}/{len(main_slots):<4d}│{bad:>2d}{near:>2d} │ {wall:4.2f} │{vox:>4d} │"
               f"{best:>5d}  │{trv:4.1f}")
        flags = []
        if hc is not None:
            HL = int(hc.count[0][s0:s1].sum())
            HR = int(hc.count[1][s0:s1].sum())
            d = (L + R) - (HL + HR)
            row += f" │{HL:>3d}{HR:>3d} {d:>+4d}"
            if HL + HR >= 4 and (L + R) < 0.5 * (HL + HR):
                flags.append("THIN-vs-human")
            if (L + R) > 1.6 * (HL + HR) + 2:
                flags.append("DENSE-vs-human")
        if bad:
            flags.append(f"✗{bad}")
        if len(main_slots) >= 4 and hit < 0.5 * len(main_slots):
            flags.append("MAIN-unanswered")
        if vox >= 3 and (L + R) == 0:
            flags.append("VOX-silent-map")
        if best >= lat.sub * 2 and not sc.onset[s0:s1].any():
            flags.append("BURST-no-onsets")
        if L + R == 0 and E >= 5:
            flags.append("EMPTY-loud")
        row += " │ " + " ".join(flags)
        lines.append(row)
    return lines


# ----------------------------------------------------------------------------- arrays
def to_arrays(m: MapData, sc: SongCols, mc: MapCols, lat: Lattice, hc: MapCols | None):
    T = lat.n
    song_cols = [
        ("kit_kick", sc.kit[:, 0]), ("kit_snare", sc.kit[:, 1]),
        ("kit_hat", sc.kit[:, 2]), ("kit_crash", sc.kit[:, 3]),
        ("bass_midi", sc.midi["bass"]), ("lead_midi", sc.midi["other"]),
        ("vox_midi", sc.midi["vocals"]),
        ("bass_sus", sc.sus["bass"]), ("lead_sus", sc.sus["other"]),
        ("vox_sus", sc.sus["vocals"]),
        ("gtr", sc.gtr), ("pno", sc.pno), ("energy", sc.energy),
        ("onset", sc.onset),
        ("main", np.array([MAIN_CODE.get(str(x), 0) for x in sc.main])),
    ]
    song = np.stack([np.asarray(v, dtype=float) for _, v in song_cols], axis=1)

    def map_arr(cols: MapCols):
        parts = [cols.grid.reshape(T, 24).astype(float)]
        names = [f"c{i}_{k}" for i in range(12) for k in ("color", "dir")]
        parts.append(cols.bombs[:, None].astype(float)); names.append("bombs")
        lanes = np.array([[1.0 if ch == "█" else 0.0 for ch in str(l)] for l in cols.lanes])
        parts.append(lanes); names += [f"wall_lane{i}" for i in range(4)]
        for c in (0, 1):
            parts.append(np.array([1.0 if a else 0.0 for a in cols.arc[c]])[:, None])
            names.append(f"arc_{_HAND[c]}")
        for c in (0, 1):
            parts.append(np.array([1.0 if a else 0.0 for a in cols.chain[c]])[:, None])
            names.append(f"chain_{_HAND[c]}")
        parts.append(cols.align_ms[:, None]); names.append("align_ms")
        for c in (0, 1):
            parts.append(cols.trv[c][:, None]); names.append(f"trv_{_HAND[c]}")
            parts.append(cols.rot[c][:, None]); names.append(f"rot_{_HAND[c]}")
        return np.concatenate(parts, axis=1), names
    mp, mnames = map_arr(mc)
    out = dict(t_sec=lat.t_sec, beat=lat.beat,
               bar=(lat.beat // BEATS_PER_BAR + 1).astype(int),
               section=np.array([str(x) for x in sc.section]),
               song=song, song_names=np.array([n for n, _ in song_cols]),
               lyric=np.array([str(x) for x in sc.lyric]),
               map=mp, map_names=np.array(mnames),
               bpm=m.bpm, offset=m.offset, sub=lat.sub)
    if hc is not None:
        out["human"], _ = map_arr(hc)
    return out


# ----------------------------------------------------------------------------- cli
def _bar_range(spec: str, lat: Lattice) -> tuple[int, int]:
    a, _, b = spec.partition("-")
    per = lat.sub * BEATS_PER_BAR
    return (int(a) - 1) * per, (int(b) if b else int(a)) * per


def resolve_vs(vs: str, sid: str | None) -> pathlib.Path:
    """`--vs auto` → the corpus map of THIS song; `--vs 1f8d6` (a corpus id, 5 hex) → that
    song's corpus map; anything else is a zip path. P2b: the corpus holds ONE human map per
    id and the crawl was rating-sorted with upvote ratio ≥ 0.8 (data/download.py), so
    `data/raw/<id>.zip` IS the top-rated human map we have of that song — the tutor."""
    if vs == "auto":
        return REPO / "data" / "raw" / f"{sid}.zip"
    p = pathlib.Path(vs)
    if not p.exists() and re.fullmatch(r"[0-9a-f]{4,6}", vs):
        return REPO / "data" / "raw" / f"{vs}.zip"
    return p


def build(map_path: pathlib.Path, song_spec: str | None, sub: int, vs: str | None,
          perceive: bool = False):
    m = load_map(map_path)
    sid, audio, how = resolve_song(song_spec, map_path)
    song = load_song(sid, audio, perceive=perceive)
    vsm = None
    if vs:
        vp = resolve_vs(vs, sid)
        if vp.exists():
            vsm = load_map(vp)
        else:
            song.notes.append(f"--vs {vp} not found")
    lat = make_lattice(m, sub, _end_beat(m, song, vsm))
    sc = song_columns(song, lat, m)
    mc = map_columns(m, lat, song)
    hc = None
    if vsm is not None:
        # the human map on OUR lattice: re-time its beats through both clocks
        hm = MapData(path=vsm.path, bpm=m.bpm, offset=m.offset,
                     notes=[Note(m.beat_of(vsm.t(n.beat)), n.x, n.y, n.color, n.direction,
                                 src_beat=n.beat) for n in vsm.notes],
                     bombs=[dict(b, b=m.beat_of(vsm.t(b["b"]))) for b in vsm.bombs],
                     walls=[dict(w, b=m.beat_of(vsm.t(w["b"])), d=w["d"] * m.bpm / vsm.bpm)
                            for w in vsm.walls],
                     arcs=[dict(a, b=m.beat_of(vsm.t(a["b"])), tb=m.beat_of(vsm.t(a["tb"])))
                           for a in vsm.arcs],
                     chains=[dict(c, b=m.beat_of(vsm.t(c["b"])), tb=m.beat_of(vsm.t(c["tb"])))
                             for c in vsm.chains])
        hc = map_columns(hm, lat, song)
    return m, song, how, vsm, lat, sc, mc, hc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("map", type=pathlib.Path)
    ap.add_argument("--song", help="song id (1f333), or the audio path; default: guess from filename")
    ap.add_argument("--bars", help="bar range to print, e.g. 33-36 (mandatory above 32 bars)")
    ap.add_argument("--sub", type=int, default=4, help="rows per beat: 4 (1/16), 8, 12 (triplets)")
    ap.add_argument("--vs", help="human map on the same rows: a zip path, a corpus id (1f8d6 → "
                                 "data/raw/1f8d6.zip), or 'auto' (this song's corpus map)")
    ap.add_argument("--sections", action="store_true", help="one row per bar — the triage page")
    ap.add_argument("--npz", type=pathlib.Path, help="write the arrays")
    ap.add_argument("--csv", type=pathlib.Path, help="write the text rows as CSV")
    ap.add_argument("--perceive", action="store_true", help="run any missing perception tool (slow)")
    ap.add_argument("--all", action="store_true", help="print every bar even above 32 (big)")
    a = ap.parse_args()

    m, song, how, vsm, lat, sc, mc, hc = build(a.map, a.song, a.sub, a.vs, a.perceive)
    print("\n".join(header_lines(m, song, sc, mc, lat, how, vsm)))
    n_bars = int(math.ceil(lat.n / (lat.sub * BEATS_PER_BAR)))
    if a.npz:
        np.savez(a.npz, **to_arrays(m, sc, mc, lat, hc))
        print(f"# wrote {a.npz}  (song[T,F] map[T,C]{' human[T,C]' if hc is not None else ''}, T={lat.n})")
    if a.sections or not (a.bars or a.all or a.npz or a.csv):
        print()
        print("\n".join(render_sections(m, sc, mc, lat, hc)))
        if not a.bars:
            print(f"\n# {n_bars} bars. Zoom with --bars a-b where the flags are.")
    if a.bars or a.all:
        s0, s1 = _bar_range(a.bars, lat) if a.bars else (0, lat.n)
        if not a.bars and n_bars > 32 and not a.all:
            print(f"# {n_bars} bars: pass --bars a-b (or --all).")
        else:
            print()
            print("\n".join(render_rows(m, sc, mc, lat, s0, s1, hc)))
    if a.csv:
        rows = render_rows(m, sc, mc, lat, 0, lat.n, hc)
        a.csv.write_text("\n".join(rows))
        print(f"# wrote {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
