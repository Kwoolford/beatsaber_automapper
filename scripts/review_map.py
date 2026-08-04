#!/usr/bin/env python
"""★ AUTOMATED MAP REVIEW — the stems against our notes, as timestamped findings.

Kyle, 2026-08-04: *"did you work on improving your own eval suite? So you could
look at demucs compared to our generated note placement? The end goal is so that
I can review less and do more in depth reviews when I do. I want to empower you."*

This is that tool. Everything else in the suite reports **cohort statistics** — a
number per map, aggregated over songs. Those numbers cannot tell you *where to
listen*, which is why every real defect this project found still required him to
play a map and describe a moment.

This produces the other thing: **a ranked list of specific timestamps with a
reason**, computed from the seeded Demucs stem cache against the map's own notes.
It is what lets an agent do the first pass so a human review can start from
"go listen at 3:32" instead of "listen to the whole song".

Six detectors, each tuned to a defect this project has actually confirmed:

  STARVED     a window where the music is busy and we are far below the human
              (or far below our own song-median when no human map exists).
              Found Fallen Kingdom's 210-230s collapse: 10 notes vs the human's 28.
  MAPPING_SILENCE  notes placed where almost no stem is active. Found the same
              song's 240-250s outro: 13 notes over a soft vocal with 1 onset.
  MISSED_HIT  a k>=3 multi-instrument coincidence with no note within 50ms.
              Humans map these 72-85% of the time.
  OFFBEAT     our nearest note to a k>=3 event sits ~half a beat away -- on a real
              onset, just the wrong one. Invisible to axis A8 by construction.
  PHRASE_HOLE a sung phrase with a >1s stretch carrying no notes.
  ENDING      the map's last note vs the last hit of the CARRYING instrument.
              Found Hunger ending 0.16s (one 16th) past the final drum hit, on a
              residual bass onset -- the "very small delay" Kyle still heard after
              BEAT_END_RESOLVE removed the orphaned half-double.

⚠️These are POINTERS, not verdicts. A finding means "a human ear should check
here", not "this is wrong" -- restraint over a quiet passage is correct mapping,
and every regularity-rewarding metric in this project turned out to be
metronome-gameable. Rank, look, describe. Do not tune against this file.

Usage:
    python scripts/review_map.py --song 1f8d6 --map path/to.zip
    python scripts/review_map.py --song 1f333 --map A.zip --top 15 --html out.html
"""

from __future__ import annotations

import argparse
import collections
import html as _html
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from beatsaber_automapper.evaluation import alignment, scorecard  # noqa: E402
from calibrate_playfeel import load_expert_only  # noqa: E402
from eval_coincidence import events_for  # noqa: E402
from eval_phrase_abandon import vocal_phrases  # noqa: E402

STEMS = ("bass", "drums", "other", "vocals")
WIN = 10.0
TOL = 0.050


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def stems_of(song_id: str) -> dict[str, np.ndarray]:
    f = REPO / "outputs" / "stem_onset_cache" / f"{song_id}.npz"
    if not f.exists():
        sys.exit(f"no stem cache for {song_id!r}")
    d = np.load(f, allow_pickle=True)
    return {s: np.sort(np.asarray(d[f"onsets_{s}"], dtype=float))
            for s in STEMS if f"onsets_{s}" in d.files}


def near(arr: np.ndarray, t: float, tol: float) -> bool:
    if len(arr) == 0:
        return False
    i = int(np.searchsorted(arr, t))
    return any(abs(t - arr[j]) <= tol for j in (i - 1, i) if 0 <= j < len(arr))


def nearest_signed(arr: np.ndarray, t: float) -> float:
    if len(arr) == 0:
        return np.inf
    i = int(np.searchsorted(arr, t))
    c = [arr[j] for j in (i - 1, i) if 0 <= j < len(arr)]
    return min(c, key=lambda x: abs(t - x)) - t if c else np.inf


def review(song: str, ours: np.ndarray, bpm: float,
           human: np.ndarray | None) -> list[dict]:
    stems = stems_of(song)
    times, ks = events_for(song, 0.030)
    beat = 60.0 / bpm
    allon = np.sort(np.concatenate(list(stems.values())))
    end = float(max(ours.max(), times.max()))
    F: list[dict] = []

    # ---- 1/2. per-window density vs the music and vs the human -------------
    # ⚠️`arange(0, end - WIN, WIN)` DROPS THE FINAL WINDOW, which is exactly where
    # outros live -- MAPPING_SILENCE fired on 0 of 24 songs because of it, while
    # Fallen Kingdom demonstrably places 13 notes over a 240-250s outro carrying a
    # single stem onset. Cover the tail. Caught 2026-08-04.
    windows = np.arange(0, end, WIN)
    med_ours = np.median([((ours >= t) & (ours < t + WIN)).sum()
                          for t in windows] or [0])
    for t0 in windows:
        t1 = t0 + WIN
        n_o = int(((ours >= t0) & (ours < t1)).sum())
        act = {s: int(((v >= t0) & (v < t1)).sum()) for s, v in stems.items()}
        busy = sum(act.values())
        if human is not None:
            n_h = int(((human >= t0) & (human < t1)).sum())
            if n_h >= 12 and n_o <= 0.55 * n_h:
                F.append(dict(kind="STARVED", t=t0, sev=(n_h - n_o) / max(n_h, 1),
                              msg=f"{n_o} notes vs the human's {n_h} over {WIN:.0f}s "
                                  f"(music busy: {busy} stem onsets)"))
        elif busy >= 40 and n_o <= 0.5 * med_ours:
            F.append(dict(kind="STARVED", t=t0, sev=1.0 - n_o / max(med_ours, 1),
                          msg=f"{n_o} notes against a song-median {med_ours:.0f}, "
                              f"while the music runs {busy} stem onsets"))
        if busy <= 4 and n_o >= 6:
            F.append(dict(kind="MAPPING_SILENCE", t=t0, sev=n_o / 10.0,
                          msg=f"{n_o} notes over {WIN:.0f}s with only {busy} stem "
                              f"onsets — we are mapping near-silence"))

    # ---- 3/4. multi-instrument events: missed, or answered on the offbeat ---
    e3 = times[ks >= 3]
    miss = [t for t in e3 if not near(ours, t, TOL)]
    for t in miss:
        d = nearest_signed(ours, t)
        ph = abs((d + beat / 2) % beat - beat / 2)
        k = int(ks[np.argmin(np.abs(times - t))])
        if ph >= 0.35 * beat:
            F.append(dict(kind="OFFBEAT", t=t, sev=0.5 + 0.1 * k,
                          msg=f"{k}-instrument hit; our nearest note is "
                              f"{abs(d) * 1000:.0f}ms away (~half a beat) — on a "
                              f"real onset, but the wrong one"))
        else:
            F.append(dict(kind="MISSED_HIT", t=t, sev=0.3 + 0.15 * k,
                          msg=f"{k} instruments hit together and we place nothing "
                              f"within 50ms (humans map these 72-85% of the time)"))

    # ---- 5. sung phrases with a hole ---------------------------------------
    for s, e in vocal_phrases(song, 1.2, 2.0):
        n = ours[(ours >= s) & (ours <= e)]
        pts = np.concatenate(([s], n, [e]))
        gaps = np.diff(pts)
        if len(gaps) and gaps.max() > 1.0:
            i = int(gaps.argmax())
            F.append(dict(kind="PHRASE_HOLE", t=float(pts[i]), sev=float(gaps.max()) / 2.0,
                          msg=f"{gaps.max():.2f}s with no notes while the singer is "
                              f"still going (phrase {mmss(s)}–{mmss(e)})"))

    # ---- 6. the ending, against the CARRYING instrument ---------------------
    # Hunger: the human's last note lands on the final DRUM hit; ours goes one
    # 16th further on a residual bass onset. "Last onset of any stem" is the wrong
    # reference -- a decaying bass or a held vocal outlasts the pulse.
    # ★VALIDATED COHORT-WIDE (13 songs, 2026-08-04): human mappers end the map on
    # the carrier's final hit — median (human_end − carrier_end) = **+0.00 s**,
    # with 1f3d7 +0.01, 1f333 +0.01, 1fbfb +0.02, 1fb44 +0.06, 1f7f1 +0.00. Ours
    # scatters: median −0.30 s, range −5.55 to +19.30. So the reference is right.
    # ⚠️But "stops early" is ONLY a defect where the HUMAN does not also stop early:
    # on 1f9a0 the human ends 10.35 s before the last drum hit and 1fb71 5.68 s,
    # so measuring against the carrier alone would flag correct restraint. Compare
    # to the human whenever one exists.
    carrier = max(("drums", "bass"), key=lambda s: len(stems.get(s, [])))
    if len(stems.get(carrier, [])):
        c_last = float(stems[carrier].max())
        d = ours.max() - c_last
        # ⚠️The +0.00 fallback for songs with no human map is the MEDIAN of n=13,
        # and that distribution has a long negative tail (+0.69, −1.68, −1.89,
        # −5.68, −10.35 all occur). So a "SHORT OF" finding on a song without a
        # human map is WEAK evidence — some human mappers really do stop seconds
        # early. Findings carrying `the human` in the message are the solid ones.
        ref, refname = (float(human.max()) - c_last, "the human") if human is not None \
            else (0.0, "the human norm (+0.00s, n=13, weak)")
        rel = d - ref
        if abs(rel) > 0.25:
            word = "PAST" if rel > 0 else "SHORT OF"
            F.append(dict(kind="ENDING", t=float(ours.max()), sev=min(abs(rel), 2.0),
                          msg=f"ends {abs(rel):.2f}s {word} where {refname} ends, "
                              f"relative to the final {carrier} hit ({mmss(c_last)})"
                              f" [ours {d:+.2f}s, ref {ref:+.2f}s]"))
    return F


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--human", default=None)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--html", default=None)
    a = ap.parse_args()

    L = scorecard._load_any(pathlib.Path(a.map))
    if not L:
        sys.exit(f"could not load {a.map}")
    bpm = float(L[1])
    ours = np.sort(np.asarray(alignment.note_times(L[0], bpm), dtype=float))

    hp = pathlib.Path(a.human) if a.human else REPO / "data" / "raw" / f"{a.song}.zip"
    human = None
    if hp.exists():
        H = load_expert_only(hp)
        if H:
            human = np.sort(np.asarray(alignment.note_times(H[0], float(H[1])), dtype=float))

    F = review(a.song, ours, bpm, human)
    counts = collections.Counter(f["kind"] for f in F)
    print(f"=== REVIEW: {a.song}  ({pathlib.Path(a.map).name}, bpm {bpm:g}) ===")
    print(f"  {len(ours)} distinct note times"
          + (f", human {len(human)}" if human is not None else ", no human map")
          + f"   |   findings: {dict(counts)}")
    print(f"\n  top {a.top} by severity — GO LISTEN AT THESE TIMESTAMPS:\n")
    F.sort(key=lambda f: -f["sev"])
    for f in F[:a.top]:
        print(f"  {mmss(f['t']):>9s}  {f['kind']:<16s} {f['msg']}")
    if not F:
        print("  (nothing flagged)")

    print("\n  ⚠️ POINTERS, NOT VERDICTS. Restraint over a quiet passage is correct")
    print("     mapping. Rank, look, describe — do not tune against this file.")

    if a.html:
        rows = "\n".join(
            f"<tr><td class=t>{mmss(f['t'])}</td><td class=k>{f['kind']}</td>"
            f"<td>{_html.escape(f['msg'])}</td></tr>" for f in F[:a.top])
        pathlib.Path(a.html).write_text(
            "<meta charset=utf-8><style>body{font:14px/1.5 system-ui;margin:2rem;max-width:60rem}"
            "table{border-collapse:collapse;width:100%}td{padding:.35rem .6rem;"
            "border-bottom:1px solid #8883}.t{font-variant-numeric:tabular-nums;white-space:nowrap}"
            ".k{font-weight:600;white-space:nowrap}</style>"
            f"<h1>Review — {_html.escape(a.song)}</h1><p>{len(ours)} note times, bpm {bpm:g}</p>"
            f"<table>{rows}</table>")
        print(f"\n  wrote {a.html}")


if __name__ == "__main__":
    main()
