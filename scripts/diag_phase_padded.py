#!/usr/bin/env python
"""Is our phase estimator biased, or is P0.4 an artifact of the CORPUS CONVENTION?

**The trap this exists to avoid.** P0.4 measures our bar grid sitting **0.053 beats
early** of the human mapper's grid (18/18 songs). The obvious fix is to correct the
phase toward theirs. But:

  * `data/eval_songset/<id>.ogg` is **byte-identical** to the audio inside
    `data/raw/<id>.zip` (verified), and
  * **all 23 human maps carry `_songTimeOffset` = 0.**

⇒On this cohort the "true" phase is **0 by file convention** -- the audio was authored
so that beat 0 lands at t=0. So *any* rule that pushes our phase toward zero will look
perfect here **and will not generalise to audio that was never cut that way**, which is
precisely the "map any song you'd like" case. ★**A correction validated only on
corpus audio would be validated against a convention, not against the music.**

**The control: PAD THE AUDIO.** Prepend `PAD_S` seconds of silence -- not a multiple of
a slot -- and the music is unchanged while the convention is broken: the true downbeat
now sits at `PAD_S`, not at 0. Then

    if the estimator TRACKS the pad (phi_padded - PAD_S ~= phi_original), it is
       measuring the music, and the -0.053 bias is a genuine, correctable estimator
       error.
    if the estimator IGNORES the pad (phi_padded ~= phi_original), it is locking onto
       t=0 -- i.e. onto the convention -- and "our grid is early" is an artifact that
       says nothing about arbitrary audio.

⚠️Costs a fresh 6-stem Demucs pass per padded song (the event cache is keyed by song
id), so this runs on a handful of songs, not the cohort. It is a MECHANISM test: the
effect is already established at n=23.

Usage:
    python scripts/diag_phase_padded.py --songs 1f767 1f333 1f8d6 1f913
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import statistics as st
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
AM = REPO / "agent_mapper"
sys.path.insert(0, str(REPO / "src"))

# ★Deliberately NOT a multiple of a 1/4-beat slot at any cohort tempo (slots run
# 77-161 ms), so a "correct" answer cannot be reached by rounding.
PAD_S = 0.137


def pad_audio(src: pathlib.Path, dst: pathlib.Path, pad_s: float) -> bool:
    """Prepend `pad_s` of silence. Music identical, convention broken."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-t", f"{pad_s}", "-i",
         "anullsrc=channel_layout=stereo:sample_rate=44100",
         "-i", str(src), "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
         "-map", "[out]", str(dst)],
        capture_output=True, text=True)
    return r.returncode == 0 and dst.exists()


def fitted_phase(audio: pathlib.Path, name: str) -> float | None:
    """Run the builder's own grid fit on this audio and return its phase, in seconds."""
    # ⚠️`--name` is a FLAG, not positional, and the session lands in
    # `agent_mapper/sessions/<name>/session.json`.
    out = subprocess.run(
        [sys.executable, str(AM / "mapctl.py"), "init", str(audio), "--name", name,
         "--fresh"],
        capture_output=True, text=True, cwd=REPO)
    f = AM / "sessions" / name / "session.json"
    if out.returncode != 0 or not f.exists():
        tail = (out.stderr or out.stdout).strip().splitlines()[-2:]
        print(f"    init failed: {tail}")
        return None
    return float(json.loads(f.read_text()).get("phase", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*",
                    default=["1f767", "1f333", "1f8d6", "1f913"])
    ap.add_argument("--pad", type=float, default=PAD_S)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="phasepad_"))
    rows = []
    print(f"pad = {a.pad * 1000:.0f} ms of leading silence\n")
    print(f"{'song':8s}{'bpm':>6s}{'phi_orig':>10s}{'phi_pad':>10s}"
          f"{'tracked':>10s}{'residual':>11s}")
    print("-" * 55)
    for sid in a.songs:
        src = REPO / "data" / "eval_songset" / f"{sid}.ogg"
        ec = REPO / "outputs" / "event_cache" / f"{sid}.6s.json"
        if not src.exists() or not ec.exists():
            print(f"{sid:8s}  missing audio or event cache")
            continue
        d = json.loads(ec.read_text())
        bpm, phi0 = float(d["bpm"]), float(d["phase"])

        dst = tmp / f"{sid}_pad.ogg"
        if not pad_audio(src, dst, a.pad):
            print(f"{sid:8s}  ffmpeg padding failed")
            continue
        phi1 = fitted_phase(dst, f"padtest_{sid}")
        if phi1 is None:
            print(f"{sid:8s}  phase fit failed")
            continue

        spb = 60.0 / bpm
        slot = spb / 4.0
        # If the estimator tracks the music, phi_padded - pad should equal phi_orig,
        # modulo one slot (the grid is periodic, so only the remainder is meaningful).
        def wrap(x):
            return (x + slot / 2) % slot - slot / 2
        tracked = wrap(phi1 - a.pad - phi0)
        ignored = wrap(phi1 - phi0)
        print(f"{sid:8s}{bpm:6.0f}{phi0 * 1000:>9.0f}m{phi1 * 1000:>9.0f}m"
              f"{tracked * 1000:>9.0f}m{ignored * 1000:>10.0f}m")
        rows.append(dict(song=sid, bpm=bpm, phi_orig=phi0, phi_pad=phi1,
                         resid_if_tracked=tracked, resid_if_ignored=ignored,
                         slot_s=slot))

    if not rows:
        print("\nno songs measured")
        return 1

    mt = st.median([abs(r["resid_if_tracked"]) for r in rows])
    mi = st.median([abs(r["resid_if_ignored"]) for r in rows])
    print("-" * 55)
    print(f"median |residual| if the estimator TRACKS the pad : {mt * 1000:6.1f} ms")
    print(f"median |residual| if it IGNORES the pad           : {mi * 1000:6.1f} ms")

    print("\nVERDICT")
    if mt < mi:
        print("  ✅THE ESTIMATOR TRACKS THE MUSIC. Padding moved its phase with the")
        print("     audio, so the fit is measuring the song and not the file's t=0")
        print("     convention. ⇒P0.4's 0.053-beat bias is a REAL estimator error and")
        print("     correcting it should generalise to arbitrary audio.")
    else:
        print("  🔴THE ESTIMATOR LOCKS ONTO t=0, NOT THE MUSIC. Padding barely moved")
        print("     its phase, so on corpus audio it is being graded against a")
        print("     convention (every human map here has _songTimeOffset = 0).")
        print("     ⇒**Any phase 'correction' tuned on this cohort would be fitting")
        print("     the convention and would NOT transfer to a song Kyle brings.**")
        print("     Fix the estimator's music-tracking first; do not ship a shift.")

    if a.json:
        p = pathlib.Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(rows=rows, pad_s=a.pad,
                                     med_resid_tracked=mt,
                                     med_resid_ignored=mi), indent=2))
        print(f"\nwrote {p}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
