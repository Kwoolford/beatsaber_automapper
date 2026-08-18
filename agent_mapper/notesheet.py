#!/usr/bin/env python
"""THE NOTESHEET — the song as a readable score, on one page.

★**Why this exists.** On 2026-08-17 Kyle judged the maps and gave a defect list rather
than a preference: *"very slow, slightly off beat, doing drops at the wrong time, not
following the main vocals, random bursts of really fast non flowy notes… the nps is
generally wasted on every few non main notes."* Then the directive:

> *"make the visibility suite the top priority… so visibly great that you can go back
> through and evaluate through me instead of making another evaluation metric."*

Two levers with strong numeric evidence (`BEAT_GRID_PHASE`, 74 songs better and 0
worse; an agent-authored map at human `ebpm_burst` and zero parity violations) both
failed to reach his ear. ⇒**A passed DoD is evidence about the metric, not about the
map.** The answer is not a seventh axis, it is to make the map legible enough that *he*
is the evaluator.

**The perception tools already existed and were never time-aligned into one picture** —
`melody.py` knows the pitch of every onset, `percussion.py` knows which drum hit,
`structure.py` knows where the sections are, `lyrics.py` knows the words and when they
are sung. Each printed its own table on its own time axis. This puts them on **one**.

## What it draws
Systems, stacked down the page like an orchestral score. Each system is 8 bars wide and
carries, top to bottom:

- the **section banner** — letter, role (`DROP` / `build` / `breakdown` …), bar range
- **VOX** — every pitched vocal onset at its own pitch, with the lyric beneath it
- **LEAD** — the `other` stem's top line (a salience peak; see the confidence note)
- **BASS** — the bass line
- **KIT** — crash / hat / snare / kick as a drum tab, strike size by velocity

⚠️**It draws what the perception tools actually found, including where they found
nothing.** A stem with no trackable pitch renders as an empty lane and says so in the
header rather than silently drawing a flat line — on 1f333 the vocals are screamed and
pitch-track at coverage 0.28, and a viewer must be able to see that the blank lane is
the *song*, not a bug.

★**V2 will draw our map on top of these lanes** (HIT / MISSED / WASTED), which is why
every lane hue here is cool — teal, periwinkle, violet — and warm and green are spent
on nothing. Those are reserved for the map's three verdicts, so the two layers can
never be confused for each other.

Usage:
    python agent_mapper/notesheet.py data/eval_songset/1f8d6.ogg
    python agent_mapper/notesheet.py <audio.ogg> --out page.html --bars 8
"""

from __future__ import annotations

import argparse
import html
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "notesheets"

# --- geometry, in SVG user units -------------------------------------------------
W = 1160                 # logical width of a system; the SVG scales to its container
GUT = 74                 # left gutter holding the lane labels
PAD_R = 12
LANE_H = 52              # a melodic lane
LYRIC_H = 15
KIT_ROW = 11
BANNER_H = 20
RULER_H = 16
MAP_H = 24               # the map lane, drawn only with --map

MELODIC = (("vocals", "VOX", "vox"), ("other", "LEAD", "lead"), ("bass", "BASS", "bass"))
KIT_ROWS = (("crash", "crash"), ("hat", "hat"), ("snare", "snare"), ("kick", "kick"))


def _mmss(t: float) -> str:
    t = max(t, 0.0)
    return f"{int(t // 60)}:{t % 60:04.1f}"


def collect(audio: pathlib.Path, force: bool = False,
            map_zip: pathlib.Path | None = None, main_rule: str | None = None) -> dict:
    """Everything the page draws, on one time axis. Each tool is cached separately."""
    import brief as _brief
    import melody as _mel
    import percussion as _perc
    import structure as _struct

    a = _brief.analyse(audio, force)
    g = _brief.grid(a)
    mel = _mel.analyse(audio, force)
    perc = _perc.analyse(audio, force)
    sec = _struct.analyse(audio, force)
    _struct.roles(sec["sections"], a)

    # Pitch range per stem over the WHOLE song. A per-system range would make the
    # contour lie: the same drawn height would mean a different note in every system.
    span = {}
    for stem, _, _ in MELODIC:
        ms = np.array([e["midi"] for e in mel["stems"].get(stem, [])], dtype=float)
        span[stem] = ((float(np.percentile(ms, 2)), float(np.percentile(ms, 98)))
                      if len(ms) > 4 else None)

    try:
        words = _brief.lyric_words(audio.stem)
    except Exception:
        words = []

    d = {"song": audio.stem, "title": None, "grid": g, "dur": a["dur"], "r": a.get("r"),
         "melody": mel, "perc": perc, "sections": sec["sections"],
         "span": span, "words": words, "overlay": None}

    if map_zip is not None:
        import overlay as _ov
        notes, map_bpm = _ov.load_map(map_zip)
        d["overlay"] = _ov.classify(notes, d, main_rule or _ov.MAIN_DEFAULT)
        d["overlay"]["map_name"] = map_zip.stem
        d["overlay"]["map_bpm"] = map_bpm
    return d


# --- SVG pieces ------------------------------------------------------------------

def _x(t: float, t0: float, t1: float) -> float:
    return GUT + (t - t0) / max(t1 - t0, 1e-9) * (W - GUT - PAD_R)


def _lane(d: dict, stem: str, cls: str, t0: float, t1: float, top: float) -> list[str]:
    """One melodic lane: a mark per pitched onset, placed at its own pitch."""
    out = []
    sp = d["span"][stem]
    ev = [e for e in d["melody"]["stems"].get(stem, []) if t0 <= e["t"] < t1]
    if sp is None:
        return out
    lo, hi = sp
    for e in ev:
        frac = float(np.clip((e["midi"] - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
        y = top + (LANE_H - 6) * (1.0 - frac) + 3
        x = _x(e["t"], t0, t1)
        w = max(_x(e["t"] + min(e["dur"], 0.9), t0, t1) - x, 3.0)
        out.append(f'<rect class="n {cls}" x="{x:.1f}" y="{y - 2:.1f}" '
                   f'width="{w:.1f}" height="4" rx="2"><title>{e["name"]} '
                   f'{_mmss(e["t"])}</title></rect>')
    return out


def _kit(d: dict, t0: float, t1: float, top: float) -> list[str]:
    out = []
    hits = [h for h in d["perc"]["hits"] if t0 <= h["t"] < t1]
    for row, (piece, cls) in enumerate(KIT_ROWS):
        y = top + row * KIT_ROW + KIT_ROW / 2
        for h in hits:
            if h["piece"] != piece:
                continue
            x = _x(h["t"], t0, t1)
            r = 1.6 + 1.9 * min(h["vel"], 1.2) / 1.2
            out.append(f'<circle class="k {cls}" cx="{x:.1f}" cy="{y:.1f}" '
                       f'r="{r:.1f}"><title>{piece} {_mmss(h["t"])}</title></circle>')
    return out


def _system(d: dict, b0: int, nbars: int) -> str:
    """One system: `nbars` bars of every lane, plus its ruler and section banner."""
    import brief as _brief

    g = d["grid"]
    t0 = _brief.bar_time(g, b0)
    t1 = t0 + nbars * g["bar_s"]

    top = BANNER_H + RULER_H
    map_top = None
    if d.get("overlay"):
        map_top, top = top, top + MAP_H + 8
    lanes: list[tuple[str, str, float]] = []
    for stem, label, cls in MELODIC:
        lanes.append((label, cls, top))
        top += LANE_H + (LYRIC_H if stem == "vocals" else 0) + 6
    kit_top = top
    height = kit_top + len(KIT_ROWS) * KIT_ROW + 8

    s: list[str] = []

    # section banners — a section can start mid-system, so clip each to this window
    for sec in d["sections"]:
        a, b = max(sec["t0"], t0), min(sec["t1"], t1)
        if b - a <= 0.01:
            continue
        xa, xb = _x(a, t0, t1), _x(b, t0, t1)
        role = sec["role"]
        hot = " hot" if role in ("DROP", "peak") else ""
        s.append(f'<rect class="ban{hot}" x="{xa:.1f}" y="2" width="{max(xb-xa,2):.1f}" '
                 f'height="{BANNER_H - 6}" rx="3" />')
        if xb - xa > 54:
            s.append(f'<text class="banl{hot}" x="{xa + 6:.1f}" y="{BANNER_H - 8}">'
                     f'{html.escape(sec["label"])} · {html.escape(role)}</text>')

    # ruler: a bar line every bar, a beat tick every beat, numbers on the bar
    for i in range(nbars + 1):
        bt = t0 + i * g["bar_s"]
        x = _x(bt, t0, t1)
        s.append(f'<line class="bar" x1="{x:.1f}" y1="{BANNER_H}" x2="{x:.1f}" '
                 f'y2="{height - 6}" />')
        if i < nbars:
            for beat in range(1, 4):
                bx = _x(bt + beat * g["spb"], t0, t1)
                s.append(f'<line class="beat" x1="{bx:.1f}" y1="{BANNER_H + RULER_H - 4}"'
                         f' x2="{bx:.1f}" y2="{height - 6}" />')
            s.append(f'<text class="barn" x="{x + 3:.1f}" y="{BANNER_H + 11}">'
                     f'{b0 + i}</text>')
    s.append(f'<text class="tc" x="{GUT - 6:.1f}" y="{BANNER_H + 11}">{_mmss(t0)}</text>')

    # lanes
    for (stem, label, cls), (_, _, ltop) in zip(MELODIC, lanes):
        s.append(f'<rect class="bed" x="{GUT}" y="{ltop}" width="{W - GUT - PAD_R}" '
                 f'height="{LANE_H}" rx="2" />')
        s.append(f'<text class="ll {cls}" x="{GUT - 8}" y="{ltop + LANE_H / 2 + 3:.1f}">'
                 f'{label}</text>')
        s.extend(_lane(d, stem, cls, t0, t1, ltop))

    # lyrics under the vocal lane, thinned so words never collide
    ly_top = lanes[0][2] + LANE_H
    free_at = -1e9
    for w in d["words"]:
        if not (t0 <= w["t"] < t1):
            continue
        x = _x(w["t"], t0, t1)
        txt = str(w.get("word", "")).strip()
        # Skip a word that would land on top of the one before it. The bound is the
        # PREVIOUS word's drawn width (~5.6 px a glyph at 9.5 px Plex Sans) plus a
        # gap — using the current word's width, as a first version did, lets a short
        # word slide under a long one.
        if not txt or x < free_at:
            continue
        free_at = x + len(txt) * 5.6 + 6
        s.append(f'<text class="ly" x="{x:.1f}" y="{ly_top + 11:.1f}">'
                 f'{html.escape(txt)}</text>')

    # ★THE OVERLAY. The map goes at the TOP, directly under the ruler: it is the
    # thing under audit and everything below it is the evidence. MISSED events are
    # drawn instead inside the lane they belong to, so "not following the main vocals"
    # appears as amber gaps *in the vocal line* rather than as a number.
    if map_top is not None:
        ov = d["overlay"]
        lane_top = {"vocals": lanes[0][2], "other": lanes[1][2], "kit": kit_top}
        lane_h = {"vocals": LANE_H, "other": LANE_H, "kit": len(KIT_ROWS) * KIT_ROW}
        s.append(f'<rect class="bed" x="{GUT}" y="{map_top}" '
                 f'width="{W - GUT - PAD_R}" height="{MAP_H}" rx="2" />')
        s.append(f'<text class="ll mapl" x="{GUT - 8}" '
                 f'y="{map_top + MAP_H / 2 + 3:.1f}">MAP</text>')
        for v in ov["verdicts"]:
            if not (t0 <= v["t"] < t1):
                continue
            x = _x(v["t"], t0, t1)
            s.append(f'<rect class="mn {v["v"]}" x="{x - 1.3:.1f}" y="{map_top + 4}" '
                     f'width="2.6" height="{MAP_H - 8}" rx="1.3">'
                     f'<title>{v["v"].upper()} on {v["on"]} · {_mmss(v["t"])}</title>'
                     f'</rect>')
        for m in ov["missed"]:
            if not (t0 <= m["t"] < t1) or m["lane"] not in lane_top:
                continue
            x = _x(m["t"], t0, t1)
            y = lane_top[m["lane"]]
            s.append(f'<line class="miss" x1="{x:.1f}" y1="{y + 1:.1f}" x2="{x:.1f}" '
                     f'y2="{y + lane_h[m["lane"]] - 1:.1f}">'
                     f'<title>MISSED {m["src"]} · {_mmss(m["t"])}</title></line>')

    s.append(f'<text class="ll kitl" x="{GUT - 8}" '
             f'y="{kit_top + len(KIT_ROWS) * KIT_ROW / 2 + 3:.1f}">KIT</text>')
    for row, (piece, _) in enumerate(KIT_ROWS):
        s.append(f'<text class="kr" x="{GUT - 8}" '
                 f'y="{kit_top + row * KIT_ROW + KIT_ROW - 2:.1f}">{piece}</text>')
    s.extend(_kit(d, t0, t1, kit_top))

    return (f'<div class="sys"><svg viewBox="0 0 {W} {height}" width="{W}" '
            f'height="{height}" role="img" aria-label="bars {b0} to {b0 + nbars - 1}">'
            + "".join(s) + "</svg></div>")


CSS = """
:root{
  --ground:#F6F6F9; --surface:#FFFFFF; --bed:#EEEEF4; --rule:#D9D9E4;
  --ink:#181A21; --ink-2:#5A5F70; --ink-3:#8B90A2;
  --vox:#1E9E99; --lead:#4B6FD0; --bass:#7A5FC4; --kit:#6B7285;
  --hot:#C2410C; --hot-bed:#FBE6DA;
  --hit:#2E8B57; --missed:#C98A0B; --wasted:#C0392B;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#101219; --surface:#171A22; --bed:#1D212B; --rule:#2C313E;
    --ink:#E8E9EF; --ink-2:#9AA0B2; --ink-3:#6C7285;
    --vox:#5BC8C4; --lead:#8AA6F0; --bass:#B49BEE; --kit:#7A8194;
    --hot:#F2884B; --hot-bed:#3A2417;
    --hit:#5FCB8C; --missed:#F0B429; --wasted:#F2685A;
  }
}
:root[data-theme="dark"]{
  --ground:#101219; --surface:#171A22; --bed:#1D212B; --rule:#2C313E;
  --ink:#E8E9EF; --ink-2:#9AA0B2; --ink-3:#6C7285;
  --vox:#5BC8C4; --lead:#8AA6F0; --bass:#B49BEE; --kit:#7A8194;
  --hot:#F2884B; --hot-bed:#3A2417;
  --hit:#5FCB8C; --missed:#F0B429; --wasted:#F2685A;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  font-size:14px; line-height:1.5;
}
.wrap{max-width:1240px; margin:0 auto; padding:28px 18px 64px}
header{border-bottom:1px solid var(--rule); padding-bottom:18px; margin-bottom:22px}
h1{
  font-family:"Instrument Serif",Georgia,serif; font-weight:400;
  font-size:44px; line-height:1.05; margin:0 0 2px; letter-spacing:-.01em;
  text-wrap:balance;
}
.sub{color:var(--ink-2); margin:0 0 16px; max-width:64ch}
.facts{
  display:flex; flex-wrap:wrap; gap:8px 10px; list-style:none; padding:0; margin:0;
}
.facts li{
  background:var(--surface); border:1px solid var(--rule); border-radius:4px;
  padding:5px 9px; font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px;
  font-variant-numeric:tabular-nums; color:var(--ink-2);
}
.facts b{color:var(--ink); font-weight:600}
.facts li.warn{border-color:var(--hot); background:var(--hot-bed); color:var(--ink)}
.legend{
  display:flex; flex-wrap:wrap; gap:6px 16px; margin:16px 0 0; padding:0; list-style:none;
  font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif; font-size:11.5px;
  letter-spacing:.06em; text-transform:uppercase; color:var(--ink-2);
}
.legend i{display:inline-block; width:16px; height:4px; border-radius:2px;
  margin-right:6px; vertical-align:middle}
.scroll{overflow-x:auto; padding-bottom:4px}
.sys{background:var(--surface); border:1px solid var(--rule); border-radius:5px;
  margin:0 0 9px; padding:4px 0}
.sys svg{display:block; width:100%; min-width:860px; height:auto}
.bed{fill:var(--bed)}
.bar{stroke:var(--rule); stroke-width:1}
.beat{stroke:var(--rule); stroke-width:.5; opacity:.55}
.n{opacity:.95}
.n.vox{fill:var(--vox)} .n.lead{fill:var(--lead)} .n.bass{fill:var(--bass)}
.k{fill:var(--kit)}
/* A crash is the most reliable "something new starts here" marker in recorded music,
   so it reads as a ring rather than a dot — visible without borrowing a lane's hue. */
.k.crash{fill:none; stroke:var(--kit); stroke-width:1.2}
.ban{fill:var(--bed)}
.ban.hot{fill:var(--hot-bed)}
text{font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums}
.barn{font-size:9px; fill:var(--ink-3)}
.tc{font-size:9.5px; fill:var(--ink-2); text-anchor:end}
.banl{font-size:10px; fill:var(--ink-2); letter-spacing:.08em}
.banl.hot{fill:var(--hot); font-weight:600}
.ll{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif; font-size:10px;
  letter-spacing:.1em; text-anchor:end}
.ll.vox{fill:var(--vox)} .ll.lead{fill:var(--lead)} .ll.bass{fill:var(--bass)}
.ll.kitl{fill:var(--kit)}
.ll.mapl{fill:var(--ink)}
.mn.hit{fill:var(--hit)}
.mn.wasted{fill:var(--wasted)}
.miss{stroke:var(--missed); stroke-width:1.4; opacity:.7}
.tally{display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:10px;
  margin:16px 0 0; padding:0; list-style:none}
.tally li{background:var(--surface); border:1px solid var(--rule); border-left-width:3px;
  border-radius:4px; padding:9px 11px}
.tally .v{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:22px;
  font-variant-numeric:tabular-nums; line-height:1.1; display:block}
.tally .l{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif; font-size:11px;
  letter-spacing:.08em; text-transform:uppercase; color:var(--ink-2)}
.tally .w{font-size:12px; color:var(--ink-2); display:block; margin-top:3px}
.tally li.hit{border-left-color:var(--hit)} .tally li.hit .v{color:var(--hit)}
.tally li.missed{border-left-color:var(--missed)} .tally li.missed .v{color:var(--missed)}
.tally li.wasted{border-left-color:var(--wasted)} .tally li.wasted .v{color:var(--wasted)}
.rule{background:var(--surface); border:1px solid var(--rule); border-radius:4px;
  padding:12px 14px; margin:16px 0 0; font-size:13px; color:var(--ink-2)}
.rule b{color:var(--ink)}
.rule code{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:12px;
  background:var(--bed); padding:1px 5px; border-radius:3px; color:var(--ink)}
.kr{font-size:7.5px; fill:var(--ink-3); text-anchor:end}
.ly{font-family:"IBM Plex Sans",sans-serif; font-size:9.5px; fill:var(--ink-2)}
.note{color:var(--ink-2); font-size:13px; max-width:66ch; margin:26px 0 0}
.note b{color:var(--ink)}
"""


def render(d: dict, bars: int = 8) -> str:
    g = d["grid"]
    n_bars = int(d["dur"] / g["bar_s"]) + 1
    meta = d["melody"]["meta"]

    facts = [f'<li>bpm <b>{g["bpm"]:.2f}</b></li>',
             f'<li>length <b>{_mmss(d["dur"])}</b></li>',
             f'<li>bars <b>{n_bars}</b></li>',
             f'<li>sections <b>{len(d["sections"])}</b></li>']
    for stem, label, _ in MELODIC:
        m = meta.get(stem, {})
        cov = m.get("coverage")
        if cov is None:
            continue
        warn = " class=\"warn\"" if cov < 0.45 else ""
        facts.append(f'<li{warn}>{label.lower()} pitched <b>{cov:.0%}</b> '
                     f'of {m.get("onsets", 0)} onsets</li>')
    kit = d["perc"].get("counts", {})
    facts.append("<li>kit <b>" + " ".join(f"{v}&#8202;{k}" for k, v in kit.items())
                 + "</b></li>")

    low = [lbl for stem, lbl, _ in MELODIC
           if (meta.get(stem, {}).get("coverage") or 1.0) < 0.45]
    caveat = ""
    if low:
        caveat = (f'<p class="note">⚠️<b>{", ".join(low)} is drawn nearly empty because '
                  "the pitch tracker found nothing there</b>, not because the stem is "
                  "silent. Screamed or heavily distorted singing genuinely has no f0. "
                  "Read that lane's rhythm from the onsets, not its contour.</p>")

    systems = "".join(_system(d, b0, min(bars, n_bars - b0 + 1))
                      for b0 in range(1, n_bars + 1, bars)
                      if _bar_start_ok(d, b0))

    ov = d.get("overlay")
    tally = rulebox = ""
    if ov:
        dur = d["dur"]
        facts.insert(0, f'<li>map <b>{html.escape(ov["map_name"])}</b></li>')
        if abs(ov["map_bpm"] - g["bpm"]) > 0.6:
            facts.append(f'<li class="warn">map bpm <b>{ov["map_bpm"]:.2f}</b> '
                         f'vs detected {g["bpm"]:.2f} — notes drift against the lanes</li>')
        tally = f"""<ul class="tally">
  <li class="hit"><span class="v">{ov['precision']:.0%}</span>
    <span class="l">Hit</span><span class="w">{ov['hit']} of {ov['n_notes']} notes
    land on a main event &middot; {ov['n_notes']/dur:.2f} nps</span></li>
  <li class="missed"><span class="v">{1 - ov['recall']:.0%}</span>
    <span class="l">Missed</span><span class="w">{ov['n_missed']} of {ov['n_main']} main
    events have no note &mdash; we play {ov['recall']:.0%} of the main line</span></li>
  <li class="wasted"><span class="v">{ov['wasted'] / max(ov['n_notes'],1):.0%}</span>
    <span class="l">Wasted</span><span class="w">{ov['wasted']} notes with no main event
    under them, {ov['wasted_on_nothing']} of them on <b>nothing at all</b></span></li>
</ul>"""
        rulebox = ("""<p class="rule">★<b>&ldquo;Main&rdquo; is a guess, and it is the thing
  to argue with.</b> Right now a main event is <code>""" + html.escape(ov["rule"])
  + """</code>: every pitched vocal onset, every kick and snare, and &mdash; only where the
  vocal rests for two beats or more &mdash; the lead line. Hats, toms, bass runs under a vocal
  and anything unpitched are <b>not</b> main; a note landing on one of those still counts as
  wasted, but the tooltip says what it hit, and a note on <em>nothing</em> is counted
  separately. Tolerance is &plusmn;""" + f"{ov['tol']*1000:.0f}"
  + """&nbsp;ms. <b>If the colours look wrong to you, the rule is what is wrong</b> &mdash;
  tell me which events you would have called main and I will change this line, not the map.</p>""")

    legend = "".join(
        f'<li><i style="background:var(--{cls})"></i>{lbl}</li>'
        for _, lbl, cls in MELODIC) + '<li><i style="background:var(--kit)"></i>KIT</li>'
    if ov:
        legend = ('<li><i style="background:var(--hit)"></i>hit</li>'
                  '<li><i style="background:var(--missed)"></i>missed</li>'
                  '<li><i style="background:var(--wasted)"></i>wasted</li>') + legend

    sub = ("Our map drawn against the music under it. The top lane is the map; "
           "everything below is the evidence, and an amber tick is a main event we "
           "left unplayed." if ov else
           "Every pitched onset, every drum strike and every word, on one time axis — "
           "what the song is doing, before any map is laid over it.")
    name = d.get("title") or d["song"]
    title = (f"{name} Overlay" if ov else f"{name} Notesheet")

    return f"""<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;600&family=Instrument+Serif&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>{html.escape(name)}</h1>
  <p class="sub">{sub}</p>
  <ul class="facts">{"".join(facts)}</ul>
  <ul class="legend">{legend}</ul>
  {tally}
  {rulebox}
</header>
<div class="scroll">{systems}</div>
{caveat}
<p class="note">Each system is {bars} bars. Lane height is pitch, scaled to that stem's
own range across the whole song — so a mark that sits higher <b>is</b> a higher note,
in this system and every other one. Kit strike size is velocity. Hover any mark for its
note name and timestamp.</p>
</div>
"""


def _bar_start_ok(d: dict, b0: int) -> bool:
    import brief as _brief
    return _brief.bar_time(d["grid"], b0) < d["dur"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--bars", type=int, default=8, help="bars per system")
    ap.add_argument("--force", action="store_true", help="recompute every analysis")
    ap.add_argument("--map", type=pathlib.Path, default=None,
                    help="a map zip to draw over the score as HIT/MISSED/WASTED")
    ap.add_argument("--main", default=None,
                    help="which events count as main (see overlay.py)")
    ap.add_argument("--name", default=None, help="the song's real name, for the heading")
    a_ = ap.parse_args()
    if not a_.audio.exists():
        print(f"no such audio: {a_.audio}", file=sys.stderr)
        return 2

    d = collect(a_.audio, a_.force, a_.map, a_.main)
    d["title"] = a_.name
    tag = f".{a_.map.stem}" if a_.map else ""
    out = a_.out or (OUT / f"{a_.audio.stem}{tag}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(d, a_.bars))

    m = d["melody"]["meta"]
    print(f"{d['song']}: {_mmss(d['dur'])}, {d['grid']['bpm']:.2f} bpm, "
          f"{len(d['sections'])} sections, {len(d['perc']['hits'])} kit hits, "
          f"{sum(len(v) for v in d['melody']['stems'].values())} pitched notes "
          f"(" + ", ".join(f"{s} {m[s]['coverage']:.2f}" for s, _, _ in MELODIC
                           if s in m) + ")")
    if d["overlay"]:
        o = d["overlay"]
        print(f"  overlay: HIT {o['precision']:.1%} of {o['n_notes']} notes | "
              f"MISSED {o['n_missed']} of {o['n_main']} main events "
              f"(we play {o['recall']:.1%}) | WASTED {o['wasted']} "
              f"({o['wasted_on_nothing']} on nothing)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
