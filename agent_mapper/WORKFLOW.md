# Building a map as an agent — the loop

Repo: `/home/kyle/repos/beatsaber_automapper`. `source .venv/bin/activate` first.
All commands are `python agent_mapper/<tool>`.

## 0. Transcribe the vocals once (slow, cached)
```bash
python agent_mapper/lyrics.py data/eval_songset/1f333.ogg --lines
```
Instrumental songs legitimately return nothing — that is an answer, not a failure.

## 0b. THE OTHER THREE PERCEPTION AXES (all cached after the first run)
```bash
python agent_mapper/structure.py  <audio>            # ★sections, and which REPEAT
python agent_mapper/melody.py     <audio>            # pitch: what note, which way it moves
python agent_mapper/percussion.py <audio>            # which drum is hitting
```
Each takes `--validate` and will tell you **on this song** whether to trust it —
a screamed vocal has no pitch, and a drum track too repetitive to carry information
says so instead of guessing. See `docs/perception_scorecard.md` for what each control
established and what is still missing.

## 1. READ THE WHOLE SONG BEFORE PLACING ANYTHING
```bash
python agent_mapper/brief.py data/eval_songset/1f333.ogg          # 8-bar timeline
```
You get: the grid, per-stem onset density per 8 bars, an energy column, the lyric for
each phrase, and ★**the lyric repeat map** — which lines are sung more than once and in
which bars. That last one is the structure, read rather than inferred.

**Decide the plan here, in prose, before touching the map**: which instrument carries
each section, how dense it should be, where the player should breathe. This is the part
only you can do — the generator has a 16-note window and cannot know that bar 129 is a
breakdown or that bars 59/115/195 are the same chorus.

## 2. ZOOM IN where you are about to write
```bash
python agent_mapper/brief.py <audio> --bars 59-66
```
One row per stem per bar, sixteen cells to a bar. `mapctl` accepts notes in **exactly
these cells**, so what you read is what you write.

## 3. BUILD, section by section
```bash
python agent_mapper/mapctl.py init <audio> --name hunger

# bulk: follow an instrument over a range
python agent_mapper/mapctl.py auto hunger --bars 33-56 --follow vocals --lead L --wide
python agent_mapper/mapctl.py auto hunger --bars 33-56 --follow drums --every 3 --lead R

# hand-place the moments that matter
python agent_mapper/mapctl.py add hunger --from phrase.txt
```
`phrase.txt` is `<bar>.<slot> <L|R> <col 0-3> <row 0-2> <dir>`; dir is
`U D L R UL UR DL DR X`. A bad line rejects the whole file — a half-applied phrase is
worse than none, because you cannot see which half landed.

**Layering is normal**: follow the drums, then add the vocal line on top. `auto`
assigns hands over the **merged** timeline and enforces the human per-hand floor
(~150 ms), so it will *skip* onsets no hand can reach and tell you how many.

## 3b. ★MAP A SECTION ONCE, THEN REUSE IT
```bash
python agent_mapper/mapctl.py plan  hunger                 # the section work list
python agent_mapper/mapctl.py auto  hunger --bars 55-74 --follow drums --wide
python agent_mapper/mapctl.py reuse hunger --label D       # -> 113-130 and 194-215
```
`reuse` copies the mapped instance of a section to its repeats, truncated to each
target's own length. ⚠️**`--vary` defaults to 0.15 on purpose** — the open question on
review set A is whether repetition reads INTENTIONAL or LAZY, and a byte-identical
repeat is the definition of lazy. Use `--vary 0` only when you want an exact copy.

## 4. CHECK BEFORE YOU EXPORT
```bash
python agent_mapper/mapctl.py check hunger      # parity, doubles, violations
python agent_mapper/mapctl.py view hunger --bars 59-62   # your notes vs the stems
python agent_mapper/mapctl.py status hunger     # coverage, and what is still empty
```
A parity violation is not a rough edge, it is an unplayable map.

## 5. TWEAK ON FEEDBACK — the loop Kyle asked for
Feedback is almost always about a *place* and a *feeling*. Translate it, redo that
range only, re-check:

| he says | what to do |
|---|---|
| "the chorus is too empty" | `clear --bars 57-72` then `auto --follow drums` (drop `--every`) |
| "too busy / exhausting there" | `clear`, re-`auto` with `--every 2` or a sparser stem |
| "it ignores the singing" | `auto --follow vocals` over that range |
| "it doesn't line up" | check the grid: `init` prints the fit `r`; a weak fit means the bar lines are wrong, and no amount of note-editing fixes that |
| "let me breathe before the drop" | `clear` the bars before it; emptiness is a choice |

```bash
python agent_mapper/mapctl.py clear hunger --bars 57-72
python agent_mapper/mapctl.py auto  hunger --bars 57-72 --follow drums --lead L --wide
python agent_mapper/mapctl.py check hunger
```

## 6. EXPORT AND INSTALL
```bash
python agent_mapper/mapctl.py export hunger --out outputs/agent_hunger.zip
python scripts/deploy_maps.py outputs/agent_hunger.zip --replace
```

---

## What to distrust in your own map

★**Onset precision is circular here.** `auto` places notes on the onsets the metric
scores against, so a high number is guaranteed and means nothing. Judge yourself on the
axes you did *not* optimise: `ebpm_burst` (per-hand speed), `travel`, `double_share`,
nps against the human Expert median of **3.91**.

★**The suite does not track Kyle's ear.** It ranks the map he called *"really empty"*
second-best and the one he graded **A+** fifth-worst. A good score is not a good map;
the only verdict that counts is him playing it. Record what he says with
`python scripts/record_verdict.py`.

⚠️**Known gaps in the current `auto`** (both measured, both open):
- `travel` **4.42 vs a human 12.53** — it uses two columns and two rows per hand, so
  the hands barely move. Hand-place the passages you want to feel big.
- `double_share` **0.000 vs a human 0.146** — it never plays both hands at one instant.
  Doubles are how a human adds density *without* speeding up either hand.
