"""Make a section HOLD an interval, instead of taking the union of two streams.

**The measurement this exists to fix** (PROGRESS.md 2026-08-20, n=23 songs)::

    pulse gap +0.186  =  vocabulary +0.077  +  ordering +0.096

Our maps score `pulse_stability` 0.329 against a human 0.514. The cause is NOT that
our note times are smeared -- we use **16 distinct intervals to the human's 347**,
and our top three cover more of the map than theirs do. We are *more* quantised than
a human. The cause is that we emit those intervals in nearly the order a shuffle
would: our lift over our own shuffled null is +0.072, the human's is +0.168, and a
human map with its rhythm randomly shuffled still holds a pulse as well as ours does
in its intended order.

So the fix is not a finer or coarser grid. It is to pick ONE interval per phrase and
run it, then change at a phrase boundary -- which is what `dominant_share` 0.469 vs
our 0.362 says a human does.

★**Density must not pay for this.** The human reaches 0.514 at **980 notes to our
641**: they are denser AND more pulsed, so pulse and density are not in tension and a
pulse bought by thinning is a failed fix, not a trade. That is why the period is
chosen FROM the phrase's own candidate count (`span / n_candidates`) rather than from
a fixed subdivision -- a busy phrase gets a fast pulse and a sparse one a slow pulse,
and the note count comes out where it started.

★**A run must be able to stop.** Emitting every lattice point from the first candidate
to the last would play straight through a rest, and `pulse_stability` would go to 1.0
-- a metronome, which is further from the human 0.514 than we are now, on the other
side. Runs break where the source events go quiet.
"""
from __future__ import annotations

# Periods offered, in grid slots. At SUBDIV=4 a slot is 1/4 beat, so these are
# 1/4, 1/2, 3/4, 1, 3/2 and 2 beats. The human's dominant intervals are 0.5 and
# 0.25 beats; 3 (a dotted eighth) is offered because our 0.75-beat excess shows the
# build already produces it -- better held deliberately than as an interleaving
# artifact.
PERIODS = (1, 2, 3, 4, 6, 8)
# A run stops when this many lattice points in a row have no source event near them.
# One is half a beat at P=2: enough to bridge a single quiet slot without inventing a
# bar of notes the song does not play. (Two put the first build 88 % above the human
# note count.)
MAX_EMPTY_RUN = 1


def _snap_cost(cands: list[int], start: int, period: int, phase: int) -> tuple[int, float]:
    """(distinct lattice points hit, mean snap distance) for one lattice."""
    hits: set[int] = set()
    total = 0.0
    for c in cands:
        k = round((c - start - phase) / period)
        lat = start + phase + k * period
        hits.add(lat)
        total += abs(c - lat)
    return len(hits), (total / len(cands) if cands else 0.0)


def _emit(cands: list[int], start: int, span: int, period: int,
          phase: int) -> list[int]:
    """The notes one (period, phase) lattice would actually play for this phrase."""
    lattice = [start + phase + k * period
               for k in range((span - phase + period - 1) // period)
               if start + phase + k * period < start + span]
    if not lattice:
        return sorted(set(cands))

    hit = set()
    for c in cands:
        k = round((c - start - phase) / period)
        hit.add(start + phase + k * period)

    out: list[int] = []
    empty = 0
    started = False
    for lat in lattice:
        if lat in hit:
            out.append(lat)
            started = True
            empty = 0
        elif started:
            empty += 1
            if empty <= MAX_EMPTY_RUN:
                # Hold the pulse across a short gap: this is what turns two hits
                # either side of a quiet slot into a RUN rather than a long IOI.
                out.append(lat)
            # Past MAX_EMPTY_RUN the run has ended; the next hit starts a new one.
    # ⚠️Trim the tail: the fill holds the pulse ACROSS a gap, but past the last source
    # event there is nothing to hold it to, and those notes would land after the
    # section ends -- outside the bar range the caller asked for.
    last_hit = max(hit)
    kept = {o for o in out if start <= o < start + span and o <= last_hit}

    # ★★BREAK THE PULSE WHERE THE MUSIC DOES, not on a tuned probability.
    # Holding the lattice and nothing else gives `pulse_stability` 0.853 against a
    # human 0.514 -- a metronome, which is further from human than the 0.329 we
    # started at, on the other side. The events the lattice explains WORST are the
    # song's own syncopations: a source event sitting a half-period away from every
    # lattice point is off-beat in the music, not noise.
    for c in cands:
        k = round((c - start - phase) / period)
        if abs(c - (start + phase + k * period)) * 2 >= period:
            kept.add(c)
    return sorted(o for o in kept if start <= o < start + span)


def quantise_phrase(cands: list[int], start: int, span: int,
                    periods: tuple[int, ...] = PERIODS) -> list[int]:
    """Lattice points to play for one phrase, as absolute slot indices.

    Every emitted point sits on one lattice, so consecutive gaps are equal by
    construction -- that is the whole mechanism.

    ★**The period is chosen by the note count it actually PRODUCES**, not by the
    events' average spacing. Anchoring on spacing looked right and was not: holding
    the pulse across quiet slots and restoring the song's syncopations both ADD
    notes, so a period picked before those steps ran overshot the section's budget by
    38 % (1031 notes against a control's 748 and a human's 746). Scoring each period
    by its own emission closes that loop exactly, with no fitted correction factor.
    """
    if len(cands) < 3:
        return sorted(set(cands))
    target = len(cands)
    best = None
    for period in periods:
        for phase in range(period):
            got = _emit(cands, start, span, period, phase)
            if not got:
                continue
            miss = abs(len(got) - target)
            dist = sum(min(abs(c - g) for g in got) for c in cands) / len(cands)
            score = (miss, dist)
            if best is None or score < best[0]:
                best = (score, got)
    return best[1] if best else sorted(set(cands))


def quantise(picks: list[tuple[int, int]], n_cells: int, bar0: int,
             phrase_bars: int = 4,
             periods: tuple[int, ...] = PERIODS) -> list[tuple[int, int]]:
    """Re-time `(bar, slot)` picks so each phrase holds one interval.

    `picks` are already snapped to the build grid; this decides WHICH of that grid's
    cells are played so that consecutive gaps repeat. Phrases are independent, which
    is what produces a change of gear at a musical boundary instead of never or
    constantly (`ioi_switch_rate` ours 25.9 vs human 14.8).
    """
    if not picks:
        return []
    idx = sorted({(b - bar0) * n_cells + s for b, s in picks})
    span = phrase_bars * n_cells
    lo, hi = idx[0], idx[-1]
    out: list[int] = []
    p0 = (lo // span) * span
    while p0 <= hi:
        cands = [i for i in idx if p0 <= i < p0 + span]
        if cands:
            out.extend(quantise_phrase(cands, p0, span, periods))
        p0 += span
    return sorted({(bar0 + i // n_cells, i % n_cells) for i in out})
