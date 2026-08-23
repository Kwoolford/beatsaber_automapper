"""Detect 2-state pendulum locks: one hand alternating between exactly two
(x, y, direction) states for >= MINRUN consecutive notes.

Motivated by Kyle's only recorded verdict-with-a-reason (2026-08-17, Hunger):
"the notes flow in a really odd way".  Reading bars 106-107 of the current
build showed both hands locked in a two-state oscillation for ~8 beats.
Every transition there is a common idiom and parity is legal, so no existing
axis can see it -- the phrase is locally valid and goes nowhere.
"""
import sys, glob, pathlib, statistics as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from audit_eval_suite import _load_generated, _load_human

MINRUN = 6   # notes == 3 full back-and-forth cycles


def load(z):
    for L in (_load_human, _load_generated):
        got = L(pathlib.Path(z))
        if got:
            return got
    return None


def pendulum_share(notes):
    """fraction of all notes sitting inside a >=MINRUN two-state alternation"""
    tot = locked = 0
    for hand in (0, 1):
        seq = [(n.x, n.y, n.direction) for n in notes if n.color == hand]
        tot += len(seq)
        i = 0
        while i + 1 < len(seq):
            a, b = seq[i], seq[i + 1]
            if a == b:
                i += 1
                continue
            j = i + 2
            while j < len(seq) and seq[j] == (a if (j - i) % 2 == 0 else b):
                j += 1
            run = j - i
            if run >= MINRUN:
                locked += run
                i = j
            else:
                i += 1
    return locked / tot if tot else 0.0


def main():
    pats = sys.argv[1:] or ['outputs/kyle_review_2026-08-19/*_FULL.zip::OURS',
                            'outputs/kyle_review_2026-08-19/*_BEFORE.zip::HUMAN']
    out = []
    for spec in pats:
        pat, _, tag = spec.partition('::')
        vals = []
        for z in sorted(glob.glob(pat)):
            got = load(z)
            if got:
                vals.append((pathlib.Path(z).stem, pendulum_share(got[0])))
        out.append((tag or pat, vals))

    for tag, vals in out:
        if not vals:
            continue
        print(f"{tag}:")
        for s, v in vals:
            print(f"   {s:22s} {v:.3f}")
        print(f"   {'MEDIAN':22s} {st.median(v for _, v in vals):.3f}\n")


if __name__ == '__main__':
    main()
