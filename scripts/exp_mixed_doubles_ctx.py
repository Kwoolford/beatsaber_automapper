import sys, pathlib, collections, random, numpy as np
sys.path.insert(0,"/home/kyle/repos/beatsaber_automapper"); sys.path.insert(0,".")
from agent_mapper.score import load_map
from exp_mixed_doubles import kind
from concurrent.futures import ProcessPoolExecutor
def ctx(p):
    try: m=load_map(p)
    except Exception: return None
    ns=m.notes
    if len(ns)<100 or any(not(0<=n.x<=3 and 0<=n.y<=2 and 0<=n.direction<=8) for n in ns): return None
    at=collections.defaultdict(lambda:{0:[],1:[]})
    for n in ns: at[round(n.beat*48)][n.color].append(n)
    ts=sorted(at); out=collections.Counter()
    last={0:None,1:None}
    for t in ts:
        h=at[t]
        if len(h[0])==1 and len(h[1])==1:
            a,b=kind(h[0][0].direction),kind(h[1][0].direction)
            if "o" not in (a,b):
                lab="mixed" if a!=b else a
                ph="beat" if t%48==0 else "8th" if t%24==0 else "other"
                # min gap since each hand's previous note (beats)
                g=min((t-last[c])/48 if last[c] is not None else 9 for c in (0,1))
                gb="fast<.5" if g<.5 else "mid<1" if g<1 else "slow"
                # crossed hands: red x > blue x
                hz="cross" if h[0][0].x>h[1][0].x else "open"
                out[(lab,ph)]+=1; out[(lab,gb)]+=1; out[(lab,hz)]+=1
                # previous same-hand direction matches => would be reset for that hand
        for c in (0,1):
            if h[c]: last[c]=t
    return out
if __name__=="__main__":
    z=sorted(pathlib.Path("/home/kyle/repos/beatsaber_automapper/data/raw").glob("*.zip"))
    random.seed(0); z=random.sample(z,1500)
    tot=collections.Counter()
    with ProcessPoolExecutor(16) as ex:
        for r in ex.map(ctx,z,chunksize=16):
            if r: tot.update(r)
    for key in ("beat","8th","other","fast<.5","mid<1","slow","cross","open"):
        n=sum(tot[(l,key)] for l in("mixed","u","d"))
        print(f"{key:8} n={n:7d} mixed {tot[('mixed',key)]/n:.3f} up {tot[('u',key)]/n:.3f} down {tot[('d',key)]/n:.3f}")
    import pickle; pickle.dump(tot,open("ctx.pkl","wb"))
