#!/usr/bin/env python3
"""2b — LEARNED feel discriminator: human vs OUR-V7 (2026-06-11).

The 06-10 gate killed the handcrafted-feature reward: AUC(human vs V7)=0.31 — the
11 hand-pooled scalars can't separate human from our generator (V7 is over-regular,
so it aces the "non-random?" test the corrupt-trained classifier learned). Per the
build plan, escalate to a LEARNED map encoder whose NEGATIVE class is our own output.

This is the v0: a small transformer over the raw note SEQUENCE (per-note dt, x, y,
dir), binary head human(1) vs V7(0). Both classes reach the model through the SAME
loaders reward_gate_poc uses — human from cached .pt (decode_events), V7 from .zip
(_load_notes_with_direction) — reduced to (beat,x,y,dir). No audio in v0; if a
token-only encoder already separates >=0.75 we don't need MERT.

Split is GROUPED BY SONG (a song's human & V7 map never straddle train/val), so the
reported held-out AUC measures human-vs-V7 discrimination, not song memorization.

DoD: held-out AUC(human vs V7) >= 0.75  ->  the learned reward direction is alive
(then: best-of-N rerank). < 0.75  ->  even a learned note-seq encoder can't see the
gap -> it's perceptual/audio-relative; add MERT conditioning or rethink.
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from beatsaber_automapper.data.swing_tokenizer import SwingEventTokenizer

MAXLEN = 512  # notes per map (truncate); maps rarely exceed this at Expert


def _seq_from_notes(notes, bpm):
    """notes: list of (beat,x,y,dir) -> [L, 12] per-note features (sorted by time)."""
    rows = sorted(notes, key=lambda r: r[0])
    spb = 60.0 / bpm
    prev_t = None
    feats = []
    for (b, x, y, d) in rows[:MAXLEN]:
        t = b * spb
        dt = 0.0 if prev_t is None else min(2.0, t - prev_t)
        prev_t = t
        x = int(np.clip(x, 0, 3)); y = int(np.clip(y, 0, 2)); d = int(np.clip(d, 0, 8))
        onehot = np.zeros(9, dtype=np.float32); onehot[d] = 1.0
        feats.append(np.concatenate([[dt, x / 3.0, y / 2.0], onehot]).astype(np.float32))
    if len(feats) < 8:
        return None
    return np.stack(feats)  # [L,12]


def load_human(pt_path, difficulty, tok):
    try:
        dd = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    diffs = dd.get("difficulties", {})
    if difficulty not in diffs:
        return None
    st = diffs[difficulty].get("swing_tokens")
    if not st:
        return None
    try:
        events = tok.decode_events(list(st))
    except Exception:
        return None
    notes = [(e.beat, e.x, e.y, e.direction) for e in events
             if 0 <= int(getattr(e, "direction", -1)) <= 8]
    bpm = float(dd.get("bpm", 120.0))
    return _seq_from_notes(notes, bpm)


def _zip_bpm(zp):
    import json, zipfile
    try:
        with zipfile.ZipFile(zp) as zf:
            nm = next((n for n in zf.namelist()
                       if pathlib.PurePosixPath(n).name.lower() == "info.dat"), None)
            if nm:
                return float(json.loads(zf.read(nm).decode("utf-8", "ignore"))
                             .get("_beatsPerMinute", 0)) or None
    except Exception:
        return None
    return None


def load_v7(zip_path, difficulty):
    from eval_contour_follow import _load_notes_with_direction
    try:
        recs = _load_notes_with_direction(pathlib.Path(zip_path), difficulty)
    except Exception:
        return None
    notes = [(b, x, y, dr) for (b, x, y, _c, dr) in recs]
    bpm = _zip_bpm(zip_path) or 120.0
    return _seq_from_notes(notes, bpm)


class NoteSeqDisc(nn.Module):
    def __init__(self, d=128, nhead=4, layers=2, ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(12, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self.register_buffer("pos", self._posenc(MAXLEN + 1, d), persistent=False)

    @staticmethod
    def _posenc(L, d):
        pe = torch.zeros(L, d)
        pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x, mask):
        # x [B,L,12], mask [B,L] True=pad
        B = x.shape[0]
        h = self.proj(x)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos[:, : h.shape[1] + 1]
        cmask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), mask], dim=1)
        h = self.enc(h, src_key_padding_mask=cmask)
        return self.head(h[:, 0])  # [B,2]


def collate(batch, device):
    seqs, labels, _ = zip(*batch)
    L = max(s.shape[0] for s in seqs)
    X = torch.zeros(len(seqs), L, 12)
    M = torch.ones(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        X[i, : s.shape[0]] = torch.from_numpy(s)
        M[i, : s.shape[0]] = False
    return X.to(device), M.to(device), torch.tensor(labels).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-glob", default="outputs/v7_cohort_2026-06-10/*.zip")
    ap.add_argument("--pt-glob", default="data/processed/*.pt")
    ap.add_argument("--difficulty", default="Expert")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ablate", choices=["none", "dt", "spatial", "dir"], default="none",
                    help="zero a feature group to see what drives separation: "
                         "dt=timing(col0), spatial=x,y(col1-2), dir=onehot(col3-11)")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--save-ckpt", type=pathlib.Path, default=None,
                    help="save final model state_dict here")
    ap.add_argument("--dump-scores", type=pathlib.Path, default=None,
                    help="write per-map {song_id, label, split, logit, p_human} JSON — "
                         "for checking whether the reward gives a usable RANKING "
                         "within the V7 cohort (saturation check)")
    args = ap.parse_args()

    def _ablate(s):
        if args.ablate == "dt":
            s = s.copy(); s[:, 0] = 0.0
        elif args.ablate == "spatial":
            s = s.copy(); s[:, 1:3] = 0.0
        elif args.ablate == "dir":
            s = s.copy(); s[:, 3:12] = 0.0
        return s

    rng = np.random.default_rng(args.seed)
    tok = SwingEventTokenizer()

    # negatives: V7 cohort (keyed by song id = zip stem)
    samples = []  # (seq, label, song_id)
    v7_ids = []
    for zp in sorted(glob.glob(args.v7_glob)):
        sid = pathlib.Path(zp).stem
        s = load_v7(zp, args.difficulty)
        if s is not None:
            samples.append((s, 0, sid)); v7_ids.append(sid)
    n_v7 = len(v7_ids)
    print(f"[data] V7 negatives: {n_v7}")
    if n_v7 < 10:
        print("!! too few V7 maps; aborting"); sys.exit(2)

    # positives: prefer the SAME song ids (matched human), then top up with others to balance
    v7_set = set(v7_ids)
    pt_by_id = {pathlib.Path(p).stem: p for p in glob.glob(args.pt_glob)}
    n_pos = 0
    matched = 0
    for sid in v7_ids:
        if sid in pt_by_id:
            s = load_human(pt_by_id[sid], args.difficulty, tok)
            if s is not None:
                samples.append((s, 1, sid)); n_pos += 1; matched += 1
    others = [pid for sid, pid in pt_by_id.items() if sid not in v7_set]
    rng.shuffle(others)
    for pid in others:
        if n_pos >= n_v7:
            break
        s = load_human(pid, args.difficulty, tok)
        if s is not None:
            samples.append((s, 1, pathlib.Path(pid).stem)); n_pos += 1
    print(f"[data] human positives: {n_pos} (matched-by-song {matched})")

    if args.ablate != "none":
        samples = [(_ablate(s), lab, sid) for (s, lab, sid) in samples]
        print(f"[ablate] zeroed feature group: {args.ablate}")

    # GROUP split by song id
    ids = sorted({sid for _, _, sid in samples})
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_frac))
    val_ids = set(ids[:n_val])
    train = [s for s in samples if s[2] not in val_ids]
    val = [s for s in samples if s[2] in val_ids]
    print(f"[split] train={len(train)} val={len(val)} "
          f"(val pos={sum(l for _,l,_ in val)} neg={sum(1-l for _,l,_ in val)})")

    dev = args.device
    model = NoteSeqDisc().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    from sklearn.metrics import roc_auc_score

    def epoch(data, training):
        model.train(training)
        order = list(range(len(data)))
        if training:
            rng.shuffle(order)
        bs = 32
        tot, probs, ys = 0.0, [], []
        for i in range(0, len(order), bs):
            batch = [data[j] for j in order[i:i + bs]]
            X, M, y = collate(batch, dev)
            with torch.set_grad_enabled(training):
                logits = model(X, M)
                loss = lossf(logits, y)
                if training:
                    opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(batch)
            probs.extend(torch.softmax(logits, -1)[:, 1].detach().cpu().numpy())
            ys.extend(y.cpu().numpy())
        auc = roc_auc_score(ys, probs) if len(set(ys)) > 1 else float("nan")
        return tot / max(1, len(data)), auc

    best_val_auc = 0.0
    for ep in range(args.epochs):
        tl, ta = epoch(train, True)
        vl, va = epoch(val, False)
        best_val_auc = max(best_val_auc, va)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:02d} train_loss={tl:.3f} train_auc={ta:.3f} "
                  f"val_loss={vl:.3f} val_auc={va:.3f}")

    print(f"\n[DoD] best held-out AUC(human vs V7) = {best_val_auc:.4f}   (>=0.75 PASS)")
    verdict = ("GREEN: a LEARNED note-seq encoder SEPARATES human from our V7 (AUC>=0.75) where "
               "hand-features failed (0.31). The reward direction is alive → use it: best-of-N "
               "rerank at inference, then reward-weighted fine-tune." if best_val_auc >= 0.75 else
               "RED/AMBER: even a learned note-seq encoder can't cleanly separate human from V7 "
               "(<0.75). The gap is not in the note token stream alone → add pooled-MERT audio "
               "conditioning (human-vs-V7 given the song) or reconsider the objective.")
    print("\n=== VERDICT ===\n" + verdict)

    if args.json:
        import json as _json
        args.json.write_text(_json.dumps({
            "n_v7": n_v7, "n_pos": n_pos, "matched": matched,
            "train": len(train), "val": len(val),
            "best_val_auc": best_val_auc, "verdict": verdict}, indent=2))
        print(f"\nwrote {args.json}")

    if args.save_ckpt:
        torch.save(model.state_dict(), args.save_ckpt)
        print(f"wrote {args.save_ckpt}")

    if args.dump_scores:
        import json as _json
        model.eval()
        rows = []
        bs = 32
        with torch.no_grad():
            for i in range(0, len(samples), bs):
                batch = samples[i:i + bs]
                X, M, _ = collate(batch, dev)
                logits = model(X, M)
                # margin logit (human - gen) is rank-stable even when softmax saturates
                margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
                p = torch.softmax(logits, -1)[:, 1].cpu().numpy()
                for (s, lab, sid), lg, ph in zip(batch, margin, p):
                    rows.append({"song_id": sid, "label": int(lab),
                                 "split": "val" if sid in val_ids else "train",
                                 "logit": float(lg), "p_human": float(ph)})
        args.dump_scores.write_text(_json.dumps(rows, indent=1))
        print(f"wrote {args.dump_scores} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
