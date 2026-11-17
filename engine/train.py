"""Train the stateful GRU with truncated BPTT over FULL sequences.

Each batch = B whole sequences (20,000 steps). They are processed in chunks of
--chunk steps; the hidden state is carried between chunks (detached), exactly like
row-by-row inference on the platform, so there is no train/inference mismatch.

Loss (rows with step >= 99 only):
    (1 - weighted Pearson)  averaged over t0, t1   — the competition metric on the chunk,
                                                     weights |clip(y)| as in METRIC.md
  + mse_weight * weighted MSE(pred, clip(y))       — keeps the output scale sane
Optional --event-weight a multiplies row weights by (1 + a * [i0 trade on this row]),
because scored rows are ~2x more likely on i0 trades (day-1 EDA).

After every epoch: full validation on --valid-dir with the official mask (WP on mask,
WP on all required rows), plus the best output scale (predictions are clipped to
+-2 by the metric, so the scale matters). Checkpoints go to --out (use Google Drive).

    python train.py --train-dir /content/data/train --valid-dir /content/data/valid \
        --out /content/drive/MyDrive/wunder/runs/gru128x2 --epochs 6

Training data can also be read straight from the competition parquet (all 10,607 train
sequences, no conversion step and no extra disk): --train-parquet .../datasets/train.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from common import SEQ_LEN, WARMUP
from prepare_data import read_sequence
from metric import WPAccumulator
from model import GRUNet, BaselineGRU, build_model
from mamba import MambaNet


def detach_state(h):
    """Detach a recurrent state of any shape: tensor (GRU) or nested lists/tuples (Mamba)."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    return type(h)(detach_state(x) for x in h)

TRADE_I0 = slice(48, 52)   # i0_dv0..3


# ----------------------------------------------------------------------------- loss
def weighted_pearson_loss(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """pred, y, w: (N, 2). Returns 1 - mean_k WP_k (differentiable)."""
    sw = w.sum(0).clamp_min(1e-8)
    my = (w * y).sum(0) / sw
    mp = (w * pred).sum(0) / sw
    yc, pc = y - my, pred - mp
    cov = (w * yc * pc).sum(0) / sw
    vy = (w * yc * yc).sum(0) / sw
    vp = (w * pc * pc).sum(0) / sw
    corr = cov / torch.sqrt(vy * vp + 1e-12)
    return 1.0 - corr.mean()


def chunk_loss(pred, y, step0, x, args):
    """pred, y: (B, C, 2); x: (B, C, 112) raw; step0 = first step index of the chunk."""
    C = pred.shape[1]
    steps = torch.arange(step0, step0 + C, device=pred.device)
    keep = steps >= WARMUP
    if not keep.any():
        return None
    pred, y = pred[:, keep].float(), y[:, keep]
    yc = y.clamp(-2, 2)
    w = yc.abs()
    if args.event_weight > 0:
        prev = torch.cat([x[:, :1], x[:, :-1]], 1)
        trade = (x[..., TRADE_I0] != prev[..., TRADE_I0]).any(-1)[:, keep].float()
        w = w * (1 + args.event_weight * trade)[..., None]
    pred, yc, w = pred.reshape(-1, 2), yc.reshape(-1, 2), w.reshape(-1, 2)
    loss = weighted_pearson_loss(pred, yc, w)
    if args.mse_weight > 0:
        loss = loss + args.mse_weight * ((w * (pred - yc) ** 2).sum() / w.sum().clamp_min(1e-8))
    return loss


# ----------------------------------------------------------------------------- data
class NpySeqs:
    """Sequences from prepare_data.py output (memory-mapped .npy)."""

    def __init__(self, folder, max_seqs=None):
        self.X = np.load(Path(folder) / "features.npy", mmap_mode="r")
        self.Y = np.load(Path(folder) / "targets.npy", mmap_mode="r")
        self.n = self.X.shape[0] if max_seqs is None else min(max_seqs, self.X.shape[0])
        self.name = f"{folder} (npy, {self.X.dtype})"

    def fetch(self, idx):
        idx = np.sort(np.asarray(idx))
        return np.asarray(self.X[idx], np.float32), np.asarray(self.Y[idx], np.float32)


class ParquetSeqs:
    """Whole sequences read straight from a competition parquet (one row group = one sequence).

    Each sequence is validated like prepare_data.py (20,000 rows, one seq_ix, steps in order).
    Several threads decode row groups in parallel (pyarrow releases the GIL); every thread
    has its own ParquetFile handles.
    extra: a second parquet whose sequences are added (e.g. valid.parquet), minus the sequences
    whose seq_ix appears in exclude_from (e.g. valid_sub.parquet, kept for validation)."""

    def __init__(self, path, max_seqs=None, workers=2, extra=None, exclude_from=None):
        path = str(path)
        self.items = [(path, rg) for rg in range(pq.ParquetFile(path).num_row_groups)]
        if max_seqs is not None:
            self.items = self.items[:max_seqs]
        desc = path
        if extra:
            skip = set(seq_ids(exclude_from)) if exclude_from else set()
            ids = seq_ids(extra)
            add = [(str(extra), rg) for rg, sid in enumerate(ids) if sid not in skip]
            self.items += add
            desc += f" + {len(add)} of {len(ids)} sequences of {extra}"
            if exclude_from:
                desc += f" (without the {len(ids) - len(add)} that are in {Path(exclude_from).name})"
        self.n = len(self.items)
        self.name = f"{desc} (parquet, read on the fly by {workers} threads)"
        self._local = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=workers)

    def _read(self, item):
        path, rg = item
        pfs = getattr(self._local, "pfs", None)
        if pfs is None:
            pfs = self._local.pfs = {}
        if path not in pfs:
            pfs[path] = pq.ParquetFile(path)
        _, x, y, _ = read_sequence(pfs[path], int(rg), with_scored=False)
        return x, y

    def fetch(self, idx):
        parts = list(self._pool.map(self._read, [self.items[i] for i in np.sort(np.asarray(idx))]))
        return np.stack([p[0] for p in parts]), np.stack([p[1] for p in parts])


def seq_ids(path):
    """seq_ix of every row group (= sequence) of a competition parquet, in file order."""
    pf = pq.ParquetFile(str(path))
    return [int(pf.read_row_group(rg, columns=["seq_ix"]).column(0)[0].as_py()) for rg in range(pf.num_row_groups)]


def load_init(model, spec):
    """Initialise from our checkpoint(s): PATH[+PATH...]; several are weight-averaged (SWA)."""
    sts = [torch.load(p, map_location="cpu", weights_only=False)["model"] for p in spec.split("+")]
    avg = {k: sum(s[k].float() for s in sts) / len(sts) for k in sts[0]}
    model.load_state_dict(avg)


def feature_stats(src, n_seqs=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(src.n, min(n_seqs, src.n), replace=False))
    full, _ = src.fetch(idx)
    X = full[:, ::3]                                         # thin rows for RAM
    D = np.diff(full[:10], axis=1, prepend=full[:10, :1])
    mu = np.concatenate([X.reshape(-1, 112).mean(0), D.reshape(-1, 112).mean(0)])
    sd = np.concatenate([X.reshape(-1, 112).std(0), D.reshape(-1, 112).std(0)])
    return mu.astype(np.float32), np.maximum(sd, 1e-6).astype(np.float32)


class Prefetcher:
    """Fetches the next batches of whole sequences in a background thread; .wait = seconds the
    training loop spent waiting for data (if large, the GPU is starved by I/O)."""

    def __init__(self, fetch, batches, depth=2):
        self.q = queue.Queue(maxsize=depth)
        self.wait = 0.0
        self.t = threading.Thread(target=self._run, args=(fetch, batches), daemon=True)
        self.t.start()

    def _run(self, fetch, batches):
        try:
            for idx in batches:
                self.q.put(fetch(idx))
        except BaseException as e:      # noqa: BLE001  surface reader errors in the main thread
            self.q.put(e)
            return
        self.q.put(None)

    def __iter__(self):
        while True:
            t = time.perf_counter()
            item = self.q.get()
            self.wait += time.perf_counter() - t
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def predict_sequences(model, Xm, idx, device, batch=16, chunk=5000):
    model.eval()
    out = np.zeros((len(idx), SEQ_LEN, 2), np.float32)
    for b in range(0, len(idx), batch):
        x = torch.from_numpy(np.asarray(Xm[idx[b:b + batch]], np.float32)).to(device)
        h, prev, parts = None, None, []
        for s in range(0, SEQ_LEN, chunk):
            xc = x[:, s:s + chunk]
            p, h = model(xc, h, prev)
            prev = xc[:, -1]
            parts.append(p.float())
        out[b:b + batch] = torch.cat(parts, 1).cpu().numpy()
    model.train()
    return out


def evaluate(model, Xv, Yv, Mv, device, n_seqs=None):
    n = Xv.shape[0] if n_seqs is None else min(n_seqs, Xv.shape[0])
    idx = np.arange(n)
    P = predict_sequences(model, Xv, idx, device)
    need = np.arange(SEQ_LEN) >= WARMUP
    Y = np.asarray(Yv[:n])
    M = np.asarray(Mv[:n]) & need if Mv is not None else None
    res = {"seqs": n}
    sd = P[:, need].reshape(-1, 2).std(0) + 1e-12
    res["pred_std"] = sd.round(4).tolist()
    best = (-1.0, 1.0)
    # scale predictions so that their std equals c (per target), pick c on the mask
    for c in (0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5):
        acc = WPAccumulator(2)
        acc.update(Y, P * (c / sd), M) if M is not None else acc.update(Y[:, need], (P * (c / sd))[:, need])
        res[f"WP_mask_std{c}"] = round(acc.score(), 5)
        if acc.score() > best[0]:
            best = (acc.score(), c)
    acc_all = WPAccumulator(2)
    acc_all.update(Y[:, need], (P * (best[1] / sd))[:, need])
    acc_raw = WPAccumulator(2)
    acc_raw.update(Y, P, M) if M is not None else acc_raw.update(Y[:, need], P[:, need])
    res.update(WP_mask_raw=round(acc_raw.score(), 5), WP_mask_best=round(best[0], 5),
               best_target_std=best[1], output_scale=(best[1] / sd).round(5).tolist(),
               WP_all_at_best=round(acc_all.score(), 5))
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=None, help="prepare_data.py output folder")
    ap.add_argument("--train-parquet", default=None, help="or: train.parquet itself (read on the fly)")
    ap.add_argument("--extra-parquet", default=None,
                    help="with --train-parquet: also train on the sequences of this parquet (e.g. valid.parquet) ...")
    ap.add_argument("--exclude-seqs-from", default=None,
                    help="... except those whose seq_ix is in this parquet (e.g. valid_sub.parquet, used for validation)")
    ap.add_argument("--init-from", default=None,
                    help="start from our checkpoint(s) PATH[+PATH...] (averaged); the architecture args must match")
    ap.add_argument("--loader-workers", type=int, default=2, help="threads decoding parquet row groups")
    ap.add_argument("--keep-epochs", action="store_true", help="also save epoch_<k>.pt after every epoch")
    ap.add_argument("--valid-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--proj", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="gru and baseline: dropout (training only) on the GRU input, between the GRU layers, on the head input")
    ap.add_argument("--input-noise", type=float, default=0.0,
                    help="gru and baseline: Gaussian noise on the inputs (training only; gru: standardized, baseline: raw)")
    ap.add_argument("--mse-weight", type=float, default=0.1)
    ap.add_argument("--event-weight", type=float, default=0.0)
    ap.add_argument("--max-train-seqs", type=int, default=None)
    ap.add_argument("--valid-seqs", type=int, default=None, help="limit validation size for speed")
    ap.add_argument("--arch", choices=["gru", "baseline", "mamba"], default="gru",
                    help="gru = our GRU (raw+deltas, proj, MLP head); baseline = organizers' architecture; "
                         "mamba = Mamba-2 SSM (raw+deltas)")
    ap.add_argument("--d-model", type=int, default=64, help="mamba: residual width")
    ap.add_argument("--n-layers", type=int, default=2, help="mamba: number of Mamba-2 layers")
    ap.add_argument("--d-state", type=int, default=16, help="mamba: SSM state size N")
    ap.add_argument("--expand", type=int, default=2, help="mamba: d_inner = expand * d_model")
    ap.add_argument("--headdim", type=int, default=16, help="mamba: head size P (heads = d_inner / P)")
    ap.add_argument("--d-conv", type=int, default=4, help="mamba: causal conv width")
    ap.add_argument("--block-len", type=int, default=50, help="mamba: scan block length")
    ap.add_argument("--out-norm", action="store_true", help="mamba: RMSNorm before out_proj (Mamba-2), ~10%% slower")
    ap.add_argument("--init-onnx", default=None, help="baseline.onnx to start from (requires --arch baseline)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not args.no_amp
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} amp={use_amp}", torch.cuda.get_device_name(0) if device == "cuda" else "")

    if bool(args.train_dir) == bool(args.train_parquet):
        raise SystemExit("give exactly one of --train-dir / --train-parquet")
    if args.extra_parquet and not args.train_parquet:
        raise SystemExit("--extra-parquet needs --train-parquet")
    if args.init_from and args.init_onnx:
        raise SystemExit("give at most one of --init-from / --init-onnx")
    train_src = (ParquetSeqs(args.train_parquet, args.max_train_seqs, args.loader_workers, args.extra_parquet,
                             args.exclude_seqs_from) if args.train_parquet
                 else NpySeqs(args.train_dir, args.max_train_seqs))
    Xv = np.load(Path(args.valid_dir) / "features.npy", mmap_mode="r")
    Yv = np.load(Path(args.valid_dir) / "targets.npy", mmap_mode="r")
    mp = Path(args.valid_dir) / "is_scored.npy"
    Mv = np.load(mp, mmap_mode="r") if mp.exists() else None
    n_train = train_src.n
    print(f"train seqs {n_train} from {train_src.name}; valid seqs {Xv.shape[0]}")

    if args.arch == "baseline":
        model = BaselineGRU(args.hidden, args.dropout, args.input_noise)
        if args.init_onnx:
            model.load_onnx(args.init_onnx)
            print(f"initialised from {args.init_onnx}")
        model = model.to(device)
    elif args.arch == "mamba":
        if args.init_onnx:
            raise SystemExit("--init-onnx needs --arch baseline")
        model = MambaNet(args.d_model, args.n_layers, args.d_state, args.expand, args.headdim,
                         args.d_conv, args.head, args.block_len, args.out_norm).to(device)
    else:
        if args.init_onnx:
            raise SystemExit("--init-onnx needs --arch baseline")
        model = GRUNet(args.proj, args.hidden, args.layers, args.head, args.dropout, args.input_noise).to(device)
    groups = model.param_groups(args.wd) if hasattr(model, "param_groups") else model.parameters()
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(n_train / args.batch) * math.ceil(SEQ_LEN / args.chunk)
    total = steps_per_epoch * args.epochs
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_score, history = 0, -1.0, []

    ck = out / "last.pt"
    if args.resume and ck.exists():
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch, best_score, history = st["epoch"] + 1, st["best_score"], st["history"]
        print(f"resumed from epoch {st['epoch']}, best {best_score:.5f}")
    elif args.init_from:
        load_init(model, args.init_from)                    # includes mu/sd of our GRU: keep them
        print(f"initialised from {args.init_from}")
    elif hasattr(model, "mu"):
        mu, sd = feature_stats(train_src)
        model.mu.copy_(torch.from_numpy(mu))
        model.sd.copy_(torch.from_numpy(sd))
    if (args.init_onnx or args.init_from) and start_epoch == 0:
        ev0 = evaluate(model, Xv, Yv, Mv, device, args.valid_seqs)
        ev0.update(epoch=-1, train_minutes=0.0)
        history.append(ev0)
        best_score = ev0["WP_mask_best"]
        print("VALID (start = initial weights)", json.dumps(ev0), flush=True)
        # keep the starting point as best.pt, so fine-tuning can only improve on it
        torch.save(dict(model=model.state_dict(), cfg=model.cfg, epoch=-1, best_score=best_score,
                        history=history, output_scale=ev0["output_scale"], args=vars(args)), out / "best.pt")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    for epoch in range(start_epoch, args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        perm = rng.permutation(n_train)
        batches = [perm[i:i + args.batch] for i in range(0, n_train, args.batch)]
        t0, rows, run_loss, n_loss = time.time(), 0, 0.0, 0
        loader = Prefetcher(train_src.fetch, batches)
        for bi, (xb, yb) in enumerate(loader):
            x = torch.from_numpy(xb).to(device, non_blocking=True)
            y = torch.from_numpy(yb).to(device, non_blocking=True)
            h, prev = None, None
            for s in range(0, SEQ_LEN, args.chunk):
                xc, yc = x[:, s:s + args.chunk], y[:, s:s + args.chunk]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    pred, h = model(xc, h, prev)
                loss = chunk_loss(pred, yc, s, xc, args)
                h, prev = detach_state(h), xc[:, -1]
                if loss is None:
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                run_loss += loss.item()
                n_loss += 1
            rows += xb.shape[0] * SEQ_LEN
            if (bi + 1) % args.log_every == 0 or bi + 1 == len(batches):
                el = time.time() - t0
                print(f"ep {epoch} batch {bi + 1}/{len(batches)} loss {run_loss / max(n_loss, 1):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {rows / el / 1e6:.2f}M rows/s "
                      f"data-wait {loader.wait / el:.0%} "
                      f"eta {el / (bi + 1) * (len(batches) - bi - 1) / 60:.1f} min", flush=True)
                run_loss, n_loss = 0.0, 0

        ev = evaluate(model, Xv, Yv, Mv, device, args.valid_seqs)
        ev.update(epoch=epoch, train_minutes=round((time.time() - t0) / 60, 1))
        history.append(ev)
        print("VALID", json.dumps(ev), flush=True)
        state = dict(model=model.state_dict(), cfg=model.cfg, opt=opt.state_dict(), sched=sched.state_dict(),
                     epoch=epoch, best_score=max(best_score, ev["WP_mask_best"]), history=history,
                     output_scale=ev["output_scale"], args=vars(args))
        torch.save(state, out / "last.pt")
        if args.keep_epochs:
            torch.save(state, out / f"epoch_{epoch}.pt")
        if ev["WP_mask_best"] > best_score:
            best_score = ev["WP_mask_best"]
            torch.save(state, out / "best.pt")
            print(f"  new best {best_score:.5f} -> {out / 'best.pt'}")
        (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"done. best WP(mask) = {best_score:.5f}")


if __name__ == "__main__":
    main()
