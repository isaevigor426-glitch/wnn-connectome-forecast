"""Training of one recurrent model (PyTorch).

Loss |y| * (p - y)^2 with targets clipped to [-2, 2]: its minimiser is E[|y| y | x] / E[|y| | x],
the optimum of the |y|-weighted Pearson correlation. Sequences of 20 000 steps are cut into chunks
of 1000 with the hidden state carried over (truncated BPTT); warm-up steps are excluded from the loss.

Example:
    python -m wnn.train --data data --cell lstm --active 5000 --seed 0 --out lstm_s0.pt
"""
import argparse
import time

import numpy as np
import torch
from torch import nn

from .data import family_weights, select_active
from .features import event_rate
from .metric import VALID_MIX, WARMUP, family_of
from .models import RecurrentNet, detach_state


class Concat:
    """Indexes several (n_i, steps, features) arrays as one; returns float32."""

    def __init__(self, parts):
        self.parts, self.off = parts, np.cumsum([0] + [len(p) for p in parts])

    def __len__(self):
        return int(self.off[-1])

    def __getitem__(self, idx):
        idx = np.sort(np.atleast_1d(idx)); out = []
        for k, p in enumerate(self.parts):
            m = (idx >= self.off[k]) & (idx < self.off[k + 1])
            if m.any():
                out.append(np.asarray(p[idx[m] - self.off[k]], np.float32))
        return np.concatenate(out)


def train(X, Y, weights, cell="gru", hidden=128, epochs=5, lr=2e-3, batch=32, chunk=1000, seed=0, device="cuda", log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = RecurrentNet(cell, X[np.array([0])].shape[-1], hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n, steps = len(X), Y.shape[1]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (n // batch) * (steps // chunk), pct_start=0.05)
    w_all = torch.from_numpy(np.asarray(weights, np.float32))
    for ep in range(epochs):
        net.train(); t0 = time.time(); order = rng.permutation(n); losses = []
        for b in range(n // batch):
            idx = np.sort(order[b * batch:(b + 1) * batch])
            x = torch.from_numpy(X[idx]).to(device)
            y = torch.from_numpy(np.asarray(Y[idx], np.float32)).to(device).clamp(-2, 2)
            ws = w_all[idx].to(device)[:, None, None]
            h = None
            for t in range(0, steps, chunk):
                o, h = net(x[:, t:t + chunk], h); h = detach_state(h)
                yy = y[:, t:t + chunk]
                loss = ws * yy.abs() * (o - yy) ** 2
                if t < WARMUP:
                    loss = loss[:, WARMUP - t:]
                loss = loss.mean()
                if not torch.isfinite(loss):          # skip a rare exploding chunk instead of corrupting the weights
                    opt.zero_grad(set_to_none=True); sched.step(); h = None; continue
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); losses.append(loss.item())
        log(f"epoch {ep + 1}/{epochs}  loss {np.mean(losses):.4f}  {time.time() - t0:.0f}s")
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with train_X.npy/train_Y.npy and validrest_X.npy/validrest_Y.npy")
    ap.add_argument("--cell", choices=["gru", "lstm"], default="lstm")
    ap.add_argument("--active", type=int, default=5000, help="use the N most active train sequences (0 = all train)")
    ap.add_argument("--no-valid", action="store_true", help="do not add the 1573 valid sequences to training")
    ap.add_argument("--no-family-weights", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    XT, YT = np.load(f"{a.data}/train_X.npy", mmap_mode="r"), np.load(f"{a.data}/train_Y.npy", mmap_mode="r")
    famT = family_of(YT)
    ids = np.arange(len(XT))
    if a.active:
        rate = np.array([event_rate(XT[s]) for s in range(len(XT))])
        ids = select_active(rate, famT, a.active)
    X = Concat([_Subset(XT, ids)] + ([] if a.no_valid else [np.load(f"{a.data}/validrest_X.npy", mmap_mode="r")]))
    Y = np.concatenate([np.asarray(YT[ids])] + ([] if a.no_valid else [np.load(f"{a.data}/validrest_Y.npy")]))
    fam = family_of(Y)
    w = np.ones(len(Y), np.float32) if a.no_family_weights else family_weights(fam, VALID_MIX)
    net = train(X, Y, w, cell=a.cell, epochs=a.epochs, seed=a.seed, device="cuda" if torch.cuda.is_available() else "cpu")
    torch.save(net.state_dict(), a.out)


class _Subset:
    def __init__(self, X, ids):
        self.X, self.ids = X, np.asarray(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.X[self.ids[np.asarray(i)]]


if __name__ == "__main__":
    main()
