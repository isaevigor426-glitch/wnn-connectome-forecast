"""Stateful GRU, pure-numpy inference (no torch / onnxruntime needed at scoring time).

Weights are read from weights.npz next to this file (written by export.py).
Per row: raw 112 features + delta vs previous row -> standardize/clip -> Linear+SiLU
-> stacked GRU cells (PyTorch gate order r, z, n) -> MLP head -> * output_scale.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CLIP_IN = 10.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


class PredictionModel:
    def __init__(self):
        w = np.load(Path(__file__).resolve().parent / "weights.npz")
        f32 = lambda k: np.ascontiguousarray(w[k], dtype=np.float32)  # noqa: E731
        self.mu, self.inv_sd = f32("mu"), (1.0 / f32("sd")).astype(np.float32)
        self.W_in, self.b_in = f32("inp.weight").T.copy(), f32("inp.bias")
        self.L = int(w["layers"])
        self.H = int(w["hidden"])
        self.W_ih = [f32(f"gru.weight_ih_l{i}").T.copy() for i in range(self.L)]
        self.W_hh = [f32(f"gru.weight_hh_l{i}").T.copy() for i in range(self.L)]
        self.b_ih = [f32(f"gru.bias_ih_l{i}") for i in range(self.L)]
        self.b_hh = [f32(f"gru.bias_hh_l{i}") for i in range(self.L)]
        self.W1, self.b1 = f32("head1.weight").T.copy(), f32("head1.bias")
        self.W2, self.b2 = f32("head2.weight").T.copy(), f32("head2.bias")
        self.scale = f32("output_scale")
        self.seq_ix = None
        self.prev = None
        self.h = None

    def _reset(self):
        self.h = [np.zeros(self.H, np.float32) for _ in range(self.L)]
        self.prev = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self._reset()
        s = np.asarray(data_point.state, dtype=np.float32)
        d = s - self.prev if self.prev is not None else np.zeros_like(s)
        self.prev = s
        f = np.concatenate([s, d])
        f = np.clip((f - self.mu) * self.inv_sd, -CLIP_IN, CLIP_IN)
        z = _silu(f @ self.W_in + self.b_in)
        x = z
        H = self.H
        for i in range(self.L):
            h = self.h[i]
            gi = x @ self.W_ih[i] + self.b_ih[i]
            gh = h @ self.W_hh[i] + self.b_hh[i]
            rz = _sigmoid(gi[:2 * H] + gh[:2 * H])
            n = np.tanh(gi[2 * H:] + rz[:H] * gh[2 * H:])
            h = (1.0 - rz[H:]) * n + rz[H:] * h
            self.h[i] = h
            x = h
        o = _silu(np.concatenate([x, z]) @ self.W1 + self.b1) @ self.W2 + self.b2
        if not data_point.need_prediction:
            return None
        out = (o * self.scale).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=2.0, neginf=-2.0)
