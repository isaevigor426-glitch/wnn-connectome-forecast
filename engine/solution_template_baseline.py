"""Teammate's 2-layer GRU (baseline architecture: GRU 112->128, GRU 128->128, Linear 128->2),
pure-numpy stateful inference.

Input: the 112 raw features in the official order (no extra features).
Output: head(h2) * target_std + target_mean  (de-normalized), then * output_scale.
weights.npz holds the weights (PyTorch gate order r, z, n); config.json holds output_scale.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_step(x, h, W_ih, W_hh, b_ih, b_hh, H):
    """Works for a single row (x: (D,), h: (H,)) and for a batch (x: (B, D), h: (B, H))."""
    gi = x @ W_ih + b_ih
    gh = h @ W_hh + b_hh
    rz = _sigmoid(gi[..., :2 * H] + gh[..., :2 * H])
    n = np.tanh(gi[..., 2 * H:] + rz[..., :H] * gh[..., 2 * H:])
    z = rz[..., H:]
    return (1.0 - z) * n + z * h


class Weights:
    def __init__(self, folder=HERE):
        w = np.load(Path(folder) / "weights.npz")
        f = lambda k: np.ascontiguousarray(w[k], dtype=np.float32)  # noqa: E731
        self.H = int(f("gru0.weight_hh_l0").shape[1])
        self.layers = [(f(f"gru{i}.weight_ih_l0").T.copy(), f(f"gru{i}.weight_hh_l0").T.copy(),
                        f(f"gru{i}.bias_ih_l0"), f(f"gru{i}.bias_hh_l0")) for i in range(2)]
        self.W_head, self.b_head = f("head.weight").T.copy(), f("head.bias")
        self.t_mean, self.t_std = f("target_mean"), f("target_std")
        cfg = json.loads((Path(folder) / "config.json").read_text())
        self.scale = np.asarray(cfg.get("output_scale", [1.0, 1.0]), np.float32)

    def output(self, h_last):
        return ((h_last @ self.W_head + self.b_head) * self.t_std + self.t_mean) * self.scale


class PredictionModel:
    def __init__(self):
        self.w = Weights()
        self.seq_ix = None
        self.h = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.h = [np.zeros(self.w.H, np.float32) for _ in range(2)]
        x = np.asarray(data_point.state, dtype=np.float32)
        for i, layer in enumerate(self.w.layers):
            self.h[i] = gru_step(x, self.h[i], *layer, self.w.H)
            x = self.h[i]
        if not data_point.need_prediction:
            return None
        out = self.w.output(x).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=2.0, neginf=-2.0)
