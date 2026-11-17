"""Mamba-2 SSM, pure-numpy stateful inference (no torch needed at scoring time).

weights.npz is written by export.py. To keep the per-row cost low, the norm weights are
folded into the following matrices, the causal conv uses a ring buffer with pre-rotated
weights, and SiLU of z and of the conv output is computed in one call.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CLIP_IN = 10.0
EPS = 1e-5


class _Layer:
    """One Mamba-2 layer. The per-head step size dt is expanded to per-channel inside W_in
    (rows repeated P times), so the state is a flat matrix stored as (N, H*P) (long axis
    contiguous) and no broadcasting over heads is needed at run time. Only numpy ufuncs
    and matmuls are used on the hot path (np.sum/np.clip/np.outer are slow Python wrappers)."""

    def __init__(self, w, i, d_model, d_inner, N, H, P, K, out_norm):
        f = lambda k: np.ascontiguousarray(w[f"layers.{i}.{k}"], dtype=np.float32)  # noqa: E731
        self.d_inner, self.N, self.K, self.out_norm = d_inner, N, K, out_norm
        self.conv_dim = cd = d_inner + 2 * N
        W = f("in_proj.weight") * f("norm.weight")[None, :]                 # (d_inner + cd + H, d_model)
        W_dt = np.repeat(W[d_inner + cd:], P, axis=0)                       # (d_inner, d_model)
        self.W_in = np.ascontiguousarray(np.concatenate([W[:d_inner + cd], W_dt]).T)
        self.bias = np.concatenate([np.zeros(d_inner, np.float32), f("conv.bias"),
                                    np.repeat(f("dt_bias"), P)]).astype(np.float32)
        wT = f("conv.weight")[:, 0, :].T                                    # (K, cd), oldest first
        # ring buffer: after writing slot `pos`, slot j holds chronological index (j - pos - 1) % K
        self.W_roll = [np.ascontiguousarray(wT[[(j - pos - 1) % K for j in range(K)]]) for pos in range(K)]
        self.A = np.repeat(-np.exp(f("A_log").astype(np.float64)), P).astype(np.float32)   # (d_inner,)
        self.D = np.repeat(f("D"), P).astype(np.float32)
        W_out = f("out_proj.weight")
        if out_norm:
            W_out = W_out * f("out_norm.weight")[None, :]
        self.W_out = np.ascontiguousarray(W_out.T)
        self.inv_d_model, self.inv_d_inner = 1.0 / d_model, 1.0 / d_inner
        self.tmp = np.empty((K, cd), np.float32)
        self.e = np.empty(d_inner + cd, np.float32)
        self.reset()

    def reset(self):
        self.buf = np.zeros((self.K, self.conv_dim), np.float32)
        self.pos = 0
        self.h = np.zeros((self.N, self.d_inner), np.float32)

    def step(self, u):
        di, cd, N = self.d_inner, self.conv_dim, self.N
        r = u * (1.0 / math.sqrt(float(u @ u) * self.inv_d_model + EPS))
        proj = r @ self.W_in                                   # [z | xBC | dt per channel]
        pos = self.pos
        self.buf[pos] = proj[di:di + cd]                       # raw xBC into the ring buffer
        np.multiply(self.buf, self.W_roll[pos], out=self.tmp)
        np.add.reduce(self.tmp, axis=0, out=proj[di:di + cd])  # causal depthwise conv
        self.pos = (pos + 1) % self.K
        proj += self.bias                                      # conv bias and dt bias at once
        zc = proj[:di + cd]
        e = self.e
        np.negative(zc, out=e)
        np.exp(e, out=e)
        e += 1.0
        np.divide(zc, e, out=zc)                               # silu(z) and silu(conv)
        x = zc[di:2 * di]
        v = proj[di + cd:]
        np.minimum(v, 20.0, out=v)                             # softplus(v) == v beyond 20
        dt = np.log1p(np.exp(v))                               # softplus, per channel
        h = self.h
        h *= np.exp(dt * self.A)                               # decay, broadcast over the N rows
        h += np.multiply.outer(zc[2 * di:2 * di + N], x * dt)  # + B ⊗ (dt*x)
        y = zc[2 * di + N:di + cd] @ h                         # C · h
        y += self.D * x
        y *= zc[:di]                                           # gate silu(z)
        if self.out_norm:
            y *= 1.0 / math.sqrt(float(y @ y) * self.inv_d_inner + EPS)
        return u + y @ self.W_out


class PredictionModel:
    def __init__(self):
        w = np.load(Path(__file__).resolve().parent / "weights.npz")
        f = lambda k: np.ascontiguousarray(w[k], dtype=np.float32)  # noqa: E731
        n_layers, d_model, d_inner, N, H, P, K = (int(w[k]) for k in
                                                  ("n_layers", "d_model", "d_inner", "d_state", "n_heads",
                                                   "headdim", "d_conv"))
        self.mu, self.inv_sd = f("mu"), (1.0 / f("sd")).astype(np.float32)
        self.W_inp, self.b_inp = np.ascontiguousarray(f("inp.weight").T), f("inp.bias")
        out_norm = bool(int(w["out_norm"])) if "out_norm" in w else True
        self.layers = [_Layer(w, i, d_model, d_inner, N, H, P, K, out_norm) for i in range(n_layers)]
        self.W1 = np.ascontiguousarray((f("head1.weight") * f("norm_f.weight")[None, :]).T)
        self.b1 = f("head1.bias")
        scale = f("output_scale")
        self.W2 = np.ascontiguousarray(f("head2.weight").T * scale[None, :])   # output scale folded in
        self.b2 = (f("head2.bias") * scale).astype(np.float32)
        self.inv_d_model = 1.0 / d_model
        self.n = self.mu.shape[0] // 2
        self.f = np.zeros(self.mu.shape[0], np.float32)          # [raw | delta]
        self.prev = np.zeros(self.n, np.float32)
        self.g = np.empty(self.mu.shape[0], np.float32)
        self.seq_ix = None
        self.first = True

    def _reset(self):
        for layer in self.layers:
            layer.reset()
        self.first = True

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self._reset()
        s = np.asarray(data_point.state, dtype=np.float32)
        n, f, g = self.n, self.f, self.g
        f[:n] = s
        if self.first:
            f[n:] = 0.0
            self.first = False
        else:
            np.subtract(s, self.prev, out=f[n:])
        self.prev[:] = s
        np.subtract(f, self.mu, out=g)
        g *= self.inv_sd
        np.minimum(g, CLIP_IN, out=g)
        np.maximum(g, -CLIP_IN, out=g)
        u = g @ self.W_inp + self.b_inp
        for layer in self.layers:
            u = layer.step(u)
        r = u * (1.0 / math.sqrt(float(u @ u) * self.inv_d_model + EPS))
        hdn = r @ self.W1 + self.b1
        np.maximum(hdn, 0.0, out=hdn)
        o = hdn @ self.W2 + self.b2
        if not data_point.need_prediction:
            return None
        if not (math.isfinite(float(o[0])) and math.isfinite(float(o[1]))):
            o = np.nan_to_num(o, nan=0.0, posinf=2.0, neginf=-2.0)
        return o.astype(np.float32, copy=False)
