"""Mamba-2 (selective state space model) for row-by-row prediction.

Pure PyTorch (no mamba-ssm CUDA kernels, so it installs anywhere). Training uses the
chunked "SSD" scan from the Mamba-2 paper (Dao & Gu, 2024, `ssd_minimal_discrete`):
the sequence is cut into blocks; inside a block the recurrence is computed as small
matmuls, between blocks the state is propagated. The state is carried across training
chunks exactly like the GRU hidden state, so training == row-by-row inference.

Per layer and step (pre-norm residual block):
    zxbcdt = W_in · rmsnorm(u)                      -> z, xBC, dt
    xBC    = silu(causal depthwise conv_K(xBC))     -> x (H heads x P), B (N), C (N)
    dt     = softplus(dt + dt_bias)                 (one step size per head)
    h      = exp(dt·A) · h + dt · x ⊗ B             (h: H x P x N, A < 0 per head)
    y      = h · C + D · x
    out    = W_out · (y · silu(z));  u = u + out       (optional rmsnorm before W_out: --out-norm)
Each head has its own decay exp(dt·A): with A in [-16, -1] and dt in [5e-4, 0.1] the
memory spans ~1 to ~2000 steps, which covers the target horizon (100-1000 steps).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CLIP_IN, N_RAW, GRUNet


def _f32(t: torch.Tensor) -> torch.Tensor:
    """Upcast half precision to fp32 for the scan; leave fp32/fp64 as they are."""
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def segsum(x: torch.Tensor) -> torch.Tensor:
    """x: (..., T) -> (..., T, T); out[..., i, j] = sum_{k=j+1..i} x[..., k] for i >= j, -inf above."""
    T = x.size(-1)
    x = x[..., None].expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd_scan(X, A, B, C, block_len: int, initial_state=None):
    """Chunked scan of h_t = exp(A_t) h_{t-1} + X_t ⊗ B_t,  y_t = h_t · C_t.

    X: (b, L, h, p)   input already multiplied by dt
    A: (b, L, h)      log-decay per step (= dt * A <= 0)
    B, C: (b, L, h, n)
    initial_state: (b, h, p, n) or None.  L must be a multiple of block_len.
    Returns y (b, L, h, p) and the final state (b, h, p, n).
    """
    b, L, h, p = X.shape
    n = B.shape[-1]
    c = L // block_len
    X = X.reshape(b, c, block_len, h, p)
    B = B.reshape(b, c, block_len, h, n)
    C = C.reshape(b, c, block_len, h, n)
    A = A.reshape(b, c, block_len, h).permute(0, 3, 1, 2)            # (b, h, c, l)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. outputs from inputs inside the same block
    Lmat = torch.exp(segsum(A)).permute(0, 2, 1, 3, 4)                # (b, c, h, l, s)
    CB = torch.einsum("bclhn,bcshn->bchls", C, B)
    Y_diag = torch.einsum("bchls,bcshp->bclhp", CB * Lmat, X)

    # 2. state at the end of each block from that block's inputs
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)           # (b, h, c, l)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. propagate states across blocks, starting from the carried-in state
    if initial_state is None:
        initial_state = torch.zeros_like(states[:, :1])
    else:
        initial_state = initial_state[:, None].to(states.dtype)
    states = torch.cat([initial_state, states], dim=1)                # (b, c+1, h, p, n)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[..., -1], (1, 0))))  # (b, h, c+1, c+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. outputs from the state that enters each block
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, torch.exp(A_cumsum))
    return (Y_diag + Y_off).reshape(b, L, h, p), final_state


class Mamba2Layer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, headdim: int = 16,
                 d_conv: int = 4, block_len: int = 50, out_norm: bool = False,
                 dt_min: float = 5e-4, dt_max: float = 0.1):
        super().__init__()
        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0, "expand*d_model must be divisible by headdim"
        self.H, self.P, self.N, self.K = self.d_inner // headdim, headdim, d_state, d_conv
        self.conv_dim = self.d_inner + 2 * d_state
        self.block_len = block_len
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner + self.conv_dim + self.H, bias=False)
        self.conv = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv, groups=self.conv_dim, bias=True)
        dt = torch.exp(torch.rand(self.H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))      # softplus^-1(dt)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.H))
        # Mamba-2 normalizes y*silu(z) before out_proj; off by default (Mamba-1 style): ~10% faster
        # numpy inference, which matters for the 1-CPU time limit.
        self.out_norm = RMSNorm(self.d_inner) if out_norm else nn.Identity()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u, state=None):
        """u: (b, L, d_model); state = (conv_state (b, K-1, conv_dim), ssm_state (b, H, P, N)) or None."""
        b, L, _ = u.shape
        H, P, N, K = self.H, self.P, self.N, self.K
        zxbcdt = self.in_proj(self.norm(u))
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.conv_dim, H], dim=-1)

        if state is None:
            conv_state = xBC.new_zeros(b, K - 1, self.conv_dim)
            ssm_state = None
        else:
            conv_state, ssm_state = state
        xBC_full = torch.cat([conv_state.to(xBC.dtype), xBC], dim=1)       # (b, K-1+L, conv_dim)
        new_conv_state = xBC_full[:, xBC_full.shape[1] - (K - 1):]
        xBC = F.silu(self.conv(xBC_full.transpose(1, 2)).transpose(1, 2))   # (b, L, conv_dim)
        x, Bm, Cm = torch.split(xBC, [self.d_inner, N, N], dim=-1)

        with torch.autocast(device_type=u.device.type, enabled=False):      # the scan runs in fp32
            dt = F.softplus(_f32(dt) + self.dt_bias)                       # (b, L, H)
            A = -torch.exp(self.A_log)                                     # (H,)
            x = _f32(x).reshape(b, L, H, P)
            Bm = _f32(Bm)[:, :, None, :].expand(b, L, H, N)
            Cm = _f32(Cm)[:, :, None, :].expand(b, L, H, N)
            Xs, dts = x * dt[..., None], dt
            pad = (-L) % self.block_len
            if pad:  # dt = 0 on padded steps: no decay, no input -> the final state stays exact
                Xs = F.pad(Xs, (0, 0, 0, 0, 0, pad))
                dts = F.pad(dts, (0, 0, 0, pad))
                Bm = F.pad(Bm, (0, 0, 0, 0, 0, pad))
                Cm = F.pad(Cm, (0, 0, 0, 0, 0, pad))
            y, new_ssm_state = ssd_scan(Xs, dts * A, Bm, Cm, self.block_len, ssm_state)
            y = y[:, :L] + x * self.D[:, None]
            y = y.reshape(b, L, self.d_inner) * F.silu(_f32(z))
        out = self.out_proj(self.out_norm(y))
        return u + out, (new_conv_state, new_ssm_state)


class MambaNet(nn.Module):
    """112 raw features + 1-step deltas -> standardize -> Linear -> n x Mamba2Layer -> RMSNorm -> MLP -> [t0, t1]."""

    def __init__(self, d_model: int = 64, n_layers: int = 2, d_state: int = 16, expand: int = 2,
                 headdim: int = 16, d_conv: int = 4, head: int = 64, block_len: int = 50,
                 out_norm: bool = False, **_):
        super().__init__()
        self.cfg = dict(arch="mamba", d_model=d_model, n_layers=n_layers, d_state=d_state, expand=expand,
                        headdim=headdim, d_conv=d_conv, head=head, block_len=block_len, out_norm=out_norm)
        d_in = 2 * N_RAW
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.inp = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([Mamba2Layer(d_model, d_state, expand, headdim, d_conv, block_len, out_norm)
                                     for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.head1 = nn.Linear(d_model, head)
        self.head2 = nn.Linear(head, 2)

    def forward(self, x, h=None, prev=None):
        f = GRUNet.make_inputs(x, prev)
        f = ((f - self.mu) / self.sd).clamp(-CLIP_IN, CLIP_IN)
        u = self.inp(f)
        new_state = []
        for i, layer in enumerate(self.layers):
            u, s = layer(u, None if h is None else h[i])
            new_state.append(s)
        out = self.head2(F.relu(self.head1(self.norm_f(u))))
        return out, new_state

    def param_groups(self, weight_decay: float):
        """AdamW groups: no weight decay on A_log, D, dt_bias, norms and biases (as in Mamba)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if p.ndim < 2 or name.endswith(("A_log", ".D", "dt_bias")):
                no_decay.append(p)
            else:
                decay.append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
