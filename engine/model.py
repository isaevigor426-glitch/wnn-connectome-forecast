"""Stateful GRU model. Everything here must be reproducible step-by-step in numpy
(see solution_template.py), so only Linear / SiLU / GRU are used.

Input per row: 112 raw features + 112 one-step deltas (delta = 0 at step 0),
standardized with train statistics and clipped to +-CLIP_IN.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_RAW = 112
CLIP_IN = 10.0


class GRUNet(nn.Module):
    """dropout / input_noise act only in training (model.train()); inference is unchanged, so
    the numpy and numba solutions need nothing new. dropout is applied to the GRU input, between
    the GRU layers and to the head input; input_noise is Gaussian noise added to the
    standardized features."""

    def __init__(self, proj: int = 96, hidden: int = 128, layers: int = 2, head: int = 64,
                 dropout: float = 0.0, input_noise: float = 0.0):
        super().__init__()
        self.cfg = dict(proj=proj, hidden=hidden, layers=layers, head=head, dropout=dropout, input_noise=input_noise)
        d_in = 2 * N_RAW
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.inp = nn.Linear(d_in, proj)
        self.gru = nn.GRU(proj, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head1 = nn.Linear(hidden + proj, head)
        self.head2 = nn.Linear(head, 2)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.input_noise = input_noise

    def param_groups(self, wd: float):
        """Weight decay on matrices only (not on biases)."""
        decay = [p for p in self.parameters() if p.dim() >= 2]
        rest = [p for p in self.parameters() if p.dim() < 2]
        return [{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}]

    @staticmethod
    def make_inputs(x: torch.Tensor, prev: torch.Tensor | None) -> torch.Tensor:
        """x: (B, T, 112) raw chunk; prev: (B, 112) last raw row before the chunk or None."""
        first = x[:, :1] if prev is None else prev[:, None]
        d = torch.diff(x, dim=1, prepend=first)
        return torch.cat([x, d], -1)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None, prev: torch.Tensor | None = None):
        f = self.make_inputs(x, prev)
        f = ((f - self.mu) / self.sd).clamp(-CLIP_IN, CLIP_IN)
        if self.training and self.input_noise > 0:
            f = f + self.input_noise * torch.randn_like(f)
        z = self.act(self.inp(f))
        o, h = self.gru(self.drop(z), h)
        out = self.head2(self.act(self.head1(self.drop(torch.cat([o, z], -1)))))
        return out, h


class BaselineGRU(nn.Module):
    """The organizers' baseline architecture: GRU(112->128) -> GRU(128->128) -> Linear(128->2),
    raw features in, no normalization. Can be initialised from baseline.onnx and fine-tuned.
    The hidden state is a (2, B, 128) tensor (one slice per GRU) so the training loop is shared."""

    def __init__(self, hidden: int = 128, dropout: float = 0.0, input_noise: float = 0.0, **_):
        super().__init__()
        self.cfg = dict(arch="baseline", hidden=hidden, dropout=dropout, input_noise=input_noise)
        self.gru0 = nn.GRU(N_RAW, hidden, batch_first=True)
        self.gru1 = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(dropout)          # training only: between the GRUs and before the head
        self.input_noise = input_noise           # training only: Gaussian noise on the raw features

    def param_groups(self, wd: float):
        """Weight decay on matrices only (not on biases)."""
        decay = [p for p in self.parameters() if p.dim() >= 2]
        rest = [p for p in self.parameters() if p.dim() < 2]
        return [{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}]

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None, prev: torch.Tensor | None = None):
        h0, h1 = (None, None) if h is None else (h[0:1].contiguous(), h[1:2].contiguous())
        if self.training and self.input_noise > 0:
            x = x + self.input_noise * torch.randn_like(x)
        o, h0 = self.gru0(x, h0)
        o, h1 = self.gru1(self.drop(o), h1)
        return self.head(self.drop(o)), torch.cat([h0, h1], 0)

    @torch.no_grad()
    def load_onnx(self, path: str) -> None:
        """Copy weights from the starter-pack baseline.onnx (ONNX gate order z,r,h -> torch r,z,n;
        the ONNX GRUs use linear_before_reset=1, which is PyTorch's formulation)."""
        import numpy as np
        import onnx
        from onnx import numpy_helper

        m = onnx.load(path)
        init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
        grus = [n for n in m.graph.node if n.op_type == "GRU"]
        assert len(grus) == 2, "expected two GRU nodes"

        def zrh_to_rzn(w):
            z, r, n = np.split(w, 3, axis=0)
            return np.concatenate([r, z, n], 0)

        for node, gru in zip(grus, (self.gru0, self.gru1)):
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            assert attrs.get("linear_before_reset", 0) == 1
            W, R, B = (init[node.input[k]] for k in (1, 2, 3))
            H = W.shape[1] // 3
            gru.weight_ih_l0.copy_(torch.from_numpy(zrh_to_rzn(W[0])))
            gru.weight_hh_l0.copy_(torch.from_numpy(zrh_to_rzn(R[0])))
            gru.bias_ih_l0.copy_(torch.from_numpy(zrh_to_rzn(B[0, :3 * H])))
            gru.bias_hh_l0.copy_(torch.from_numpy(zrh_to_rzn(B[0, 3 * H:])))
        mm = [n for n in m.graph.node if n.op_type == "MatMul"][0]
        add = [n for n in m.graph.node if n.op_type == "Add"][-1]
        Wh = init[mm.input[1]] if mm.input[1] in init else init[mm.input[0]]
        bh = next(init[i] for i in add.input if i in init)
        self.head.weight.copy_(torch.from_numpy(Wh.T.copy()))
        self.head.bias.copy_(torch.from_numpy(bh.copy()))


def build_model(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    arch = cfg.pop("arch", "gru")
    if arch == "baseline":
        return BaselineGRU(**cfg)
    if arch == "mamba":
        from mamba import MambaNet
        return MambaNet(**cfg)
    return GRUNet(**cfg)
