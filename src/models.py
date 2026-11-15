"""Recurrent models (PyTorch). Two stacked GRU or LSTM layers and a linear head, raw 112 features in."""
import torch
from torch import nn


class RecurrentNet(nn.Module):
    """GRU/LSTM(n_in -> hidden) x 2 layers -> Linear(hidden -> 2). Stateful: pass the hidden state between chunks."""

    def __init__(self, cell="gru", n_in=112, hidden=128):
        super().__init__()
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.cell = cell
        self.rnn = rnn(n_in, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, h=None):
        y, h = self.rnn(x, h)
        return self.head(y), h


def detach_state(h):
    """Cuts the graph at a chunk boundary (truncated BPTT) for GRU (tensor) and LSTM (tuple) states."""
    return tuple(v.detach() for v in h) if isinstance(h, tuple) else h.detach()


class GRUStep(nn.Module):
    """One-row step of a GRU RecurrentNet with explicit per-layer states — the interface used for ONNX export."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, h0, h1):
        y, h = self.net.rnn(x, torch.cat([h0, h1], 0))
        return self.net.head(y), h[0:1], h[1:2]
