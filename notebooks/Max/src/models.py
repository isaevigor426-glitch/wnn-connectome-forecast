import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, model='lstm', hidden=128):
        super().__init__()
        self.kind, self.hidden = model, hidden
        self.rnn = (nn.LSTM if model == 'lstm' else nn.GRU)(120 if model == 'gru_ev' else 112, hidden, 2, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, state=None):
        y, state = self.rnn(x.float(), state)
        return self.head(y), state


def detach(state):
    return tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()
