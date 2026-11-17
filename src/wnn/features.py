"""Causal hand-crafted features (numpy only). At step t only rows 0..t are used.

"Price events" are steps where any of the 22 price columns of instrument i0 (columns 0-21)
changes. The target changes only on such steps, so they are the natural clock of the problem.
"""
import numpy as np

N_STEPS = 20000
PRICE_COLS = slice(0, 22)


def event_rate(x, stride=4):
    """Share of steps (every `stride`-th) on which any i0 price column changes — the activity of a sequence."""
    xs = np.asarray(x[::stride, PRICE_COLS], np.float32)
    return float((np.diff(xs, axis=0) != 0).any(axis=1).mean())


def event_features(x, n=N_STEPS):
    """Eight event features for every step of one sequence x of shape (n_steps, 112).

    0-4: sum of the book shift (mean change of the 22 price columns) over the last 1, 2, 3, 5, 10 events;
    5:   log(1 + steps since the last event), log(1 + n) before the first event;
    6-7: number of events in the last 50 and 200 steps.
    """
    x = np.asarray(x, np.float32)
    m_ = len(x)
    d = np.diff(x[:, PRICE_COLS], axis=0, prepend=x[:1, PRICE_COLS])
    is_ev = (d != 0).any(axis=1)
    is_ev[0] = False
    shift = np.where(is_ev, d.mean(axis=1), 0.0)
    ev = np.flatnonzero(is_ev)
    cnt = np.cumsum(is_ev)
    cs = np.concatenate([[0.0], np.cumsum(shift[ev])])
    out = np.zeros((m_, 8), np.float32)
    t = np.arange(m_)
    for i, k in enumerate((1, 2, 3, 5, 10)):
        out[:, i] = cs[cnt] - cs[np.maximum(cnt - k, 0)]
    last = ev[np.maximum(cnt - 1, 0)] if len(ev) else np.zeros(m_, int)
    out[:, 5] = np.log1p(np.where(cnt > 0, t - last, n))
    c0 = np.concatenate([[0], cnt])
    for i, w in ((6, 50), (7, 200)):
        out[:, i] = c0[t + 1] - c0[np.maximum(t + 1 - w, 0)]
    return out
