"""Competition metric and target-family utilities (numpy only).

The score is a |y|-weighted Pearson correlation computed over *all* scored rows of *all*
sequences at once (not per sequence, not per batch). Targets and predictions are clipped
to [-2, 2]; the final score is the mean over the two targets t0 and t1.
"""
import numpy as np

UNITS = np.array([0.1254, 0.2413, 0.4685, 0.6569])        # target step size of families F0..F3
VALID_MIX = np.array([0.2125, 0.1853, 0.2301, 0.3721])     # family shares in the validation split
WARMUP = 99                                                # steps 0..98 are warm-up, never scored
CLIP = 2.0


def weighted_corr(y, p, w):
    """Weighted Pearson correlation of 1-D arrays y and p with weights w."""
    y, p, w = (np.asarray(a, np.float64).ravel() for a in (y, p, w))
    sw = w.sum()
    my, mp = (w * y).sum() / sw, (w * p).sum() / sw
    vy, vp = (w * (y - my) ** 2).sum(), (w * (p - mp) ** 2).sum()
    if vy <= 0 or vp <= 0:
        return 0.0
    return float((w * (y - my) * (p - mp)).sum() / np.sqrt(vy * vp))


def score(y, p, mask=None):
    """Competition score for arrays of shape (..., 2): mean over targets of |y|-weighted Pearson."""
    y = np.clip(np.asarray(y, np.float64), -CLIP, CLIP)
    p = np.clip(np.asarray(p, np.float64), -CLIP, CLIP)
    if mask is not None:
        y, p = y[mask], p[mask]
    return float(np.mean([weighted_corr(y[..., j], p[..., j], np.abs(y[..., j])) for j in (0, 1)]))


class CorrAccumulator:
    """Streaming version of `score`: add data in chunks, read the score at the end.

    Keeps six weighted sufficient statistics per target, so 39M rows never need to sit in memory.
    """

    def __init__(self):
        self.t = np.zeros((2, 6))

    def add(self, y, p, mask=None, weight=1.0):
        y = np.clip(np.asarray(y, np.float64), -CLIP, CLIP)
        p = np.clip(np.asarray(p, np.float64), -CLIP, CLIP)
        if mask is not None:
            y, p = y[mask], p[mask]
        for j in (0, 1):
            yy, pp = y[..., j].ravel(), p[..., j].ravel()
            a = np.abs(yy) * weight
            self.t[j] += [a.sum(), (a * yy).sum(), (a * pp).sum(), (a * yy * yy).sum(), (a * pp * pp).sum(), (a * yy * pp).sum()]
        return self

    def __iadd__(self, other):
        self.t += other.t
        return self

    def value(self):
        sw = self.t[:, 0]
        my, mp = self.t[:, 1] / sw, self.t[:, 2] / sw
        cov = self.t[:, 5] / sw - my * mp
        vy, vp = self.t[:, 3] / sw - my ** 2, self.t[:, 4] / sw - mp ** 2
        return float(np.mean(cov / np.sqrt(np.maximum(vy * vp, 1e-16))))


def family_of(Y):
    """Family (0..3) of each sequence from its targets: the unit closest to the smallest non-zero |t0|.

    Y: array (n_seq, n_steps, 2). Every target value of a sequence is a multiple of its unit.
    """
    a = np.abs(np.clip(np.asarray(Y)[..., 0], -CLIP, CLIP)).astype(np.float64)
    smallest = np.where(a > 0, a, 9.0).min(axis=1)
    return np.abs(smallest[:, None] - UNITS[None]).argmin(axis=1)


def mixed_score(P, Y, M, fam, mix=VALID_MIX):
    """Score with each family re-weighted to the shares `mix`.

    A held-out set of a few hundred sequences has a random family mix; re-weighting removes
    that source of noise. P, Y: (n_seq, n_steps, 2); M: (n_seq, n_steps) scored-row mask.
    """
    share = np.bincount(fam, minlength=4) / len(fam)
    acc = CorrAccumulator()
    post = np.arange(Y.shape[1]) >= WARMUP
    for f in range(4):
        idx = np.where(fam == f)[0]
        if len(idx) == 0:
            continue
        m = M[idx] & post[None]
        acc.add(Y[idx], P[idx], m, weight=mix[f] / share[f])
    return acc.value()
