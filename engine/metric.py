"""Global Weighted Pearson (WP) — local reimplementation of METRIC.md.

Official definition (starter pack METRIC.md):
    y = clip(t, -2, 2), p = clip(pred, -2, 2), w = |y|   (zero targets have zero weight)
    weighted Pearson per target in float64; 0 if total weight < 1e-8 or either
    weighted std <= 1e-8; result bounded to [-1, 1]; score = mean over t0, t1.
"abs_target" is the official mode and the default. "uniform" (w = 1) is kept only
as a diagnostic: the gap between the two shows how much the large moves dominate.

Pooling: all scored rows of all sequences together (no per-sequence averaging),
separately for t0 and t1, then the mean of the two.
"""
from __future__ import annotations

import numpy as np

from common import CLIP

WEIGHT_MODES = ("abs_target", "uniform")   # first = official


def compute_weights(y_clipped: np.ndarray, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones_like(y_clipped, dtype=np.float64)
    if mode == "abs_target":
        return np.abs(y_clipped).astype(np.float64)
    raise ValueError(f"unknown weight mode {mode!r}; choose from {WEIGHT_MODES}")


EPS = 1e-8


def _finish(sw: float, cov: float, vy: float, vp: float) -> float:
    """cov, vy, vp are weighted (divided by sw) moments."""
    if sw < EPS or vy <= EPS * EPS or vp <= EPS * EPS:   # std <= 1e-8
        return 0.0
    return float(np.clip(cov / np.sqrt(vy * vp), -1.0, 1.0))


def weighted_pearson(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson for 1-D arrays (already clipped)."""
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    w = np.asarray(w, np.float64)
    sw = w.sum()
    if sw < EPS:
        return 0.0
    my = (w * y).sum() / sw
    mp = (w * p).sum() / sw
    cov = (w * (y - my) * (p - mp)).sum() / sw
    vy = (w * (y - my) ** 2).sum() / sw
    vp = (w * (p - mp) ** 2).sum() / sw
    return _finish(sw, cov, vy, vp)


class WPAccumulator:
    """Streaming version: feed chunks, get the pooled score at the end.

    Keeps 6 float64 sums per target, so 37M+ rows need no extra memory.
    Uses a shifted formulation (subtracting the first chunk's means) to keep
    the one-pass sums numerically stable.
    """

    def __init__(self, n_targets: int = 2, weight_mode: str = "abs_target"):
        self.n = n_targets
        self.mode = weight_mode
        self.s = np.zeros((n_targets, 6), np.float64)  # sw, swy, swp, swyy, swpp, swyp
        self.shift = None
        self.rows = 0

    def update(self, y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None) -> None:
        # float32 first, as in the reference implementation
        y = np.asarray(y, np.float32).astype(np.float64).reshape(-1, self.n)
        p = np.asarray(p, np.float32).astype(np.float64).reshape(-1, self.n)
        if mask is not None:
            m = np.asarray(mask, bool).reshape(-1)
            y, p = y[m], p[m]
        if len(y) == 0:
            return
        y = np.clip(y, -CLIP, CLIP)
        p = np.clip(p, -CLIP, CLIP)
        if self.shift is None:
            self.shift = (y.mean(0), p.mean(0))
        for j in range(self.n):
            w = compute_weights(y[:, j], self.mode)
            yy = y[:, j] - self.shift[0][j]
            pp = p[:, j] - self.shift[1][j]
            self.s[j] += [w.sum(), (w * yy).sum(), (w * pp).sum(),
                          (w * yy * yy).sum(), (w * pp * pp).sum(), (w * yy * pp).sum()]
        self.rows += len(y)

    def per_target(self) -> list[float]:
        out = []
        for sw, swy, swp, swyy, swpp, swyp in self.s:
            if sw < EPS:
                out.append(0.0)
                continue
            my, mp = swy / sw, swp / sw
            cov = swyp / sw - my * mp
            vy = swyy / sw - my * my
            vp = swpp / sw - mp * mp
            out.append(_finish(sw, cov, vy, vp))
        return out

    def score(self) -> float:
        return float(np.mean(self.per_target()))


def score(y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None,
          weight_mode: str = "abs_target") -> dict:
    """One-shot helper: y, p of shape (N, 2)."""
    acc = WPAccumulator(y.shape[-1], weight_mode)
    acc.update(y, p, mask)
    pt = acc.per_target()
    return {"t0": pt[0], "t1": pt[1], "score": float(np.mean(pt)), "rows": acc.rows}
