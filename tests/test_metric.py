import numpy as np
import pytest

from wnn.metric import UNITS, CorrAccumulator, family_of, mixed_score, score, weighted_corr


def test_unweighted_matches_numpy():
    rng = np.random.default_rng(0); y, p = rng.standard_normal(1000), rng.standard_normal(1000)
    assert weighted_corr(y, p, np.ones_like(y)) == pytest.approx(np.corrcoef(y, p)[0, 1])


def test_invariant_to_positive_scale_without_clipping():
    rng = np.random.default_rng(1); y = rng.standard_normal((200, 2)) * 0.3; p = y + rng.standard_normal((200, 2)) * 0.3
    assert score(y, p) == pytest.approx(score(y, 0.2 * p))


def test_clipping_changes_score_for_large_predictions():
    rng = np.random.default_rng(2); y = np.clip(rng.standard_normal((2000, 2)), -2, 2); p = 3 * (y + rng.standard_normal((2000, 2)))
    assert score(y, p) != pytest.approx(score(y, p / 10))


def test_rows_with_zero_target_have_no_weight():
    y = np.array([[1.0, -1.0], [0.0, 0.0], [2.0, -0.5]]); p = np.array([[0.5, -0.2], [99.0, -99.0], [1.0, -0.1]])
    p2 = p.copy(); p2[1] = [-99.0, 99.0]
    assert score(y, p) == pytest.approx(score(y, p2))


def test_accumulator_equals_direct_computation():
    rng = np.random.default_rng(3); y = rng.standard_normal((5, 300, 2)); p = y + rng.standard_normal((5, 300, 2))
    acc = CorrAccumulator()
    for i in range(5):
        acc.add(y[i], p[i])
    assert acc.value() == pytest.approx(score(y, p))


def test_family_detection():
    Y = np.zeros((4, 50, 2))
    for f in range(4):
        Y[f, :, 0] = UNITS[f] * np.resize([0, 1, -2, 3], 50)
    assert family_of(Y).tolist() == [0, 1, 2, 3]


def test_mixed_score_equals_score_when_mix_matches():
    rng = np.random.default_rng(4); n, L = 8, 400
    fam = np.repeat([0, 1, 2, 3], 2)
    Y = (rng.integers(-3, 4, (n, L, 2)) * UNITS[fam][:, None, None]).astype(float)
    P = Y + rng.standard_normal((n, L, 2)); M = np.ones((n, L), bool)
    expected = score(Y[:, 99:], P[:, 99:])
    assert mixed_score(P, Y, M, fam, mix=np.full(4, 0.25)) == pytest.approx(expected)

