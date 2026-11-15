import numpy as np

from wnn.data import family_weights, holdout_ids, select_active, valid_train_ids
from wnn.features import event_features, event_rate


def _sequence(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    regime = np.cumsum(rng.random(n) < 0.3)                     # prices change on ~30% of steps, all 22 columns at once
    x = rng.standard_normal((n, 112)).astype(np.float32)
    x[:, :22] = rng.standard_normal((regime.max() + 1, 22))[regime]
    return x


def test_event_features_are_causal():
    x = _sequence(); full = event_features(x, n=20000); part = event_features(x[:1500], n=20000)
    assert np.allclose(full[:1500], part)


def test_event_features_shape_and_finite():
    f = event_features(_sequence())
    assert f.shape == (3000, 8) and np.isfinite(f).all()


def test_event_rate_matches_construction():
    assert 0.2 < event_rate(_sequence(20000), stride=1) < 0.4


def test_select_active_keeps_family_proportions_and_picks_active():
    rng = np.random.default_rng(0); fam = rng.integers(0, 4, 1000); rate = rng.random(1000)
    ids = select_active(rate, fam, n=400)
    assert abs(len(ids) - 400) <= 2
    for f in range(4):
        chosen, all_f = rate[ids][fam[ids] == f], rate[fam == f]
        assert chosen.min() >= np.sort(all_f)[::-1][len(chosen) - 1] - 1e-12


def test_family_weights_reach_target_mix():
    fam = np.array([0] * 10 + [1] * 30 + [2] * 30 + [3] * 30); mix = np.array([0.4, 0.2, 0.2, 0.2])
    w = family_weights(fam, mix)
    assert np.isclose(w.mean(), 1.0)
    assert np.allclose(np.bincount(fam, weights=w) / w.sum(), mix)


def test_holdout_split_is_disjoint_and_complete():
    h, r = holdout_ids(), valid_train_ids()
    assert len(h) == 300 and len(r) == 1573 and not set(h) & set(r)
