import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from mgw.bench import metrics as m

ORG = ["Brain", "Heart", "Liver", "Gut"]


def _labels(n, organs, seed=0):
    return np.random.default_rng(seed).choice(organs, n)


def test_identity_coupling_is_one_and_product_is_zero():
    lab = _labels(200, ORG)
    Q = m.onehot(lab)
    kb, K = m.score(np.eye(200), Q, Q, min_bins=1)
    assert K == 4 and kb == pytest.approx(1.0)
    kb0, _ = m.score(np.full((200, 200), 1 / 200 ** 2), Q, Q, min_bins=1)
    assert kb0 == pytest.approx(0.0, abs=1e-12)


def test_hard_coupling_matches_adjusted_balanced_accuracy():
    rng = np.random.default_rng(1)
    true = _labels(300, ORG, 1)
    targ = _labels(250, ORG, 2)
    # a hard coupling: each query point sends all its mass to one target point
    j = rng.integers(0, 250, 300)
    P = np.zeros((300, 250)); P[np.arange(300), j] = 1
    Q, T = m.onehot(true), m.onehot(targ)
    kb, _ = m.score(P, Q, T, min_bins=1)
    assert kb == pytest.approx(balanced_accuracy_score(true, targ[j], adjusted=True))
    assert (m.hard_prediction(P, T) == m.organ_codes(targ[j])).all()


def test_two_organs_is_youden_j():
    true = np.array(["Brain"] * 60 + ["Heart"] * 40)
    pred = np.array(["Brain"] * 45 + ["Heart"] * 15 + ["Heart"] * 30 + ["Brain"] * 10)
    Q, H = m.onehot(true), m.onehot(pred)
    kb, K = m.kappa_bal(Q.T @ H, m.present(Q, 1), m.present(H, 1))
    sens, spec = 45 / 60, 30 / 40
    assert K == 2 and kb == pytest.approx(sens + spec - 1)


def test_kappa_bal_is_chance_adjusted_agreement_after_row_balancing():
    rng = np.random.default_rng(3)
    C = rng.random((len(m.ORGANS), len(m.ORGANS)))
    pres = m.SCORED.copy()
    kb, K = m.kappa_bal(C, pres, pres)
    Cs = C[np.ix_(pres, pres)]
    B = Cs / Cs.sum(1, keepdims=True) / K                  # every row the same mass, total 1
    po, pe = np.trace(B), float(B.sum(1) @ B.sum(0))       # chance agreement after balancing is 1/K
    assert pe == pytest.approx(1 / K) and kb == pytest.approx((po - pe) / (1 - pe))


def test_full_resolution_composition_counts_bins():
    # two tiles: tile 0 has 3 Brain bins + 1 Heart bin, tile 1 has 2 Heart bins
    comp = m.composition(["Brain", "Brain", "Brain", "Heart", "Heart", "Heart"], [0, 0, 0, 0, 1, 1], 2)
    assert comp[0, m.ORGAN_INDEX["Brain"]] == 3 and comp[0, m.ORGAN_INDEX["Heart"]] == 1
    # identity coupling against itself: Brain bins all land on tile 0 (3/4 Brain), Heart split
    C = m.confusion(np.eye(2), comp, comp)
    b, h = m.ORGAN_INDEX["Brain"], m.ORGAN_INDEX["Heart"]
    assert C[b, b] == pytest.approx(3 * 0.75) and C[h, h] == pytest.approx(1 * 0.25 + 2 * 1.0)
    assert C.sum() == pytest.approx(6)


def test_unscored_labels_never_count():
    lab = np.array(["Brain"] * 50 + ["Other"] * 50 + ["Excluded"] * 20)
    Q = m.onehot(lab)
    kb, K = m.score(np.eye(120), Q, Q, min_bins=1)
    assert K == 1 and np.isnan(kb)


def test_target_presence_override_keeps_missing_organs_in_K():
    # the target section has no Heart, but the target embryo has: Heart stays in K and scores 0
    q = m.onehot(["Brain"] * 50 + ["Heart"] * 50)
    t = m.onehot(["Brain"] * 100)
    P = np.full((100, 100), 1e-4)
    _, K_own = m.score(P, q, t, min_bins=1)
    kb, K = m.score(P, q, t, min_bins=1, present_t=m.present(m.onehot(["Brain", "Heart"]), 1))
    assert K_own == 1 and K == 2 and kb == pytest.approx(0.0)
