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
    kb = m.kappa_scores(m.confusion(np.eye(200), Q, Q), m.present(Q, 1), m.present(Q, 1))
    assert kb["K"] == 4 and kb["kappa_bal"] == pytest.approx(1.0) and kb["kappa"] == pytest.approx(1.0)
    P = np.full((200, 200), 1 / 200 ** 2)
    kb0 = m.kappa_scores(m.confusion(P, Q, Q), m.present(Q, 1), m.present(Q, 1))
    assert kb0["kappa_bal"] == pytest.approx(0.0, abs=1e-12)


def test_hard_coupling_matches_adjusted_balanced_accuracy():
    rng = np.random.default_rng(1)
    true = _labels(300, ORG, 1)
    targ = _labels(250, ORG, 2)
    # a hard coupling: each query point sends all its mass to one target point
    j = rng.integers(0, 250, 300)
    P = np.zeros((300, 250)); P[np.arange(300), j] = 1
    Q, T = m.onehot(true), m.onehot(targ)
    kb = m.kappa_scores(m.confusion(P, Q, T), m.present(Q, 1), m.present(T, 1))
    ref = balanced_accuracy_score(true, targ[j], adjusted=True)
    assert kb["kappa_bal"] == pytest.approx(ref)


def test_two_organs_is_youden_j():
    true = np.array(["Brain"] * 60 + ["Heart"] * 40)
    pred = np.array(["Brain"] * 45 + ["Heart"] * 15 + ["Heart"] * 30 + ["Brain"] * 10)
    Q, H = m.onehot(true), m.onehot(pred)
    kb = m.kappa_scores(Q.T @ H, m.present(Q, 1), m.present(H, 1))
    sens, spec = 45 / 60, 30 / 40
    assert kb["kappa_bal"] == pytest.approx(sens + spec - 1)


def test_oms_is_observed_agreement_after_row_balancing():
    rng = np.random.default_rng(3)
    C = rng.random((len(m.ORGANS), len(m.ORGANS)))
    pres = m.SCORED.copy()
    kb = m.kappa_scores(C, pres, pres)
    Cs = C[np.ix_(pres, pres)]
    B = Cs / Cs.sum(1, keepdims=True) / pres.sum()        # every row the same mass, total 1
    assert np.trace(B) == pytest.approx(kb["oms"])
    # chance agreement after balancing is exactly 1/K
    assert float(B.sum(1) @ B.sum(0)) == pytest.approx(1 / pres.sum())


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
    kb = m.kappa_scores(m.confusion(np.eye(120), Q, Q), m.present(Q, 1), m.present(Q, 1))
    assert kb["K"] == 1 and np.isnan(kb["kappa_bal"])


def test_foldover_identity_reflection_and_local_fold():
    rng = np.random.default_rng(4)
    x = rng.random((400, 2))
    eye = np.eye(400)
    assert m.fold_overs(x, x, eye)["foldover"] == 0
    refl = m.fold_overs(x, x * [-1, 1], eye)
    assert refl["foldover"] == 0 and refl["orientation"] == -1
    # fold the left fifth over onto the right
    y = x.copy()
    left = y[:, 0] < 0.2
    y[left, 0] = 0.4 - y[left, 0]
    fo = m.fold_overs(x, y, eye)
    assert 0.05 < fo["foldover"] < 0.4 and fo["folded_points"].any()


def test_sharpness_and_topk():
    P = np.eye(10) / 10
    assert m.sharpness(P) == pytest.approx(1.0)
    assert m.sharpness(np.full((10, 10), 0.01)) == pytest.approx(10.0)
    rng = np.random.default_rng(5)
    P = rng.random((30, 20))
    ir, vr, ic, vc = m.topk(P, 5)
    assert np.allclose(np.sort(vr, 1), np.sort(P, 1)[:, -5:])
    assert np.allclose(np.sort(vc, 1), np.sort(P.T, 1)[:, -5:])


def test_label_scores_identity_full_and_reduced():
    rng = np.random.default_rng(60)          # not the stream of the labels, or every tile is pure
    bins = _labels(1000, ORG, 6)
    tile = rng.integers(0, 100, 1000)
    tile = np.unique(tile, return_inverse=True)[1]
    n = tile.max() + 1
    comp = m.composition(bins, tile, n)
    red = m.onehot(m.majority(comp))
    s, C_full, C_red = m.label_scores(np.eye(n), comp, comp, red, red, min_bins=1)
    assert s["kappa_bal_red"] == pytest.approx(1.0)
    assert s["kappa_bal"] < 1.0           # impure tiles: full resolution sees the mixing
    assert C_full.sum() == pytest.approx(1000)
    ps, _, _ = m.prior_scores(comp, comp, red, red, min_bins=1)
    assert ps["kappa_bal"] == pytest.approx(0.0, abs=1e-9)


def test_foldover_3d_target_plane_and_fold():
    rng = np.random.default_rng(7)
    x = rng.random((400, 2))
    eye = np.eye(400)
    # the query laid into a tilted plane in 3D: no fold-over, orientation undefined
    R = np.array([[1.0, 0.0], [0.0, 0.8], [0.0, 0.6]])
    flat = m.fold_overs(x, x @ R.T, eye)
    assert flat["foldover"] == 0 and flat["orientation"] == 0
    # fold the left fifth back over the plane
    y = x.copy()
    left = y[:, 0] < 0.2
    y[left, 0] = 0.4 - y[left, 0]
    fo = m.fold_overs(x, np.c_[y, 0.05 * left], eye)
    assert 0.05 < fo["foldover"] < 0.4


def test_target_presence_override_keeps_missing_organs_in_K():
    # the target section has no Heart, but the target embryo has: Heart stays in K and scores 0
    q = m.onehot(["Brain"] * 50 + ["Heart"] * 50)
    t = m.onehot(["Brain"] * 100)
    P = np.full((100, 100), 1e-4)
    own = m.label_scores(P, q, t, q, t, min_bins=1)[0]
    emb = m.present(m.onehot(["Brain", "Heart"]), 1)
    over = m.label_scores(P, q, t, q, t, min_bins=1, present_t=(emb, emb))[0]
    assert own["K"] == 1 and over["K"] == 2
    assert over["oms"] == pytest.approx(0.5) and over["kappa_bal"] == pytest.approx(0.0)
