import itertools

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from mgw import stack3d as s3
from mgw.bench import metrics, temporal as T


def _toy(seed=0, n1=40, n2=55):
    rng = np.random.default_rng(seed)
    xa, xb = rng.normal(size=(n1, 2)), rng.normal(size=(n2, 2)) * 1.2
    ea, eb = rng.normal(size=(n1, 5)), rng.normal(size=(n2, 5)) + 0.3
    return xa, xb, ea, eb


def _reference_loop(ea, eb, xa, xb, alpha, beta, gamma, eps, max_iter):
    """DeST_OT's LogSinkhorn_iteration (destot_opt.py), transcribed to numpy: one dual update per gradient."""
    C, C1, C2 = cdist(ea, eb), cdist(ea, ea), cdist(eb, eb)
    M1, M2 = cdist(xa, xa) * C1, cdist(xb, xb) * C2
    N1, N2 = C.shape
    nC = np.linalg.norm(C)
    p1, p2 = np.sqrt(nC) / np.linalg.norm(M1) * np.sqrt(N1), np.sqrt(nC) / np.linalg.norm(M2) * np.sqrt(N1)
    r2, r1 = nC / np.linalg.norm(M2 ** 2) * (N1 ** 1 / 2), nC / np.linalg.norm(M1 ** 2) * (N1 / N2 ** (1 / 2))
    g1, g2 = np.ones(N1) / N1, np.ones(N2) / N1
    P = np.outer(g1, g2)
    f, g = np.zeros(N1), np.zeros(N2)
    ubc = gamma / (gamma + eps)
    cost = lambda f, g, G: -(G - f[:, None] - g[None, :]) / eps
    for _ in range(max_iter):
        orig = r1 * (M1 ** 2 @ P) + r2 * (P @ M2 ** 2)
        gw = -2 * ((p1 * M1) @ P @ (p2 * M2).T) + (p1 * M1) ** 2 @ P @ np.ones((N2, N2))
        G = (1 - alpha) * C + alpha * (beta * orig + (1 - beta) * gw)
        f = ubc * (f + eps * (np.log(g1) - logsumexp(cost(f, g, G), axis=1)))
        g = g + eps * (np.log(g2) - logsumexp(cost(f, g, G), axis=0))
        P = np.exp(cost(f, g, G))
    return P


@pytest.mark.parametrize("alpha,beta,gamma,eps", [(0.2, 0.5, 50, 0.1), (0.99, 0.5, 1, 0.1), (0.6, 0.0, 10, 0.5),
                                                  (0.6, 1.0, 10, 0.5)])
def test_reference_parity(alpha, beta, gamma, eps):
    xa, xb, ea, eb = _toy(1)
    n1, n2 = len(xa), len(xb)
    P_ref = _reference_loop(ea, eb, xa, xb, alpha, beta, gamma, eps, 100)
    g1, g2 = np.full(n1, 1 / n1), np.full(n2, 1 / n1)
    c = T.destot_costs(ea, eb, xa, xb, g1, g2, "reference")
    P, info = T.destot(c, g1, g2, alpha=alpha, beta=beta, gamma=gamma, epsilon=eps, outer_maxit=100, inner_steps=1,
                       inner_maxit=1, inner_tol=0.0, tol=0.0, prox=0.0, anneal=0)
    assert info["n_outer"] == 100
    assert np.abs(P - P_ref).sum() / P_ref.sum() < 1e-3


def test_gradient_matches_autodiff_and_four_tensor():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    try:
        xa, xb, ea, eb = _toy(2, 6, 7)
        rng = np.random.default_rng(0)
        g1, g2 = rng.random(6) + 0.5, rng.random(7) + 0.5
        g1, g2 = g1 / g1.sum(), 1.3 * g2 / g2.sum()
        c = T.destot_costs(ea, eb, xa, xb, g1, g2, "balanced")
        P = rng.random((6, 7)) * 0.05
        # the quartet as the paper's 4-tensor sum
        L = (c.Q1[:, None, :, None] - c.Q2[None, :, None, :]) ** 2           # i j' k l'
        assert 0.5 * np.einsum("ijkl,ij,kl->", L, P, P) == pytest.approx(T.energy_terms(P, c.C, c.Q1, c.Q2, c.T1,
                                                                                         c.T2)[2])
        for alpha, beta in itertools.product((0.0, 0.3, 1.0), (0.0, 0.4, 1.0)):
            E = lambda X: ((1 - alpha) * jnp.sum(c.C * X) + alpha * (
                beta * 0.5 * (jnp.sum((c.T1 @ X) * X) + jnp.sum((X @ c.T2) * X))
                + (1 - beta) * 0.5 * (X.sum(1) @ (c.Q1 ** 2) @ X.sum(1) + X.sum(0) @ (c.Q2 ** 2) @ X.sum(0)
                                      - 2 * jnp.sum((c.Q1 @ X @ c.Q2) * X))))
            auto = np.asarray(jax.grad(E)(jnp.asarray(P)))
            assert np.allclose(T.gradient(P, c.C, c.Q1, c.Q2, c.T1, c.T2, alpha, beta), auto, atol=1e-10)
            assert float(E(jnp.asarray(P))) == pytest.approx(T.energy(P, c.C, c.Q1, c.Q2, c.T1, c.T2, alpha, beta))
    finally:
        jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("norm", ["balanced", "reference"])
def test_semirelaxed_marginals(norm):
    xa, xb, ea, eb = _toy(3)
    rng = np.random.default_rng(3)
    comp_a, comp_b = rng.integers(1, 9, (len(xa), 1)), rng.integers(1, 9, (len(xb), 1))
    g1, g2 = T.masses(comp_a, comp_b, "bins")
    assert g1.sum() == pytest.approx(1) and g2.sum() == pytest.approx(comp_b.sum() / comp_a.sum())
    c = T.destot_costs(ea, eb, xa, xb, g1, g2, norm)
    P, info = T.destot(c, g1, g2, alpha=0.4, beta=0.5, gamma=10, epsilon=0.05 if norm == "balanced" else 0.5)
    assert info["converged"] and info["finite"]
    assert np.abs(P.sum(0) - g2).sum() / g2.sum() < 1e-4
    assert (P.sum(1) - g1).sum() == pytest.approx(g2.sum() - g1.sum(), rel=1e-4)


def test_balanced_norm_terms_are_order_one():
    xa, xb, ea, eb = _toy(4, 300, 400)
    g1, g2 = np.full(300, 1 / 300), np.full(400, 1.5 / 400)
    c = T.destot_costs(ea, eb, xa, xb, g1, g2, "balanced")
    P0 = np.outer(g1, g2) / g1.sum()
    trip = c.T1 @ P0 + P0 @ c.T2
    quart = ((c.Q1 ** 2) @ P0.sum(1))[:, None] - 2 * (c.Q1 @ P0 @ c.Q2)
    for X in (trip, quart):
        assert 0.1 < np.abs(X).mean() < 10


def test_large_gamma_alpha0_matches_balanced_wot():
    from mgw.bench import selection
    xa, xb, ea, eb = _toy(5, 30, 30)
    g = np.full(30, 1 / 30)
    c = T.destot_costs(ea, eb, xa, xb, g, g, "balanced")
    P, _ = T.destot(c, g, g, alpha=0.0, gamma=np.inf, epsilon=0.05, tol=1e-7, outer_maxit=5000)
    Pw, _ = selection.wot(c.C, g, g, epsilon=0.05)
    assert np.abs(P - Pw).sum() < 1e-3


def _sim1d(n=51, w1=-10, w2=5):
    """The paper's 1D simulation: positions -N..N, type A left of the pivot (w1 at t1, w2 at t2), B right; features
    interpolate within each type (A grows from 16 to 31 spots, B shrinks from 35 to 20)."""
    N = n // 2
    s = np.arange(-N, N + 1, dtype=float)[:, None]
    rng = np.random.default_rng(0)
    v = rng.normal(size=(4, 4))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    va, vb = [np.r_[x, np.zeros(4)] for x in v[:2]], [np.r_[np.zeros(4), x] for x in v[2:]]

    def feats(w):
        typ = np.where(s[:, 0] <= w, 0, 1)
        F = np.zeros((n, 8))
        for i, x in enumerate(s[:, 0]):
            if typ[i] == 0:
                lam = (x + N) / (w + N)
                F[i] = (1 - lam) * va[0] + lam * va[1]
            else:
                lam = (x - w) / (N - w)
                F[i] = (1 - lam) * vb[0] + lam * vb[1]
        return F, typ
    (Fa, ta), (Fb, tb) = feats(w1), feats(w2)
    return s, Fa, ta, Fb, tb


def _organ_comp(typ):
    comp = np.zeros((len(typ), len(metrics.ORGANS)))
    names = [o for o in metrics.ORGANS if o not in ("Other", "Excluded")][:2]
    comp[np.arange(len(typ)), [metrics.ORGAN_INDEX[names[t]] for t in typ]] = 1
    return comp


def test_sim1d_growth_signs_and_distortion():
    s, Fa, ta, Fb, tb = _sim1d()
    g1, g2 = T.masses(_organ_comp(ta), _organ_comp(tb), "uniform")
    c = T.destot_costs(Fa, Fb, s, s, g1, g2, "balanced")
    P, _ = T.destot(c, g1, g2, alpha=0.2, beta=0.5, gamma=0.3, epsilon=0.05)
    P0, _ = T.destot(c, g1, g2, alpha=0.0, gamma=0.3, epsilon=0.05)
    rho = P.sum(1) / g1 - 1
    # A grows 16 -> 31 spots, B shrinks 35 -> 20
    assert rho[ta == 0].mean() == pytest.approx(31 / 16 - 1, abs=0.05)
    assert rho[ta == 1].mean() == pytest.approx(20 / 35 - 1, abs=0.05)
    w1, w2 = T.organ_weights(_organ_comp(ta), g1), T.organ_weights(_organ_comp(tb), g2)
    pres = w1.sum(0) > 0
    d = T.growth_distortion(rho, w1, w2.sum(0), pres, w2.sum(0) > 0)[1]
    d0 = T.growth_distortion(P0.sum(1) / g1 - 1, w1, w2.sum(0), pres, w2.sum(0) > 0)[1]
    assert d <= d0 * 1.05 + 1e-3


def test_growth_distortion_matches_reference_definition():
    """Uniform masses, one organ per tile: DeST_OT's metrics.growth_distortion_metric_helper (with and without T)."""
    rng = np.random.default_rng(6)
    n1, n2 = 30, 45
    ta, tb = rng.integers(0, 3, n1), rng.integers(0, 3, n2)
    P = rng.random((n1, n2))
    P *= (1 / n1) / P.sum(0)                                               # columns g2 = 1/n1
    xi = P.sum(1) - 1 / n1
    # the reference, no transitions
    ref = 0.0
    for p in np.intersect1d(ta, tb):
        m1, m2 = (ta == p).sum(), (tb == p).sum()
        ref += ((xi[ta == p] - (m2 - m1) / m1 / n1) ** 2).sum()
    ref *= n1
    # with the inferred transitions: T column-normalized, m_t <- T m_t
    Tm = np.array([[P[np.ix_(ta == p, tb == q)].sum() for q in range(3)] for p in range(3)])
    Tm /= Tm.sum(0)
    mt = Tm @ np.array([(tb == q).sum() for q in range(3)])
    ref_t = n1 * sum(((xi[ta == p] - (mt[p] - (ta == p).sum()) / (ta == p).sum() / n1) ** 2).sum() for p in range(3))
    comp_a, comp_b = _organ3(ta), _organ3(tb)
    g1, g2 = T.masses(comp_a, comp_b, "uniform")
    w1, w2 = T.organ_weights(comp_a, g1), T.organ_weights(comp_b, g2)
    gd, gd_nt = T.growth_distortion(P.sum(1) / g1 - 1, w1, w2.sum(0), w1.sum(0) > 0, w2.sum(0) > 0)
    assert gd_nt == pytest.approx(ref)
    assert gd == pytest.approx(ref_t)


def _organ3(typ):
    names = [o for o in metrics.ORGANS if o not in ("Other", "Excluded")][:3]
    comp = np.zeros((len(typ), len(metrics.ORGANS)))
    comp[np.arange(len(typ)), [metrics.ORGAN_INDEX[names[t]] for t in typ]] = 1
    return comp


def test_procrustes_migration_zero_for_rigid_copy():
    rng = np.random.default_rng(7)
    xa = rng.normal(size=(50, 2))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([1, -1])   # rotation + reflection
    xb = xa @ R + np.array([3.0, -1.0])
    Rf, t = T.weighted_procrustes(xa, xb, np.eye(50) / 50)
    assert np.abs(xb @ Rf + t - xa).max() < 1e-8


def test_topk_roundtrip():
    rng = np.random.default_rng(8)
    P = rng.random((20, 15)) ** 8
    idx, vals, cov = T.topk_columns(P, 5)
    D = T.dense_from_topk(idx, vals, 20)
    assert D.sum() == pytest.approx(vals.sum(dtype=np.float64), rel=1e-5)
    assert cov == pytest.approx(D.sum() / P.sum(), rel=1e-5)
    assert np.allclose(np.sort(D, 0)[-5:], np.sort(P, 0)[-5:], rtol=1e-6)
    idx, vals, cov = T.topk_columns(P, 50)
    assert cov == pytest.approx(1.0, rel=1e-6)


def test_run_names_roundtrip():
    for m, p in [("destot", dict(alpha=0.2, beta=0.5, gamma=50.0, epsilon=0.01)), ("expr", dict(gamma=10.0,
                 epsilon=0.003)), ("fgw_bal", dict(alpha=0.5, epsilon=0.005)), ("moscot", dict(alpha=0.8,
                 epsilon=0.01)), ("label", dict(gamma=250.0, epsilon=0.01))]:
        assert T.parse_run(T.run_name(m, p)) == (m, p)


def test_chain_pairs():
    from mgw.bench import config
    cfg = config.load()
    pairs = T.chain_pairs(T.load_chains(cfg.chains_path()), cfg.raw["temporal"]["stage_days"])
    assert [p.id for p in pairs] == [f"H{k}" for k in range(1, 7)] + [f"M{k}" for k in range(1, 7)]
    assert all(p.dt > 0 for p in pairs)
    assert pairs[0].a == "CS12-13_E2S1" and pairs[6].dataset == "MOSTA"
    assert T.parse_pairs("H2,E9.5_E2S1:E10.5_E1S1", cfg.raw["temporal"]["stage_days"], pairs)[1].dt == 1.0
