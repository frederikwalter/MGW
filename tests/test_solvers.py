"""The two solver properties the harness relies on: one balanced solve serves both directions (P and P^T), and
the mirror image of a query gives the same FGW coupling (so A does not try it)."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from mgw.bench import solvers


@pytest.fixture(scope="module")
def two_sets():
    rng = np.random.default_rng(0)
    xa = rng.random((70, 2))
    xb = xa[:60] @ np.array([[0.8, -0.6], [0.6, 0.8]]) + rng.normal(0, 0.02, (60, 2))
    fa = np.c_[xa, rng.normal(size=(70, 3))]
    fb = np.c_[xb, rng.normal(size=(60, 3))]
    M = cdist(fa, fb)
    return xa, xb, M / np.quantile(M, 0.99)


P_PARAMS = solvers.FGWParams(alpha=0.5, epsilon=5e-3, inner_maxit=3000, outer_maxit=200)


def test_fgw_transpose_symmetry(two_sets):
    xa, xb, M = two_sets
    P_ab, _ = solvers.fgw(xa, xb, M, P_PARAMS)
    P_ba, _ = solvers.fgw(xb, xa, M.T, P_PARAMS)
    assert np.abs(P_ab - P_ba.T).sum() / P_ab.sum() < 1e-3
    o_ab = solvers.fgw_objective(xa, xb, M, P_ab, 0.5)
    o_ba = solvers.fgw_objective(xb, xa, M.T, P_ba, 0.5)
    assert o_ab == pytest.approx(o_ba, rel=1e-4)


def test_fgw_mirror_invariance(two_sets):
    xa, xb, M = two_sets
    P, _ = solvers.fgw(xa, xb, M, P_PARAMS)
    P_m, _ = solvers.fgw(xa * [-1, 1], xb, M, P_PARAMS)
    assert np.abs(P - P_m).sum() / P.sum() < 1e-4


def test_fgw_info_and_marginals(two_sets):
    xa, xb, M = two_sets
    P, info = solvers.fgw(xa, xb, M, P_PARAMS)
    assert {"converged", "n_outer", "outer_stopped", "inner_converged", "seconds"} <= set(info)
    assert np.allclose(P.sum(1), 1 / len(xa), rtol=1e-2)


def test_cost_scale_subsample_close():
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(300, 5)), rng.normal(size=(400, 5))
    exact = np.quantile(cdist(A, B), 0.99)
    assert solvers.cost_scale(A, B, max_entries=30_000) == pytest.approx(exact, rel=0.05)


P_FAST = solvers.FGWParams(alpha=0.5, epsilon=5e-3, inner_maxit=1000, outer_maxit=40)


@pytest.fixture(scope="module")
def slab_pair():
    """3D: b covers only the lower half (in z) of a's box; features follow the position."""
    rng = np.random.default_rng(2)
    xa = rng.random((160, 3)) * [1, 0.8, 0.6]
    xb = rng.random((110, 3)) * [1, 0.8, 0.3]
    f = lambda x: np.c_[np.sin(3 * x[:, 0]), np.cos(4 * x[:, 1]), 3 * x[:, 2]]
    M = cdist(f(xa) + rng.normal(0, .05, (160, 3)), f(xb) + rng.normal(0, .05, (110, 3)))
    return xa, xb, M / np.quantile(M, 0.99), xa[:, 2] < 0.3


def test_unbalanced_fgw_3d_rejects_the_unmatched_depth(slab_pair):
    xa, xb, M, inside = slab_pair
    P, info = solvers.fgw(xa, xb, M, P_FAST, tau_a=0.95, tau_b=0.95)
    assert np.isfinite(P).all() and "ugw_fallback" in info
    rows = P.sum(1) * len(xa)
    assert rows[~inside].mean() < 0.5 * rows[inside].mean()          # a's mass outside b's depth range is rejected
    # equal tau on both sides: one solve serves both directions
    P2, _ = solvers.fgw(xb, xa, M.T, P_FAST, tau_a=0.95, tau_b=0.95)
    assert np.abs(P - P2.T).sum() / P.sum() < 0.05


def test_stage2_aligners_and_masses(slab_pair):
    from mgw.bench import volume
    xa, xb, M, inside = slab_pair
    Fa, Fb = xa * [1, 1, 3], xb * [1, 1, 3]
    a = np.r_[np.full(80, 2.0), np.ones(80)]
    for aligner in ("fgw", "linear"):
        P, obj, info = volume.stage2(aligner, xa, xb, Fa, Fb, 1.0, P_PARAMS, a=a / a.sum(), tau=0.95,
                                     aligner_eps=5e-3)
        rej_a, rej_b, mass = volume.rejected(P, a, None)
        assert P.shape == (160, 110) and np.isfinite(obj) and 0 <= rej_a < 1 and 0 <= rej_b < 1 and mass > 0
    with pytest.raises(ValueError):
        volume.stage2("bogus", xa, xb, Fa, Fb, 1.0, P_PARAMS)
