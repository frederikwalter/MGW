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
