import numpy as np
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
#from ot.gromov import gwloss, init_matrix  # POT, for diagnostics only
import jax
import jax.numpy as jnp
from ott.geometry import geometry

def gw_distance(C1, C2, p, q):
    """
    Compute the (squared) GW objective value given cost matrices and coupling.
    Useful for monitoring, not for optimization itself.
    """
    # GW objective: sum_{i,i',j,j'} (C1_{ii'} - C2_{jj'})^2 * P_{ij} P_{i'j'}
    term = (C1[:, :, None, None] - C2[None, None, :, :]) ** 2
    val = np.sum(term * (p[:, None, :, None] * p[None, :, None, :]))  # if p==P (not just marginals)
    return val

def solve_gw(C1, C2, a=None, b=None,
             epsilon=1e-2,
             numItermax=500,
             tol=1e-6,
             verbose=True,
             use_ott=True,
             normalize=False,
            ):
    """
    Solve entropic Gromov–Wasserstein using OTT-JAX (preferred) or POT fallback.
    Compatible with recent OTT-JAX API (requires linear_solver).
    """
    n1, n2 = C1.shape[0], C2.shape[0]
    if a is None: a = np.ones(n1) / n1
    if b is None: b = np.ones(n2) / n2

    # normalize costs to comparable scale
    if normalize:
        q99 = np.quantile(np.concatenate([C1.ravel(), C2.ravel()]), 0.99)
        if q99 > 0:
            C1 = C1 / q99
            C2 = C2 / q99

    if use_ott:
        try:
            from ott.problems.quadratic import quadratic_problem
            from ott.solvers.linear.sinkhorn import Sinkhorn
            from ott.solvers.quadratic.gromov_wasserstein import GromovWasserstein
        except ImportError:
            use_ott = False
            if verbose:
                print("[solve_gw] Falling back to POT (OTT-JAX not installed).")

    if use_ott:
        # convert to JAX
        C1_j = jnp.asarray(C1, dtype=jnp.float32)
        C2_j = jnp.asarray(C2, dtype=jnp.float32)
        a_j  = jnp.asarray(a, dtype=jnp.float32)
        b_j  = jnp.asarray(b, dtype=jnp.float32)

        # geometries for the two cost matrices
        geom_x = geometry.Geometry(cost_matrix=C1_j, epsilon=None)
        geom_y = geometry.Geometry(cost_matrix=C2_j, epsilon=None)

        # quadratic problem
        prob = quadratic_problem.QuadraticProblem(geom_x, geom_y, a=a_j, b=b_j)

        # linear solver (Sinkhorn) required in OTT ≥0.5
        linear_solver = Sinkhorn(max_iterations=numItermax,
                                 threshold=tol)

        # Gromov–Wasserstein solver
        solver = GromovWasserstein(
            linear_solver=linear_solver,
            epsilon=epsilon,
            max_iterations=numItermax,
            threshold=tol
        )

        # solve
        out = solver(prob)
        P = np.array(out.matrix)
        if verbose:
            print(f"[OTT] GW converged: loss={float(out.primal_cost):.4e}, "
                  f"ε={epsilon}, iters={out.n_iters}")
        return P

    '''
    # ---- fallback: POT ----
    import ot
    if epsilon > 0:
        P = ot.gromov.entropic_gromov_wasserstein(
            C1, C2, a, b, loss_fun='square_loss',
            epsilon=epsilon, verbose=verbose, max_iter=numItermax)
    else:
        P = ot.gromov.gromov_wasserstein(
            C1, C2, a, b, loss_fun='square_loss',
            verbose=verbose, max_iter=numItermax
            )
    '''
    return P

_GW_SOLVERS = {}


def _gw_solver(epsilon, inner_tol, outer_tol, inner_maxit, outer_maxit, jit):
    """One (jitted) GW solver per setting: jax caches its compilation per input shape, so repeated solves of the
    same size compile once instead of on every call."""
    key = (float(epsilon), float(inner_tol), float(outer_tol), int(inner_maxit), int(outer_maxit), bool(jit))
    if key not in _GW_SOLVERS:
        lin = Sinkhorn(max_iterations=inner_maxit, threshold=inner_tol)
        gw = GromovWasserstein(linear_solver=lin, epsilon=epsilon, max_iterations=outer_maxit, threshold=outer_tol)
        _GW_SOLVERS[key] = jax.jit(gw) if jit else gw
    return _GW_SOLVERS[key]


def solve_gw_ott(C1, C2, a=None, b=None,
                        epsilon=1e-2, inner_tol=1e-7, outer_tol=1e-7,
                         tau_a=1, tau_b=1,
                        inner_maxit=4000, outer_maxit=1000, verbose=True,
                         jit=True, M=None, alpha=1.0, return_info=False):
    """Entropic GW between C1 and C2. With a cross cost M (n1 x n2) it is fused GW,
    with alpha weighting the linear term <M, P> against the quadratic one.
    return_info=True returns (P, info) with the solver's convergence flag and number of outer iterations."""
    n1 = C1.shape[0]
    n2 = C2.shape[0]
    
    if a is None: a = np.ones(n1)/n1
    if b is None: b = np.ones(n2)/n2
    
    # JAX arrays
    C1_j = jnp.asarray(C1, dtype=jnp.float32)
    C2_j = jnp.asarray(C2, dtype=jnp.float32)
    a_j  = jnp.asarray(a,  dtype=jnp.float32)
    b_j  = jnp.asarray(b,  dtype=jnp.float32)
    
    # Intra-domain geometries
    geom_x = geometry.Geometry(cost_matrix=C1_j, epsilon=epsilon)
    geom_y = geometry.Geometry(cost_matrix=C2_j, epsilon=epsilon)
    
    # Cross-domain geometry for the fused term (OTT: fused_penalty multiplies the linear term)
    geom_xy = None if M is None else geometry.Geometry(cost_matrix=jnp.asarray(M, dtype=jnp.float32), epsilon=epsilon)

    # Quadratic GW problem
    prob = quadratic_problem.QuadraticProblem(geom_x, geom_y, geom_xy=geom_xy, fused_penalty=alpha,
                                              a=a_j, b=b_j, tau_a=tau_a, tau_b=tau_b)
    
    out = _gw_solver(epsilon, inner_tol, outer_tol, inner_maxit, outer_maxit, jit)(prob)
    P = np.array(out.matrix)
    if return_info:
        costs = np.asarray(out.costs)
        done = costs[costs != -1]
        lin = np.asarray(out.linear_convergence)[:len(done)]
        # OTT's flag needs the outer loop to stop before max_iterations *and* every inner Sinkhorn to converge
        return P, dict(converged=bool(out.converged), n_outer=int(len(done)),
                       outer_stopped=bool(len(done) < outer_maxit),
                       inner_converged=float(np.mean(lin)) if len(lin) else float("nan"),
                       reg_cost=float(done[-1]) if len(done) else float("nan"))
    
    '''
    # Diagnostics (POT’s explicit GW loss; optional)
    constC, hC1, hC2 = init_matrix(C1, C2, a, b, loss_fun="square_loss")
    loss_val = gwloss(constC, hC1, hC2, P)
    if verbose:
        print(f"[OTT] GW  ε={epsilon:.2e}  iters={getattr(out,'n_iters',-1)}  loss≈{loss_val:.4e}")
    '''
    return P

