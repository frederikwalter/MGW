"""Spatial alignment of two sections with moscot's AlignmentProblem (fused Gromov-Wasserstein).

Needs moscot (a project dependency).
"""
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def align_moscot(A, B, *, joint_attr="X_ortho", spatial_key="spatial", spatial_key_B=None, alpha=0.5, epsilon=1e-2,
                 rank=-1, cost="sq_euclidean", tau_a=1.0, tau_b=1.0, normalize_spatial=True, verbose=True,
                 **solve_kwargs):
    """Align section A (source) to section B (target) with moscot.

    The quadratic term compares the spatial distances within each section (`obsm[spatial_key]`, each
    normalized by moscot), the linear term the distances between `obsm[joint_attr]` of the two
    sections, which must be a shared feature space (e.g. orthologs.shared_embedding). `alpha` in (0, 1]
    weights the two: 1 is pure spatial GW, alpha -> 0 the pure linear problem. `rank > 0` uses the
    low-rank solver. Marginals are uniform; `tau_a` / `tau_b` < 1 relax them (unbalanced OT). Extra keyword
    arguments go to `AlignmentProblem.solve`.

    The coordinates may have any dimension. If A and B differ (a 2D section against a 3D stack), the
    lower-dimensional side is padded with zeros. moscot then standardizes each side with one scalar over
    all entries, padding included, so pass `normalize_spatial=False` with coordinates already in one scale.

    Returns (P, info): the dense coupling (n_A x n_B, rows in A's order, columns in B's) and a dict
    with the solver's `converged`, `cost` and `iterations`.
    """
    from moscot.problems.generic import GWProblem
    from moscot.problems.space import AlignmentProblem

    if alpha == 1:
        joint_attr = None  # pure GW
    if joint_attr is not None:
        for a, name in ((A, "A"), (B, "B")):
            if joint_attr not in a.obsm:
                raise KeyError(f"{name} has no obsm[{joint_attr!r}]")
        if A.obsm[joint_attr].shape[1] != B.obsm[joint_attr].shape[1]:
            raise ValueError(f"obsm[{joint_attr!r}] dims differ: {A.obsm[joint_attr].shape[1]} vs "
                             f"{B.obsm[joint_attr].shape[1]}; the linear term needs a shared space")

    # Only what moscot needs: the two sections stacked, with their coordinates and shared features
    src, tgt = "src", "tgt"
    xa = np.asarray(A.obsm[spatial_key], dtype=np.float64)
    xb = np.asarray(B.obsm[spatial_key_B or spatial_key], dtype=np.float64)
    d = max(xa.shape[1], xb.shape[1])
    pad = lambda x: np.hstack([x, np.zeros((len(x), d - x.shape[1]))])
    obsm = {spatial_key: np.vstack([pad(xa), pad(xb)])}
    if joint_attr is not None:
        obsm[joint_attr] = np.vstack([A.obsm[joint_attr], B.obsm[joint_attr]]).astype(np.float64)
    batch = pd.Categorical([src] * A.n_obs + [tgt] * B.n_obs, categories=[src, tgt])
    # moscot refuses an AnnData without variables; X is an empty placeholder
    obs = pd.DataFrame({"batch": batch}, index=[f"A_{i}" for i in range(A.n_obs)] + [f"B_{i}" for i in range(B.n_obs)])
    adata = ad.AnnData(sp.csr_matrix((len(obs), 1), dtype=np.float32), obs=obs, obsm=obsm)

    if joint_attr is None:
        # AlignmentProblem always adds a linear term (a PCA of X without joint_attr), so pure GW goes
        # through GWProblem; scale_cost ('mean' by default) makes each section's distances scale-free
        xy = {"attr": "obsm", "key": spatial_key}
        prob = GWProblem(adata).prepare(key="batch", x_attr=xy, y_attr=xy, policy="sequential", cost=cost)
        prob = prob.solve(epsilon=epsilon, rank=rank, tau_a=tau_a, tau_b=tau_b, **solve_kwargs)
    else:
        prob = AlignmentProblem(adata).prepare(batch_key="batch", spatial_key=spatial_key, joint_attr=joint_attr,
                                                policy="sequential", normalize_spatial=normalize_spatial, cost=cost)
        prob = prob.solve(alpha=alpha, epsilon=epsilon, rank=rank, tau_a=tau_a, tau_b=tau_b, **solve_kwargs)
    sol = prob[src, tgt].solution
    P = np.asarray(sol.transport_matrix, dtype=np.float64)
    info = dict(converged=bool(sol.converged), cost=float(sol.cost),
                iterations=int(getattr(sol, "n_iters", -1) or -1))
    if verbose:
        print(f"[moscot] {A.n_obs:,} x {B.n_obs:,}, alpha {alpha}, epsilon {epsilon}, rank {rank}: "
              f"converged {info['converged']}, cost {info['cost']:.4e}")
    return P, info
