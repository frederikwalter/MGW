"""2D (and 2D -> 3D) couplings without notebook globals: FGW and the null aligners.

Every solver returns (P, info) with P (n_a x n_b) and info a dict (converged, iterations, seconds). The objective
of each aligner is what a pipeline selects candidates by:
- fgw:         alpha * GW(Ca/max, Cb/max, P) + (1 - alpha) * <M, P>   (fgw_objective, as in the 3D notebook)
- linear:      <M, P>                                                  (features only, no geometry)
- spatial_gw:  GW(Ca/max, Cb/max, P)                                   (geometry only, no features)
- scot:        GW(Fa/max, Fb/max, P), F the feature distances          (SCOT: feature geometry only)
"""
import time
from dataclasses import dataclass, fields

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from mgw import stack3d as s3
from mgw.gw import solve_gw_ott


@dataclass(frozen=True)
class FGWParams:
    alpha: float = 0.5          # weight of the GW term; 1 - alpha on the cross cost
    epsilon: float = 5e-3
    inner_maxit: int = 3000
    outer_maxit: int = 200
    inner_tol: float = 1e-7
    outer_tol: float = 1e-7

    @classmethod
    def from_mapping(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in names})


# ---------- geometry helpers
def centred(xy, scale=None):
    """Centre a point set and divide by `scale` (default: its RMS radius)."""
    xy = np.asarray(xy, dtype=np.float64)
    xy = xy - xy.mean(0)
    return xy / (scale or s3.rms_radius(xy))


def light(xy, emb):
    """A minimal AnnData for the solvers: coordinates and one embedding."""
    a = ad.AnnData(obs=pd.DataFrame(index=[f"p{i}" for i in range(len(xy))]))
    a.obsm["spatial"], a.obsm["X_emb"], a.obsm["X_pca"] = np.asarray(xy, float), np.asarray(emb, float), np.asarray(emb, float)
    return a


def dist_norm(x):
    """Pairwise Euclidean distances divided by their maximum."""
    C = cdist(x, x)
    return C / C.max()


def gw_energy(C1, C2, P):
    p, q = P.sum(1), P.sum(0)
    return float(p @ (C1 * C1) @ p + q @ (C2 * C2) @ q - 2 * ((C1 @ P) * (P @ C2)).sum())


def fgw_objective(xa, xb, M, P, alpha, Ca=None, Cb=None):
    """The FGW objective of P (normalized to mass 1) with max-normalized distances (Ca, Cb: precomputed)."""
    Ca = dist_norm(xa) if Ca is None else Ca
    Cb = dist_norm(xb) if Cb is None else Cb
    P = P / P.sum()
    return alpha * gw_energy(Ca, Cb, P) + (1 - alpha) * float((M * P).sum())


def cost_scale(Fq, Ft, q=0.99, max_entries=20_000_000, seed=0):
    """The q quantile of the feature distances between Fq and Ft, from a fixed subsample of Ft if large."""
    Ft = np.asarray(Ft)
    n_cols = max(1, min(len(Ft), max_entries // max(len(Fq), 1)))
    if n_cols < len(Ft):
        Ft = Ft[np.random.default_rng(seed).choice(len(Ft), n_cols, replace=False)]
    return float(np.quantile(cdist(Fq, Ft), q))


# ---------- solvers
def fgw(xa, xb, M, p: FGWParams, *, tau_a=1.0, tau_b=1.0, Ca=None, Cb=None):
    """Entropic fused GW of two point sets (OTT), distances max-normalized: the notebook's solve_2d."""
    Ca = dist_norm(xa) if Ca is None else Ca
    Cb = dist_norm(xb) if Cb is None else Cb
    t0 = time.perf_counter()
    P, info = solve_gw_ott(Ca, Cb, M=M, alpha=(1 - p.alpha) / p.alpha, epsilon=p.epsilon, tau_a=tau_a, tau_b=tau_b,
                           inner_maxit=p.inner_maxit, outer_maxit=p.outer_maxit, inner_tol=p.inner_tol,
                           outer_tol=p.outer_tol, verbose=False, return_info=True)
    info["seconds"] = time.perf_counter() - t0
    return np.asarray(P, dtype=np.float64), info


def gw(C1, C2, p: FGWParams, epsilon=None):
    """Entropic GW (no linear term) on two max-normalized cost matrices."""
    t0 = time.perf_counter()
    P, info = solve_gw_ott(C1, C2, epsilon=epsilon or p.epsilon, inner_maxit=p.inner_maxit,
                           outer_maxit=p.outer_maxit, inner_tol=p.inner_tol, outer_tol=p.outer_tol,
                           verbose=False, return_info=True)
    info["seconds"] = time.perf_counter() - t0
    return np.asarray(P, dtype=np.float64), info


def linear_ot(M, epsilon=5e-3, maxit=3000, tol=1e-7):
    """Entropic OT on the cross cost only (uniform marginals)."""
    import jax.numpy as jnp
    from ott.geometry import geometry
    from ott.problems.linear import linear_problem
    from ott.solvers.linear.sinkhorn import Sinkhorn
    t0 = time.perf_counter()
    geom = geometry.Geometry(cost_matrix=jnp.asarray(M, dtype=jnp.float32), epsilon=epsilon)
    out = Sinkhorn(max_iterations=maxit, threshold=tol)(linear_problem.LinearProblem(geom))
    P = np.asarray(out.matrix, dtype=np.float64)
    return P, dict(converged=bool(out.converged), n_outer=int(out.n_iters), seconds=time.perf_counter() - t0)


def rigid_icp(xq, xt, fq, ft, P0, lam=0.5, n_iter=15, epsilon=5e-3):
    """Rigid placement only: OT-ICP of the target onto the query (rotation or reflection, the lower cost), started
    from P0. Both sides are scaled to an RMS radius of 1 (a rigid map cannot fix size). Returns the ICP coupling."""
    t0 = time.perf_counter()
    reg = s3.register_pair(centred(xq), centred(xt), fq, ft, P0, flip="auto", lam=lam, n_iter=n_iter,
                           epsilon=epsilon)
    return np.asarray(reg["P"], dtype=np.float64), dict(converged=True, n_outer=len(reg["history"]),
                                                       reflected=bool(reg["reflected"]),
                                                       seconds=time.perf_counter() - t0)


def moscot_pair(xa, fa, xb, fb, alpha=0.8, epsilon=1e-2):
    """FGW of two sections of one embryo on their shared embedding (moscot, spatial weight alpha)."""
    from validation.moscot_align import align_moscot
    P, _ = align_moscot(light(xa, fa), light(xb, fb), joint_attr="X_emb", alpha=alpha, epsilon=epsilon,
                        verbose=False)
    return P
