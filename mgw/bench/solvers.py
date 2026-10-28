"""Couplings without notebook globals: entropic FGW (OTT) and the intra-domain costs it runs on.

fgw returns (P, info) with P (n_a x n_b) and info a dict (converged, iterations, seconds). The objective a pipeline
selects candidates by is alpha * GW(Ca, Cb, P) + (1 - alpha) * <M, P> (fgw_objective, as in the 3D notebook), with
Ca, Cb max-normalized intra-domain costs:
- Euclidean distances (dist_norm): FGW;
- shortest paths in a Euclidean kNN graph (knn_geodesic): Riemannian FGW;
- geodesics under the pull-back metric of a map phi from coordinates to features (pullback_geodesic; mgw_cost trains
  phi on one section): MGW.
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


def cost_scale(Fq, Ft, q=0.99, max_entries=20_000_000, seed=0, metric="euclidean"):
    """The q quantile of the feature distances (cdist metric) between Fq and Ft, from a fixed subsample of Ft if
    large."""
    Ft = np.asarray(Ft)
    n_cols = max(1, min(len(Ft), max_entries // max(len(Fq), 1)))
    if n_cols < len(Ft):
        Ft = Ft[np.random.default_rng(seed).choice(len(Ft), n_cols, replace=False)]
    return float(np.quantile(cdist(Fq, Ft, metric), q))


# ---------- solvers
def _gw_unbalanced(Ca, Cb, tau_a, tau_b, unbalanced_correction, **kw):
    """solve_gw_ott, retried once without OTT's unbalanced correction if the unbalanced solve returns NaN (it does
    at small epsilon and tau <= 0.9 in float32). info["ugw_fallback"] records the retry."""
    P, info = solve_gw_ott(Ca, Cb, tau_a=tau_a, tau_b=tau_b, unbalanced_correction=unbalanced_correction,
                           verbose=False, return_info=True, **kw)
    info["ugw_fallback"] = False
    if (tau_a < 1 or tau_b < 1) and unbalanced_correction and not np.isfinite(np.asarray(P)).all():
        P, info = solve_gw_ott(Ca, Cb, tau_a=tau_a, tau_b=tau_b, unbalanced_correction=False, verbose=False,
                               return_info=True, **kw)
        info["ugw_fallback"] = True
    return P, info


def fgw(xa, xb, M, p: FGWParams, *, tau_a=1.0, tau_b=1.0, Ca=None, Cb=None, a=None, b=None,
        unbalanced_correction=True, epsilon=None):
    """Entropic fused GW of two point sets (OTT), distances max-normalized: the notebook's solve_2d. Any dimension
    (2D sections, 3D volumes); masses a, b (default uniform) and tau < 1 for an unbalanced solve (§1e)."""
    Ca = dist_norm(xa) if Ca is None else Ca
    Cb = dist_norm(xb) if Cb is None else Cb
    t0 = time.perf_counter()
    P, info = _gw_unbalanced(Ca, Cb, tau_a, tau_b, unbalanced_correction, a=a, b=b, M=M,
                             alpha=(1 - p.alpha) / p.alpha, epsilon=epsilon or p.epsilon, inner_maxit=p.inner_maxit,
                             outer_maxit=p.outer_maxit, inner_tol=p.inner_tol, outer_tol=p.outer_tol)
    info["seconds"] = time.perf_counter() - t0
    return np.asarray(P, dtype=np.float64), info


def moscot_pair(xa, fa, xb, fb, alpha=0.8, epsilon=1e-2):
    """FGW of two sections of one embryo on their shared embedding (moscot, spatial weight alpha)."""
    from validation.moscot_align import align_moscot
    P, _ = align_moscot(light(xa, fa), light(xb, fb), joint_attr="X_emb", alpha=alpha, epsilon=epsilon,
                        verbose=False)
    return P


# ---------- intra-domain costs beyond Euclidean (Riemannian FGW, MGW)
def _torch_device(device=None):
    import torch
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _max_norm(D):
    D = np.maximum(np.asarray(D, dtype=np.float64), 0.0)
    np.fill_diagonal(D, 0.0)
    if not np.isfinite(D).all():
        raise ValueError(f"{(~np.isfinite(D)).sum()} non-finite geodesic distances")
    return D / D.max()


def knn_geodesic(X, k=12):
    """Shortest-path distances in the Euclidean kNN graph of X (any dimension), divided by their maximum."""
    from scipy.sparse.csgraph import dijkstra
    from mgw import geometry
    return _max_norm(dijkstra(geometry.knn_graph(np.asarray(X, dtype=np.float64), k=k), directed=False))


def pullback_geodesic(phi, X, k=12, eps=1e-2, device=None, dtype=None):
    """Geodesic distances under the pull-back metric J_phi^T J_phi of phi (a callable on n x d tensors of `dtype`,
    default float64), along the kNN graph of X, divided by their maximum: MGW's intra-domain cost
    (mgw.mgw_align_core)."""
    import torch
    from mgw import geometry
    X = np.asarray(X, dtype=np.float32 if dtype == torch.float32 else np.float64)
    G = geometry.pullback_metric_field(phi, torch.as_tensor(X, device=_torch_device(device)), eps=eps).cpu()
    return _max_norm(geometry.geodesic_distances_fast(X, G, geometry.knn_graph(X.astype(np.float64), k=k)))


def mgw_cost(xy, F, p, seed=0, device=None, log=None):
    """MGW's cost of one section: phi (coordinates -> features, z-scored per column) is trained on the section alone,
    then pullback_geodesic. p: the config's mgw block (widths, lr, niter, knn_k, geodesic_eps)."""
    import torch
    from mgw import models
    dev = _torch_device(device)
    X = centred(xy)
    F = np.asarray(F, dtype=np.float64)
    Y = (F - F.mean(0)) / np.where(F.std(0) > 0, F.std(0), 1.0)
    torch.manual_seed(int(seed))
    phi = models.PhiModel(X.shape[1], Y.shape[1], widths=tuple(p["widths"])).double()
    phi = models.train_phi(phi, torch.from_numpy(X), torch.from_numpy(Y), lr=p["lr"], niter=p["niter"],
                           print_every=p["niter"] + 1, device=dev)
    phi.eval()
    if log:
        with torch.no_grad():
            mse = float(((phi(torch.from_numpy(X).to(dev)).cpu().numpy() - Y) ** 2).mean())
        log(f"    mgw phi: {len(X)} points, {p['niter']} steps, mse {mse:.4f}")
    return pullback_geodesic(phi, X, p["knn_k"], p["geodesic_eps"], dev, torch.float64)
