"""Stacks as 3D measures, stage 2 between two of them, and the way back to the real spots.

- The measure of a stack: the field's sampled volume (equal masses, the field's features at the samples), or the
  point cloud: the registered stack points with masses proportional to the thickness of each section's slab (half the
  gap to each neighbour; an end section gets its inner half only; slabs are floored at 5 % of the mean, so a zero gap
  does not delete a section), the real features, subsampled per section in proportion to its spots when there are
  more than N points.
- Stage 2: one unbalanced FGW solve per pair of measures, with the same tau on both sides, so the coupling P (a x b)
  and its transpose serve both directions. The aligner sets the intra-volume cost (max-normalized):
  fgw: Euclidean distances; rfgw: shortest paths in a Euclidean kNN graph of the samples (Riemannian FGW); mgw:
  geodesics under the pull-back metric of the field's features (the field is MGW's phi).
- Back to the spots: A_q (query spots x query samples) spreads every real spot over its k nearest samples of its own
  volume with Gaussian weights at the sample spacing (rows sum to 1); B_t (target samples x target spots) pools the
  bins of the k nearest real target spots (1/k each). The scored coupling of a query section is A_q P, against the
  target samples carrying B_t T (compositions).
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from mgw.bench import solvers


def section_index(ST):
    return np.repeat(np.arange(len(ST["names"])), np.diff(ST["offsets"]))


# ---------- the point cloud (no field)
def slab_masses(z, floor=0.05):
    """Mass per section proportional to its slab: half the gap to each neighbour (floored at `floor` x the mean)."""
    g = np.diff(np.asarray(z, dtype=np.float64))
    slab = 0.5 * (np.r_[0.0, g] + np.r_[g, 0.0])
    if slab.sum() <= 0:
        return np.full(len(slab), 1 / len(slab))
    slab = np.maximum(slab, floor * slab.mean())
    return slab / slab.sum()


def points(ST, n_max, seed=0):
    """The point-cloud measure of a stack: dict(xyz, F, mass, rows (into ST), section). At most n_max points."""
    off = ST["offsets"]
    n_k = np.diff(off)
    rng = np.random.default_rng(seed)
    if n_k.sum() > n_max:
        quota = np.maximum(np.floor(n_max * n_k / n_k.sum()).astype(int), 1)
        rows = np.concatenate([off[k] + np.sort(rng.choice(n_k[k], quota[k], replace=False)) for k in range(len(n_k))])
    else:
        rows = np.arange(off[-1])
    sec = section_index(ST)[rows]
    slab = slab_masses(ST["z"])
    cnt = np.bincount(sec, minlength=len(n_k))
    mass = slab[sec] / cnt[sec]
    return dict(xyz=ST["xyz"][rows], F=ST["emb"]["X_ortho"][rows], mass=mass / mass.sum(), rows=rows, section=sec)


# ---------- the measures and the way back to the real spots
def spread(xq, xs, k, h):
    """A (n_q x n_s, sparse): every query point over its k nearest samples, Gaussian weights at spacing h."""
    k = min(k, len(xs))
    d, j = cKDTree(xs).query(xq, k=k)
    d, j = d.reshape(len(xq), k), j.reshape(len(xq), k)
    w = np.exp(-0.5 * (d / h) ** 2)
    w[w.sum(1) <= 1e-300, 0] = 1.0                       # far from every sample: the nearest one
    w /= w.sum(1, keepdims=True)
    return sp.csr_matrix((w.ravel(), (np.repeat(np.arange(len(xq)), k), j.ravel())), shape=(len(xq), len(xs)))


def pool(xs, xt, k):
    """B (n_s x n_t, sparse): every sample pools its k nearest real points (1/k each)."""
    k = min(k, len(xt))
    j = cKDTree(xt).query(xs, k=k)[1].reshape(len(xs), k)
    return sp.csr_matrix((np.full(j.size, 1 / k), (np.repeat(np.arange(len(xs)), k), j.ravel())),
                         shape=(len(xs), len(xt)))


def nearest_in_section(xq, xs, sec_s, sec_q):
    """A for the point cloud: every query spot carries its nearest kept point of its own section (the identity when
    nothing was dropped)."""
    cols = np.empty(len(xq), dtype=np.int64)
    for s in np.unique(sec_q):
        cand = np.flatnonzero(sec_s == s)
        q = np.flatnonzero(sec_q == s)
        cols[q] = cand[cKDTree(xs[cand]).query(xq[q])[1]]
    return sp.csr_matrix((np.ones(len(xq)), (np.arange(len(xq)), cols)), shape=(len(xq), len(xs)))


def measure(cfg, root, seed, pair_id, embryo, ST, representation, n, variant="full", log=print):
    """One side of stage 2: the field's sampled volume (equal masses, f at the samples) or the point cloud.
    Returns dict(kind, X, F, mass (None: uniform), spacing, ...)."""
    from mgw.bench import field
    if representation == "field":
        fld = field.get_field(cfg, root, seed, pair_id, embryo, ST, ST["emb"]["X_ortho"], variant, log)
        vol = field.get_volume(cfg, root, seed, pair_id, embryo, fld, n, variant)
        return dict(kind="field", X=vol["xyz"], F=vol["F"], mass=None, spacing=float(vol["spacing"]),
                    accept=float(vol["accept"]), field=fld)
    pc = points(ST, n, seed)
    return dict(kind="points", X=pc["xyz"], F=pc["F"], mass=pc["mass"], spacing=field.spot_spacing(ST),
                accept=np.nan, rows=pc["rows"], section=pc["section"])


def query_map(m, ST, s, k):
    """A_q of section s of ST: its spots x the measure's points (rows sum to 1)."""
    off = ST["offsets"]
    xq = ST["xyz"][off[s]:off[s + 1]]
    if m["kind"] == "field":
        return spread(xq, m["X"], k, m["spacing"])
    return nearest_in_section(xq, m["X"], m["section"], np.full(len(xq), s))


def target_pool(m, ST, k):
    """B_t: the measure's points x the real spots of ST (the samples pool their k nearest spots; the point cloud's
    points are spots)."""
    if m["kind"] == "field":
        return pool(m["X"], ST["xyz"], k)
    n = len(m["rows"])
    return sp.csr_matrix((np.ones(n), (np.arange(n), m["rows"])), shape=(n, len(ST["xyz"])))


# ---------- stage 2
def intra_cost(aligner, m, cfg):
    """The max-normalized intra-volume cost of a measure m for an aligner (fgw, rfgw, mgw)."""
    X = m["X"]
    if aligner == "fgw":
        return solvers.dist_norm(X)
    if aligner == "rfgw":
        return solvers.knn_geodesic(X, cfg.raw["rfgw"]["knn_k"])
    if aligner == "mgw":
        import torch
        from mgw.bench.field import FeatureMap
        fld = m["field"]
        p = cfg.raw["mgw"]
        # the field's own normalized coordinates (one scale for all axes, so the metric only changes by a constant)
        Xn = fld._x(X).cpu().numpy()
        return solvers.pullback_geodesic(FeatureMap(fld.net).eval(), Xn, p["knn_k"], p["geodesic_eps"], fld.device,
                                         torch.float32)
    raise ValueError(f"stage 2 aligner {aligner!r}")


def stage2(Ca, Cb, Fa, Fb, scale, p, *, a=None, b=None, tau=1.0, epsilon=None, unbalanced_correction=True,
           metric="euclidean"):
    """One unbalanced FGW coupling of two 3D measures (rows: a) with intra costs Ca, Cb; the cross cost is
    cdist(Fa, Fb, metric) / scale (one scale per pair and metric). Returns (P, objective, info)."""
    M = cdist(Fa, Fb, metric) / scale
    P, info = solvers.fgw(None, None, M, p, Ca=Ca, Cb=Cb, epsilon=epsilon, unbalanced_correction=unbalanced_correction,
                          a=a, b=b, tau_a=tau, tau_b=tau)
    if not np.isfinite(P).all():
        raise FloatingPointError("stage 2 returned a non-finite coupling")
    return P, solvers.fgw_objective(None, None, M, P, p.alpha, Ca=Ca, Cb=Cb), info


def rejected(P, a=None, b=None):
    """(rejected share of side a, of side b, total transported mass), masses normalized to 1."""
    a = np.full(P.shape[0], 1 / P.shape[0]) if a is None else np.asarray(a) / np.sum(a)
    b = np.full(P.shape[1], 1 / P.shape[1]) if b is None else np.asarray(b) / np.sum(b)
    r, c = P.sum(1), P.sum(0)
    return float(np.clip(a - r, 0, None).sum()), float(np.clip(b - c, 0, None).sum()), float(P.sum())
