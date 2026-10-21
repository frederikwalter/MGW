"""Stacks as 3D measures, stage 2 between two of them, and the way back to the real spots
(docs/st_3d_alignment_strategies.md §1d–§1f, §2c, §2d).

- Sub-stacks (G1 leaves a section out, G2 keeps every other one): the sections keep their registered positions in
  the full stack's frame; where two kept sections were not neighbours, their displacement pairs come from a direct
  coupling of the two (`stack.skip_coupling`, cached per embryo next to the stack).
- The point cloud (§1e, the ablation without a field): the registered stack points with masses proportional to the
  thickness of each section's slab (half the gap to each neighbour; an end section gets its inner half only; slabs
  are floored at 5 % of the mean, so a zero gap does not delete a section), the real features, subsampled per section
  in proportion to its spots when there are more than N points.
- Stage 2: one solve per pair of measures, FGW (or the nulls: feature-only OT, geometry-only GW, SCOT) on
  max-normalized Euclidean distances in any dimension, unbalanced on both sides with the same tau, so the coupling
  P (a x b) and its transpose serve both directions.
- Back to the spots (§1f): A_q (query spots x query samples) spreads every real spot over its k nearest samples of
  its own volume with Gaussian weights at the sample spacing (rows sum to 1); B_t (target samples x target spots)
  pools the bins of the k nearest real target spots (1/k each). The scored coupling of a query section is A_q P,
  against the target samples carrying B_t T (compositions), B_t Y (held-out genes) and B_t L (laterality).
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from mgw import stack3d as s3
from mgw.bench import solvers, stack as bstack, store

PER_POINT = ("xyz", "section", "labels", "comp", "Y", "L")


# ---------- sub-stacks
def skip_pairs(cfg, root, seed, embryo, ST, log=print):
    """{(k, k + 2): (i, j, w)} displacement pairs (indices local to the two sections) of the direct coupling of every
    pair of sections two apart, cached as cache/stacks/<embryo>/seed<s>_<stack hash>_skip.npz."""
    f = Path(root) / "cache" / "stacks" / embryo / f"seed{seed}_{cfg.stack_hash()}_skip.npz"
    n = len(ST["names"])
    if f.exists():
        z = np.load(f)
        return {(k, k + 2): (z[f"p{k}"][:, 0].astype(np.int64), z[f"p{k}"][:, 1].astype(np.int64), z[f"p{k}"][:, 2])
                for k in range(n - 2)}
    params = bstack.StackParams.from_mapping(cfg.raw["stack"])
    off, X = ST["offsets"], ST["emb"]["X_feat"]
    out = {}
    for k in range(n - 2):
        a, b = k, k + 2
        log(f"  skip coupling {ST['names'][a]} x {ST['names'][b]}")
        P = bstack.skip_coupling(ST["reg_xy"][a], ST["reg_xy"][b], X[off[a]:off[a + 1]], X[off[b]:off[b + 1]], params)
        out[a, b] = s3.displacement_pairs(P)
    store.savez(f, **{f"p{a}": np.c_[i, j, w] for (a, _), (i, j, w) in out.items()})
    return out


def substack(ST, keep, skips=None, z=None):
    """The stack restricted to the sections `keep` (indices in stack order, increasing), in the full stack's frame.

    Neighbouring kept sections take the full stack's displacement pairs if they were neighbours there, else the
    skip pairs {(a, b): (i, j, w)} (local indices). z: the kept sections' depths (default: the full stack's).
    Per-point arrays are sliced; `rows` maps the substack's points to the full stack's."""
    off, keep = ST["offsets"], list(keep)
    rows = np.concatenate([np.arange(off[k], off[k + 1]) for k in keep])
    n_obs = [off[k + 1] - off[k] for k in keep]
    new_off = np.r_[0, np.cumsum(n_obs)]
    zk = np.asarray(ST["z"], dtype=np.float64)[keep] if z is None else np.asarray(z, dtype=np.float64)
    out = dict(names=[ST["names"][k] for k in keep], z=zk, offsets=new_off, rows=rows, frame=ST["frame"],
               emb={e: F[rows] for e, F in ST["emb"].items()},
               reg_xy=[ST["xyz"][off[k]:off[k + 1], :2] for k in keep])
    for key in PER_POINT:
        if key in ST:
            out[key] = np.asarray(ST[key])[rows].copy()
    out["xyz"][:, 2] = np.repeat(zk, n_obs)
    pairs = []
    for m in range(len(keep) - 1):
        a, b = keep[m], keep[m + 1]
        if b == a + 1:
            i, j, w = ST["pairs"][a]
            i, j = i - off[a], j - off[b]
        else:
            i, j, w = skips[a, b]
        pairs.append((i + new_off[m], j + new_off[m + 1], w))
    out["pairs"] = pairs
    return out


def section_index(ST):
    return np.repeat(np.arange(len(ST["names"])), np.diff(ST["offsets"]))


# ---------- the point cloud (§1e ablation)
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


# ---------- back to the real spots (§1f)
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
def stage2(aligner, Xa, Xb, Fa, Fb, scale, p, *, a=None, b=None, tau=1.0, epsilon=None, aligner_eps=None,
           unbalanced_correction=True):
    """One coupling of two 3D measures (rows: a). The cross cost is cdist(Fa, Fb) / scale (one scale per pair).
    Returns (P, objective, info)."""
    M = cdist(Fa, Fb) / scale
    Ca, Cb = solvers.dist_norm(Xa), solvers.dist_norm(Xb)
    kw = dict(a=a, b=b, tau_a=tau, tau_b=tau)
    if aligner == "fgw":
        P, info = solvers.fgw(None, None, M, p, Ca=Ca, Cb=Cb, epsilon=epsilon,
                              unbalanced_correction=unbalanced_correction, **kw)
        obj = solvers.fgw_objective(None, None, M, P, p.alpha, Ca=Ca, Cb=Cb)
    elif aligner == "linear":
        P, info = solvers.linear_ot(M, epsilon or aligner_eps, p.inner_maxit, p.inner_tol, **kw)
        obj = float((M * P).sum() / P.sum())
    elif aligner in ("spatial_gw", "scot"):
        C1, C2 = (Ca, Cb) if aligner == "spatial_gw" else (solvers.dist_norm(Fa), solvers.dist_norm(Fb))
        P, info = solvers.gw(C1, C2, p, epsilon or aligner_eps, unbalanced_correction=unbalanced_correction, **kw)
        obj = solvers.gw_energy(C1, C2, P / P.sum())
    else:
        raise ValueError(f"stage 2 aligner {aligner!r}")
    if not np.isfinite(P).all():
        raise FloatingPointError(f"stage 2 ({aligner}) returned a non-finite coupling")
    return P, obj, info


def rejected(P, a=None, b=None):
    """(rejected share of side a, of side b, total transported mass), masses normalized to 1."""
    a = np.full(P.shape[0], 1 / P.shape[0]) if a is None else np.asarray(a) / np.sum(a)
    b = np.full(P.shape[1], 1 / P.shape[1]) if b is None else np.asarray(b) / np.sum(b)
    r, c = P.sum(1), P.sum(0)
    return float(np.clip(a - r, 0, None).sum()), float(np.clip(b - c, 0, None).sum()), float(P.sum())


# ---------- consistency of a query embryo's sections (§2d)
def plane_normal(Y, w=None):
    """Unit normal of the weighted best-fit plane through points Y (n x 3)."""
    w = np.ones(len(Y)) if w is None else np.asarray(w, dtype=np.float64)
    c = w @ Y / w.sum()
    return np.linalg.svd((Y - c) * np.sqrt(w / w.sum())[:, None], full_matrices=False)[2][2]


def parallel_sections(images, z_query):
    """images: per query section (stack order) its spots' barycentric images in the target (n x 3) and row masses.
    Returns (angle in degrees of each section's image normal to the mean normal, Spearman rho between the query's
    own z and the images' centroids along the mean normal: |rho| = 1 keeps the order)."""
    normals = [plane_normal(Y, w) if len(Y) >= 3 else np.full(3, np.nan) for Y, w in images]
    N = np.array(normals)
    ok = np.isfinite(N).all(1)
    if not ok.any():
        return np.full(len(images), np.nan), np.nan
    ref = N[ok][0]
    N[ok] *= np.sign(N[ok] @ ref)[:, None]
    mean = N[ok].mean(0)
    mean /= np.linalg.norm(mean)
    ang = np.degrees(np.arccos(np.clip(np.abs(N @ mean), 0, 1)))
    cen = np.array([w @ Y / w.sum() @ mean if len(Y) else np.nan for Y, w in images])
    rho = spearmanr(z_query, cen)[0] if len(images) >= 3 and np.isfinite(cen).all() else np.nan
    return ang, float(rho)
