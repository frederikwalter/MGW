"""A 3D stack from the serial sections of one embryo, and cutting planes through it.

Stage 1 registers neighbouring sections rigidly (OT-ICP), estimates the gaps between them and
stacks them in 3D. Stage 2 cuts virtual sections along arbitrary planes by displacement
interpolation along the couplings of neighbouring sections, and fits a plane to where a coupling
puts a section in the stack.

Conventions: coordinates are row vectors. A rigid map is (R, t) with y = x @ R + t; as a 3 x 3
homogeneous matrix T = [[R, 0], [t, 1]] it acts as [x, 1] @ T. A plane is z = z0 + g . (xy - c),
with c the stack centre and g = tan(theta_u) u + tan(theta_v) v for the stack's in-plane principal
axes u, v.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from mgw import geometry, util


# ---------- section order
def infer_order(D, start=None):
    """Shortest open path through all sections (Held-Karp, exact) for a symmetric dissimilarity D.

    Feasible up to about 15 sections. Returns the order as a list of indices.
    """
    D = np.asarray(D, dtype=np.float64)
    n = len(D)
    if n <= 2:
        return list(range(n))
    full = (1 << n) - 1
    cost = np.full((1 << n, n), np.inf)
    prev = np.full((1 << n, n), -1, dtype=np.int64)
    for j in range(n):
        if start is None or j == start:
            cost[1 << j, j] = 0.0
    for mask in range(1, 1 << n):
        for j in range(n):
            c = cost[mask, j]
            if not np.isfinite(c):
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                m2 = mask | (1 << k)
                if c + D[j, k] < cost[m2, k]:
                    cost[m2, k], prev[m2, k] = c + D[j, k], j
    j, mask, path = int(np.argmin(cost[full])), full, []
    while j >= 0:
        path.append(j)
        j, mask = int(prev[mask, j]), mask & ~(1 << j)
    return path[::-1]


def path_length(D, order):
    return float(sum(D[a, b] for a, b in zip(order[:-1], order[1:])))


# ---------- rigid registration
def homogeneous(R, t):
    d = len(t)
    T = np.eye(d + 1)
    T[:d, :d], T[d, :d] = R, t
    return T


def apply(T, x):
    d = x.shape[1]
    return x @ T[:d, :d] + T[d, :d]


def procrustes(blocks, reflect=None):
    """Rigid map of moving points onto fixed ones, pooled over blocks of (X_fixed, Y_moving, P).

    P (n_X x n_Y) weights the pairs. reflect: None for the best orthogonal map, False for a rotation,
    True for a reflection. Returns (R, t) with Y @ R + t ~ X.
    """
    W = sum(P.sum() for _, _, P in blocks)
    xbar = sum(P.sum(1) @ X for X, _, P in blocks) / W
    ybar = sum(P.sum(0) @ Y for _, Y, P in blocks) / W
    H = sum((Y - ybar).T @ (P.T @ (X - xbar)) for X, Y, P in blocks)  # d x d
    U, _, Vt = np.linalg.svd(H)
    if reflect is not None and (np.linalg.det(U @ Vt) < 0) != reflect:
        U[:, -1] *= -1
    R = U @ Vt
    return R, xbar - ybar @ R


def _median_scale(C):
    return C / (np.median(C) + 1e-12)


def rigid_ot_icp(xa, xb, fa, fb, P0, *, lam=0.5, n_iter=15, epsilon=5e-3, reflect=False, tol=1e-5, verbose=False):
    """Register section b onto section a: alternate a weighted Procrustes fit and a linear OT re-solve.

    The OT cost is lam * |x_a - (x_b R + t)|^2 + (1 - lam) * |f_a - f_b|^2, each term scaled to a median of 1,
    so once the sections roughly overlap a point can only pair with nearby points of similar expression.
    `reflect` fits a reflection instead of a rotation (a section mounted mirrored).
    Returns dict(R, t, P, cost, history), with xb @ R + t in a's frame and P (n_a x n_b).
    """
    xa, xb = np.asarray(xa, dtype=np.float64), np.asarray(xb, dtype=np.float64)
    Cf = _median_scale(cdist(fa, fb, "sqeuclidean"))
    P = np.asarray(P0, dtype=np.float64)
    hist, R, t = [], None, None
    for it in range(n_iter):
        R, t = procrustes([(xa, xb, P)], reflect=reflect)
        Cs = cdist(xa, xb @ R + t, "sqeuclidean")
        C = lam * Cs / (np.median(Cs) + 1e-12) + (1 - lam) * Cf
        P_new = util._solve_linear_ott(C, epsilon=epsilon)
        P_new = P_new / P_new.sum()
        cost = float((C * P_new).sum())
        rmsd = float(np.sqrt((Cs * P_new).sum()))
        hist.append(dict(iter=it, cost=cost, rmsd=rmsd))
        if verbose:
            print(f"  icp {it}: cost {cost:.4f}, rmsd {rmsd:.4f}")
        delta = np.abs(P_new - P).sum()
        P = P_new
        if delta < tol:
            break
    return dict(R=R, t=t, P=P, cost=hist[-1]["cost"], rmsd=hist[-1]["rmsd"], history=pd.DataFrame(hist))


def register_pair(xa, xb, fa, fb, P0, *, flip="auto", **kw):
    """rigid_ot_icp as a rotation and (flip="auto") as a reflection; keeps the lower OT cost.

    flip: "auto", False (rotation only) or True (reflection only). Returns the chosen result with
    `reflected` and both costs.
    """
    kinds = {"auto": (False, True), False: (False,), True: (True,)}[flip]
    runs = {k: rigid_ot_icp(xa, xb, fa, fb, P0, reflect=k, **kw) for k in kinds}
    best = min(runs, key=lambda k: runs[k]["cost"])
    out = runs[best]
    out["reflected"] = best
    out["cost_rotation"] = runs[False]["cost"] if False in runs else np.nan
    out["cost_reflection"] = runs[True]["cost"] if True in runs else np.nan
    return out


def compose_to_reference(pair_T, ref):
    """Transforms into the reference frame from the pairwise ones.

    pair_T[k] maps section k+1 into section k's frame (k = 0 .. n-2, sections in stack order).
    Returns T[k] mapping section k into the frame of section `ref`.
    """
    n = len(pair_T) + 1
    T = [None] * n
    T[ref] = np.eye(pair_T[0].shape[0]) if pair_T else np.eye(3)
    for k in range(ref + 1, n):          # k -> k-1 -> ... -> ref
        T[k] = pair_T[k - 1] @ T[k - 1]
    for k in range(ref - 1, -1, -1):     # k -> k+1 -> ... -> ref
        T[k] = np.linalg.inv(pair_T[k]) @ T[k + 1]
    return T


def global_sweeps(xy, T, couplings, n_sweeps=2, ref=None, flips=None):
    """Re-fit every section except `ref` to both registered neighbours at once (limits drift along the stack).

    xy: raw coordinates per section; couplings[k]: n_k x n_{k+1}; flips[k]: whether section k is mirrored
    relative to the reference. Returns the updated T.
    """
    n = len(xy)
    ref = n // 2 if ref is None else ref
    flips = flips or [False] * n
    T = [t.copy() for t in T]
    for _ in range(n_sweeps):
        for k in [k for k in range(n) if k != ref]:
            blocks = []
            if k > 0:
                blocks.append((apply(T[k - 1], xy[k - 1]), xy[k], couplings[k - 1]))
            if k < n - 1:
                blocks.append((apply(T[k + 1], xy[k + 1]), xy[k], couplings[k].T))
            T[k] = homogeneous(*procrustes(blocks, reflect=flips[k]))
    return T


# ---------- gaps between sections
def rms_radius(xy):
    xy = np.asarray(xy, dtype=np.float64)
    return float(np.sqrt(((xy - xy.mean(0)) ** 2).sum(1).mean()))


def inplane_variogram(xy_list, f_list, n_pairs=200_000, n_bins=40, r_max=None, seed=0):
    """Mean feature distance against in-plane distance, pooled over sections (made non-decreasing).

    Returns a DataFrame (r, gamma, n).
    """
    rng = np.random.default_rng(seed)
    rs, ds = [], []
    per = max(n_pairs // len(xy_list), 1)
    for xy, f in zip(xy_list, f_list):
        # half the pairs at random, half among near neighbours, so short distances are resolved
        i, j = rng.integers(0, len(xy), per // 2), rng.integers(0, len(xy), per // 2)
        k = min(30, len(xy) - 1)
        nn = cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]
        a = rng.integers(0, len(xy), per - per // 2)
        b = nn[a, rng.integers(0, k, len(a))]
        i, j = np.r_[i, a], np.r_[j, b]
        keep = i != j
        rs.append(np.linalg.norm(xy[i[keep]] - xy[j[keep]], axis=1))
        ds.append(np.linalg.norm(f[i[keep]] - f[j[keep]], axis=1))
    r, d = np.concatenate(rs), np.concatenate(ds)
    r_max = r_max or np.quantile(r, 0.5)
    edges = np.r_[0, np.quantile(r[r <= r_max], np.linspace(0, 1, n_bins + 1)[1:])]
    idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_bins - 1)
    ok = r <= r_max
    g = np.bincount(idx[ok], d[ok], n_bins) / np.maximum(np.bincount(idx[ok], minlength=n_bins), 1)
    centres = np.bincount(idx[ok], r[ok], n_bins) / np.maximum(np.bincount(idx[ok], minlength=n_bins), 1)
    vg = pd.DataFrame(dict(r=centres, gamma=np.maximum.accumulate(g), n=np.bincount(idx[ok], minlength=n_bins)))
    return vg[vg.n > 0].reset_index(drop=True)


def estimate_gap(xa, xb, fa, fb, vg):
    """Gap d between two registered sections under local isotropy.

    Each point of a is paired with its nearest point of b in the registered frame (and vice versa), at in-plane
    offset r_i and feature distance D_i. d solves mean_i gamma(sqrt(r_i^2 + d^2)) = mean_i D_i, with gamma the
    in-plane variogram. Returns dict(gap, D, gamma_r, saturated); saturated means D lies above the variogram's
    range and the gap is a lower bound.
    """
    r_list, D_list = [], []
    for (x1, f1), (x2, f2) in (((xa, fa), (xb, fb)), ((xb, fb), (xa, fa))):
        dist, j = cKDTree(x2).query(x1)
        r_list.append(dist)
        D_list.append(np.linalg.norm(f1 - f2[j], axis=1))
    r, D = np.concatenate(r_list), float(np.concatenate(D_list).mean())
    gamma = lambda s: np.interp(s, vg.r, vg.gamma)
    g_at_r = float(gamma(r).mean())
    if D <= g_at_r:
        return dict(gap=0.0, D=D, gamma_r=g_at_r, saturated=False)
    lo, hi = 0.0, float(vg.r.max())
    if float(gamma(np.sqrt(r ** 2 + hi ** 2)).mean()) < D:
        return dict(gap=hi, D=D, gamma_r=g_at_r, saturated=True)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(gamma(np.sqrt(r ** 2 + mid ** 2)).mean()) < D:
            lo = mid
        else:
            hi = mid
    return dict(gap=0.5 * (lo + hi), D=D, gamma_r=g_at_r, saturated=False)


# ---------- the stack
def displacement_pairs(P, offset_a=0, offset_b=0):
    """Sparse pairs of a coupling: each row's peak and each column's peak, weighted by their mass.

    Returns (i, j, w) with global indices (offsets added) and w summing to 1.
    """
    P = np.asarray(P)
    ia = np.arange(P.shape[0]); ja = P.argmax(1)
    jb = np.arange(P.shape[1]); ib = P.argmax(0)
    ij = np.unique(np.c_[np.r_[ia, ib], np.r_[ja, jb]], axis=0)
    w = P[ij[:, 0], ij[:, 1]]
    return ij[:, 0] + offset_a, ij[:, 1] + offset_b, w / w.sum()


def principal_axes(xy):
    """Centre and in-plane principal axes (u major, v minor) of a point cloud."""
    c = xy.mean(0)
    _, _, Vt = np.linalg.svd(xy - c, full_matrices=False)
    u, v = Vt[0], Vt[1]
    if np.linalg.det(np.c_[u, v]) < 0:
        v = -v
    return c, u, v


def stack_graph(xyz, section, pairs, k_in=12):
    """kNN graph within each section plus one edge per displacement pair between neighbouring sections.

    A plain 3D kNN graph has almost no edges between sections when the gaps exceed the bin spacing, so the
    geodesics would cross between sections at a few arbitrary points. Edge weights are Euclidean lengths.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    rows, cols = [], []
    for s in np.unique(section):
        idx = np.flatnonzero(section == s)
        G = geometry.knn_graph(xyz[idx, :2], k=min(k_in, len(idx) - 1)).tocoo()
        rows.append(idx[G.row]); cols.append(idx[G.col])
    for i, j, _ in pairs:
        rows += [i, j]; cols += [j, i]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    w = np.linalg.norm(xyz[rows] - xyz[cols], axis=1) + 1e-12
    G = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)
    return geometry._connect_components(xyz, G)


# ---------- planes and virtual sections
def plane(z0, theta_u=0.0, theta_v=0.0):
    """A plane in stack coordinates; angles in degrees."""
    return dict(z0=float(z0), theta_u=float(theta_u), theta_v=float(theta_v))


def plane_frame(pl, frame):
    """Normal n, in-plane basis e1, e2 (3D) and the gradient g of the plane (see the module docstring)."""
    c, u, v = frame["centre"], frame["u"], frame["v"]
    g = np.tan(np.radians(pl["theta_u"])) * u + np.tan(np.radians(pl["theta_v"])) * v
    e1 = np.r_[u, g @ u]; e1 /= np.linalg.norm(e1)
    e2 = np.r_[v, g @ v]; e2 -= (e2 @ e1) * e1; e2 /= np.linalg.norm(e2)
    n = np.cross(e1, e2)
    return n, e1, e2, g


def plane_coords(xyz, pl, frame):
    """Coordinates of 3D points in the plane's basis (2D), and their signed height above it."""
    n, e1, e2, g = plane_frame(pl, frame)
    o = np.r_[frame["centre"], pl["z0"]]
    d = xyz - o
    return np.c_[d @ e1, d @ e2], d @ n


def _voronoi_reduce(xy, w, max_obs, lloyd_iter=3, seed=0):
    """Assign points to <= max_obs tiles (random seeds, Lloyd steps). Returns the tile of each point."""
    n = len(xy)
    if n <= max_obs:
        return np.arange(n)
    seeds = np.random.default_rng(seed).choice(n, size=max_obs, replace=False)
    assign = cKDTree(xy[seeds]).query(xy)[1]
    for _ in range(lloyd_iter):
        assign = np.unique(assign, return_inverse=True)[1].ravel()
        ww = np.bincount(assign, w)
        centres = np.stack([np.bincount(assign, w * c) for c in xy.T], 1) / ww[:, None]
        assign = cKDTree(centres).query(xy)[1]
    return np.unique(assign, return_inverse=True)[1].ravel()


class VirtualSection:
    """A section cut from the stack along a plane.

    Each displacement pair (i in section k, j in section k+1) is a segment in 3D; where it crosses the plane,
    at t in [0, 1], it gives a point carrying (1 - t) * value_i + t * value_j. The points are merged into
    about max_obs tiles. `interp(M)` returns any per-stack-point matrix M on the tiles; `labels(lab)` the
    mass-weighted majority label per tile.
    """

    def __init__(self, xyz, pairs, pl, frame, max_obs=1000, lloyd_iter=3, seed=0):
        self.plane, self.frame = pl, frame
        n, e1, e2, g = plane_frame(pl, frame)
        c = frame["centre"]
        f = lambda p: p[:, 2] - pl["z0"] - (p[:, :2] - c) @ g   # signed height (along z)
        I, J, T, W = [], [], [], []
        last = len(pairs) - 1
        self.n_pairs = []
        for k, (i, j, w) in enumerate(pairs):
            fi, fj = f(xyz[i]), f(xyz[j])
            # half-open in t: a plane through a section belongs to the interval that starts there
            hit = ((fi <= 0) & (fj > 0)) | ((fi >= 0) & (fj < 0))
            if k == last:
                hit |= (fj == 0) & (fi != 0)
            t = np.where(hit, fi / np.where(fi - fj == 0, 1, fi - fj), 0.0)
            I.append(i[hit]); J.append(j[hit]); T.append(np.clip(t[hit], 0, 1)); W.append(w[hit])
            self.n_pairs.append(len(i))
        self.i, self.j, self.t, self.w = (np.concatenate(a) for a in (I, J, T, W))
        self.coverage = min(len(self.i) / max(np.median(self.n_pairs), 1), 1.0)
        if len(self.i) == 0:
            self.xyz = np.zeros((0, 3)); self.xy = np.zeros((0, 2)); self.tile = np.zeros(0, dtype=int)
            self.n_obs = 0
            return
        p = (1 - self.t)[:, None] * xyz[self.i] + self.t[:, None] * xyz[self.j]
        xy, _ = plane_coords(p, pl, frame)
        self.tile = _voronoi_reduce(xy, self.w, max_obs, lloyd_iter, seed)
        self.n_obs = int(self.tile.max()) + 1
        A = sp.csr_matrix((self.w, (self.tile, np.arange(len(self.tile)))), shape=(self.n_obs, len(self.tile)))
        self._A = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
        self.xyz = self._A @ p
        self.xy = self._A @ xy

    def interp(self, M):
        """Values of a per-stack-point matrix (n_stack x d), dense or sparse, on the tiles."""
        if self.n_obs == 0:
            return np.zeros((0, M.shape[1]))
        Mi, Mj = M[self.i], M[self.j]
        if sp.issparse(M):
            V = sp.diags(1 - self.t) @ Mi + sp.diags(self.t) @ Mj
            return np.asarray((self._A @ V).toarray())
        return self._A @ ((1 - self.t)[:, None] * Mi + self.t[:, None] * Mj)

    def interp_rescaled(self, F, centre=None):
        """Features on the tiles, interpolated without shrinking towards the mean.

        Mixing two points, (1 - t) f_i + t f_j, and averaging the points of a tile both pull the features
        towards the mean: the mix of two different cells is less distinct than either. A query then sits
        closer to it on average (E|q - f|² grows with |f - mean|²), so planes between sections, where more
        is mixed, look better than planes through them. Here the mixed deviation from the stack mean is
        rescaled to the mixed length of the deviations it came from. Only the length changes, not the
        direction.
        """
        if self.n_obs == 0:
            return np.zeros((0, F.shape[1]))
        F = np.asarray(F, dtype=np.float64)
        m = F.mean(0) if centre is None else centre
        di, dj = F[self.i] - m, F[self.j] - m
        mix = self._A @ ((1 - self.t)[:, None] * di + self.t[:, None] * dj)
        length = self._A @ ((1 - self.t) * np.linalg.norm(di, axis=1) + self.t * np.linalg.norm(dj, axis=1))
        return m + mix * (length / (np.linalg.norm(mix, axis=1) + 1e-12))[:, None]

    def interp_cost(self, Fq, F):
        """Cross cost (n_q x n_tiles) between query features Fq and the tiles, interpolated as a cost.

        Each crossing point costs (1 - t) |q - f_i| + t |q - f_j|; tiles average their points by mass.
        This is never smaller than the cost to the mixed features (convexity) and leans the other way:
        a query that matches one endpoint well is pulled up by the other, so planes through real sections
        win. Kept for comparison; `interp_rescaled` is the unbiased choice.
        """
        if self.n_obs == 0:
            return np.zeros((len(Fq), 0))
        D = (1 - self.t) * cdist(Fq, F[self.i]) + self.t * cdist(Fq, F[self.j])
        return np.asarray((self._A @ D.T).T)

    def labels(self, lab):
        """Mass-weighted majority of the nearer endpoint's label in each tile."""
        lab = np.asarray(lab, dtype=str)
        near = np.where(self.t < 0.5, lab[self.i], lab[self.j])
        classes, codes = np.unique(near, return_inverse=True)
        votes = sp.csr_matrix((self.w, (self.tile, codes)), shape=(self.n_obs, len(classes))).toarray()
        return classes[votes.argmax(1)]

    def source_section(self, section):
        """Section of the nearer endpoint, per tile (majority)."""
        return self.labels(section)


def z_from_sections(z, obj):
    """z between the real sections, from one objective per section (lower is better).

    The vertex of the parabola through the best section and its two neighbours, clipped to between them.
    Returns (z, edge); edge is True when the best section is the first or the last, and z is then its z.
    """
    z, obj = np.asarray(z, dtype=np.float64), np.asarray(obj, dtype=np.float64)
    k = int(np.nanargmin(obj))
    if k == 0 or k == len(z) - 1:
        return float(z[k]), True
    a, b, _ = np.polyfit(z[k - 1:k + 2], obj[k - 1:k + 2], 2)
    zv = -b / (2 * a) if a > 0 else z[k]
    return float(np.clip(zv, z[k - 1], z[k + 1])), False


def plane_from_coupling(xyz, P, frame, section=None):
    """Fit a plane to where a coupling (n_query x n_stack) puts the query in the stack.

    Each query point goes to its barycentric image; a weighted PCA gives the normal. Returns
    dict(plane, rms, normal, bary, mass_per_section).
    """
    P = np.asarray(P, dtype=np.float64)
    w = P.sum(1)
    bary = (P @ xyz) / (w[:, None] + 1e-30)
    c3 = w @ bary / w.sum()
    Bc = (bary - c3) * np.sqrt(w / w.sum())[:, None]
    _, _, Vt = np.linalg.svd(Bc, full_matrices=False)
    n = Vt[2] * np.sign(Vt[2][2] or 1.0)
    g = -n[:2] / n[2]
    c = frame["centre"]
    z0 = c3[2] + g @ (c - c3[:2])
    pl = plane(z0, np.degrees(np.arctan(g @ frame["u"])), np.degrees(np.arctan(g @ frame["v"])))
    rms = float(np.sqrt((w * ((bary - c3) @ n) ** 2).sum() / w.sum()))
    mass = None
    if section is not None:
        mass = pd.Series(P.sum(0)).groupby(np.asarray(section)).sum()
        mass = mass / mass.sum()
    return dict(plane=pl, rms=rms, normal=n, bary=bary, mass_per_section=mass)


# ---------- features
def joint_embedding(ln_list, n_hvg=2000, n_comps=30, exclude=None):
    """One embedding for log-normalized sections of one species: the n_hvg most variable shared genes of
    each section (minus `exclude`), z-scored within each section, then a joint PCA. Returns a list of arrays."""
    from sklearn.utils.extmath import randomized_svd

    genes = ln_list[0].var_names
    for a in ln_list[1:]:
        genes = genes.intersection(a.var_names)
    keep = np.zeros(len(genes), dtype=bool)
    Xs = [a[:, genes].X for a in ln_list]
    for X in Xs:
        mean = np.asarray(X.mean(0)).ravel()
        sq = np.asarray(X.multiply(X).mean(0)).ravel() if sp.issparse(X) else (X ** 2).mean(0)
        keep[np.argsort(-(sq - mean ** 2))[:n_hvg]] = True
    if exclude is not None:
        keep &= ~np.isin(np.asarray(genes), np.asarray(exclude))
    Z = []
    for X in Xs:
        X = np.asarray(X[:, keep].toarray() if sp.issparse(X) else X[:, keep], dtype=np.float32)
        Z.append((X - X.mean(0)) / (X.std(0) + 1e-6))
    J = np.vstack(Z)
    J -= J.mean(0)
    U, S, _ = randomized_svd(J, n_components=n_comps, random_state=0)
    E = (U * S).astype(np.float64)
    return np.split(E, np.cumsum([len(z) for z in Z])[:-1])
