"""Scores of a coupling P (n_query x n_target) against the organ labels (docs/st_3d_alignment_strategies.md §2).

The headline is organ-balanced Cohen's kappa (kappa_bal, §2a), computed from the soft organ x organ matrix C:
C_ab is the mass of P that goes from query organ a to target organ b.

- Full resolution (the headline): every bin of a query tile carries that tile's row of P, so the query side is the
  tile x organ bin-count matrix Q. A target tile's mass is split over organs by its bins' composition (T, rows
  normalized). C = Q^T P_rows T, with P_rows = P with rows normalized to 1.
- Reduced: one-hot majority labels on both sides.

From C, restricted to the K organs present on both sides (scored organs only; `Other` and `Excluded` are never
scored; the pipelines take the target side's organs from the whole target embryo, `present_t`):
- OMS: mean over organs of C_oo / sum_b C_ob (the share of an organ's mass that lands on the same organ);
- kappa_bal = (OMS - 1/K) / (1 - 1/K): Cohen's kappa after rescaling every row to the same mass; the soft version
  of chance-adjusted balanced accuracy, Youden's J for K = 2;
- Cohen's kappa on the unscaled C (reported, not ranked).
"""
import numpy as np
import scipy.sparse as sp

from mgw import crossspecies as cs

ORGANS = tuple(sorted(set(cs.HARMONIZE.values()) | {"Other", cs.EXCLUDED}))
ORGAN_INDEX = {o: i for i, o in enumerate(ORGANS)}
SCORED = np.array([o not in cs.NOT_SCORED for o in ORGANS])


# ---------- label matrices
def organ_codes(labels):
    """Index into ORGANS of each label (unknown labels count as Other)."""
    other = ORGAN_INDEX["Other"]
    return np.array([ORGAN_INDEX.get(l, other) for l in np.asarray(labels, dtype=str)], dtype=np.int64)


def onehot(labels):
    codes = organ_codes(labels)
    M = np.zeros((len(codes), len(ORGANS)), dtype=np.float64)
    M[np.arange(len(codes)), codes] = 1.0
    return M


def composition(bin_labels, tile, n_tiles):
    """Tile x organ bin counts from the harmonized label and the tile of every full-resolution bin."""
    codes = organ_codes(bin_labels)
    return sp.csr_matrix((np.ones(len(codes)), (np.asarray(tile), codes)),
                         shape=(n_tiles, len(ORGANS))).toarray()


def majority(comp):
    """Majority organ of each row of a composition matrix (names)."""
    return np.asarray(ORGANS)[np.asarray(comp).argmax(1)]


def row_normalize(M):
    M = np.asarray(M, dtype=np.float64)
    s = M.sum(1, keepdims=True)
    return M / np.where(s > 0, s, 1.0)


# ---------- the organ x organ matrix
def confusion(P, Q, T):
    """C (organs x organs): the mass of P from query organ a to target organ b.

    P: n_q x n_t coupling (any marginals). Q: n_q x organs, the query's mass per organ (bin counts at full
    resolution, one-hot at reduced). T: n_t x organs, the target composition (normalized per row here).
    Every query tile spreads its Q row over the targets in proportion to its row of P.
    """
    P = np.asarray(P, dtype=np.float64)
    return np.asarray(Q, dtype=np.float64).T @ (row_normalize(P) @ row_normalize(T))


def present(M, min_count):
    """Organs present on one side: scored, with at least min_count (bins or tiles)."""
    return SCORED & (np.asarray(M).sum(0) >= min_count)


def kappa_scores(C, present_q, present_t):
    """OMS, kappa_bal and Cohen's kappa from C, on the K organs present on both sides.

    Returns dict(K, oms, kappa_bal, kappa, acc, recall {organ: m_o}, mass_unscored). `acc` is the observed
    agreement of the unscaled restricted C (soft accuracy). `mass_unscored` is the share of the shared query
    organs' mass that goes to target organs outside the K.
    """
    C = np.asarray(C, dtype=np.float64)
    shared = np.flatnonzero(present_q & present_t)
    K = len(shared)
    out = dict(K=K, oms=np.nan, kappa_bal=np.nan, kappa=np.nan, acc=np.nan, recall={}, mass_unscored=np.nan)
    if K == 0:
        return out
    Cs = C[np.ix_(shared, shared)]
    row = Cs.sum(1)
    ok = row > 0
    m = np.where(ok, np.diag(Cs) / np.where(ok, row, 1.0), 0.0)
    total_rows = C[shared].sum()
    out["recall"] = {ORGANS[o]: float(v) for o, v in zip(shared, m)}
    out["mass_unscored"] = float(1 - Cs.sum() / total_rows) if total_rows > 0 else np.nan
    out["oms"] = float(m.mean())
    if K >= 2:
        out["kappa_bal"] = float((out["oms"] - 1 / K) / (1 - 1 / K))
    N = Cs.sum()
    if N > 0:
        po = np.trace(Cs) / N
        pe = float(Cs.sum(1) @ Cs.sum(0)) / N ** 2
        out["acc"] = float(po)
        out["kappa"] = float((po - pe) / (1 - pe)) if pe < 1 else np.nan
    return out


def coverage(recall, share):
    """Share of the shared organs whose counterpart receives at least `share` of their mass."""
    v = np.asarray(list(recall.values()))
    return float((v >= share).mean()) if len(v) else np.nan


def hard_prediction(P, T):
    """Arg-max label transfer: each query tile gets the organ that receives most of its row of P; and the
    share of the row it holds (confidence)."""
    post = row_normalize(P) @ row_normalize(T)
    return post.argmax(1), post.max(1)


def label_scores(P, Q_full, T_full, Q_red, T_red, *, min_bins=5, min_tiles=1, coverage_share=0.1,
                 present_t=None):
    """All label metrics of one coupling, at full and reduced resolution, soft and hard (arg-max).

    Q_full / T_full: tile x organ bin counts of query / target; Q_red / T_red: one-hot majority labels.
    present_t: (full, reduced) organ masks of the target side, e.g. of the whole target embryo, so that a target
    section without an organ cannot drop it from K; default: the organs of T_full / T_red.
    Returns (flat dict of scalars, C_full, C_red).
    """
    pq_f, pq_r = present(Q_full, min_bins), present(Q_red, min_tiles)
    if present_t is None:
        pt_f, pt_r = present(T_full, min_bins), present(T_red, min_tiles)
    else:
        pt_f, pt_r = present_t
    C_full = confusion(P, Q_full, T_full)
    C_red = confusion(P, Q_red, T_red)
    out = {}
    for tag, C, pq, pt in (("", C_full, pq_f, pt_f), ("_red", C_red, pq_r, pt_r)):
        k = kappa_scores(C, pq, pt)
        out.update({f"K{tag}": k["K"], f"oms{tag}": k["oms"], f"kappa_bal{tag}": k["kappa_bal"],
                    f"kappa{tag}": k["kappa"], f"acc{tag}": k["acc"],
                    f"coverage{tag}": coverage(k["recall"], coverage_share),
                    f"mass_unscored{tag}": k["mass_unscored"]})
        if tag == "":
            out["recall"] = k["recall"]
    # hard: the predicted organ of each query tile, carried by all its bins
    pred, conf = hard_prediction(P, T_red)
    H = np.zeros((len(pred), len(ORGANS)))
    H[np.arange(len(pred)), pred] = 1.0
    for tag, Q, pq in (("", Q_full, pq_f), ("_red", Q_red, pq_r)):
        k = kappa_scores(np.asarray(Q).T @ H, pq, pt_r)
        out.update({f"acc_hard{tag}": k["acc"], f"bal_acc_hard{tag}": k["oms"], f"kappa_bal_hard{tag}": k["kappa_bal"]})
    out["confidence"] = float(np.mean(conf))
    return out, C_full, C_red


def prior_scores(Q_full, T_full, Q_red, T_red, **kw):
    """The organ-frequency prior: every query tile sends its mass to the target's organ frequencies (P = a b^T)."""
    n_q, n_t = len(Q_full), len(T_full)
    P = np.full((n_q, n_t), 1.0 / (n_q * n_t))
    return label_scores(P, Q_full, T_full, Q_red, T_red, **kw)


# ---------- plausibility of the map (label-free)
def sharpness(P):
    """Effective number of partners per query point, exp(entropy of its row), averaged."""
    R = row_normalize(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(R > 0, R * np.log(R), 0.0).sum(1)
    return float(np.exp(H).mean())


def barycentric(P, xt):
    P = np.asarray(P, dtype=np.float64)
    return (P @ np.asarray(xt, dtype=np.float64)) / (P.sum(1, keepdims=True) + 1e-300)


def neighbourhoods(xq, k=16):
    """Indices of each query point's k nearest query points (itself included)."""
    from scipy.spatial import cKDTree
    xq = np.asarray(xq, dtype=np.float64)
    return cKDTree(xq).query(xq, k=min(k + 1, len(xq)))[1]


def fold_overs(xq, xt, P, k=16, nbr=None):
    """Fold-overs of the barycentric map of a 2D query into a 2D target section or a 3D target volume.

    Every query point gets the local Jacobian J of the map: the least-squares affine fit of its k nearest
    neighbours' barycentric images (a fit over a neighbourhood, so the blur of an entropic coupling does not decide
    the sign as it would for single triangles).
    - 2D target: a point folds over when det J has the opposite sign from the majority, so a global reflection
      counts as orientation, not as fold-over.
    - 3D target: the image is a surface; its normal J e1 x J e2 carries the orientation of the query's own axes. A
      point folds over when its normal points against the mean normal (this assumes a near-planar image: one bent
      by more than 90 degrees would count as folded). The orientation is undefined in 3D and reported as 0.
    Returns dict(foldover: share of points, orientation: +1/-1/0, cond_median: median condition number of J,
    folded_points: bool per point).
    """
    xq = np.asarray(xq, dtype=np.float64)
    nbr = neighbourhoods(xq, k) if nbr is None else nbr
    y = barycentric(P, xt)
    X = xq[nbr] - xq[nbr].mean(1, keepdims=True)                 # (n, k, 2)
    Y = y[nbr] - y[nbr].mean(1, keepdims=True)                   # (n, k, 2 or 3)
    XtX = np.einsum("nki,nkj->nij", X, X)
    XtY = np.einsum("nki,nkj->nij", X, Y)
    ok = np.abs(np.linalg.det(XtX)) > 1e-18
    J = np.zeros((len(xq), 2, y.shape[1]))                       # row i: the image of query axis i
    J[ok] = np.linalg.solve(XtX[ok], XtY[ok])
    if y.shape[1] == 2:
        s = np.linalg.det(J)
        orient = 1 if np.sum(np.sign(s)) >= 0 else -1
    else:
        nrm = np.cross(J[:, 0], J[:, 1])
        unit = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
        s = nrm @ unit[ok].mean(0)
        orient = 1
    folded = (np.sign(s) != orient) & ok
    if y.shape[1] != 2:
        orient = 0
    sv = np.linalg.svd(J[ok], compute_uv=False)
    cond = sv[:, 0] / np.maximum(sv[:, 1], 1e-12)
    return dict(foldover=float(folded.mean()), orientation=orient,
                cond_median=float(np.median(cond)) if len(cond) else np.nan, folded_points=folded)


def topk(P, k):
    """The k largest entries per row (indices int32, values float32) and per column."""
    P = np.asarray(P)
    k_r, k_c = min(k, P.shape[1]), min(k, P.shape[0])
    ir = np.argpartition(-P, k_r - 1, axis=1)[:, :k_r]
    vr = np.take_along_axis(P, ir, 1)
    ic = np.argpartition(-P, k_c - 1, axis=0)[:k_c, :].T
    vc = np.take_along_axis(P.T, ic, 1)
    return ir.astype(np.int32), vr.astype(np.float32), ic.astype(np.int32), vc.astype(np.float32)


def map_scores(xq, xt, P, k=16, nbr=None):
    """Label-free scores of a coupling: fold-overs, Jacobian condition, sharpness, orientation (2D or 3D target)."""
    fo = fold_overs(xq, xt, P, k, nbr=nbr)
    return dict(foldover=fo["foldover"], cond_median=fo["cond_median"], orientation=fo["orientation"],
                sharpness=sharpness(P)), fo["folded_points"]


def coexpression(P, Yq, Yt, seed=0):
    """Median per-gene r of held-out genes, and with the target's rows shuffled (reduced resolution)."""
    if Yq is None or Yt is None:
        return dict(coexpr_r=np.nan, coexpr_r0=np.nan)
    r, r0 = cs.coexpression(np.asarray(P, dtype=np.float64), np.asarray(Yq, dtype=np.float64),
                            np.asarray(Yt, dtype=np.float64), seed=seed)
    return dict(coexpr_r=float(np.nanmedian(r)), coexpr_r0=float(np.nanmedian(r0)))


# ---------- pair-level summaries
def pair_mean(df, value, by_direction="direction"):
    """The symmetric pair score: the mean of the per-direction means."""
    return float(df.groupby(by_direction)[value].mean().mean()) if len(df) else np.nan
