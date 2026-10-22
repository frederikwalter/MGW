"""The one score of a coupling P (n_query x n_target): organ-balanced Cohen's kappa (kappa_bal).

C is the soft organ x organ matrix at full resolution: C_ab is the mass of P that goes from query organ a to target
organ b. Every bin of a query tile carries that tile's row of P, so the query side is the tile x organ bin-count
matrix Q; a target tile's mass is split over organs by its bins' composition (T, rows normalized):
C = Q^T P_rows T, with P_rows = P with rows normalized to 1.

From C, restricted to the K organs present on both sides (`Other` and `Excluded` are never scored):
- m_o = C_oo / sum_b C_ob, the share of organ o's mass that lands on the same organ; OMS = the mean of m_o;
- kappa_bal = (OMS - 1/K) / (1 - 1/K): Cohen's kappa after rescaling every row to the same mass, i.e. the soft
  version of chance-adjusted balanced accuracy (0 = chance, 1 = every organ to the same organ); Youden's J for K = 2.
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


# ---------- the organ x organ matrix and kappa_bal
def confusion(P, Q, T):
    """C (organs x organs): the mass of P from query organ a to target organ b.

    P: n_q x n_t coupling (any marginals). Q: n_q x organs, the query's mass per organ (bin counts at full
    resolution). T: n_t x organs, the target composition (normalized per row here).
    """
    P = np.asarray(P, dtype=np.float64)
    return np.asarray(Q, dtype=np.float64).T @ (row_normalize(P) @ row_normalize(T))


def present(M, min_count):
    """Organs present on one side: scored, with at least min_count bins."""
    return SCORED & (np.asarray(M).sum(0) >= min_count)


def kappa_bal(C, present_q, present_t):
    """(kappa_bal, K) of C on the K organs present on both sides; kappa_bal is NaN for K < 2."""
    C = np.asarray(C, dtype=np.float64)
    shared = np.flatnonzero(present_q & present_t)
    K = len(shared)
    if K < 2:
        return np.nan, K
    Cs = C[np.ix_(shared, shared)]
    row = Cs.sum(1)
    m = np.where(row > 0, np.diag(Cs) / np.where(row > 0, row, 1.0), 0.0)
    return float((m.mean() - 1 / K) / (1 - 1 / K)), K


def score(P, Q, T, *, min_bins=5, present_t=None):
    """kappa_bal of a coupling P between a query (bin counts Q) and a target (compositions T).

    present_t: the organ mask of the target side, e.g. of the whole target embryo, so that a target section without
    an organ cannot drop it from K; default: the organs of T. Returns (kappa_bal, K).
    """
    pt = present(T, min_bins) if present_t is None else present_t
    return kappa_bal(confusion(P, Q, T), present(Q, min_bins), pt)


def hard_prediction(P, T):
    """Arg-max label transfer: each query tile gets the organ that receives most of its row of P (organ index)."""
    return (row_normalize(P) @ row_normalize(T)).argmax(1)
