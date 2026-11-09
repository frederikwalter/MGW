"""Within-species alignment across time, DeST-OT style (Halmos et al. 2025, Cell Systems 16:101160).

A coupling P (n1 x n2) of a section at t1 to one at t2: P_ij' is the mass of t2 tile j' that descends from t1 tile i.
Semi-relaxed OT: every t2 tile has an ancestor (P^T 1 = g2, hard), a t1 tile may grow or shrink (KL(P 1 | g1),
weight gamma). The growth vector is xi = P 1 - g1, the relative growth rho = P 1 / g1 - 1.

Objective (paper Eqs. 13-14, 19), with r = P 1, c = P^T 1 and squares entrywise:
    (1 - a) <C, P>                                                           doublet: expression across the slices
  + a b     1/2 (<T1 P, P> + <P T2, P>)                                     triplet: T_i = s_i M_i^2
  + a (1-b) 1/2 (r^T Q1^2 r + c^T Q2^2 c - 2 <Q1 P Q2, P>)                  quartet: GW on Q_i = q_i M_i
  + gamma KL(r | g1) - epsilon H(P)       s.t. c = g2
with the merged feature-spatial matrices M_i = C_i * D_i (intra-slice expression distance times spatial distance).

Optimisation: the reference implementation (github.com/raphael-group/DeST_OT, destot_opt.py) runs log-domain
semi-relaxed Sinkhorn on the cost G = grad E(P), refreshing the gradient after every dual update, a fixed-point
iteration. On real sections at small epsilon or large alpha that iteration cycles (P flips between two couplings) and
stops in worse local minima. `destot` keeps the same updates and adds (both optional, off = the reference):
- a proximal term prox KL(P | P_k) per refresh (mirror descent with a finite step: Sinkhorn on G - prox log P_k with
  entropy epsilon + prox), prox doubled when the change in P stalls;
- entropy annealed from eps0 down to epsilon over the first refreshes.
Its fixed points are the reference's. It stops when P changes by less than `tol` (relative L1) per refresh.

Term weights (`norm`):
- "reference": DeST_OT's constants on raw distances: q_i = sqrt(|C|_F) / |M_i|_F sqrt(n1); s1 = |C|_F / |M1^2|_F
  n1 / sqrt(n2) and s2 = |C|_F / |M2^2|_F n1 / 2 (the code's `N1**1/2`, kept for parity).
- "balanced": C divided by its mean under a x b (a, b the normalised masses), M_i divided by its RMS under a x a /
  b x b, and the triplet scaled by 1 / mean mass per tile (s1 = n2 / sum g2, s2 = n1 / sum g1), so every gradient
  term is O(1) like C and epsilon, alpha, beta mean the same on every pair. Without the 1 / mean-mass factor the
  triplet's gradient is O(1/n) of the others.

Masses: "bins" (a Voronoi tile weighs its bins; g1 sums to 1, g2 to the tissue-area ratio bins2 / bins1, so the
global growth is observed, not fixed) or "uniform" (the paper: 1/n1 per tile on both sides).

Metrics (the paper's, generalised to weighted tiles; with uniform masses and one organ per tile they equal
DeST_OT's metrics.py):
- growth distortion: sum_p sum_i w_ip (rho_i - rho_p)^2, w_ip = the mass of t1 tile i in organ p. `growth_dist`
  infers the organ transitions (rho_p = the organ's mean rho: the T* of Prop. 2, the within-organ variance of
  growth); `growth_dist_nt` assumes none (rho_p = m2(p) / m1(p) - 1, organs on both sides only).
- migration (Eq. 47 as in the reference: expected distance, not squared): P-weighted Procrustes (reflections allowed:
  sections may be mounted mirrored), then the mass-weighted mean over t1 tiles of the expected distance to their
  descendants, in units of t1's RMS radius. `spread` (Eq. 48): the RMS distance of a tile's descendants from their
  barycentre, in units of t2's RMS radius (frame-free).
- labels: kappa_bal t2 -> t1 (ancestor organ; the main one) and t1 -> t2 (bench/metrics.py), `diag_mass` the share
  of the coupling between the same organ; the organ x organ coupling F1^T P F2 is kept (T_org).
- gene r (selection.gene_r) on held-out genes; Spearman rho of the growth rate J = log(1 + rho) / dt with
  proliferation and apoptosis scores (moscot's marker sets) on t1.
"""
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from mgw import stack3d as s3
from mgw.bench import metrics, prepare, selection

METHODS = ("destot", "expr", "label", "fgw_bal", "moscot")


# ---------- pairs
def stage_of(sec):
    return sec.rsplit("_", 1)[0]


def species_of(sec):
    return "human" if sec.startswith("CS") else "mouse"


def dataset_of(sec):
    return "HESTA" if species_of(sec) == "human" else "MOSTA"


@dataclass(frozen=True)
class Pair:
    id: str
    species: str
    a: str          # t1 section
    b: str          # t2 section
    dt: float       # days

    @property
    def dataset(self):
        return dataset_of(self.a)

    @property
    def key(self):
        return f"temporal_{self.a}__{self.b}"


def make_pair(pid, a, b, stage_days):
    sp_a, sp_b = species_of(a), species_of(b)
    if sp_a != sp_b:
        raise ValueError(f"{a} -> {b}: not one species")
    dt = stage_days.get(stage_of(b), np.nan) - stage_days.get(stage_of(a), np.nan)
    return Pair(pid, sp_a, a, b, float(dt))


def chain_pairs(chains, stage_days):
    """Adjacent sections of each species' chain: H1.. for human, M1.. for mouse."""
    out = []
    for species, prefix in (("human", "H"), ("mouse", "M")):
        secs = list(chains.get(species) or [])
        out += [make_pair(f"{prefix}{k + 1}", a, b, stage_days) for k, (a, b) in enumerate(zip(secs, secs[1:]))]
    return out


def parse_pairs(spec, stage_days, known=()):
    """--pairs: pair ids of the chain (H2,M1) and/or explicit a:b section pairs (id a__b, the same in every task)."""
    known = {p.id: p for p in known}
    out = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok in known:
            out.append(known[tok])
        elif ":" in tok:
            a, b = tok.split(":")
            out.append(make_pair(f"{a}__{b}", a, b, stage_days))
        else:
            raise ValueError(f"unknown pair {tok!r} (ids: {sorted(known)}; or a:b)")
    return out


def load_chains(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------- features
def embed(A, B, n_hvg, n_comps, exclude=None, scale="joint", seed=0):
    """A shared PCA of two log-normalized sections of one species on their shared genes: the union of each
    section's n_hvg most variable genes minus `exclude`. scale="joint" z-scores each gene over both sections
    together (DeST-OT's joint PCA, which keeps the shift between time points); "section" within each section
    (orthologs.shared_embedding). Returns (EA, EB, genes)."""
    from mgw import orthologs
    if scale == "section":
        return orthologs.shared_embedding(A, B, None, n_hvg=n_hvg, n_comps=n_comps, exclude=exclude,
                                          random_state=seed)
    import scipy.sparse as sp
    from sklearn.utils.extmath import randomized_svd
    from mgw import crossspecies as cs
    genes = np.intersect1d(A.var_names, B.var_names)
    XA, XB = A[:, genes].X, B[:, genes].X
    keep = np.zeros(len(genes), dtype=bool)
    for X in (XA, XB):
        keep[np.argsort(-cs.gene_var(X))[:n_hvg]] = True
    if exclude is not None:
        keep &= ~np.isin(genes, np.asarray(exclude))
    J = np.vstack([cs.dense(XA[:, keep]), cs.dense(XB[:, keep])])
    J = (J - J.mean(0)) / (J.std(0) + 1e-6)
    U, S, _ = randomized_svd(J, n_components=n_comps, random_state=seed)
    E = (U[:, :n_comps] * S[:n_comps]).astype(np.float64)
    return E[:A.n_obs], E[A.n_obs:], genes[keep]


def marker_scores(adata, organism, seed=0):
    """(proliferation, apoptosis) scores of every tile (scanpy score_genes on moscot's marker sets)."""
    import scanpy as sc
    from moscot.utils import data as mdata
    out = []
    for genes in (mdata.proliferation_markers(organism), mdata.apoptosis_markers(organism)):
        genes = [g for g in genes if g in adata.var_names]
        if len(genes) < 5:
            out.append(np.full(adata.n_obs, np.nan))
            continue
        a = adata.copy()
        sc.tl.score_genes(a, genes, score_name="s", random_state=seed)
        out.append(a.obs["s"].to_numpy(np.float64))
    return out


def feature_path(pair, cfg, seed, root):
    return prepare.feature_file(root, pair.key, seed, cfg.temporal_feature_hash())


def pair_features(pair, cfg, seed, root, force=False):
    """Embedding, held-out genes and marker scores of a pair, from the all-gene counts. Returns the file."""
    from mgw import crossspecies as cs
    out = feature_path(pair, cfg, seed, root)
    if out.exists() and not force:
        return out
    f = cfg.raw["temporal"]["features"]
    sh = cfg.section_hash()
    A, B = prepare.load_counts_all(root, pair.a, seed, sh), prepare.load_counts_all(root, pair.b, seed, sh)
    genes = np.intersect1d(A.var_names, B.var_names)
    ev = cs.pick_eval_genes(A[:, genes].X, B[:, genes].X, genes, f["n_eval_genes"])
    EA, EB, emb_genes = embed(A, B, f["n_hvg"], f["pca_comp"], exclude=ev, scale=f["scale"], seed=seed)
    pa, aa = marker_scores(A, pair.species, seed)
    pb, ab = marker_scores(B, pair.species, seed)
    prepare.atomic_savez(out, emb_a=EA.astype(np.float32), emb_b=EB.astype(np.float32),
                         Ya=cs.dense(A[:, list(ev)].X).astype(np.float16),
                         Yb=cs.dense(B[:, list(ev)].X).astype(np.float16), eval_genes=np.asarray(ev, dtype=str),
                         emb_genes=np.asarray(emb_genes, dtype=str), n_shared=len(genes),
                         prolif_a=pa, apopt_a=aa, prolif_b=pb, apopt_b=ab,
                         pitch_a=A.uns["bin_pitch"], pitch_b=B.uns["bin_pitch"])
    return out


@dataclass
class PairData:
    pair: Pair
    sa: prepare.Section
    sb: prepare.Section
    emb_a: np.ndarray
    emb_b: np.ndarray
    Ya: np.ndarray
    Yb: np.ndarray
    scores: dict = field(default_factory=dict)      # prolif_a, apopt_a, prolif_b, apopt_b
    pitch: tuple = (np.nan, np.nan)
    n_shared: int = 0


def load_pair(pair, cfg, seed, root):
    sh = cfg.section_hash()
    z = np.load(feature_path(pair, cfg, seed, root), allow_pickle=False)
    sa, sb = prepare.load_section(root, pair.a, seed, sh), prepare.load_section(root, pair.b, seed, sh)
    if len(z["emb_a"]) != sa.n or len(z["emb_b"]) != sb.n:
        raise RuntimeError(f"{pair.id}: features and section caches disagree; rerun prepare")
    return PairData(pair, sa, sb, z["emb_a"].astype(np.float64), z["emb_b"].astype(np.float64),
                    z["Ya"].astype(np.float32), z["Yb"].astype(np.float32),
                    {k: z[k].astype(np.float64) for k in ("prolif_a", "apopt_a", "prolif_b", "apopt_b")},
                    (float(z["pitch_a"]), float(z["pitch_b"])), int(z["n_shared"]))


# ---------- masses and costs
def masses(comp_a, comp_b, mode="bins"):
    """(g1, g2): g1 sums to 1, g2 to bins_b / bins_a ("bins") or n2 / n1 ("uniform")."""
    ba, bb = np.asarray(comp_a, np.float64).sum(1), np.asarray(comp_b, np.float64).sum(1)
    if mode == "uniform":
        n1 = len(ba)
        return np.full(n1, 1 / n1), np.full(len(bb), 1 / n1)
    return ba / ba.sum(), bb / ba.sum()


@dataclass
class Costs:
    C: np.ndarray       # n1 x n2
    Q1: np.ndarray      # quartet matrices
    Q2: np.ndarray
    T1: np.ndarray      # triplet matrices (already squared and scaled)
    T2: np.ndarray


def destot_costs(emb_a, emb_b, xa, xb, g1, g2, norm="balanced"):
    """DeST-OT's C, merged feature-spatial matrices M_i = C_i * D_i, and the term weights of `norm`."""
    C = cdist(emb_a, emb_b)
    M1 = cdist(emb_a, emb_a) * cdist(xa, xa)
    M2 = cdist(emb_b, emb_b) * cdist(xb, xb)
    n1, n2 = C.shape
    if norm == "reference":
        nC = np.linalg.norm(C)
        q1, q2 = np.sqrt(nC) / np.linalg.norm(M1) * np.sqrt(n1), np.sqrt(nC) / np.linalg.norm(M2) * np.sqrt(n1)
        s1 = nC / np.linalg.norm(M1 ** 2) * (n1 / np.sqrt(n2))
        s2 = nC / np.linalg.norm(M2 ** 2) * (n1 / 2)       # DeST_OT: N1**1/2
        return Costs(C, q1 * M1, q2 * M2, s1 * M1 ** 2, s2 * M2 ** 2)
    if norm != "balanced":
        raise ValueError(f"norm {norm!r}")
    a, b = g1 / g1.sum(), g2 / g2.sum()
    C = C / (a @ C @ b)
    M1 = M1 / np.sqrt(a @ M1 ** 2 @ a)
    M2 = M2 / np.sqrt(b @ M2 ** 2 @ b)
    s1, s2 = n2 / g2.sum(), n1 / g1.sum()
    return Costs(C, M1, M2, s1 * M1 ** 2, s2 * M2 ** 2)


# ---------- the solver
def _jnp():
    import jax.numpy as jnp
    return jnp


def gradient(P, C, Q1, Q2, T1, T2, alpha, beta, column_term=True):
    """grad E(P) (the module docstring's energy without KL and entropy). jax or numpy arrays.

    column_term=False drops the quartet's (Q2^2 c) 1^T: constant in every column under P^T 1 = g2, so it moves no
    coupling, only the column duals (the solver and the reference leave it out)."""
    r = P.sum(1)
    trip = T1 @ P + P @ T2
    quart = ((Q1 * Q1) @ r)[:, None] - 2 * (Q1 @ P @ Q2)
    if column_term:
        quart = quart + ((Q2 * Q2) @ P.sum(0))[None, :]
    return (1 - alpha) * C + alpha * (beta * trip + (1 - beta) * quart)


def energy_terms(P, C, Q1, Q2, T1, T2):
    """(doublet, triplet, quartet) of P, numpy, float64."""
    P = np.asarray(P, np.float64)
    r, c = P.sum(1), P.sum(0)
    doublet = float((C * P).sum())
    triplet = 0.5 * float(((T1 @ P) * P).sum() + ((P @ T2) * P).sum())
    quartet = 0.5 * float(r @ (Q1 * Q1) @ r + c @ (Q2 * Q2) @ c - 2 * ((Q1 @ P @ Q2) * P).sum())
    return doublet, triplet, quartet


def energy(P, C, Q1, Q2, T1, T2, alpha, beta):
    d, t, q = energy_terms(P, C, Q1, Q2, T1, T2)
    return (1 - alpha) * d + alpha * (beta * t + (1 - beta) * q)


@lru_cache(maxsize=8)
def _solver(linear):
    """The jitted loop. linear: alpha = 0, the cost is C throughout (no gradient, no intra-slice matrices)."""
    import jax
    jnp = _jnp()
    from jax.scipy.special import logsumexp

    @jax.jit
    def solve(C, Q1, Q2, T1, T2, g1, g2, alpha, beta, gamma, eps, prox, prox_max, patience, eps0, anneal, maxit, tol,
              inner_steps, inner_maxit, inner_tol):
        lg1, lg2 = jnp.log(g1), jnp.log(g2)
        grad = (lambda P: C) if linear else (lambda P: gradient(P, C, Q1, Q2, T1, T2, alpha, beta, False))
        eps0 = jnp.maximum(eps0, eps)
        decay = (eps / eps0) ** (1.0 / jnp.maximum(anneal, 1))

        def sinkhorn(f, g, G, e):
            """Semi-relaxed log-domain Sinkhorn on cost G with entropy e, warm-started: at least inner_steps updates,
            then (inner_tol > 0) until the row duals move less than inner_tol * e."""
            ubc = jnp.where(jnp.isinf(gamma), 1.0, gamma / (gamma + e))

            def body(st):
                i, f, g, _ = st
                fn = ubc * (e * lg1 - e * logsumexp((g[None, :] - G) / e, axis=1))
                gn = e * lg2 - e * logsumexp((fn[:, None] - G) / e, axis=0)
                return i + 1, fn, gn, (jnp.abs(fn - f).max() / e).astype(jnp.float32)

            def cond(st):
                i, _, _, delta = st
                return (i < inner_steps) | ((inner_tol > 0) & (delta > inner_tol) & (i < inner_maxit))
            i, f, g, delta = jax.lax.while_loop(cond, body, (jnp.int32(0), f, g, jnp.float32(jnp.inf)))
            return f, g, i

        def body(state):
            k, f, g, logP, gap, best, stall, lam, n_in = state
            # the proximal step argmin <G, P> + lam KL(P | P_k) + gamma KL(P1 | g1) - e H(P) s.t. P^T 1 = g2 is
            # Sinkhorn on G - lam log P_k with entropy e + lam (mirror descent with step 1 / lam; lam = 0 is the
            # reference's fixed-point iteration); e anneals from eps0 to eps over `anneal` refreshes
            e = jnp.maximum(eps, eps0 * decay ** k)
            Gp = grad(jnp.exp(logP)) - lam * logP
            f, g, i = sinkhorn(f, g, Gp, e + lam)
            logPn = (f[:, None] + g[None, :] - Gp) / (e + lam)
            Pn, P = jnp.exp(logPn), jnp.exp(logP)
            gap = (jnp.abs(Pn - P).sum() / Pn.sum()).astype(jnp.float32)
            # lam doubles when the change in P has not improved by 5% for `patience` refreshes at the final eps
            settled = e <= eps * 1.0001
            better = gap < 0.95 * best
            stall = jnp.where(settled & ~better, stall + 1, 0)
            best = jnp.where(settled & better, gap, jnp.where(settled, best, jnp.inf))
            bump = (stall >= patience) & (lam > 0)
            lam = jnp.where(bump, jnp.minimum(2 * lam, prox_max), lam)
            stall = jnp.where(bump, 0, stall)
            best = jnp.where(bump, gap, best)
            return k + 1, f, g, logPn, jnp.where(settled, gap, jnp.inf), best, stall, lam, n_in + i

        def cond(state):
            k, gap = state[0], state[4]
            return (k < maxit) & (gap > tol)

        logP0 = jnp.log(g1)[:, None] + jnp.log(g2)[None, :] - jnp.log(g1.sum())
        z1, z2 = jnp.zeros_like(g1), jnp.zeros_like(g2)
        inf = jnp.float32(jnp.inf)
        state = (jnp.int32(0), z1, z2, logP0, inf, inf, jnp.int32(0), jnp.float32(prox), jnp.int32(0))
        k, f, g, logP, gap, best, stall, lam, n_in = jax.lax.while_loop(cond, body, state)
        return jnp.exp(logP), k, gap, lam, n_in
    return solve


def destot(costs, g1, g2, *, alpha, beta=0.5, gamma=50.0, epsilon=0.1, outer_maxit=3000, inner_steps=5,
           inner_maxit=1, inner_tol=0.0, tol=5e-5, prox=0.0, prox_max=1.0, patience=100, eps0=0.0, anneal=0):
    """DeST-OT's semi-relaxed coupling (n1 x n2, float64) and info. gamma = inf is balanced in g1.

    Each gradient refresh runs inner_steps Sinkhorn updates (then, with inner_tol > 0, more until the duals settle,
    at most inner_maxit) of a proximal step with weight prox (cost units) and entropy annealed from eps0 to epsilon
    over `anneal` refreshes; the loop stops when P changes by less than tol (relative L1) at the final epsilon.
    prox doubles (up to prox_max) when that change stalls for `patience` refreshes. inner_steps = 1, prox = 0 and no
    annealing is the reference's single loop."""
    jnp = _jnp()
    f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)
    linear = float(alpha) == 0.0
    z = jnp.zeros((1, 1), dtype=jnp.float32)
    mats = (z, z, z, z) if linear else tuple(f32(m) for m in (costs.Q1, costs.Q2, costs.T1, costs.T2))
    t0 = time.perf_counter()
    P, k, gap, lam, n_in = _solver(linear)(
        f32(costs.C), *mats, f32(g1), f32(g2), float(alpha), float(beta), float(gamma), float(epsilon),
        0.0 if linear else float(prox), float(prox_max), int(patience), float(eps0) if not linear else 0.0,
        int(anneal), int(outer_maxit), float(tol), int(inner_steps), int(inner_maxit), float(inner_tol))
    P = np.asarray(P, dtype=np.float64)
    info = dict(converged=bool(float(gap) <= tol), n_outer=int(k), n_inner=int(n_in), gap=float(gap),
                prox_end=float(lam), seconds=time.perf_counter() - t0, finite=bool(np.isfinite(P).all()))
    return P, info


def kl(r, g):
    r, g = np.asarray(r, np.float64), np.asarray(g, np.float64)
    m = r > 0
    return float((r[m] * np.log(r[m] / g[m])).sum() - r.sum() + g.sum())


# ---------- runs: a method and its params
def run_name(method, params):
    keys = {"destot": "abge", "expr": "ge", "label": "ge", "fgw_bal": "ae", "moscot": "ae"}[method]
    full = dict(a="alpha", b="beta", g="gamma", e="epsilon")
    return method + "_" + "_".join(f"{k}{params[full[k]]:g}" for k in keys)


def parse_run(name):
    method = rest = None
    for m in sorted(METHODS, key=len, reverse=True):
        if name.startswith(m + "_"):
            method, rest = m, name[len(m) + 1:]
            break
    if method is None:
        raise ValueError(f"run {name!r}: unknown method")
    full = dict(a="alpha", b="beta", g="gamma", e="epsilon")
    return method, {full[t[0]]: float(t[1:]) for t in rest.split("_")}


def grid(cfg, methods=METHODS):
    """[(method, params)] of the sweep."""
    t = cfg.raw["temporal"]
    out = []
    if "destot" in methods:
        sw = t["sweep"]
        pts = [dict(alpha=float(a), beta=float(b), gamma=float(g), epsilon=float(e)) for a in sw["alpha"] for b in sw["beta"]
               for g in sw["gamma"] for e in sw["epsilon"]]
        for p in sw["extra"]:
            if set(p) != {"alpha", "beta", "gamma", "epsilon"}:
                raise ValueError(f"temporal.sweep.extra: {p!r} needs exactly alpha, beta, gamma, epsilon")
            pts.append({k: float(v) for k, v in p.items()})
        out += [("destot", p) for p in pts]
    bl = t["baselines"]
    for m in ("expr", "label"):
        if m in methods:
            out += [(m, dict(gamma=g, epsilon=e)) for g in bl[m]["gamma"] for e in bl[m]["epsilon"]]
    if "fgw_bal" in methods:
        out += [("fgw_bal", dict(alpha=a, epsilon=bl["fgw_bal"]["epsilon"])) for a in bl["fgw_bal"]["alpha"]]
    if "moscot" in methods:
        out += [("moscot", dict(alpha=a, epsilon=bl["moscot"]["epsilon"])) for a in bl["moscot"]["alpha"]]
    seen, uniq = set(), []
    for m, p in out:
        if run_name(m, p) not in seen:
            seen.add(run_name(m, p))
            uniq.append((m, p))
    return uniq


class Problem:
    """One pair, ready to solve: masses and (lazily) the cost matrices of each method."""

    def __init__(self, data: PairData, cfg):
        self.data, self.cfg = data, cfg
        t = cfg.raw["temporal"]
        self.g1, self.g2 = masses(data.sa.comp, data.sb.comp, t["masses"])
        self._destot, self._label = None, None

    @property
    def destot_costs(self):
        if self._destot is None:
            d = self.data
            self._destot = destot_costs(d.emb_a, d.emb_b, d.sa.xy, d.sb.xy, self.g1, self.g2,
                                        self.cfg.raw["temporal"]["norm"])
        return self._destot

    @property
    def label_costs(self):
        if self._label is None:
            M = selection.label_cost(self.data.sa.comp, self.data.sb.comp)
            z = np.zeros((1, 1))
            self._label = Costs(M, z, z, z, z)
        return self._label

    def to_semirelaxed(self, P):
        """A coupling of another method with its columns rescaled to g2 (the semi-relaxed convention)."""
        c = P.sum(0)
        return P * np.where(c > 0, self.g2 / np.where(c > 0, c, 1), 0.0)[None, :]

    def solve(self, method, params):
        s = self.cfg.raw["temporal"]["solver"]
        kw = dict(s)
        if method == "destot":
            return destot(self.destot_costs, self.g1, self.g2, **params, **kw)
        if method in ("expr", "label"):
            costs = self.destot_costs if method == "expr" else self.label_costs
            return destot(costs, self.g1, self.g2, alpha=0.0, beta=0.0, gamma=params["gamma"],
                          epsilon=params["epsilon"], **kw)
        if method == "fgw_bal":
            return self._fgw(params)
        if method == "moscot":
            return self._moscot(params)
        raise ValueError(method)

    def _fgw(self, params):
        from mgw.bench import solvers
        d, fp = self.data, self.cfg.raw["fgw"]
        C = cdist(d.emb_a, d.emb_b)
        M = C / np.quantile(C, self.cfg.raw["features"]["m_scale_quantile"])
        p = solvers.FGWParams.from_mapping({**fp, "alpha": params["alpha"], "epsilon": params["epsilon"]})
        P, info = solvers.fgw(d.sa.xy, d.sb.xy, M, p, a=self.g1 / self.g1.sum(), b=self.g2 / self.g2.sum())
        info = dict(converged=bool(info["converged"]), n_outer=int(info["n_outer"]), gap=np.nan,
                    seconds=info["seconds"], finite=bool(np.isfinite(P).all()), outer_stopped=info["outer_stopped"],
                    inner_converged=info["inner_converged"], ugw_fallback=info["ugw_fallback"])
        return self.to_semirelaxed(P), info

    def _moscot(self, params):
        import anndata as ad
        from moscot.problems.spatiotemporal import SpatioTemporalProblem
        d, bl = self.data, self.cfg.raw["temporal"]["baselines"]["moscot"]
        n1, n2 = d.sa.n, d.sb.n
        obs = pd.DataFrame(dict(time=np.r_[np.zeros(n1), np.ones(n2)],
                                proliferation=np.r_[d.scores["prolif_a"], d.scores["prolif_b"]],
                                apoptosis=np.r_[d.scores["apopt_a"], d.scores["apopt_b"]]),
                           index=[f"t{i}" for i in range(n1 + n2)])
        emb = np.vstack([d.emb_a, d.emb_b]).astype(np.float32)
        a = ad.AnnData(emb, obs=obs)                     # moscot wants variables; the embedding is the joint term
        a.obsm["spatial"] = np.vstack([d.sa.xy, d.sb.xy])
        a.obsm["X_emb"] = emb
        t0 = time.perf_counter()
        prob = SpatioTemporalProblem(a)
        ok = np.isfinite(obs.proliferation).all() and np.isfinite(obs.apoptosis).all()
        if ok:
            prob.proliferation_key, prob.apoptosis_key = "proliferation", "apoptosis"
        prob = prob.prepare(time_key="time", joint_attr="X_emb")
        prob = prob.solve(alpha=params["alpha"], epsilon=params["epsilon"], tau_a=bl["tau_a"], tau_b=bl["tau_b"])
        sol = prob[(0.0, 1.0)].solution
        P = np.asarray(sol.transport_matrix, dtype=np.float64)
        info = dict(converged=bool(sol.converged), n_outer=-1, gap=np.nan, seconds=time.perf_counter() - t0,
                    finite=bool(np.isfinite(P).all()), moscot_marginals=bool(ok))
        return self.to_semirelaxed(P), info


# ---------- metrics
def weighted_procrustes(xa, xb, P):
    """(R, t) with xb @ R + t ~ xa under P, reflections allowed."""
    return s3.procrustes([(np.asarray(xa, np.float64), np.asarray(xb, np.float64), P)], reflect=None)


def organ_weights(comp, g):
    """Mass of every tile in every organ, in g's units (n x organs)."""
    return metrics.row_normalize(comp) * np.asarray(g)[:, None]


def growth_distortion(rho, w1, m2, present1, present2):
    """(with inferred transitions, without transitions) on organ weights w1 (n1 x organs) and t2's organ masses m2."""
    m1 = w1.sum(0)
    out = []
    for transitions in (True, False):
        tot = 0.0
        for p in np.flatnonzero(present1 if transitions else present1 & present2):
            wp = w1[:, p]
            ref = (wp @ rho) / m1[p] if transitions else m2[p] / m1[p] - 1
            tot += float(wp @ (rho - ref) ** 2)
        out.append(tot)
    return tuple(out)


def evaluate(P, prob: Problem, seed=0, genes=True):
    """(row, extras) of a coupling: the metrics of the module docstring; extras hold xi, J and T_org."""
    from scipy.stats import spearmanr
    d, g1, g2 = prob.data, prob.g1, prob.g2
    mb = prob.cfg.raw["temporal"]["min_bins"]
    P = np.asarray(P, np.float64)
    r, c = P.sum(1), P.sum(0)
    xi, rho = r - g1, r / g1 - 1
    w1, w2 = organ_weights(d.sa.comp, g1), organ_weights(d.sb.comp, g2)
    pres1, pres2 = metrics.present(d.sa.comp, mb), metrics.present(d.sb.comp, mb)
    gd, gd_nt = growth_distortion(rho, w1, w2.sum(0), pres1, pres2)
    m1 = w1.sum(0)
    both = pres1 & pres2
    rho_bar = np.where(m1 > 0, (w1.T @ rho) / np.where(m1 > 0, m1, 1), 0)
    rho_obs = np.where(m1 > 0, w2.sum(0) / np.where(m1 > 0, m1, 1) - 1, 0)
    organ_err = float(np.sqrt((m1[both] * (rho_bar[both] - rho_obs[both]) ** 2).sum() / m1[both].sum())) \
        if both.any() else np.nan
    # migration and spread
    xa, xb = d.sa.xy, d.sb.xy
    R, t = weighted_procrustes(xa, xb, P)
    Pr = P / np.where(r > 0, r, 1)[:, None]
    mig_i = (Pr * cdist(xa, xb @ R + t)).sum(1)
    live = r > 0
    migration_px = float((g1[live] @ mig_i[live]) / g1[live].sum())
    bary = Pr @ xb
    spread_i = np.maximum((Pr * (xb ** 2).sum(1)[None, :]).sum(1) - (bary ** 2).sum(1), 0)
    spread = float(np.sqrt((g1[live] @ spread_i[live]) / g1[live].sum()) / s3.rms_radius(xb))
    # labels
    F1, F2 = metrics.row_normalize(d.sa.comp), metrics.row_normalize(d.sb.comp)
    T_org = F1.T @ P @ F2
    sc_ = metrics.SCORED
    diag = float(np.trace(T_org[np.ix_(sc_, sc_)]) / max(T_org[np.ix_(sc_, sc_)].sum(), 1e-30))
    k_ba, K = metrics.score(P.T, d.sb.comp, d.sa.comp, min_bins=mb)
    k_ab, _ = metrics.score(P, d.sa.comp, d.sb.comp, min_bins=mb)
    # growth rates and markers
    J = np.log(np.maximum(1 + rho, 1e-3)) / (d.pair.dt if np.isfinite(d.pair.dt) and d.pair.dt > 0 else 1.0)
    rho_s = {}
    for k in ("prolif", "apopt"):
        s = d.scores[f"{k}_a"]
        ok = np.isfinite(s)
        rho_s[k] = float(spearmanr(J[ok], s[ok]).statistic) if ok.sum() > 10 and np.std(J[ok]) > 0 else np.nan
    row = dict(growth_dist=gd, growth_dist_nt=gd_nt, organ_growth_err=organ_err,
               migration=migration_px / s3.rms_radius(xa), migration_px=migration_px, spread=spread,
               reflected=bool(np.linalg.det(R) < 0), kappa_ba=k_ba, kappa_ab=k_ab, K=K, diag_mass=diag,
               mass_t2=float(g2.sum()), col_err=float(np.abs(c - g2).sum() / g2.sum()), kl_rows=kl(r, g1),
               rho_cv=float(np.sqrt(g1 @ (rho - g1 @ rho / g1.sum()) ** 2 / g1.sum())),
               dead_share=float(g1[rho < -0.5].sum() / g1.sum()),
               rho_J_prolif=rho_s["prolif"], rho_J_apopt=rho_s["apopt"])
    if genes:
        row["gene_r"] = selection.gene_r(P, d.Ya, d.Yb, seed=seed)
    return row, dict(xi=xi.astype(np.float32), J=J.astype(np.float32), T_org=T_org.astype(np.float32))


# ---------- result tables (bench/scripts/temporal_align.py)
def runs_dir(out, seed):
    from pathlib import Path
    return Path(out) / "runs" / f"seed{seed}"


def load_runs(out, seed):
    """All sweep rows of a seed (one table per pair), one frame."""
    files = sorted(runs_dir(out, seed).glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def load_final(out):
    """All final rows (one table per pair and seed), one frame, or None."""
    from pathlib import Path
    files = sorted((Path(out) / "final").glob("*/runs_seed*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def topk_columns(P, k):
    """Each column's k largest entries: (rows k x n2 int32, values k x n2 float32, covered share of the mass)."""
    P = np.asarray(P)
    k = min(k, P.shape[0])
    idx = np.argpartition(-P, k - 1, axis=0)[:k]
    vals = np.take_along_axis(P, idx, axis=0)
    return idx.astype(np.int32), vals.astype(np.float32), float(vals.sum() / max(P.sum(), 1e-30))


def dense_from_topk(idx, vals, n1):
    P = np.zeros((n1, idx.shape[1]), dtype=np.float64)
    np.put_along_axis(P, idx.astype(np.int64), vals.astype(np.float64), axis=0)
    return P
