"""Cross-species alignment of the chain sections, scored by how well it commutes with the temporal couplings.

Rungs: the cross pairs X1-X10 of the series (bench/configs/series.yaml, tier full), each a (human stage, mouse
stage); the section of a stage is the one of bench/configs/chains.yaml. A cross coupling Q_r (human x mouse) per rung.

Loops: two rungs r = (h, m), r' = (h', m') of the ladder where each species advances at most one stage and at
least one does: 6 squares (both advance) and 6 triangles (one species stays, identity on that side). Starting from
a tile of one species' earlier section, two paths reach the other species' later section:
    from human:  path 1 = K_H(h -> h') K(Q_r')        (own species first, then across)
                 path 2 = K(Q_r) K_M(m -> m')         (across first, then forward in time)
    from mouse:  path 1 = K_M(m -> m') K(Q_r'^T),   path 2 = K(Q_r^T) K_H(h -> h')
K(P) is P with its rows normalized (a Markov kernel); a row with almost no mass (below `drop_floor` x the mean: a
tile that dies in a semi-relaxed temporal coupling, or one an unbalanced cross coupling drops) gets the target's
masses, which say nothing. If the alignments commute, both paths send a tile to the same place.

Commutativity (per start tile i, two distributions K1[i], K2[i] over the target tiles at positions y, centred and in
units of the target's RMS radius, D = cdist(y, y)):
- comm: the energy distance sqrt(2 E|X - Y| - E|X - X'| - E|Y - Y'|), X ~ K1[i], Y ~ K2[i], mass-weighted mean over
  the start tiles (bins). 0 iff the two distributions agree. A metric on distributions that credits near misses.
- sharp: the same distance of each path from the uninformative kernel (every row = the target masses), averaged over
  the two paths. A diffuse coupling commutes trivially, so comm is read against sharp (comm_rel = comm / sharp) and
  against the rungs' accuracy.
- tv_organ: total variation between the organ compositions the two paths reach; bary: distance between the two
  barycentres (target RMS radii); overlap: sum_j min(K1_ij, K2_ij).

Methods (run names): fgw_a<alpha> balanced FGW on the ortholog embedding (alpha the GW weight; 0 linear OT, 1 GW on
space only), ufgw_a<alpha>_t<tau> the same unbalanced with tau_a = tau_b = tau, label_wot / label_fgw_a<alpha> on the
organ-label cost (references: they use the annotation), moscot_a<alpha> moscot's FGW. Unbalanced runs are named by the
marginal penalty rho (ufgw_a<alpha>_r<rho>, KL weight in cost units; OTT's tau = rho / (rho + epsilon)).

scripts: bench/scripts/cross_align.py; evaluation: demos/cross_alignment.ipynb.
"""
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from mgw import stack3d as s3
from mgw.bench import metrics, prepare, selection, temporal as tp

METHODS = ("fgw", "ufgw", "label_wot", "label_fgw", "moscot")
REFERENCES = ("label_wot", "label_fgw")       # use the organ labels: plotted, never candidates
DIRECTIONS = ("h", "m")                        # start from human / from mouse


# ---------- rungs and loops
@dataclass(frozen=True)
class Rung:
    id: str
    h: str              # human section
    m: str              # mouse section
    stage_h: str
    stage_m: str
    ih: int             # index of the stage in the human chain
    im: int

    @property
    def key(self):
        return f"crossloop_{self.h}__{self.m}"


@dataclass(frozen=True)
class Loop:
    id: str             # "X3-X5"
    r1: Rung
    r2: Rung
    th: str             # temporal pair of the human step (H1..), "" if human stays
    tm: str

    @property
    def kind(self):
        return "square" if self.th and self.tm else "triangle"

    def start(self, direction):
        return self.r1.h if direction == "h" else self.r1.m

    def target(self, direction):
        return self.r2.m if direction == "h" else self.r2.h

    def intermediates(self, direction):
        """(path 1's intermediate section, path 2's), "" where a path has none."""
        if direction == "h":
            return (self.r2.h if self.th else ""), (self.r1.m if self.tm else "")
        return (self.r2.m if self.tm else ""), (self.r1.h if self.th else "")

    def steps(self, direction):
        """[(label, kind, from, to)] of path 1 and of path 2; kind cross or temporal."""
        r1, r2 = self.r1, self.r2
        if direction == "h":
            p1 = ([(self.th, "temporal", r1.h, r2.h)] if self.th else []) + [(r2.id, "cross", r2.h, r2.m)]
            p2 = [(r1.id, "cross", r1.h, r1.m)] + ([(self.tm, "temporal", r1.m, r2.m)] if self.tm else [])
        else:
            p1 = ([(self.tm, "temporal", r1.m, r2.m)] if self.tm else []) + [(r2.id, "cross", r2.m, r2.h)]
            p2 = [(r1.id, "cross", r1.m, r1.h)] + ([(self.th, "temporal", r1.h, r2.h)] if self.th else [])
        return p1, p2


def rungs(cfg):
    """The ladder's rungs on the chain sections, in ladder order."""
    from mgw.bench import series as sr
    chains = tp.load_chains(cfg.chains_path())
    ser = sr.load(cfg.series_path())
    idx_h = {tp.stage_of(s): (k, s) for k, s in enumerate(chains["human"])}
    idx_m = {tp.stage_of(s): (k, s) for k, s in enumerate(chains["mouse"])}
    out = []
    for pid, hs, ms in selection.rungs(ser, "full"):
        if hs not in idx_h or ms not in idx_m:
            raise ValueError(f"rung {pid}: stage {hs} or {ms} has no section in the chains")
        (ih, h), (im, m) = idx_h[hs], idx_m[ms]
        out.append(Rung(pid, h, m, hs, ms, ih, im))
    return out


def loops(rung_list):
    """Every rung pair where each species advances by at most one stage and one at least: squares and triangles.
    The temporal pair ids are those of temporal.chain_pairs (H1 = human chain step 0 -> 1)."""
    out = []
    for i, r1 in enumerate(rung_list):
        for r2 in rung_list[i + 1:]:
            dh, dm = r2.ih - r1.ih, r2.im - r1.im
            if dh in (0, 1) and dm in (0, 1) and dh + dm > 0:
                out.append(Loop(f"{r1.id}-{r2.id}", r1, r2, f"H{r1.ih + 1}" if dh else "", f"M{r1.im + 1}" if dm else ""))
    return out


def loop_table(loop_list):
    return pd.DataFrame([dict(loop=l.id, kind=l.kind, r1=l.r1.id, r2=l.r2.id, th=l.th, tm=l.tm,
                              h1=l.r1.h, m1=l.r1.m, h2=l.r2.h, m2=l.r2.m) for l in loop_list])


def rung_table(rung_list):
    return pd.DataFrame([dict(rung=r.id, h=r.h, m=r.m, stage_h=r.stage_h, stage_m=r.stage_m) for r in rung_list])


# ---------- features
def feature_path(rung, cfg, seed, root):
    return prepare.feature_file(root, rung.key, seed, cfg.feature_hash())


def cross_features(rung, cfg, seed, root, ortho, force=False):
    """The ortholog embedding of one human and one mouse section (prepare.pair_features' cross recipe on single
    sections: genes z-scored within each section) and the held-out 1:1 orthologs. Returns the file."""
    from mgw import crossspecies as cs
    from mgw import orthologs
    out = feature_path(rung, cfg, seed, root)
    if out.exists() and not force:
        return out
    f, sh = cfg.raw["features"], cfg.section_hash()
    A, B = prepare.load_counts(root, rung.h, seed, sh), prepare.load_counts(root, rung.m, seed, sh)
    h2m = dict(zip(ortho.human, ortho.mouse))
    gh, gm = orthologs.restrict_to_orthologs(A.var_names, B.var_names, ortho)
    eval_h = cs.pick_eval_genes(A[:, gh].X, B[:, gm].X, gh, f["n_eval_genes"])
    eval_m = np.array([h2m[g] for g in eval_h])
    EA, EB, emb_genes = orthologs.shared_embedding(A, B, ortho, n_hvg=f["n_hvg"], n_comps=f["pca_comp"],
                                                   exclude=eval_h, random_state=seed)
    prepare.atomic_savez(out, emb_h=EA.astype(np.float32), emb_m=EB.astype(np.float32),
                         Yh=cs.dense(A[:, list(eval_h)].X).astype(np.float16),
                         Ym=cs.dense(B[:, list(eval_m)].X).astype(np.float16),
                         eval_h=np.asarray(eval_h, dtype=str), eval_m=np.asarray(eval_m, dtype=str),
                         emb_genes=np.asarray(emb_genes, dtype=str), n_orthologs=len(gh))
    return out


@dataclass
class RungData:
    rung: Rung
    sh: prepare.Section
    sm: prepare.Section
    emb_h: np.ndarray
    emb_m: np.ndarray
    Yh: np.ndarray
    Ym: np.ndarray
    n_orthologs: int


def load_rung(rung, cfg, seed, root):
    sh = cfg.section_hash()
    z = np.load(feature_path(rung, cfg, seed, root), allow_pickle=False)
    a, b = prepare.load_section(root, rung.h, seed, sh), prepare.load_section(root, rung.m, seed, sh)
    if len(z["emb_h"]) != a.n or len(z["emb_m"]) != b.n:
        raise RuntimeError(f"{rung.id}: features and section caches disagree; rerun prepare")
    return RungData(rung, a, b, z["emb_h"].astype(np.float64), z["emb_m"].astype(np.float64),
                    z["Yh"].astype(np.float32), z["Ym"].astype(np.float32), int(z["n_orthologs"]))


# ---------- runs
def run_name(method, params):
    if method == "label_wot":
        return method
    name = f"{method}_a{params['alpha']:g}"
    return name + (f"_r{params['rho']:g}" if method == "ufgw" else "")


def parse_run(name):
    if name == "label_wot":
        return "label_wot", {}
    for m in sorted(METHODS, key=len, reverse=True):
        if name.startswith(m + "_"):
            parts = {t[0]: float(t[1:]) for t in name[len(m) + 1:].split("_")}
            return m, {dict(a="alpha", r="rho")[k]: v for k, v in parts.items()}
    raise ValueError(f"run {name!r}: unknown method")


def grid(cfg, methods=METHODS):
    """[(method, params)] of the sweep (cross.sweep)."""
    sw = cfg.raw["cross"]["sweep"]
    out = []
    if "fgw" in methods:
        out += [("fgw", dict(alpha=float(a))) for a in sw["alpha"]]
    if "ufgw" in methods:
        out += [("ufgw", dict(alpha=float(a), rho=float(r))) for a in sw["alpha"] for r in sw["rho"]]
    if "label_wot" in methods:
        out.append(("label_wot", {}))
    if "label_fgw" in methods:
        out += [("label_fgw", dict(alpha=float(a))) for a in sw["label_fgw_alpha"]]
    if "moscot" in methods:
        out += [("moscot", dict(alpha=float(a))) for a in sw["moscot_alpha"]]
    return out


def is_reference(method):
    return method in REFERENCES


@lru_cache(maxsize=8)
def _sinkhorn_unbalanced(tau, tol, maxit):
    import jax
    from ott.geometry import geometry
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    solver = sinkhorn.Sinkhorn(max_iterations=maxit, threshold=tol)

    @jax.jit
    def solve(M, a, b, eps):
        prob = linear_problem.LinearProblem(geometry.Geometry(cost_matrix=M, epsilon=eps), a=a, b=b, tau_a=tau,
                                            tau_b=tau)
        out = solver(prob)
        return out.matrix, out.converged, out.n_iters
    return solve


def linear_ot(M, a, b, epsilon, tau=1.0, tol=1e-5, maxit=20000):
    """Entropic OT on a cross cost, balanced (selection.wot) or with KL-relaxed marginals (tau < 1)."""
    if tau >= 1:
        return selection.wot(M, a, b, epsilon, tol, maxit)
    import jax.numpy as jnp
    f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)
    P, conv, it = _sinkhorn_unbalanced(float(tau), tol, maxit)(f32(M), f32(a), f32(b), epsilon)
    P = np.asarray(P, dtype=np.float64)
    if not np.isfinite(P).all():
        raise FloatingPointError("non-finite coupling")
    return P, dict(converged=bool(conv), n_iters=int(it))


class RungProblem:
    """One rung, ready to solve: masses, the ortholog cross cost, the label cost and the intra-section distances."""

    def __init__(self, data: RungData, cfg):
        from mgw.bench import solvers
        self.data, self.cfg = data, cfg
        self.a, self.b = selection.masses(data.sh.comp), selection.masses(data.sm.comp)
        C = cdist(data.emb_h, data.emb_m)
        self.M = C / np.quantile(C, cfg.raw["features"]["m_scale_quantile"])
        self.Ca, self.Cb = solvers.dist_norm(data.sh.xy), solvers.dist_norm(data.sm.xy)
        self._L = None

    @property
    def L(self):
        if self._L is None:
            self._L = selection.label_cost(self.data.sh.comp, self.data.sm.comp)
        return self._L

    def solve(self, method, params):
        """(Q, info): Q human x mouse."""
        from mgw.bench import solvers
        c = self.cfg.raw["cross"]
        eps = c["epsilon"]
        if method == "moscot":
            d = self.data
            t0 = time.perf_counter()
            Q = solvers.moscot_pair(d.sh.xy, d.emb_h, d.sm.xy, d.emb_m, alpha=params["alpha"],
                                    epsilon=c["moscot_epsilon"])
            return np.asarray(Q, np.float64), dict(converged=True, outer_stopped=True, n_outer=-1,
                                                   seconds=time.perf_counter() - t0)
        alpha = params.get("alpha", 0.0)
        tau = params["rho"] / (params["rho"] + eps) if "rho" in params else 1.0
        M = self.L if method in REFERENCES else self.M
        t0 = time.perf_counter()
        if alpha == 0:
            Q, info = linear_ot(M, self.a, self.b, eps, tau, c["wot_tol"], c["wot_maxit"])
            info = dict(converged=info["converged"], outer_stopped=info["converged"], n_outer=0, n_iters=info["n_iters"])
        else:
            p = solvers.FGWParams.from_mapping({**self.cfg.raw["fgw"], "alpha": alpha, "epsilon": eps})
            Q, info = solvers.fgw(None, None, None if alpha == 1 else M, p, tau_a=tau, tau_b=tau, Ca=self.Ca,
                                  Cb=self.Cb, a=self.a, b=self.b)
            info = {k: info[k] for k in ("converged", "n_outer", "outer_stopped", "inner_converged", "ugw_fallback")
                    if k in info}
        info["seconds"] = time.perf_counter() - t0
        return Q, info

    def objective(self, Q, alpha, method):
        from mgw.bench import solvers
        M = self.L if method in REFERENCES else self.M
        Qn = Q / Q.sum()
        return float(alpha * solvers.gw_energy(self.Ca, self.Cb, Qn) + (1 - alpha) * (M * Qn).sum())


def evaluate_rung(Q, prob: RungProblem, method, params, seed=0):
    """Accuracy of a cross coupling: kappa_bal both ways, diag mass, held-out ortholog gene r, mass kept."""
    d = prob.data
    mb = prob.cfg.raw["cross"]["min_bins"]
    k_hm, K = metrics.score(Q, d.sh.comp, d.sm.comp, min_bins=mb)
    k_mh, _ = metrics.score(Q.T, d.sm.comp, d.sh.comp, min_bins=mb)
    F1, F2 = metrics.row_normalize(d.sh.comp), metrics.row_normalize(d.sm.comp)
    T = F1.T @ Q @ F2
    sc = metrics.SCORED
    diag = float(np.trace(T[np.ix_(sc, sc)]) / max(T[np.ix_(sc, sc)].sum(), 1e-30))
    r, c = Q.sum(1), Q.sum(0)
    alpha = params.get("alpha", 0.0)
    return dict(kappa_hm=k_hm, kappa_mh=k_mh, kappa=(k_hm + k_mh) / 2, K=K, diag_mass=diag,
                gene_r=selection.gene_r(Q, d.Yh, d.Ym, seed=seed), mass=float(Q.sum()),
                marg_err=float(np.abs(r - prob.a).sum() + np.abs(c - prob.b).sum()),
                objective=prob.objective(Q, alpha, "moscot" if method == "moscot" else method)
                if method != "moscot" else np.nan,
                dropped_h=float(prob.a[r < prob.cfg.raw["cross"]["drop_floor"] * r.mean()].sum()),
                dropped_m=float(prob.b[c < prob.cfg.raw["cross"]["drop_floor"] * c.mean()].sum()))


# ---------- kernels, paths, commutativity
def kernel(P, b, floor=1e-3):
    """(K, dropped mask): P with rows normalized; rows below floor x the mean row mass get b / sum b."""
    P = np.asarray(P, dtype=np.float32)
    r = P.sum(1)
    dead = r < floor * r.mean()
    K = P / np.where(dead, 1.0, r)[:, None]
    if dead.any():
        K[dead] = (np.asarray(b, np.float32) / np.sum(b))[None, :]
    return K, dead


def path_kernels(loop, direction, KQ, KT):
    """(path 1 factors, path 2 factors): lists of kernels whose product is each path. KQ: {rung id: (K h->m,
    K m->h)}; KT: {temporal pair id: K t1->t2}."""
    p1, p2 = loop.steps(direction)
    side = 0 if direction == "h" else 1          # a cross step always leaves the start species

    def k(step):
        label, kind = step[0], step[1]
        return KT[label] if kind == "temporal" else KQ[label][side]
    return [k(s) for s in p1], [k(s) for s in p2]


def _jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


@lru_cache(maxsize=1)
def _energy_fn():
    jax, jnp = _jax()
    hi = jax.lax.Precision.HIGHEST

    @jax.jit
    def f(K1, K2, y, b, F):
        sq = (y ** 2).sum(1)
        D = jnp.sqrt(jnp.maximum(sq[:, None] + sq[None, :] - 2 * jnp.matmul(y, y.T, precision=hi), 0.0))
        K1D, K2D = jnp.matmul(K1, D, precision=hi), jnp.matmul(K2, D, precision=hi)
        e12, e11, e22 = (K1D * K2).sum(1), (K1D * K1).sum(1), (K2D * K2).sum(1)
        Db = jnp.matmul(D, b, precision=hi)
        bDb = b @ Db
        ed = 2 * e12 - e11 - e22
        s1 = 2 * jnp.matmul(K1, Db, precision=hi) - e11 - bDb
        s2 = 2 * jnp.matmul(K2, Db, precision=hi) - e22 - bDb
        tv = 0.5 * jnp.abs(jnp.matmul(K1 - K2, F, precision=hi)).sum(1)
        bary = jnp.sqrt(((jnp.matmul(K1 - K2, y, precision=hi)) ** 2).sum(1))
        overlap = jnp.minimum(K1, K2).sum(1)
        return ed, s1, s2, tv, bary, overlap
    return f


@lru_cache(maxsize=1)
def _product_fn():
    jax, jnp = _jax()
    return jax.jit(lambda A, B: jnp.matmul(A, B, precision=jax.lax.Precision.HIGHEST))


def product(factors):
    """The product of a list of kernels, on the device (float32, full precision)."""
    _, jnp = _jax()
    out = jnp.asarray(factors[0], dtype=jnp.float32)
    for K in factors[1:]:
        out = _product_fn()(out, jnp.asarray(K, dtype=jnp.float32))
    return out


def energy_commutativity(K1, K2, y, b, F):
    """Per start tile (numpy float64): energy distance between the paths, each path's distance from the
    uninformative kernel, organ TV, barycentre distance, overlap. y: target positions (RMS units), b: target masses,
    F: target organ compositions (rows normalized)."""
    _, jnp = _jax()
    f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)
    bn = np.asarray(b, np.float64) / np.sum(b)
    out = _energy_fn()(f32(K1), f32(K2), f32(y), f32(bn), f32(F))
    ed, s1, s2, tv, bary, ov = (np.asarray(o, dtype=np.float64) for o in out)
    root = lambda v: np.sqrt(np.maximum(v, 0.0))
    return dict(comm=root(ed), sharp1=root(s1), sharp2=root(s2), tv_organ=tv, bary=bary, overlap=ov)


def target_frame(xy):
    """Positions centred and in units of their RMS radius."""
    xy = np.asarray(xy, np.float64)
    xy = xy - xy.mean(0)
    return xy / s3.rms_radius(xy)


@dataclass
class SectionInfo:
    xy: np.ndarray
    comp: np.ndarray
    w: np.ndarray       # bin masses (sum 1)


def section_info(sec):
    return SectionInfo(sec.xy, sec.comp, selection.masses(sec.comp))


def evaluate_loop(loop, direction, KQ, KT, sections, per_tile=False):
    """(row, per-tile dict or None) of one loop and direction. sections: {name: SectionInfo}."""
    t0 = time.perf_counter()
    f1, f2 = path_kernels(loop, direction, KQ, KT)
    K1, K2 = product(f1), product(f2)
    start, tgt = sections[loop.start(direction)], sections[loop.target(direction)]
    res = energy_commutativity(K1, K2, target_frame(tgt.xy), tgt.w, metrics.row_normalize(tgt.comp))
    w = start.w / start.w.sum()
    mean = lambda v: float(w @ v)
    sharp = (res["sharp1"] + res["sharp2"]) / 2
    row = dict(loop=loop.id, kind=loop.kind, direction=direction, start=loop.start(direction),
               target=loop.target(direction), comm=mean(res["comm"]), sharp=mean(sharp), sharp1=mean(res["sharp1"]),
               sharp2=mean(res["sharp2"]), tv_organ=mean(res["tv_organ"]), bary=mean(res["bary"]),
               overlap=mean(res["overlap"]), seconds=time.perf_counter() - t0)
    row["comm_rel"] = row["comm"] / row["sharp"] if row["sharp"] > 0 else np.nan
    tiles = None
    if per_tile:
        K1n, K2n = np.asarray(K1, np.float64), np.asarray(K2, np.float64)
        Fs = metrics.row_normalize(start.comp) * w[:, None]
        Ft = metrics.row_normalize(tgt.comp)
        tiles = dict(comm=res["comm"].astype(np.float32), sharp=sharp.astype(np.float32), K1=K1n, K2=K2n,
                     flow1=(Fs.T @ K1n @ Ft).astype(np.float32), flow2=(Fs.T @ K2n @ Ft).astype(np.float32))
    return row, tiles


# ---------- temporal couplings
def temporal_runs(tdir, seed, pairs):
    """Run names with a dense final coupling for every one of `pairs` (temporal_align.py final)."""
    runs = None
    for p in pairs:
        have = {f.name[:-len(f"_seed{seed}.npz")] for f in (Path(tdir) / "final" / p).glob(f"*_seed{seed}.npz")}
        runs = have if runs is None else runs & have
    return sorted(runs or ())


def temporal_kernels(tdir, run, seed, pairs, floor=1e-3):
    """{pair: (K t1 -> t2 float32, dropped share of t1's mass)} of one temporal run."""
    out = {}
    for p in pairs:
        z = np.load(Path(tdir) / "final" / p / f"{run}_seed{seed}.npz", allow_pickle=False)
        K, dead = kernel(z["P"], z["g2"], floor)
        g1 = np.asarray(z["g1"], np.float64)
        out[p] = (K, float(g1[dead].sum() / g1.sum()))
    return out


def temporal_sections(tdir, seed, pair):
    """(a, b) section names of a temporal pair (from its final files)."""
    f = next((Path(tdir) / "final" / pair).glob(f"*_seed{seed}.npz"))
    z = np.load(f, allow_pickle=False)
    return str(z["a"]), str(z["b"])


# ---------- the viewer's arrays
def topk_rows(K, k):
    """Each row's k largest entries: (idx uint16, mass uint8 relative to the row maximum, the row maximum, kept
    share of the row's mass). Mass of entry j of row i ~ val[i, j] / 255 * max[i]."""
    K = np.asarray(K, dtype=np.float32)
    k = min(k, K.shape[1])
    idx = np.argpartition(-K, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(K, idx, axis=1)
    mx = vals.max(1, keepdims=True)
    q = np.round(255 * vals / np.where(mx > 0, mx, 1)).astype(np.uint8)
    kept = (vals.sum(1) / np.maximum(K.sum(1), 1e-30)).astype(np.float32)
    return idx.astype(np.uint16), q, mx[:, 0].astype(np.float32), kept


def view_arrays(loop, direction, KQ, KT, sections, k=48, tiles=None):
    """The arrays of the loop viewer (mgw/bench/loopview.py): each section's positions, organs and frame transform
    into the start section's frame, and per start tile the top-k of each path's intermediate and final kernel."""
    f1, f2 = path_kernels(loop, direction, KQ, KT)
    i1, i2 = loop.intermediates(direction)
    start, tgt = loop.start(direction), loop.target(direction)
    if tiles is None:
        _, tiles = evaluate_loop(loop, direction, KQ, KT, sections, per_tile=True)
    out = dict(loop=loop.id, kind=loop.kind, direction=direction, start=start, target=tgt, inter1=i1, inter2=i2,
               organs=np.asarray(metrics.ORGANS), comm=tiles["comm"], sharp=tiles["sharp"])
    p1, p2 = loop.steps(direction)
    out["steps1"] = np.asarray([f"{s[0]}:{s[1]}:{s[2]}:{s[3]}" for s in p1])
    out["steps2"] = np.asarray([f"{s[0]}:{s[1]}:{s[2]}:{s[3]}" for s in p2])
    # each path's factors (p1a, p1b; p2a, p2b; a one-step path has only "a"), composed in the viewer: the factors are
    # sharper than their product, so their top-k keeps far more of the mass than the product's would
    for tag, fs in (("p1", f1), ("p2", f2)):
        for j, K in zip("ab", fs):
            out[f"{tag}{j}_idx"], out[f"{tag}{j}_val"], out[f"{tag}{j}_max"], out[f"{tag}{j}_kept"] = topk_rows(K, k)
    out["w"] = sections[start].w.astype(np.float32)
    # frames: every section into the start's frame, along the path that reaches it (Procrustes, reflections allowed)
    xs = sections[start].xy
    frames = {start: (np.eye(2), np.zeros(2))}
    if i1:
        frames[i1] = tp.weighted_procrustes(xs, sections[i1].xy, np.asarray(f1[0], np.float64))
    if i2:
        frames[i2] = tp.weighted_procrustes(xs, sections[i2].xy, np.asarray(f2[0], np.float64))
    K12 = (np.asarray(tiles["K1"], np.float64) + np.asarray(tiles["K2"], np.float64)) / 2
    frames[tgt] = tp.weighted_procrustes(xs, sections[tgt].xy, K12 * sections[start].w[:, None])
    for name in {start, tgt, i1, i2} - {""}:
        s = sections[name]
        R, t = frames[name]
        out[f"{name}|xy"] = s.xy.astype(np.float32)
        out[f"{name}|R"], out[f"{name}|t"] = np.asarray(R, np.float64), np.asarray(t, np.float64)
        out[f"{name}|organ"] = metrics.majority(s.comp)
    out["sections"] = np.asarray([n for n in (start, i1, i2, tgt) if n])
    return out


# ---------- result tables
def runs_dir(out, seed):
    return Path(out) / "runs" / f"seed{seed}"


def comm_dir(out, seed):
    return Path(out) / "comm" / f"seed{seed}"


def _concat(files):
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def load_runs(out, seed):
    """Rung rows of a seed (one table per setting)."""
    return _concat(sorted(runs_dir(out, seed).glob("*.csv")))


def load_comm(out, seed):
    """Loop rows of a seed (one table per setting)."""
    return _concat(sorted(comm_dir(out, seed).glob("*.csv")))


def load_final(out):
    """(rung rows, loop rows) of the final runs, both seeds."""
    f = Path(out) / "final"
    return _concat(sorted(f.glob("runs_seed*.csv"))), _concat(sorted(f.glob("comm_seed*.csv")))
