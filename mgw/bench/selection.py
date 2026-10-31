"""Which sections to align: one section per stage, one chain per species, chosen jointly with the cross-species pairs.

The cross pairs of the series form a ladder: rung X1 links a human and a mouse stage, the next rung moves one or both
species to their next stage. A choice of sections costs

    sum_temporal(w_cost + w |dL|) + lambda sum_cross(w_cost + w |dL|)

with w_cost the transport cost of Wasserstein OT on the organ labels and dL the difference in raw laterality L. Raw L,
not L_pair: L_pair is smoothed along the section order, which the control below uses as its ground truth. L comes from
the labels too (laterality_test/full_run.ipynb), so dL is a reweighting of the labels, not independent evidence.

The control scores what a cost sees of the sectioning plane: within a human embryo the section numbers follow the
cutting order, so a cost should rise with the distance between two sections. Gene r through a label coupling uses no
genes to build the coupling, so it can compare the chains, provided it passes that control (decide).

scripts: bench/scripts/select_sections.py; evaluation: demos/slice_selection.ipynb.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mgw.bench import metrics

REPO = Path(__file__).resolve().parents[2]
MIN_CLASSIFIED = 0.03         # docs/hesta_mosta_slide_pairs.md: below this L rests on too few spots
NO_COUNTS = {"CS23_E2S2", "CS23_E2S4"}   # log-normalized X only (series.yaml)
PANEL_MIN = 0.8               # a gene panel below this share of its stage's median is flagged
MIN_CONTROL_SECTIONS = 4      # embryos with fewer sections say little in the control
CONTROL_RHO_MIN = 0.3         # decide: gene r below this median rho cannot arbitrate between the chains
DL_WEAK = 0.25                # chain steps above this |dL| are flagged
MAIN_W = (0.0, 0.5)           # the two variants the decision compares
GRID_W = (0.0, 0.25, 0.5, 1.0)
GRID_LAMBDA = (0.0, 0.5, 1.0, 2.0)


# ---------- candidates and the ladder
def rungs(series, tier="full", stages=None):
    """[(pair id, human stage, mouse stage)] of the tier's cross pairs, in file order; stages: keep only these."""
    out = [(pid, series.pairs[pid].a.stage, series.pairs[pid].b.stage) for pid in series.tier(tier)]
    return [r for r in out if stages is None or (r[1] in stages and r[2] in stages)]


def stage_orders(ladder):
    """(human stages, mouse stages) in ladder order."""
    return list(dict.fromkeys(r[1] for r in ladder)), list(dict.fromkeys(r[2] for r in ladder))


def candidates(scores, ladder, overview=None, series=None):
    """One row per section of the laterality table: its stage, L, and why it is excluded (`excluded`, empty if it is a
    candidate) or flagged (`flag`). scores: prepare.laterality_table(); overview: data_analysis/dataset_overview.csv."""
    from mgw import crossspecies as cs
    h, m = stage_orders(ladder)
    stages = set(h) | set(m)
    t = scores.reset_index()
    names = t["key"].map(cs.key_name)
    out = pd.DataFrame(dict(section=names, dataset=t["dataset"], stage=t["stage"].astype(str), embryo=t["embryo"].astype(str),
                            number=t["section"], L=t["L_laterality"], frac_classified=t["frac_classified"],
                            n_bins=t["n_spots"]))

    def why(r):
        if pd.isna(r.number):
            return "combined 3D file"
        if r.stage not in stages:
            return "stage not in the ladder"
        if r.section in NO_COUNTS:
            return "no counts"
        if not r.frac_classified >= MIN_CLASSIFIED:
            return f"under {MIN_CLASSIFIED:.0%} classified"
        return ""

    out["excluded"] = [why(r) for r in out.itertuples()]
    out["flag"] = ""
    if overview is not None:
        panel = overview.set_index("slice").n_genes_panel
        median = overview.groupby("stage").n_genes_panel.median()
        low = out.section.map(panel) < PANEL_MIN * out.stage.map(median)
        out.loc[low, "flag"] = "small gene panel"
    if series is not None:
        in_series = {s for e in series.embryos.values() for s in e.sections if not e.name.startswith("smoke")}
        out["in_series"] = out.section.isin(in_series)
    return out.reset_index(drop=True)


def stage_sections(cand, stages):
    """{stage: [candidate sections]} in table order."""
    ok = cand[cand.excluded.fillna("") == ""]
    return {s: ok.section[ok.stage == s].tolist() for s in stages}


def edge_list(cand, ladder):
    """The section pairs to solve: rows (kind, a, b, gap). temporal: consecutive stages of a species, a the earlier;
    cross: the stage pairs of the ladder, a human, b mouse; control: within a HESTA embryo, gap the distance in section
    numbers (series lacking sections keep their true distance)."""
    h, m = stage_orders(ladder)
    sec = stage_sections(cand, h + m)
    rows = []
    for order in (h, m):
        for s, t in zip(order, order[1:]):
            rows += [("temporal", a, b, np.nan) for a in sec[s] for b in sec[t]]
    for hs, ms in dict.fromkeys((r[1], r[2]) for r in ladder):
        rows += [("cross", a, b, np.nan) for a in sec[hs] for b in sec[ms]]
    ok = cand[(cand.excluded.fillna("") == "") & (cand.dataset == "HESTA")]
    for _, g in ok.groupby(["stage", "embryo"], sort=False):
        if len(g) < MIN_CONTROL_SECTIONS:
            continue
        g = g.sort_values("number")
        for i, (a, na) in enumerate(zip(g.section, g.number)):
            rows += [("control", a, b, abs(nb - na)) for b, nb in zip(g.section.iloc[i + 1:], g.number.iloc[i + 1:])]
    return pd.DataFrame(rows, columns=["kind", "a", "b", "gap"])


# ---------- costs
def label_cost(comp_a, comp_b):
    """Organ-label cross cost: 0 same organ, 1 different organs, 0.5 when a side is Other/Excluded
    (demos/label_alignment_best.ipynb)."""
    Ka, Kb = metrics.row_normalize(comp_a)[:, metrics.SCORED], metrics.row_normalize(comp_b)[:, metrics.SCORED]
    return 1 - Ka @ Kb.T - 0.5 * (1 - np.outer(Ka.sum(1), Kb.sum(1)))


def masses(comp):
    """A tile's mass is its share of the section's bins."""
    m = np.asarray(comp, dtype=np.float64).sum(1)
    return m / m.sum()


@lru_cache(maxsize=4)
def _sinkhorn(tol, maxit):
    import jax
    from ott.geometry import geometry
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    solver = sinkhorn.Sinkhorn(max_iterations=maxit, threshold=tol)

    @jax.jit
    def solve(M, a, b, eps):
        out = solver(linear_problem.LinearProblem(geometry.Geometry(cost_matrix=M, epsilon=eps), a=a, b=b))
        return out.matrix, out.converged, out.n_iters
    return solve


def wot(M, a, b, epsilon=5e-3, tol=1e-5, maxit=20000):
    """Balanced entropic OT with masses a, b. (P, info); float32, so the tolerance must stay reachable (1e-5)."""
    import jax.numpy as jnp
    P, converged, n_iters = _sinkhorn(tol, maxit)(jnp.asarray(M, dtype=jnp.float32), jnp.asarray(a, dtype=jnp.float32),
                                                  jnp.asarray(b, dtype=jnp.float32), epsilon)
    P = np.asarray(P, dtype=np.float64)
    if not np.isfinite(P).all():
        raise FloatingPointError("non-finite coupling")
    return P, dict(converged=bool(converged), n_iters=int(n_iters),
                   marg_err=float(np.abs(P.sum(1) - a).sum() + np.abs(P.sum(0) - b).sum()))


def edge_scores(comp_a, comp_b, L_a, L_b, *, epsilon=5e-3, tol=1e-5, maxit=20000, min_bins=5):
    """(row, P) of one section pair: the label coupling's transport cost and the reference scores."""
    a, b = masses(comp_a), masses(comp_b)
    M = label_cost(comp_a, comp_b)
    P, info = wot(M, a, b, epsilon, tol, maxit)
    pa, pb = np.sum(comp_a, 0) / np.sum(comp_a), np.sum(comp_b, 0) / np.sum(comp_b)
    kab, K = metrics.score(P, comp_a, comp_b, min_bins=min_bins)
    kba, _ = metrics.score(P.T, comp_b, comp_a, min_bins=min_bins)
    return dict(w_cost=float((M * P).sum() / P.sum()), tv=float(0.5 * np.abs(pa - pb).sum()), dL=abs(L_a - L_b), K=K,
                kappa_ab=kab, kappa_ba=kba, bins_a=float(np.sum(comp_a)), bins_b=float(np.sum(comp_b)), **info), P


def transported_r(P, Yq, Yt, seed=0, chunk=2000):
    """Per gene, Pearson r over the query's tiles between Yq and Yt transported through P, and the same with Yt's tiles
    shuffled: crossspecies.coexpression in gene chunks. Yq, Yt dense, as a sparse product is about 20x slower here."""
    from mgw import crossspecies as cs
    Pn = np.asarray(P, dtype=np.float32)
    Pn = Pn / (Pn.sum(1, keepdims=True) + 1e-30)
    perm = np.random.default_rng(seed).permutation(len(Yt))
    r, r0 = [], []
    for j in range(0, Yq.shape[1], chunk):
        Yqc, Ytc = Yq[:, j:j + chunk].astype(np.float64), Yt[:, j:j + chunk]
        r.append(cs.colwise_r(Yqc, (Pn @ Ytc).astype(np.float64)))
        r0.append(cs.colwise_r(Yqc, (Pn @ Ytc[perm]).astype(np.float64)))
    return np.concatenate(r), np.concatenate(r0)


def gene_r(P, Ya, Yb, seed=0):
    """Mean per-gene r minus the shuffled one through P, averaged over the two directions; NaN genes are left out."""
    d = []
    for Pq, q, t in ((P, Ya, Yb), (P.T, Yb, Ya)):
        r, r0 = transported_r(Pq, q, t, seed=seed)
        d.append(np.nanmean(r) - np.nanmean(r0))
    return float(np.mean(d))


# ---------- the chain
def cost_table(edges, w):
    """{(a, b): w_cost + w dL} of the temporal and cross edges."""
    e = edges[edges.kind.isin(("temporal", "cross"))]
    return dict(zip(zip(e.a, e.b), e.w_cost + w * e.dL))


def chain_cost(ladder, chain_h, chain_m, cost, lam):
    """The objective of one choice. chain_h, chain_m: {stage: section}."""
    h, m = stage_orders(ladder)
    total = sum(cost[chain_h[s], chain_h[t]] for s, t in zip(h, h[1:]))
    total += sum(cost[chain_m[s], chain_m[t]] for s, t in zip(m, m[1:]))
    return total + lam * sum(cost[chain_h[hs], chain_m[ms]] for _, hs, ms in ladder)


def solve_chain(ladder, sections, cost, lam=1.0, exclude=()):
    """The exact minimum of chain_cost, by dynamic programming over the frontier (human section, mouse section) along
    the ladder. sections: {stage: [sections]}. Returns (cost, {stage: human section}, {stage: mouse section}).
    A missing edge raises KeyError, so an unfinished `costs` run can't pass unnoticed."""
    drop = set(exclude)
    secs = {s: [x for x in v if x not in drop] for s, v in sections.items()}
    _, h0, m0 = ladder[0]
    best = {(a, b): (lam * cost[a, b], {h0: a}, {m0: b}) for a in secs[h0] for b in secs[m0]}
    for (_, hp, mp), (_, hs, ms) in zip(ladder, ladder[1:]):
        new = {}
        for (a, b), (c, ch, cm) in best.items():
            for na in (secs[hs] if hs != hp else [a]):
                step_h = cost[a, na] if hs != hp else 0.0
                for nb in (secs[ms] if ms != mp else [b]):
                    cc = c + step_h + (cost[b, nb] if ms != mp else 0.0) + lam * cost[na, nb]
                    if (na, nb) not in new or cc < new[na, nb][0]:
                        new[na, nb] = (cc, {**ch, hs: na}, {**cm, ms: nb})
        best = new
    if not best:
        raise ValueError("no chain: a stage has no candidate left")
    return min(best.values(), key=lambda v: v[0])


def chain_edges(ladder, chain_h, chain_m):
    """[(kind, a, b)] of a chain."""
    h, m = stage_orders(ladder)
    out = [("temporal", chain_h[s], chain_h[t]) for s, t in zip(h, h[1:])]
    out += [("temporal", chain_m[s], chain_m[t]) for s, t in zip(m, m[1:])]
    return out + [("cross", chain_h[hs], chain_m[ms]) for _, hs, ms in ladder]


def margins(ladder, sections, cost, lam, chain_h, chain_m, total):
    """Per chosen section: what the objective rises by without it, and which section takes its place. NaN if its stage
    has no other candidate."""
    rows = []
    for species, chain in (("human", chain_h), ("mouse", chain_m)):
        for stage, sec in chain.items():
            if len(sections[stage]) < 2:
                rows.append(dict(species=species, stage=stage, section=sec, margin=np.nan, alternative=""))
                continue
            c, ah, am = solve_chain(ladder, sections, cost, lam, exclude=[sec])
            rows.append(dict(species=species, stage=stage, section=sec, margin=c - total,
                             alternative=(ah if species == "human" else am)[stage]))
    return pd.DataFrame(rows)


# ---------- the control and the decision
def control_table(edges):
    """Per embryo of the control edges: does each score rise with the distance between sections? rho: Spearman with
    gap. hit: the share of sections whose lowest score is at their nearest available section. Gene r is a similarity,
    so it enters negated."""
    c = edges[edges.kind == "control"].copy()
    if c.empty:
        return pd.DataFrame()
    c["embryo"] = c.a.str.rsplit("S", n=1).str[0]
    scores = dict(w_cost=c.w_cost, tv=c.tv, dL=c.dL, w_cost_dL=c.w_cost + 0.5 * c.dL)
    if "gene_r" in c:
        scores["gene_r"] = -c.gene_r
    c = c.assign(**{f"score_{k}": v for k, v in scores.items()})
    rows = []
    for e, g in c.groupby("embryo", sort=False):
        d = dict(embryo=e, n_pairs=len(g))
        secs = pd.unique(g[["a", "b"]].to_numpy().ravel())
        for k in scores:
            ok = g[f"score_{k}"].notna()
            d[f"rho_{k}"] = spearmanr(g[f"score_{k}"][ok], g.gap[ok])[0] if ok.sum() > 2 else np.nan
            hits = []
            for s in secs:
                x = g[((g.a == s) | (g.b == s)) & ok]
                if len(x):
                    hits.append(x.gap[x[f"score_{k}"].idxmin()] == x.gap.min())
            d[f"hit_{k}"] = float(np.mean(hits)) if hits else np.nan
        rows.append(d)
    return pd.DataFrame(rows)


def decide(control, scores):
    """The decision rule, fixed before the results. control: control_table. scores: rows (w, seed, gene_r), the mean
    gene r over the edges of chain (w, seed). Returns dict(w, reason, rho, gain, noise). Ties go to w = 0, the simpler
    score."""
    w0, w1 = MAIN_W
    rho = float(control.rho_gene_r.median()) if len(control) and "rho_gene_r" in control else np.nan
    out = dict(w=w0, rho=rho, gain=np.nan, noise=np.nan)
    if not np.isfinite(rho):
        return dict(out, reason="no gene r control, so gene r can't arbitrate: the simpler score")
    if rho < CONTROL_RHO_MIN:
        return dict(out, reason=f"gene r fails the control (median rho {rho:.2f} < {CONTROL_RHO_MIN}): the simpler score")
    p = scores.pivot_table(index="seed", columns="w", values="gene_r")
    if w0 not in p or w1 not in p or len(p.dropna()) < 2:
        return dict(out, reason="fewer than two seeds with both chains, so no noise level: the simpler score")
    p = p.dropna()
    gain = float((p[w1] - p[w0]).mean())
    noise = float(max((p[w0].max() - p[w0].min()), (p[w1].max() - p[w1].min())))
    out.update(gain=gain, noise=noise)
    if gain > noise:
        return dict(out, w=w1, reason=f"w = {w1} gains {gain:.4f} gene r, above the seed difference {noise:.4f}")
    return dict(out, reason=f"gain {gain:.4f} does not exceed the seed difference {noise:.4f}: the simpler score")


def weak_steps(chain_edges_table):
    """The chain's edges with flags: large_dL (|dL| above DL_WEAK), low_gene_r (the lowest quarter of its kind)."""
    t = chain_edges_table.copy()
    t["large_dL"] = t.dL > DL_WEAK
    if "gene_r" in t:
        t["low_gene_r"] = t.gene_r < t.groupby("kind").gene_r.transform(lambda v: v.quantile(0.25))
    return t
