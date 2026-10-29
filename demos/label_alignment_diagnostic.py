"""Why can FGW beat Wasserstein OT on kappa_bal when the cross cost is the organ labels?

For the section pairs picked by label_alignment_best.ipynb (the best Wasserstein kappa_bal per pair and direction),
re-solves the couplings and scores each one on several measures, to separate three explanations:
- the solver: Wasserstein does not reach the optimum of its own objective (OTT reports converged = False for every
  solve). Then FGW can have a lower label cost <M, P> than Wasserstein, which an optimal Wasserstein never allows.
- the entropy: OTT's FGW minimizes GW + (1 - a)/a <M, P> - eps H(P), so relative to the label term its entropy is
  eps a/(1 - a): eps/9 at a = 0.1 (sharper than Wasserstein at eps), 9 eps at a = 0.9 (blurrier). The control
  W@eps(a) is Wasserstein at that effective entropy.
- the metric: Wasserstein maximizes the tile-weighted share of mass on the same organ (plain accuracy, every tile
  1/n), kappa_bal is organ-balanced and weights query tiles by their full-resolution bins; mass to Other, Excluded
  or organs outside the shared set drops out of kappa_bal but costs 0.5 or 1 in <M, P>.
- feasibility: kappa_bal normalizes each query row of P, so a coupling that misses its marginals (unconverged
  Sinkhorn) can avoid the errors that balanced marginals force; marg_err measures this.

Couplings per picked section pair (balanced marginals): W (the notebook's Wasserstein), W-long (inner_maxit x
--long-factor), FGW(a) for each --alphas, W@eps(a) for each a. Scores per coupling and direction:
- label_cost: <M, P> / sum(P), the Wasserstein objective without entropy (lower is better);
- marg_err: L1 distance of the row and column sums to the uniform marginals (0 = feasible);
- perplexity: mean over query tiles of exp(entropy of the tile's normalized row), the number of target tiles a
  query tile is effectively spread over;
- acc_tile / acc_bin: plain accuracy on the shared organs, query tiles weighted 1/n (what <M, P> rewards) or by
  their bins (what kappa_bal counts);
- kappa_tile / kappa_bal: organ-balanced kappa with the same two weightings (kappa_bal = the benchmark's);
- off_shared: share of the shared organs' query mass (bins) that lands outside the shared organs (ignored by
  kappa_bal).
Per organ (bin-weighted): m_o, the share of organ o's mass that lands on o.

    uv run python demos/label_alignment_diagnostic.py                    # X1-X3, X5, X8, X9, alphas 0.1 0.5 0.9
    uv run python demos/label_alignment_diagnostic.py --pairs X8 --alphas 0.1

Writes diagnostic_<stem>.csv and diagnostic_<stem>_organs.csv next to the notebook's scores
(<root>/diag/label_alignment_best/) and prints a summary.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp
import numpy as np
import pandas as pd
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from mgw.bench import config, metrics, prepare, series, solvers
from mgw.bench.pipelines.common import Task

DIRS = ("a_stack", "b_stack")
SUMMARY = ["label_cost", "marg_err", "perplexity", "acc_tile", "kappa_tile", "acc_bin", "kappa_bal", "off_shared"]


def label_cost(comp_a, comp_b):
    """Organ-label cross cost (label_alignment.ipynb): 0 same organ, 1 different organs, 0.5 for Other/Excluded."""
    Ka, Kb = metrics.row_normalize(comp_a)[:, metrics.SCORED], metrics.row_normalize(comp_b)[:, metrics.SCORED]
    return 1 - Ka @ Kb.T - 0.5 * (1 - np.outer(Ka.sum(1), Kb.sum(1)))


def wasserstein(M, epsilon, maxit, tol):
    """Balanced entropic OT on M alone (label_alignment.ipynb's ot_linear), with OTT's convergence record."""
    t0 = time.perf_counter()
    geom = geometry.Geometry(cost_matrix=jnp.asarray(M, dtype=jnp.float32), epsilon=epsilon)
    out = sinkhorn.Sinkhorn(max_iterations=maxit, threshold=tol)(linear_problem.LinearProblem(geom))
    errs = np.asarray(out.errors)
    errs = errs[errs != -1]
    return np.asarray(out.matrix, dtype=np.float64), dict(
        converged=bool(out.converged), n_iters=int(out.n_iters), ott_err=float(errs[-1]) if len(errs) else np.nan,
        seconds=time.perf_counter() - t0)


def measures(task, d, q, t, Pq):
    """The scores of one coupling Pq (query tiles x target tiles) in direction d, and m_o per shared organ."""
    min_bins = task.mp["present_min_bins"]
    shared = np.flatnonzero(metrics.present(q.comp, min_bins) & task.target_presence(d))
    P = Pq / Pq.sum()
    Pr = metrics.row_normalize(P)
    row_ent = -np.sum(np.where(Pr > 0, Pr * np.log(np.where(Pr > 0, Pr, 1)), 0), axis=1)
    out = dict(K=len(shared),
               marg_err=float(np.abs(P.sum(1) - 1 / P.shape[0]).sum() + np.abs(P.sum(0) - 1 / P.shape[1]).sum()),
               perplexity=float(np.exp(row_ent).mean()))
    m_o = {}
    for w, Q in (("tile", metrics.row_normalize(q.comp)), ("bin", q.comp)):
        C = metrics.confusion(P, Q, t.comp)
        Cs = C[np.ix_(shared, shared)]
        row = Cs.sum(1)
        out[f"acc_{w}"] = float(np.trace(Cs) / Cs.sum())
        kb, _ = metrics.kappa_bal(C, metrics.present(q.comp, min_bins), task.target_presence(d))
        out["kappa_bal" if w == "bin" else "kappa_tile"] = kb
        if w == "bin":
            out["off_shared"] = float(1 - Cs.sum() / C[shared].sum())
            m_o = {metrics.ORGANS[o]: (float(np.diag(Cs)[k] / row[k]) if row[k] > 0 else 0.0,
                                       float(q.comp[:, o].sum() / q.comp[:, shared].sum()),
                                       float(t.comp[:, o].sum() / t.comp[:, shared].sum()))
                   for k, o in enumerate(shared)}
    return out, m_o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", nargs="+", default=["X1", "X2", "X3", "X5", "X8", "X9"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-obs", default="5000", help="points per section; 'none' for every bin")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--long-factor", type=int, default=10, help="W-long: inner_maxit x this")
    ap.add_argument("--eps-maxit", type=int, default=30000, help="max Sinkhorn iterations of the W@eps(a) controls")
    ap.add_argument("--tol", type=float, default=1e-3, help="differences below this count as ties in the verdicts")
    ap.add_argument("--config", default="bench.yaml")
    ap.add_argument("--pipeline", default="sec2sec_s-euc_f-euc_bal")
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    if not all(0 < a < 1 for a in args.alphas):
        raise ValueError(f"--alphas must lie in (0, 1), not {args.alphas}")

    max_obs = None if str(args.max_obs).lower() in ("none", "null") else int(args.max_obs)
    res = "all" if max_obs is None else str(max_obs)
    root = prepare.bench_root(args.root)
    cfg = config.load(args.config, [f"reduce.max_obs={'null' if max_obs is None else max_obs}"])
    pairs = series.load(cfg.series_path()).pairs
    out_dir = root / "diag" / "label_alignment_best"
    stem = f"seed{args.seed}_n{res}_bal"
    picks_csv = out_dir / f"{stem}_scores.csv"
    if not picks_csv.exists():
        raise FileNotFoundError(f"no {picks_csv}: run label_alignment_best.ipynb (balanced) first")
    nb = pd.read_csv(picks_csv)
    w_run = "s-none_f-lab_bal"

    rows, organ_rows = [], []
    for pid in args.pairs:
        t0 = time.perf_counter()
        task = Task(cfg, args.pipeline, pairs[pid], args.seed, root)
        H, Mo = task.data["a"], task.data["b"]
        p = task.fgw
        best = nb[(nb.pair == pid) & (nb.run == w_run) & nb.best]
        picks = {r.direction: (r.human, r.mouse) for r in best.itertuples()}
        ref = nb[(nb.pair == pid) & nb.best].set_index(["run", "direction"]).kappa_bal   # the notebook's kappa_bal
        print(f"{pid}: solver {p}")
        for hm in dict.fromkeys(picks.values()):
            h, m = hm
            dirs = [d for d in DIRS if picks[d] == hm]
            Ml = label_cost(H[h].comp, Mo[m].comp)
            Dh = solvers.dist_norm(task.xy(H[h])).astype(np.float32)
            Dm = solvers.dist_norm(task.xy(Mo[m])).astype(np.float32)
            sols = {"W": wasserstein(Ml, p.epsilon, p.inner_maxit, p.inner_tol),
                    "W-long": wasserstein(Ml, p.epsilon, p.inner_maxit * args.long_factor, p.inner_tol)}
            for a in args.alphas:
                sols[f"FGW({a:g})"] = solvers.fgw(None, None, Ml, dataclasses.replace(p, alpha=a), Ca=Dh, Cb=Dm)
                sols[f"W@eps({a:g})"] = wasserstein(Ml, p.epsilon * a / (1 - a), args.eps_maxit, p.inner_tol)
            del Dh, Dm
            for name, (P, info) in sols.items():
                if not np.isfinite(P).all():
                    raise FloatingPointError(f"{pid} {name}: non-finite coupling for {h} x {m}")
                alpha = float(name[4:-1]) if name.startswith("FGW") else np.nan
                eps_eff = (p.epsilon * alpha / (1 - alpha) if name.startswith("FGW")
                           else p.epsilon * float(name[6:-1]) / (1 - float(name[6:-1])) if name.startswith("W@eps")
                           else p.epsilon)
                for d in dirs:
                    q, t, Pq = (H[h], Mo[m], P) if d == "b_stack" else (Mo[m], H[h], P.T)
                    sc, m_o = measures(task, d, q, t, Pq)
                    run = (w_run if name == "W" else
                           f"{'s-euc' if alpha == 0.5 else f's-euc{alpha:g}'}_f-lab_bal" if name.startswith("FGW")
                           else None)
                    rows.append(dict(pair=pid, direction=d, query=q.name, target=t.name, coupling=name, alpha=alpha,
                                     eps_eff=eps_eff, label_cost=float((Ml * P).sum() / P.sum()), **sc,
                                     kappa_notebook=ref.get((run, d), np.nan) if run else np.nan,
                                     converged=info.get("converged"), n_iters=info.get("n_iters", info.get("n_outer")),
                                     ott_err=info.get("ott_err", np.nan),
                                     inner_converged=info.get("inner_converged", np.nan),
                                     seconds=info.get("seconds")))
                    organ_rows += [dict(pair=pid, direction=d, coupling=name, organ=o, m_o=v[0], query_share=v[1],
                                        target_share=v[2]) for o, v in m_o.items()]
                print(f"  {h} x {m} {name}: {info.get('seconds', 0):.1f} s, label cost {rows[-1]['label_cost']:.4f}, "
                      f"kappa_bal " + ", ".join(f"{r['direction']} {r['kappa_bal']:.3f}" for r in rows[-len(dirs):]))
            del sols, Ml
        print(f"{pid} done in {time.perf_counter() - t0:.0f} s\n")
        del task
        pd.DataFrame(rows).to_csv(out_dir / f"diagnostic_{stem}.csv", index=False)
        pd.DataFrame(organ_rows).to_csv(out_dir / f"diagnostic_{stem}_organs.csv", index=False)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    rep = df.dropna(subset=["kappa_notebook"])
    print(f"reproduction: max |kappa_bal - notebook| = {np.abs(rep.kappa_bal - rep.kappa_notebook).max():.2e} "
          f"over {len(rep)} couplings\n")
    print("per pair and direction")
    print(df.set_index(["pair", "direction", "coupling"])[SUMMARY].round(4).to_string(), "\n")
    print("mean over pairs and directions")
    print(df.groupby("coupling", sort=False)[SUMMARY + ["eps_eff"]].mean().round(4).to_string(), "\n")
    print("solver: W and W-long")
    print(df[df.coupling.isin(["W", "W-long"])].groupby("coupling")[["n_iters", "ott_err", "marg_err", "converged"]]
          .agg(["min", "max"]).to_string(), "\n")

    W = df[df.coupling == "W"].set_index(["pair", "direction"])
    tol = args.tol
    print(f"verdicts (counts over pair x direction; differences below {tol:g} count as ties)")
    for a in args.alphas:
        F = df[df.coupling == f"FGW({a:g})"].set_index(["pair", "direction"])
        E = df[df.coupling == f"W@eps({a:g})"].set_index(["pair", "direction"])
        n = len(F)
        print(f"  a = {a:g}: FGW kappa_bal > W: {(F.kappa_bal > W.kappa_bal + tol).sum()}/{n}; "
              f"FGW label cost < W (W not optimal for <M, P>): {(F.label_cost < W.label_cost - tol).sum()}/{n}; "
              f"FGW acc_tile > W: {(F.acc_tile > W.acc_tile + tol).sum()}/{n}; "
              f"FGW infeasible (marg_err > {tol:g}): {(F.marg_err > tol).sum()}/{n}; "
              f"W@eps({a:g}) kappa_bal >= FGW: {(E.kappa_bal >= F.kappa_bal - tol).sum()}/{n}; "
              f"W@eps({a:g}) label cost <= FGW: {(E.label_cost <= F.label_cost + tol).sum()}/{n}")
    print(f"\nsaved {out_dir / f'diagnostic_{stem}.csv'} and {out_dir / f'diagnostic_{stem}_organs.csv'}")


if __name__ == "__main__":
    main()
