"""Does the GW term help 3D FGW more at a higher resolution? (docs/cross_species_alignment_results.md, Caveats)

    python bench/scripts/fgw_scale.py run --pair X4 [--array-index i] [--n 6000,20000,all] [--runs euc,cos]
    python bench/scripts/fgw_scale.py report
    python bench/scripts/fgw_scale.py files [--pairs X1,...,X6] > files.txt

`run` solves stage 2 of pts2pts_s-euc_f-euc_ub on the stack points of one pair at several point counts n per embryo
(`all`: every spot), each time with and without the GW term (same points, slab masses, tau and epsilon), with the
Euclidean (f-euc) and the cosine (f-cos) cross cost. n = 6000 is the benchmark's own setting. Each run is scored per
(direction, query section) exactly as in demos/whole_embryo_tests.ipynb: kappa_bal, and gene r on the held-out genes.
FGW on 20 000 points needs ~20 GB of GPU memory; all spots (up to ~50 000 x 25 000 on X2, X3) ~40-60 GB and ~100 GB
of host RAM. It runs on LRZ as a job array, one task per pair (--array-index i: the i-th of --pairs). One CSV per (pair, seed, n) goes to --out (default
<root>/diag/fgw_scale), written before the next n starts, and is skipped on a rerun unless --force.
`report` writes <out>/report/table.md: kappa_bal and gene r per run and pair, the GW gain (s-euc - s-none) per n, and
the 6000-point rows against the earlier results (diag/whole_embryo on sim13, or the doc's section 3).
`files` prints the cache files (relative to the bench root) that `run` reads, for `rsync --files-from`.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_threads = os.environ.get("SLURM_CPUS_PER_TASK", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _threads)

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mgw import crossspecies as cs  # noqa: E402
from mgw.bench import config, prepare, series, solvers, volume  # noqa: E402
from mgw.bench.pipelines import stacked  # noqa: E402
from mgw.bench.pipelines.common import Task  # noqa: E402

PIPELINE = "pts2pts_s-euc_f-euc_ub"   # the 3D pipeline whose stacks, point clouds and settings are used
PAIRS = "X1,X2,X3,X4,X5,X6"
INNER_TOL = 1e-5                      # as in whole_embryo_tests: the same scores as the benchmark's 1e-7, faster
METRICS = {"euc": "euclidean", "cos": "cosine"}
# the doc's section 3 (seed 0): kappa_bal of pts2pts_s-euc_f-euc_ub and pts2pts_s-none_f-euc_ub
DOC_6K = {"X1": (0.351, 0.347), "X4": (0.440, 0.432), "X5": (0.435, 0.421)}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def context(args, pid):
    cfg = config.load(args.config, list(args.set))
    pair = series.load(cfg.series_path()).pairs[pid]
    root = prepare.bench_root(args.root)
    return cfg, pair, root


def out_dir(args, root):
    return Path(args.out) if args.out else root / "diag" / "fgw_scale"


def run_key(cfg):
    """Every setting that changes the rows of a (pair, seed, n)."""
    return config.short_hash(dict(pipeline=cfg.pipeline_hash(PIPELINE), inner_tol=INNER_TOL))


def run_name(n, n_default, space, feat):
    """sides_space_features_marginals: pts2pts at the benchmark's n, pts2pts20k, pts2ptsall."""
    sides = "pts2pts" if str(n) == str(n_default) else "pts2ptsall" if n == "all" else f"pts2pts{int(n) // 1000}k"
    return f"{sides}_s-{space}_f-{feat}_ub"


# ---------- solves and scores (as in demos/whole_embryo_tests.ipynb)
def ot(M, p, a, b, tau, epsilon=None):
    """Entropic OT on the cross cost alone, masses a, b, tau on both sides: the FGW solve without the GW term."""
    import jax.numpy as jnp
    from ott.geometry import geometry
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn
    t0 = time.perf_counter()
    geom = geometry.Geometry(cost_matrix=jnp.asarray(M, dtype=jnp.float32), epsilon=epsilon or p.epsilon)
    a, b = (jnp.asarray(x / np.sum(x), dtype=jnp.float32) for x in (a, b))
    out = sinkhorn.Sinkhorn(max_iterations=p.inner_maxit, threshold=p.inner_tol)(
        linear_problem.LinearProblem(geom, a=a, b=b, tau_a=tau, tau_b=tau))
    return np.asarray(out.matrix, dtype=np.float64), dict(converged=bool(out.converged),
                                                          seconds=time.perf_counter() - t0)


def gpu_peak_gb():
    """JAX's peak GPU memory so far in this process (GB), nan without a GPU."""
    import jax
    d = jax.local_devices()[0]
    stats = d.memory_stats() if d.platform == "gpu" else None
    return stats.get("peak_bytes_in_use", np.nan) / 1e9 if stats else np.nan


def score(task, sides, P, p, run):
    """One row per (direction, query section) of the coupling P (rows: side a's points), mapped back to the real spots
    as in pipelines/volume.py."""
    rows = []
    for d, qk, tk, Pd in (("b_stack", "a", "b", P), ("a_stack", "b", "a", P.T)):
        (STq, mq), (STt, mt) = sides[qk], sides[tk]
        qs, _ = task.sides(d)
        B = volume.target_pool(mt, STt, p["k_target"])
        T, Yt = (np.asarray(B @ STt[k]) for k in ("comp", "Y"))
        for s, name in enumerate(STq["names"]):
            q = qs[name]
            Pq = np.asarray(volume.query_map(mq, STq, s, p["k_query"]) @ Pd)
            r, r0 = cs.coexpression(Pq, q.Y, Yt, seed=task.seed)
            rows.append(dict(run=run, direction=d, query=name,
                             kappa_bal=task.score(d, q, "points", T, Pq, keep=False)["kappa_bal"],
                             gene_r=np.nanmean(r) - np.nanmean(r0)))
    return rows


# ---------- run
def cmd_run(args):
    pairs = args.pairs.split(",")
    pid = pairs[args.array_index] if args.array_index is not None else args.pair
    cfg, pair, root = context(args, pid)
    out = out_dir(args, root)
    out.mkdir(parents=True, exist_ok=True)
    if not prepare.feature_file(root, pid, args.seed, cfg.feature_hash()).exists():
        raise FileNotFoundError(f"no features for {pid} seed {args.seed} under {root}; copy the files of "
                                f"`fgw_scale.py files` or run `python -m mgw.bench prepare` first")
    import jax
    log(f"{pid} seed {args.seed}: root {root}, out {out}, device {jax.devices()[0].device_kind}, "
        f"commit {os.popen(f'git -C {REPO} rev-parse --short HEAD').read().strip()}")

    t0 = time.perf_counter()
    task = Task(cfg, PIPELINE, pair, args.seed, root, log=log)
    task.fgw = dataclasses.replace(task.fgw, inner_tol=INNER_TOL)
    p = task.spec["params"]
    stacks = {"a": stacked.get_stack(task, "a_stack"), "b": stacked.get_stack(task, "b_stack")}
    total = {k: len(ST["xyz"]) for k, ST in stacks.items()}
    Fa, Fb = (np.vstack([s.emb for s in task.data[k].values()]) for k in ("a", "b"))
    f = cfg.raw["features"]
    scales = {"euc": task.scale, "cos": solvers.cost_scale(Fa, Fb, f["m_scale_quantile"], f["m_scale_max_entries"],
                                                            seed=args.seed, metric="cosine")}
    log(f"  stage 1: {time.perf_counter() - t0:.0f} s; spots a {total['a']}, b {total['b']}; "
        f"cost scale euc {scales['euc']:.4f}, cos {scales['cos']:.4f}")

    ns = [n.strip() for n in args.n.split(",")]
    ns = sorted(ns, key=lambda n: max(total.values()) if n == "all" else int(n))
    feats = args.runs.split(",")
    key = run_key(cfg)
    for n in ns:
        f_out = out / f"{pid}_seed{args.seed}_n{n}_{key}.csv"
        if f_out.exists() and not args.force:
            log(f"  n {n}: cached ({f_out.name})")
            continue
        t0 = time.perf_counter()
        n_pts = max(total.values()) if n == "all" else int(n)
        ms = {k: volume.measure(cfg, root, args.seed, pid, task.pair.a.name if k == "a" else task.pair.b.name,
                                ST, "points", n_pts) for k, ST in stacks.items()}
        ma, mb = ms["a"], ms["b"]
        Ca, Cb = volume.intra_cost("fgw", ma, cfg), volume.intra_cost("fgw", mb, cfg)
        sides = {k: (stacks[k], ms[k]) for k in ("a", "b")}
        log(f"  n {n}: {len(ma['F'])} x {len(mb['F'])} points, intra costs {time.perf_counter() - t0:.0f} s")
        rows = []
        for feat in feats:
            metric = METRICS[feat]
            for space in ("euc", "none"):
                run = run_name(n, p["n_samples"], space, feat)
                if space == "euc":
                    P, _, info = volume.stage2(Ca, Cb, ma["F"], mb["F"], scales[feat], task.fgw, a=ma["mass"],
                                               b=mb["mass"], tau=p["tau"], epsilon=p["epsilon"],
                                               unbalanced_correction=p["unbalanced_correction"], metric=metric)
                else:
                    P, info = ot(cdist(ma["F"], mb["F"], metric) / scales[feat], task.fgw, ma["mass"], mb["mass"],
                                 p["tau"], p["epsilon"])
                rej_a, rej_b, mass = volume.rejected(P, ma["mass"], mb["mass"])
                peak = gpu_peak_gb()
                t1 = time.perf_counter()
                r = score(task, sides, P, p, run)
                del P
                meta = dict(n=n, n_a=len(ma["F"]), n_b=len(mb["F"]), solve_s=info.get("seconds", np.nan),
                            score_s=time.perf_counter() - t1, converged=info.get("converged"),
                            ugw_fallback=info.get("ugw_fallback", False), rejected_a=rej_a, rejected_b=rej_b,
                            mass=mass, gpu_peak_gb=peak)
                rows += [dict(row, **meta) for row in r]
                df = pd.DataFrame(r)
                log(f"    {run}: kappa_bal {df.kappa_bal.mean():.3f}, gene r {df.gene_r.mean():.3f} | solve "
                    f"{meta['solve_s']:.0f} s, converged {meta['converged']}, rejected a {rej_a:.3f} b {rej_b:.3f}, "
                    f"GPU peak {peak:.1f} GB")
        del Ca, Cb
        pd.DataFrame(rows).assign(pair=pid, seed=args.seed).to_csv(f_out, index=False)
        log(f"  n {n}: {(time.perf_counter() - t0) / 60:.1f} min -> {f_out.name}")
    log("done")


# ---------- report
def md_table(df, fmt="{:.3f}"):
    """A DataFrame as a Markdown table (floats by fmt; no tabulate in the env)."""
    cell = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else (  # noqa: E731
        "" if isinstance(v, (float, np.floating)) else str(v))
    head = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join([str(i)] + [cell(v) for v in row]) + " |" for i, row in zip(df.index, df.to_numpy())]
    return "\n".join(lines)


def cmd_report(args):
    cfg, _, root = context(args, args.pairs.split(",")[0])
    out = out_dir(args, root)
    files = sorted(out.glob(f"*_{run_key(cfg)}.csv"))
    if not files:
        raise FileNotFoundError(f"no results under {out}")
    res = pd.concat([pd.read_csv(f, dtype={"n": str}) for f in files], ignore_index=True)
    pairs = [p for p in args.pairs.split(",") if p in set(res.pair)]
    n_key = lambda v: np.inf if v == "all" else int(v)  # noqa: E731
    per = res.groupby(["pair", "run", "n"])[["kappa_bal", "gene_r"]].mean().reset_index()
    # run order: per n (ascending), f-euc then f-cos, s-euc then s-none
    order = (per.assign(nn=per.n.map(n_key),
                        feat=per.run.str.contains("f-cos"), space=per.run.str.contains("s-none"))
             .drop_duplicates("run").sort_values(["nn", "feat", "space"]).run.tolist())

    def table(col):
        t = per.pivot_table(index="run", columns="pair", values=col).reindex(order)[pairs]
        return t.assign(mean=t.mean(axis=1))

    md = [f"# 3D FGW at higher resolution (seed {', '.join(map(str, sorted(set(res.seed))))})", ""]
    sizes = res.groupby("n")[["n_a", "n_b", "solve_s", "gpu_peak_gb"]].max()
    sizes = sizes.loc[sorted(sizes.index, key=n_key)]
    md += ["Points per embryo (largest over the pairs), slowest solve and JAX GPU peak:", "",
           md_table(sizes, "{:.1f}"), ""]
    for col, title in (("kappa_bal", "κ_bal"), ("gene_r", "gene r")):
        md += [f"## {title}, mean over the query sections of both directions", "", md_table(table(col)),
               ""]
    k = per.pivot_table(index="pair", columns="run", values="kappa_bal").reindex(pairs)
    gain = {}
    for run in order:
        if "_s-euc_" in run and run.replace("_s-euc_", "_s-none_") in k:
            gain[run.split("_s-")[0] + " f-" + run.split("_f-")[1].split("_")[0]] = \
                k[run] - k[run.replace("_s-euc_", "_s-none_")]
    g = pd.DataFrame(gain).T
    g = g.assign(mean=g.mean(axis=1), positive=(g[pairs] > 0).sum(axis=1).astype(str) + f" of {len(pairs)}")
    md += ["## The GW term: κ_bal of s-euc − s-none (same points, masses and marginals)", "",
           md_table(g, "{:+.3f}"), ""]
    # sanity: the benchmark's n against the earlier results of the same seed
    n0 = str(cfg.raw["pipelines"][PIPELINE]["params"]["n_samples"])
    runs0 = [run_name(n0, int(n0), s, "euc") for s in ("euc", "none")]
    ref = []
    for pid in pairs:
        earlier = sorted((root / "diag" / "whole_embryo").glob(f"{pid}_seed{args.seed}_*.csv"),
                         key=lambda f: f.stat().st_mtime)
        if earlier:
            e = pd.read_csv(earlier[-1]).groupby("run").kappa_bal.mean()
            ref.append(dict(pair=pid, source=earlier[-1].name, **{f"earlier {r}": e.get(r, np.nan) for r in runs0}))
        elif pid in DOC_6K:
            ref.append(dict(pair=pid, source="doc, section 3", **dict(zip([f"earlier {r}" for r in runs0],
                                                                           DOC_6K[pid]))))
    if ref and all(r in k for r in runs0):
        ref = pd.DataFrame(ref).set_index("pair")
        for r in runs0:
            ref[f"now {r}"] = k.loc[ref.index, r]
            ref[f"Δ {r}"] = ref[f"now {r}"] - ref[f"earlier {r}"]
        md += [f"## Sanity: n = {n0} against the earlier runs (|Δ| ≲ 0.003 expected)", "",
               md_table(ref), ""]
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "table.md").write_text("\n".join(md))
    per.to_csv(rep / "per_pair.csv", index=False)
    print("\n".join(md))
    log(f"-> {rep / 'table.md'}")


# ---------- files
def cmd_files(args):
    cfg = config.load(args.config, list(args.set))
    ser = series.load(cfg.series_path())
    root = prepare.bench_root(args.root)
    fh, sh = cfg.feature_hash(), cfg.section_hash()
    paths = []
    for pid in args.pairs.split(","):
        pair = ser.pairs[pid]
        paths.append(prepare.feature_file(root, pid, args.seed, fh))
        for emb in (pair.a, pair.b):
            paths += [prepare.section_file(root, s, args.seed, sh) for s in emb.sections]
            paths.append(prepare.feature_file(root, f"stack_{emb.name}", args.seed, fh))
            paths.append(stacked.stack_path(root, cfg, args.seed, emb))
    paths = list(dict.fromkeys(paths))
    missing = [f for f in paths if not f.exists()]
    for f in paths:
        print(f.relative_to(root))
    print(f"{len(paths)} files under {root}, {len(missing)} missing here"
          + "".join(f"\n  missing: {f.relative_to(root)}" for f in missing), file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="bench.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra config overrides")
    ap.add_argument("--root", default=None, help="bench root (default $MGW_BENCH_DIR or <data>/bench)")
    ap.add_argument("--out", default=None, help="results (default <root>/diag/fgw_scale)")
    ap.add_argument("--pairs", default=PAIRS, help="the pairs of --array-index, report and files")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pair", default="X4")
    r.add_argument("--array-index", type=int, help="the i-th pair of --pairs (a SLURM array task)")
    r.add_argument("--n", default="6000,20000,all", help="points per embryo; all: every spot")
    r.add_argument("--runs", default="euc,cos", help="cross costs: euc (Euclidean), cos (cosine)")
    r.add_argument("--force", action="store_true", help="recompute cached (pair, seed, n)")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    sub.add_parser("files").set_defaults(fn=cmd_files)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
