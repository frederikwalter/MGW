"""Choose the sections of the temporal and cross-species alignments (mgw/bench/selection.py).

    python bench/scripts/select_sections.py prepare --seeds 0,1 --jobs 6     # data server, CPU, RAM
    python bench/scripts/select_sections.py costs --seed 0                   # GPU, resumable
    python bench/scripts/select_sections.py costs --seed 1
    python bench/scripts/select_sections.py chains --seeds 0,1 [--pool series] [--no-genes]

`prepare` writes the candidate table and builds the missing section and ortholog-count caches (reduce.max_obs, as in
the benchmark; finished ones are reused), `--stages` and `--available` narrow the candidates (smoke test).
`costs` solves Wasserstein OT on the labels for every temporal, cross and control edge, saving every 25 edges; a rerun
skips what is done, and --force starts over.
`chains` finds the chains, their sensitivity and margins, then gene r of the control and chain edges (the slow part,
it reads the ortholog counts and solves those edges again; resumable) and the viewer's sections.
`--pool series` restricts the chains to the embryos of series.yaml, with outputs in the subfolder series/.

Everything goes to <out>/n<max_obs> (default <root>/diag/slice_selection); demos/slice_selection.ipynb reads only that.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_threads = os.environ.get("SLURM_CPUS_PER_TASK", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _threads)

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mgw import crossspecies as cs  # noqa: E402
from mgw import orthologs  # noqa: E402
from mgw.bench import config, metrics, prepare, selection as sel, series  # noqa: E402

SAVE_EVERY = 25


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def context(args):
    cfg = config.load(args.config, list(args.set))
    ser = series.load(cfg.series_path())
    root = prepare.bench_root(args.root)
    n = cfg.raw["reduce"]["max_obs"]
    out = Path(args.out) if args.out else root / "diag" / "slice_selection"
    out = out / f"n{'all' if n is None else int(n)}"
    out.mkdir(parents=True, exist_ok=True)
    return cfg, ser, root, out


def save_csv(df, path):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_candidates(out):
    path = out / "candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"no {path}: run `select_sections.py prepare` first")
    return pd.read_csv(path, dtype={"embryo": str, "stage": str}, keep_default_na=False, na_values=[""])


def ladder_of(ser, cand):
    ok = cand[cand.excluded.fillna("") == ""]
    return sel.rungs(ser, stages=set(ok.stage))


def _prepare_one(cfg_path, overrides, root, dataset, sec, seed):
    cfg = config.load(cfg_path, list(overrides))
    t0 = time.perf_counter()
    prepare.prepare_section(dataset, sec, cfg, seed, root, orthologs.load_one2one(), prepare.laterality_table())
    return time.perf_counter() - t0


def cmd_prepare(args):
    cfg, ser, root, out = context(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    stages = set(args.stages.split(",")) if args.stages else None
    ladder = sel.rungs(ser, stages=stages)
    overview = pd.read_csv(REPO / "data_analysis" / "dataset_overview.csv")
    cand = sel.candidates(prepare.laterality_table(), ladder, overview, ser)
    hesta, mosta = prepare.data_dirs()
    if args.available:
        missing = np.array([not cs.section_path(r.dataset, r.section, hesta, mosta).exists() for r in cand.itertuples()])
        cand.loc[missing & (cand.excluded == ""), "excluded"] = "no file here (--available)"
        ladder = sel.rungs(ser, stages=stages & set(cand.stage[cand.excluded == ""]) if stages
                           else set(cand.stage[cand.excluded == ""]))
        h, m = sel.stage_orders(ladder)
        cand.loc[(cand.excluded == "") & ~cand.stage.isin(h + m), "excluded"] = "stage not in the ladder"
    save_csv(cand, out / "candidates.csv")
    ok = cand[cand.excluded == ""]
    log(f"{len(ok)} candidates of {len(cand)} sections; rungs {', '.join(r[0] for r in ladder)}; table in {out}")
    h = cfg.section_hash()
    todo = [(r.dataset, r.section, s) for s in seeds for r in ok.itertuples()
            if not (prepare.section_file(root, r.section, s, h).exists()
                    and prepare.counts_file(root, r.section, s, h).exists())]
    log(f"{len(ok) * len(seeds) - len(todo)} caches exist, {len(todo)} to build")
    failed = []
    with ProcessPoolExecutor(args.jobs) as pool:
        jobs = {pool.submit(_prepare_one, args.config, args.set, root, *t): t for t in todo}
        for n, f in enumerate(as_completed(jobs), 1):
            t = jobs[f]
            try:
                log(f"  [{n}/{len(todo)}] {t[1]} seed {t[2]}: {f.result():.0f} s")
            except Exception as e:      # one bad file shouldn't lose the others; reported and failed at the end
                failed.append(t)
                log(f"  [{n}/{len(todo)}] {t[1]} seed {t[2]} FAILED: {e!r}")
    if failed:
        raise SystemExit(f"{len(failed)} sections failed: {failed}")
    log("prepare done")


def load_sections(root, names, seed, h):
    miss = [s for s in names if not prepare.section_file(root, s, seed, h).exists()]
    if miss:
        raise FileNotFoundError(f"no section cache for {len(miss)} sections at seed {seed} (e.g. {miss[:3]}): run "
                                "`select_sections.py prepare` with the same --set on the data server")
    return {s: prepare.load_section(root, s, seed, h) for s in names}


def solve_params(cfg):
    return dict(epsilon=cfg.raw["fgw"]["epsilon"], min_bins=cfg.raw["metrics"]["present_min_bins"])


def cmd_costs(args):
    cfg, ser, root, out = context(args)
    cand = load_candidates(out)
    edges = sel.edge_list(cand, ladder_of(ser, cand))
    path = out / f"edges_seed{args.seed}.csv"
    done = pd.read_csv(path) if path.exists() and not args.force else pd.DataFrame(columns=["kind", "a", "b"])
    have = set(zip(done.kind, done.a, done.b))
    todo = [r for r in edges.itertuples() if (r.kind, r.a, r.b) not in have]
    log(f"seed {args.seed}: {len(edges)} edges ({edges.kind.value_counts().to_dict()}), {len(todo)} to solve -> {path}")
    if not todo:
        return
    L = cand.set_index("section").L.to_dict()
    secs = load_sections(root, sorted({s for r in todo for s in (r.a, r.b)}), args.seed, cfg.section_hash())
    params, rows, t0 = solve_params(cfg), [], time.perf_counter()

    def flush():
        save_csv(pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows), path)

    for n, r in enumerate(todo, 1):
        row, _ = sel.edge_scores(secs[r.a].comp, secs[r.b].comp, L[r.a], L[r.b], **params)
        rows.append(dict(kind=r.kind, a=r.a, b=r.b, gap=r.gap, **row))
        if n % SAVE_EVERY == 0 or n == len(todo):
            flush()
        if n <= 5 or n % SAVE_EVERY == 0 or n == len(todo):
            sec = time.perf_counter() - t0
            log(f"  [{n}/{len(todo)}] {sec / n:.2f} s per edge, ~{sec / n * (len(todo) - n) / 60:.0f} min left")
    d = pd.read_csv(path)
    log(f"done; converged {d.converged.mean():.1%}, largest marginal error {d.marg_err.max():.1e}")


def ortholog_counts(root, sec, dataset, seed, h, ortho, m2h):
    """(tiles x genes CSR, human gene symbols) of a section's 1:1 orthologs, without mitochondrial genes."""
    a = prepare.load_counts(root, sec, seed, h)
    keep = a.var_names.isin(set(ortho.human) if dataset == "HESTA" else set(m2h))
    genes = a.var_names[keep] if dataset == "HESTA" else a.var_names[keep].map(m2h)
    genes = pd.Index(genes)
    X = a.X[:, np.flatnonzero(keep)].tocsr()
    keep = ~genes.str.upper().str.startswith("MT-")
    return X[:, np.flatnonzero(keep)], genes[keep]


def add_gene_r(cfg, root, out, edges, cand, seed, wanted):
    """gene r of the edges in `wanted` [(kind, a, b)], kept in gene_r_seed<seed>.csv across runs."""
    path = out / f"gene_r_seed{seed}.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["a", "b", "gene_r", "n_genes"])
    have = set(zip(done.a, done.b))
    todo = list(dict.fromkeys((a, b) for _, a, b in wanted if (a, b) not in have))
    log(f"seed {seed}: gene r of {len(todo)} edges ({len(have)} done)")
    if todo:
        h = cfg.section_hash()
        secs = load_sections(root, sorted({s for e in todo for s in e}), seed, h)
        ortho = orthologs.load_one2one()
        m2h = dict(zip(ortho.mouse, ortho.human))
        ds = cand.set_index("section").dataset.to_dict()
        params = solve_params(cfg)
        counts, rows, t0 = {}, [], time.perf_counter()

        def get(s):
            if s not in counts:
                if len(counts) > 40:        # the edges are ordered, so old sections are not needed soon
                    counts.pop(next(iter(counts)))
                counts[s] = ortholog_counts(root, s, ds[s], seed, h, ortho, m2h)
            return counts[s]

        for n, (a, b) in enumerate(todo, 1):
            (Xa, ga), (Xb, gb) = get(a), get(b)
            shared = ga.intersection(gb)
            _, P = sel.edge_scores(secs[a].comp, secs[b].comp, 0.0, 0.0, **params)
            r = sel.gene_r(P, Xa[:, ga.get_indexer(shared)].toarray(), Xb[:, gb.get_indexer(shared)].toarray(), seed=seed)
            rows.append(dict(a=a, b=b, gene_r=r, n_genes=len(shared)))
            if n % 10 == 0 or n == len(todo):
                done_now = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
                save_csv(done_now, path)
                sec = time.perf_counter() - t0
                log(f"  [{n}/{len(todo)}] {sec / n:.1f} s per edge, ~{sec / n * (len(todo) - n) / 60:.0f} min left")
        done = pd.read_csv(path)
    return edges.merge(done, on=["a", "b"], how="left")


def long_chain(w, lam, seed, total, chain_h, chain_m):
    return [dict(w=w, lam=lam, seed=seed, total=total, species=sp, stage=st, section=s)
            for sp, chain in (("human", chain_h), ("mouse", chain_m)) for st, s in chain.items()]


def cmd_chains(args):
    cfg, ser, root, base = context(args)
    cand = load_candidates(base)
    out = base
    if args.pool == "series":
        cand = cand.copy()
        cand.loc[(cand.excluded.fillna("") == "") & ~cand.in_series.astype(bool), "excluded"] = "not in series.yaml (--pool series)"
        out = base / "series"
        out.mkdir(exist_ok=True)
    ladder = ladder_of(ser, cand)
    h, m = sel.stage_orders(ladder)
    sections = sel.stage_sections(cand, h + m)
    keep = {s for v in sections.values() for s in v}
    seeds = [int(s) for s in args.seeds.split(",")]
    edges = {}
    for s in seeds:
        path = base / f"edges_seed{s}.csv"
        if not path.exists():
            raise FileNotFoundError(f"no {path}: run `select_sections.py costs --seed {s}` first")
        e = pd.read_csv(path)
        edges[s] = e[e.a.isin(keep) & e.b.isin(keep)].reset_index(drop=True)
    save_csv(cand, out / "candidates.csv")

    grid, main, margin_rows = [], {}, []
    for s in seeds:
        for w in sel.GRID_W:
            cost = sel.cost_table(edges[s], w)
            for lam in sel.GRID_LAMBDA:
                total, ch, cm = sel.solve_chain(ladder, sections, cost, lam)
                grid += long_chain(w, lam, s, total, ch, cm)
                if lam == 1.0 and w in sel.MAIN_W:
                    main[w, s] = (ch, cm)
                    margin_rows.append(sel.margins(ladder, sections, cost, lam, ch, cm, total).assign(w=w, seed=s))
    grid = pd.DataFrame(grid)
    save_csv(grid, out / "sensitivity.csv")
    save_csv(grid[(grid.lam == 1.0) & grid.w.isin(sel.MAIN_W)], out / "chains.csv")
    margin = pd.concat(margin_rows, ignore_index=True)
    save_csv(margin, out / "margins.csv")
    for (w, s), (ch, cm) in main.items():
        log(f"w {w} seed {s}: human {list(ch.values())}; mouse {list(cm.values())}")

    tables = []
    for s in seeds:
        control = edges[s][edges[s].kind == "control"]
        wanted = [(c, a, b) for (w, sd), (ch, cm) in main.items() if sd == s for c, a, b in sel.chain_edges(ladder, ch, cm)]
        table = edges[s]
        if not args.no_genes:
            # always the shared cache, also under --pool series: an edge's gene r doesn't depend on the pool
            table = add_gene_r(cfg, root, base, table, cand, s, wanted + [("control", a, b) for a, b in zip(control.a, control.b)])
        save_csv(table[table.kind == "control"], out / f"control_seed{s}.csv")
        by_edge = table.set_index(["kind", "a", "b"])
        for (w, sd), (ch, cm) in main.items():
            if sd != s:
                continue
            rows = by_edge.loc[sel.chain_edges(ladder, ch, cm)].reset_index()
            tables.append(rows.assign(w=w, seed=s))
    ec = pd.concat(tables, ignore_index=True)
    ec["bin_ratio"] = ec.bins_b / ec.bins_a      # tissue bins, without reduce.drop_labels
    save_csv(ec, out / "edges_chain.csv")

    shown = sorted(set(grid.section[(grid.lam == 1.0) & grid.w.isin(sel.MAIN_W)]) | set(margin.alternative[margin.alternative != ""]))
    secs = load_sections(root, shown, seeds[0], cfg.section_hash())
    prepare.atomic_savez(out / "view.npz", **{k: v for s, sec in secs.items() for k, v in (
        (f"{s}|xy", sec.xy.astype(np.float32)), (f"{s}|label", metrics.organ_codes(sec.label).astype(np.uint8)))})
    log(f"chains done -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="bench.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra config overrides")
    ap.add_argument("--root", default=None, help="bench root (default $MGW_BENCH_DIR or <data>/bench)")
    ap.add_argument("--out", default=None, help="results (default <root>/diag/slice_selection)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--jobs", type=int, default=1, help="sections at once; a CS23 section needs tens of GB")
    p.add_argument("--stages", default=None, help="comma-separated: keep only these stages")
    p.add_argument("--available", action="store_true", help="only sections with a raw file here")
    p.set_defaults(fn=cmd_prepare)
    c = sub.add_parser("costs")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--force", action="store_true", help="start over")
    c.set_defaults(fn=cmd_costs)
    d = sub.add_parser("chains")
    d.add_argument("--seeds", default="0,1")
    d.add_argument("--pool", choices=("all", "series"), default="all")
    d.add_argument("--no-genes", action="store_true", help="skip gene r (no counts or solves)")
    d.set_defaults(fn=cmd_chains)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
