"""Align each species' sections across time, DeST-OT style (mgw/bench/temporal.py).

    python bench/scripts/temporal_align.py prepare --seeds 0,1 --jobs 4         # data server, CPU, RAM
    python bench/scripts/temporal_align.py sweep --seed 0                       # GPU, resumable
    python bench/scripts/temporal_align.py final --runs destot_a0.2_b0.5_g1_e0.01,expr_g1_e0.01 --seeds 0,1   # GPU

The pairs are the adjacent sections of bench/configs/chains.yaml (temporal.chains): H1-H6 (human), M1-M6 (mouse).
`--pairs H2,M1` narrows them, `--pairs CS14-15_E3S4:CS17_E1S8` names any same-species pair (smoke tests).

`prepare` builds, per section and seed, the section cache (as the benchmark's, reused if present) and the all-gene
counts per tile (cache/counts_all); then per pair the shared expression embedding (all shared genes, not just the
orthologs), the held-out genes and the proliferation / apoptosis scores; and writes pairs.csv and view_seed<s>.npz (sweep and final write them too when missing).
`sweep` solves the DeST-OT grid and the baselines (temporal.sweep, temporal.baselines; --methods narrows them) and
scores every coupling: one row per run in runs/seed<s>/<pair>.csv (one table per pair, so a job array can run one
pair per task), written after every solve, so a rerun skips what is done (--force starts over). Each coupling is kept as its columns' top-k (temporal.topk) with xi, J and the organ
coupling in couplings/<pair>/<run>_seed<s>.npz.
`final` solves the chosen runs (names as in runs.csv) on every pair and seed and keeps the dense couplings:
final/<pair>/<run>_seed<s>.npz (P float32, xi, J, g1, g2, coordinates, organs), the hand-off to the cross-species
step; their scores in final/<pair>/runs_seed<s>.csv.

Everything goes to <out>/n<max_obs>/<norm>_<masses>-<hash> (default <root>/diag/temporal; the hash covers the
temporal features, masses, norm and solver); demos/temporal_alignment.ipynb reads only that.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_threads = os.environ.get("SLURM_CPUS_PER_TASK", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _threads)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mgw import orthologs  # noqa: E402
from mgw.bench import config, metrics, prepare, temporal as tp  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def context(args):
    cfg = config.load(args.config, list(args.set))
    root = prepare.bench_root(args.root)
    t = cfg.raw["temporal"]
    n = cfg.raw["reduce"]["max_obs"]
    out = Path(args.out) if args.out else root / "diag" / "temporal"
    out = out / f"n{'all' if n is None else int(n)}" / f"{t['norm']}_{t['masses']}-{cfg.temporal_hash()}"
    out.mkdir(parents=True, exist_ok=True)
    settings = out / "settings.json"
    if not settings.exists():
        settings.write_text(json.dumps(cfg.temporal_key(), indent=1, default=str))
    days = t["stage_days"]
    pairs = tp.chain_pairs(tp.load_chains(cfg.chains_path()), days)
    if args.pairs:
        pairs = tp.parse_pairs(args.pairs, days, pairs)
    return cfg, root, out, pairs


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------- prepare
def _prepare_one(cfg_path, overrides, root, sec, seed):
    cfg = config.load(cfg_path, list(overrides))
    t0 = time.perf_counter()
    prepare.prepare_counts_all(tp.dataset_of(sec), sec, cfg, seed, root, orthologs.load_one2one(),
                               prepare.laterality_table())
    return time.perf_counter() - t0


def cmd_prepare(args):
    cfg, root, out, pairs = context(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    h = cfg.section_hash()
    secs = list(dict.fromkeys(s for p in pairs for s in (p.a, p.b)))
    todo = [(s, sd) for sd in seeds for s in secs
            if not (prepare.counts_all_file(root, s, sd, h).exists() and prepare.section_file(root, s, sd, h).exists())]
    log(f"{len(pairs)} pairs, {len(secs)} sections, seeds {seeds}: {len(todo)} section caches to build -> {root}")
    failed = []
    with ProcessPoolExecutor(args.jobs) as pool:
        jobs = {pool.submit(_prepare_one, args.config, args.set, root, s, sd): (s, sd) for s, sd in todo}
        for n, f in enumerate(as_completed(jobs), 1):
            s, sd = jobs[f]
            try:
                log(f"  [{n}/{len(todo)}] {s} seed {sd}: {f.result():.0f} s")
            except Exception as e:      # one bad file shouldn't lose the others; reported and failed at the end
                failed.append((s, sd))
                log(f"  [{n}/{len(todo)}] {s} seed {sd} FAILED: {e!r}")
    if failed:
        raise SystemExit(f"{len(failed)} sections failed: {failed}")
    for sd in seeds:
        describe(cfg, root, out, pairs, sd, build=True, force=args.force)
    log(f"prepare done -> {out}")


def describe(cfg, root, out, pairs, seed, build=False, force=False):
    """The pair table (pairs.csv, rows of this seed replaced) and the viewer's sections (view_seed<s>.npz): written by
    prepare (build: the pair features too) and, when missing for the seed, by sweep and final, since a results folder
    is per solver setting. Held under a file lock, so the tasks of a job array don't overwrite each other."""
    import fcntl
    with open(out / ".describe.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _describe(cfg, root, out, pairs, seed, build, force)


def _describe(cfg, root, out, pairs, seed, build, force):
    path = out / "pairs.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["seed", "pair"])
    view_path = out / f"view_seed{seed}.npz"
    if not build and view_path.exists() and set(p.id for p in pairs) <= set(old.pair[old.seed == seed]):
        return
    rows = []
    view = dict(np.load(view_path, allow_pickle=False)) if view_path.exists() else {}
    for p in pairs:
        t0 = time.perf_counter()
        if build:
            tp.pair_features(p, cfg, seed, root, force=force)
        d = tp.load_pair(p, cfg, seed, root)
        g1, g2 = tp.masses(d.sa.comp, d.sb.comp, cfg.raw["temporal"]["masses"])
        rows.append(dict(seed=seed, pair=p.id, species=p.species, a=p.a, b=p.b, stage_a=tp.stage_of(p.a),
                         stage_b=tp.stage_of(p.b), dt=p.dt, n_a=d.sa.n, n_b=d.sb.n,
                         bins_a=float(d.sa.comp.sum()), bins_b=float(d.sb.comp.sum()), mass_t2=float(g2.sum()),
                         pitch_a=d.pitch[0], pitch_b=d.pitch[1], n_shared_genes=d.n_shared,
                         seconds=time.perf_counter() - t0))
        for side, s, sec in (("a", p.a, d.sa), ("b", p.b, d.sb)):
            view[f"{s}|xy"] = sec.xy.astype(np.float32)
            view[f"{s}|organ"] = metrics.organ_codes(sec.label).astype(np.uint8)
            view[f"{s}|n_bins"] = sec.comp.sum(1).astype(np.float32)
            for k in ("prolif", "apopt"):
                view[f"{s}|{k}"] = d.scores[f"{k}_{side}"].astype(np.float32)
        log(f"  {p.id} {p.a} -> {p.b} seed {seed}: {d.sa.n} x {d.sb.n} tiles, area ratio {g2.sum():.2f}, "
            f"{d.n_shared} shared genes")
    new = pd.DataFrame(rows)
    keep = old[~((old.seed == seed) & old.pair.isin(new.pair))]
    save_csv(pd.concat([keep, new], ignore_index=True) if len(keep) else new, path)
    view["organs"] = np.asarray(metrics.ORGANS)
    prepare.atomic_savez(view_path, **view)


# ---------- sweep and final
def _solve_and_score(prob, method, params, seed):
    t0 = time.perf_counter()
    try:
        P, info = prob.solve(method, params)
        if not np.isfinite(P).all() or P.sum() <= 0:
            raise FloatingPointError("non-finite or empty coupling")
        row, extra = tp.evaluate(P, prob, seed=seed)
        row.update(info, error="")
    except Exception as e:      # a failed solve is a row, not the end of the sweep
        traceback.print_exc()
        P, extra, row = None, None, dict(error=repr(e)[:300], finite=False)
    row["seconds_total"] = time.perf_counter() - t0
    return P, extra, row


def _row_head(p, seed, method, params):
    return dict(seed=seed, pair=p.id, species=p.species, a=p.a, b=p.b, run=tp.run_name(method, params),
                method=method, alpha=params.get("alpha", 0.0 if method in ("expr", "label") else np.nan),
                beta=params.get("beta", np.nan), gamma=params.get("gamma", np.nan),
                epsilon=params.get("epsilon", np.nan))


def cmd_sweep(args):
    cfg, root, out, pairs = context(args)
    methods = tuple(args.methods.split(",")) if args.methods else tp.METHODS
    bad = set(methods) - set(tp.METHODS)
    if bad:
        raise SystemExit(f"unknown methods {sorted(bad)}; known {tp.METHODS}")
    runs = tp.grid(cfg, methods)
    describe(cfg, root, out, pairs, args.seed)
    k = cfg.raw["temporal"]["topk"]
    total = len(pairs) * len(runs)
    log(f"sweep seed {args.seed}: {len(pairs)} pairs x {len(runs)} runs ({', '.join(methods)}) -> "
        f"{tp.runs_dir(out, args.seed)}")
    n = 0
    for p in pairs:
        path = tp.runs_dir(out, args.seed) / f"{p.id}.csv"      # one table per pair: array tasks never share a file
        table = pd.read_csv(path) if path.exists() and not args.force else pd.DataFrame()
        done = set() if table.empty else set(table.run)
        rows = table.to_dict("records")
        todo = [(m, q) for m, q in runs if tp.run_name(m, q) not in done]
        n += len(runs) - len(todo)
        log(f"  {p.id}: {len(runs) - len(todo)} of {len(runs)} runs done")
        if not todo:
            continue
        prob = tp.Problem(tp.load_pair(p, cfg, args.seed, root), cfg)
        for m, q in todo:
            n += 1
            P, extra, row = _solve_and_score(prob, m, q, args.seed)
            row = {**_row_head(p, args.seed, m, q), **row}
            if P is not None and not args.no_save:
                idx, vals, cov = tp.topk_columns(P, k)
                row["topk_mass"] = cov
                prepare.atomic_savez(out / "couplings" / p.id / f"{row['run']}_seed{args.seed}.npz",
                                     idx=idx, vals=vals, n1=P.shape[0], **extra)
            rows.append(row)
            save_csv(pd.DataFrame(rows), path)
            msg = row["error"] or (f"growth dist {row['growth_dist_nt']:.3g}, mig {row['migration']:.3f}, "
                                   f"kappa {row['kappa_ba']:.3f}, gene r {row.get('gene_r', np.nan):.3f}, "
                                   f"{'conv' if row['converged'] else 'NOT conv'} ({row['n_outer']})")
            log(f"  [{n}/{total}] {p.id} {row['run']}: {msg}; {row['seconds_total']:.1f} s")
    log(f"sweep done -> {tp.runs_dir(out, args.seed)}")


def cmd_final(args):
    cfg, root, out, pairs = context(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [tp.parse_run(r.strip()) for r in args.runs.split(",") if r.strip()]
    for sd in seeds:
        describe(cfg, root, out, pairs, sd)
        for p in pairs:
            path = out / "final" / p.id / f"runs_seed{sd}.csv"      # one table per pair and seed
            table = pd.read_csv(path) if path.exists() else pd.DataFrame()
            rows = [] if table.empty else table.to_dict("records")
            done = set() if table.empty else set(table.run)
            todo = [(m, q) for m, q in runs if tp.run_name(m, q) not in done or args.force]
            if not todo:
                continue
            prob = tp.Problem(tp.load_pair(p, cfg, sd, root), cfg)
            d = prob.data
            for m, q in todo:
                P, extra, row = _solve_and_score(prob, m, q, sd)
                row = {**_row_head(p, sd, m, q), **row}
                if P is not None:
                    prepare.atomic_savez(out / "final" / p.id / f"{row['run']}_seed{sd}.npz",
                                         P=P.astype(np.float32), g1=prob.g1, g2=prob.g2, xy_a=d.sa.xy, xy_b=d.sb.xy,
                                         organ_a=metrics.organ_codes(d.sa.label).astype(np.uint8),
                                         organ_b=metrics.organ_codes(d.sb.label).astype(np.uint8),
                                         comp_a=d.sa.comp.astype(np.float32), comp_b=d.sb.comp.astype(np.float32),
                                         organs=np.asarray(metrics.ORGANS), a=p.a, b=p.b, dt=p.dt, method=m,
                                         params=json.dumps(q), **extra)
                rows = [r for r in rows if r["run"] != row["run"]] + [row]
                save_csv(pd.DataFrame(rows), path)
                msg = row["error"] or (f"growth dist {row['growth_dist_nt']:.3g}, mig {row['migration']:.3f}, "
                                       f"kappa {row['kappa_ba']:.3f}")
                log(f"  seed {sd} {p.id} {row['run']}: {msg}; {row['seconds_total']:.1f} s")
    log(f"final done -> {out / 'final'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="bench.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra config overrides")
    ap.add_argument("--root", default=None, help="bench root (default $MGW_BENCH_DIR or <data>/bench)")
    ap.add_argument("--out", default=None, help="results (default <root>/diag/temporal)")
    ap.add_argument("--pairs", default=None, help="pair ids (H2,M1) or a:b section pairs; default the chains")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--jobs", type=int, default=1, help="sections at once; a CS23 section needs tens of GB")
    p.add_argument("--force", action="store_true", help="rebuild the pair features")
    p.set_defaults(fn=cmd_prepare)
    s = sub.add_parser("sweep")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--methods", default=None, help=f"comma-separated subset of {','.join(tp.METHODS)}")
    s.add_argument("--force", action="store_true", help="start over")
    s.add_argument("--no-save", action="store_true", help="scores only, no top-k couplings")
    s.set_defaults(fn=cmd_sweep)
    f = sub.add_parser("final")
    f.add_argument("--runs", required=True, help="comma-separated run names (runs.csv `run`)")
    f.add_argument("--seeds", default="0,1")
    f.add_argument("--force", action="store_true", help="solve again what is done")
    f.set_defaults(fn=cmd_final)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
