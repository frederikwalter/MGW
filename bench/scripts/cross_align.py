"""Align human and mouse chain sections and score the alignments by commutativity (mgw/bench/crossloop.py).

    python bench/scripts/cross_align.py prepare --seeds 0,1 --jobs 4          # CPU: ortholog features per rung
    python bench/scripts/cross_align.py sweep --seed 0 --part 0/4             # GPU, resumable, one part per server
    python bench/scripts/cross_align.py final --runs fgw_a0.5,ufgw_a0.5_t0.95 --seeds 0,1   # GPU

Rungs are the cross pairs X1-X10 (series.yaml, tier full) on the sections of chains.yaml; loops are the rung pairs
where each species advances at most one stage (6 squares, 6 triangles). The temporal couplings are the dense final
ones of bench/scripts/temporal_align.py in cross.temporal_dir (every run there with all pairs is used).

`prepare` builds the ortholog embedding and held-out genes of each rung (cache/features/crossloop_<h>__<m>), from the
section caches the temporal prepare wrote, and writes rungs.csv and loops.csv.
`sweep` takes the settings of cross.sweep (every `--part i/N`-th one, so N servers split the work), and per setting
solves all rungs, scores them (runs/seed<s>/<setting>.csv), and evaluates every loop in both directions against
every temporal run (comm/seed<s>/<setting>.csv, written last: a setting with that file is done and skipped on a
rerun). Each coupling is kept as its columns' top-k in couplings/<rung>/<setting>_seed<s>.npz.
`final` solves the chosen runs on every rung and seed, keeps the dense couplings (final/<rung>/<run>_seed<s>.npz),
scores them (final/runs_seed<s>.csv, final/comm_seed<s>.csv), keeps the per-tile commutativity
(final/tiles/<loop>_<dir>/<run>__<temporal>_seed<s>.npz) and, for --view-seed, the loop viewer's arrays
(final/view/<loop>_<dir>/<run>__<temporal>.npz) for the --view-temporal runs.

Everything goes to <out>/n<max_obs>/<temporal folder>-x<hash> (default <root>/diag/cross); the hash covers the
features, solver tolerances, epsilon and drop floor, not the grids. demos/cross_alignment.ipynb reads only that.
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
from mgw.bench import config, prepare, crossloop as cl  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def temporal_folder(cfg, root):
    """The temporal results folder of cross.temporal_dir (None: the one whose final files are newest)."""
    n = cfg.raw["reduce"]["max_obs"]
    base = root / "diag" / "temporal" / f"n{'all' if n is None else int(n)}"
    td = cfg.raw["cross"]["temporal_dir"]
    if td:
        p = Path(td)
        p = p if p.is_absolute() else base / p
        if not (p / "final").is_dir():
            raise SystemExit(f"no final temporal couplings in {p}")
        return p
    cands = [d for d in base.iterdir() if (d / "final").is_dir()] if base.is_dir() else []
    if not cands:
        raise SystemExit(f"no temporal folder with final couplings under {base}")
    return max(cands, key=lambda d: max((f.stat().st_mtime for f in (d / "final").rglob("*.npz")), default=0))


def context(args):
    cfg = config.load(args.config, list(args.set))
    root = prepare.bench_root(args.root)
    tdir = temporal_folder(cfg, root)
    n = cfg.raw["reduce"]["max_obs"]
    out = Path(args.out) if args.out else root / "diag" / "cross"
    out = out / f"n{'all' if n is None else int(n)}" / f"{tdir.name}-x{cfg.cross_hash(tdir.name)}"
    out.mkdir(parents=True, exist_ok=True)
    settings = out / "settings.json"
    if not settings.exists():
        settings.write_text(json.dumps(dict(cfg.cross_key(tdir.name), temporal_dir=str(tdir)), indent=1, default=str))
    rungs = cl.rungs(cfg)
    loops = cl.loops(rungs)
    if args.loops:
        want = set(args.loops.split(","))
        unknown = want - {l.id for l in loops}
        if unknown:
            raise SystemExit(f"unknown loops {sorted(unknown)}; known {[l.id for l in loops]}")
        loops = [l for l in loops if l.id in want]
        used = {r for l in loops for r in (l.r1.id, l.r2.id)}
        rungs = [r for r in rungs if r.id in used]
    return cfg, root, out, tdir, rungs, loops


# ---------- prepare
def _features_one(cfg_path, overrides, root, rung, seed, force):
    cfg = config.load(cfg_path, list(overrides))
    t0 = time.perf_counter()
    cl.cross_features(rung, cfg, seed, root, orthologs.load_one2one(), force=force)
    return time.perf_counter() - t0


def cmd_prepare(args):
    cfg, root, out, tdir, rungs, loops = context(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    sh = cfg.section_hash()
    missing = [(s, sd) for sd in seeds for r in rungs for s in (r.h, r.m)
               if not prepare.counts_file(root, s, sd, sh).exists()]
    if missing:
        raise SystemExit(f"section caches missing (run temporal_align.py prepare first): {sorted(set(missing))}")
    todo = [(r, sd) for sd in seeds for r in rungs
            if args.force or not cl.feature_path(r, cfg, sd, root).exists()]
    log(f"{len(rungs)} rungs, {len(loops)} loops, seeds {seeds}: {len(todo)} feature files to build; temporal {tdir}")
    failed = []
    with ProcessPoolExecutor(args.jobs) as pool:
        jobs = {pool.submit(_features_one, args.config, args.set, root, r, sd, args.force): (r, sd) for r, sd in todo}
        for n, f in enumerate(as_completed(jobs), 1):
            r, sd = jobs[f]
            try:
                log(f"  [{n}/{len(todo)}] {r.id} {r.h} x {r.m} seed {sd}: {f.result():.0f} s")
            except Exception as e:
                failed.append((r.id, sd))
                log(f"  [{n}/{len(todo)}] {r.id} seed {sd} FAILED: {e!r}")
    if failed:
        raise SystemExit(f"{len(failed)} rungs failed: {failed}")
    rows = []
    for sd in seeds:
        for r in rungs:
            d = cl.load_rung(r, cfg, sd, root)
            rows.append(dict(seed=sd, rung=r.id, h=r.h, m=r.m, stage_h=r.stage_h, stage_m=r.stage_m, n_h=d.sh.n,
                             n_m=d.sm.n, bins_h=float(d.sh.comp.sum()), bins_m=float(d.sm.comp.sum()),
                             n_orthologs=d.n_orthologs))
    save_csv(pd.DataFrame(rows), out / "rungs.csv")
    save_csv(cl.loop_table(loops), out / "loops.csv")
    pairs = sorted({p for l in loops for p in (l.th, l.tm) if p})
    for sd in seeds:
        log(f"  seed {sd}: temporal runs with every pair: {cl.temporal_runs(tdir, sd, pairs)}")
    log(f"prepare done -> {out}")


# ---------- shared by sweep and final
class Inputs:
    """Rung problems, section infos and temporal kernels of one seed, built once."""

    def __init__(self, cfg, root, tdir, rungs, loops, seed, temporal=None):
        self.cfg, self.seed = cfg, seed
        c = cfg.raw["cross"]
        self.pairs = sorted({p for l in loops for p in (l.th, l.tm) if p})
        avail = cl.temporal_runs(tdir, seed, self.pairs)
        self.truns = [t for t in (temporal or avail) if t in avail]
        if temporal and set(temporal) - set(avail):
            raise SystemExit(f"temporal runs {sorted(set(temporal) - set(avail))} lack final couplings in {tdir}")
        if not self.truns:
            raise SystemExit(f"no temporal run with final couplings of all of {self.pairs} (seed {seed}) in {tdir}")
        t0 = time.perf_counter()
        self.probs = {}
        self.sections = {}
        for r in rungs:
            d = cl.load_rung(r, cfg, seed, root)
            self.probs[r.id] = cl.RungProblem(d, cfg)
            self.sections[r.h], self.sections[r.m] = cl.section_info(d.sh), cl.section_info(d.sm)
        sh = cfg.section_hash()
        self.KT = {}
        for t in self.truns:
            kt = cl.temporal_kernels(tdir, t, seed, self.pairs, c["drop_floor"])
            self.KT[t] = {p: k for p, (k, _) in kt.items()}
            for p in self.pairs:
                a, b = cl.temporal_sections(tdir, seed, p)
                for s in (a, b):
                    if s not in self.sections:
                        self.sections[s] = cl.section_info(prepare.load_section(root, s, seed, sh))
                if self.KT[t][p].shape != (len(self.sections[a].w), len(self.sections[b].w)):
                    raise RuntimeError(f"temporal {t} {p}: coupling shape and section caches disagree")
        log(f"inputs seed {seed}: {len(rungs)} rungs, {len(self.truns)} temporal runs x {len(self.pairs)} pairs "
            f"({time.perf_counter() - t0:.0f} s)")


def solve_rung(prob, method, params, seed):
    """(Q or None, row)."""
    t0 = time.perf_counter()
    try:
        Q, info = prob.solve(method, params)
        if not np.isfinite(Q).all() or Q.sum() <= 0:
            raise FloatingPointError("non-finite or empty coupling")
        row = cl.evaluate_rung(Q, prob, method, params, seed=seed)
        row.update(info, error="", finite=True)
    except Exception as e:          # a failed solve is a row, not the end of the sweep
        traceback.print_exc()
        Q, row = None, dict(error=repr(e)[:300], finite=False)
    row["seconds_total"] = time.perf_counter() - t0
    return Q, row


def run_head(seed, method, params, setting):
    return dict(seed=seed, run=setting, method=method, alpha=params.get("alpha", 0.0),
                rho=params.get("rho", np.inf), reference=cl.is_reference(method))


def kernels_of(Q, prob, floor):
    Khm, dh = cl.kernel(Q, prob.b, floor)
    Kmh, dm = cl.kernel(Q.T, prob.a, floor)
    return Khm, Kmh


def eval_loops(inp, loops, KQ, head, per_tile=False, on_tiles=None):
    rows = []
    for t in inp.truns:
        for l in loops:
            for direction in cl.DIRECTIONS:
                if l.r1.id not in KQ or l.r2.id not in KQ:
                    rows.append({**head, "temporal": t, "loop": l.id, "kind": l.kind, "direction": direction,
                                 "error": "a rung failed"})
                    continue
                row, tiles = cl.evaluate_loop(l, direction, KQ, inp.KT[t], inp.sections, per_tile=per_tile)
                rows.append({**head, "temporal": t, **row, "error": ""})
                if on_tiles is not None:
                    on_tiles(t, l, direction, tiles)
    return rows


# ---------- sweep
def cmd_sweep(args):
    cfg, root, out, tdir, rungs, loops = context(args)
    methods = tuple(args.methods.split(",")) if args.methods else cl.METHODS
    bad = set(methods) - set(cl.METHODS)
    if bad:
        raise SystemExit(f"unknown methods {sorted(bad)}; known {cl.METHODS}")
    i, n = (int(x) for x in args.part.split("/"))
    settings = cl.grid(cfg, methods)[i::n]
    cdir, rdir = cl.comm_dir(out, args.seed), cl.runs_dir(out, args.seed)
    todo = [(m, p) for m, p in settings if args.force or not (cdir / f"{cl.run_name(m, p)}.csv").exists()]
    log(f"sweep seed {args.seed} part {i}/{n}: {len(settings)} settings, {len(todo)} to do, {len(rungs)} rungs, "
        f"{len(loops)} loops -> {out}")
    if not todo:
        return
    inp = Inputs(cfg, root, tdir, rungs, loops, args.seed)
    c = cfg.raw["cross"]
    for k, (m, p) in enumerate(todo, 1):
        name = cl.run_name(m, p)
        head = run_head(args.seed, m, p, name)
        rows, KQ = [], {}
        for r in rungs:
            prob = inp.probs[r.id]
            Q, row = solve_rung(prob, m, p, args.seed)
            rows.append({**head, "rung": r.id, "h": r.h, "m": r.m, **row})
            if Q is not None:
                KQ[r.id] = kernels_of(Q, prob, c["drop_floor"])
                idx, vals, cov = cl.tp.topk_columns(Q, c["topk"])
                rows[-1]["topk_mass"] = cov
                prepare.atomic_savez(out / "couplings" / r.id / f"{name}_seed{args.seed}.npz", idx=idx, vals=vals,
                                     n1=Q.shape[0])
            msg = row["error"] or (f"kappa {row['kappa']:.3f}, gene r {row['gene_r']:.3f}, mass {row['mass']:.3f}, "
                                   f"{'conv' if row.get('converged') else 'NOT conv'}")
            log(f"  [{k}/{len(todo)}] {name} {r.id}: {msg}; {row['seconds_total']:.1f} s")
        save_csv(pd.DataFrame(rows), rdir / f"{name}.csv")
        t0 = time.perf_counter()
        crow = eval_loops(inp, loops, KQ, head)
        save_csv(pd.DataFrame(crow), cdir / f"{name}.csv")
        ok = pd.DataFrame([r for r in crow if not r["error"]])
        summary = ", ".join(f"{t.split('_e')[0]} {g.comm.mean():.3f}" for t, g in ok.groupby("temporal")) if len(ok) \
            else "no loop"
        log(f"  [{k}/{len(todo)}] {name}: comm {summary}; {time.perf_counter() - t0:.0f} s for "
            f"{len(crow)} loop evaluations")
    log(f"sweep done -> {out}")


# ---------- final
def cmd_final(args):
    cfg, root, out, tdir, rungs, loops = context(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [cl.parse_run(r.strip()) for r in args.runs.split(",") if r.strip()]
    temporal = [t for t in args.temporal.split(",") if t] if args.temporal else None
    view_t = [t for t in args.view_temporal.split(",") if t] if args.view_temporal else []
    c = cfg.raw["cross"]
    fdir = out / "final"
    for sd in seeds:
        inp = Inputs(cfg, root, tdir, rungs, loops, sd, temporal)
        vt = [t for t in view_t if t in inp.truns]
        if sd == args.view_seed and view_t and len(vt) < len(view_t):
            raise SystemExit(f"--view-temporal {sorted(set(view_t) - set(vt))} not among {inp.truns}")
        rpath, cpath = fdir / f"runs_seed{sd}.csv", fdir / f"comm_seed{sd}.csv"
        rtab = pd.read_csv(rpath).to_dict("records") if rpath.exists() else []
        ctab = pd.read_csv(cpath).to_dict("records") if cpath.exists() else []
        done = {r["run"] for r in ctab}
        for m, p in runs:
            name = cl.run_name(m, p)
            if name in done and not args.force:
                log(f"  seed {sd} {name}: done")
                continue
            head = run_head(sd, m, p, name)
            KQ, rows = {}, []
            for r in rungs:
                prob = inp.probs[r.id]
                f = fdir / r.id / f"{name}_seed{sd}.npz"
                if f.exists() and not args.force:
                    Q = np.load(f)["Q"].astype(np.float64)
                    row = {**cl.evaluate_rung(Q, prob, m, p, seed=sd), "error": "", "finite": True,
                           "seconds_total": np.nan}
                else:
                    Q, row = solve_rung(prob, m, p, sd)
                    if Q is not None:
                        d = prob.data
                        prepare.atomic_savez(f, Q=Q.astype(np.float32), a=prob.a, b=prob.b, xy_h=d.sh.xy,
                                             xy_m=d.sm.xy, comp_h=d.sh.comp.astype(np.float32),
                                             comp_m=d.sm.comp.astype(np.float32), h=r.h, m=r.m, method=m,
                                             params=json.dumps(p))
                rows.append({**head, "rung": r.id, "h": r.h, "m": r.m, **row})
                if Q is not None:
                    KQ[r.id] = kernels_of(Q, prob, c["drop_floor"])
                log(f"  seed {sd} {name} {r.id}: " + (row["error"] or f"kappa {row['kappa']:.3f}, "
                                                                     f"gene r {row['gene_r']:.3f}"))

            def keep(t, l, direction, tiles, name=name, sd=sd):
                key = f"{l.id}_{direction}"
                st = inp.sections[l.start(direction)]
                prepare.atomic_savez(fdir / "tiles" / key / f"{name}__{t}_seed{sd}.npz", comm=tiles["comm"],
                                     sharp=tiles["sharp"], flow1=tiles["flow1"], flow2=tiles["flow2"],
                                     xy=st.xy.astype(np.float32), organ=cl.metrics.majority(st.comp),
                                     start=l.start(direction), target=l.target(direction))
                if sd == args.view_seed and t in vt:
                    arrays = cl.view_arrays(l, direction, KQ, inp.KT[t], inp.sections, k=c["view_k"], tiles=tiles)
                    prepare.atomic_savez(fdir / "view" / key / f"{name}__{t}.npz", run=name, temporal=t, **arrays)

            crow = eval_loops(inp, loops, KQ, head, per_tile=True, on_tiles=keep)
            rtab = [r for r in rtab if r["run"] != name] + rows
            ctab = [r for r in ctab if r["run"] != name] + crow
            save_csv(pd.DataFrame(rtab), rpath)
            save_csv(pd.DataFrame(ctab), cpath)
            ok = pd.DataFrame([r for r in crow if not r["error"]])
            log(f"  seed {sd} {name}: mean comm {ok.comm.mean() if len(ok) else np.nan:.3f}")
    log(f"final done -> {fdir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="bench.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra config overrides")
    ap.add_argument("--root", default=None, help="bench root (default $MGW_BENCH_DIR or <data>/bench)")
    ap.add_argument("--out", default=None, help="results (default <root>/diag/cross)")
    ap.add_argument("--loops", default=None, help="loop ids (X3-X5,X3-X4); default all, rungs follow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--force", action="store_true", help="rebuild the rung features")
    p.set_defaults(fn=cmd_prepare)
    s = sub.add_parser("sweep")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--part", default="0/1", help="i/N: every N-th setting from the i-th (split across servers)")
    s.add_argument("--methods", default=None, help=f"comma-separated subset of {','.join(cl.METHODS)}")
    s.add_argument("--force", action="store_true", help="solve again what is done")
    s.set_defaults(fn=cmd_sweep)
    f = sub.add_parser("final")
    f.add_argument("--runs", required=True, help="comma-separated run names (runs/<seed>/*.csv `run`)")
    f.add_argument("--seeds", default="0,1")
    f.add_argument("--temporal", default=None, help="temporal runs to compose with (default: all with finals)")
    f.add_argument("--view-temporal", default=None, help="temporal runs to write viewer files for")
    f.add_argument("--view-seed", type=int, default=0)
    f.add_argument("--force", action="store_true", help="solve again what is done")
    f.set_defaults(fn=cmd_final)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
