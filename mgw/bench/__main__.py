"""python -m mgw.bench [--config bench.yaml] [--set key=value ...] [--root DIR] <command> ...

Commands (in the order of a run):
  check-data  list the tier's sections without a raw file (with curl commands)
  prepare     reduce sections, cache organ counts and ortholog counts, compute the features (data server, CPU)
  plan        write the task list (one line per pipeline x pair x seed) for a job array
  run         run tasks: --task i of a task list (SLURM array), or everything matching the filters
  report      the three kappa_bal tables and the label-transfer figures

Caches and results live under --root, $MGW_BENCH_DIR, or <MGW data dir>/bench.
"""
import os

# threads: the SLURM allocation, not the whole node (set before numpy is imported)
if os.environ.get("SLURM_CPUS_PER_TASK"):
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, os.environ["SLURM_CPUS_PER_TASK"])

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:          # validation/ (moscot wrapper) lives at the repo root
    sys.path.insert(0, str(REPO))


def _list(s, cast=str):
    if s is None:
        return None
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if cast is int and "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(cast(part))
    return out


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _context(args):
    from mgw.bench import config, prepare, series
    cfg = config.load(args.config, args.set)
    ser = series.load(cfg.series_path())
    root = prepare.bench_root(args.root)
    return cfg, ser, root


def _select(cfg, ser, args, tier):
    """(pipeline, pair, seed) of a tier after the --pipelines / --pairs / --seeds filters."""
    from mgw.bench.pipelines import applies
    pipes = _list(getattr(args, "pipelines", None)) or list(cfg.raw["pipelines"])
    pairs = _list(getattr(args, "pairs", None)) or ser.tier(tier)
    seeds = _list(getattr(args, "seeds", None), int) or list(cfg.raw["seeds"][tier])
    out = []
    for pid in pairs:
        pair = ser.pairs[pid]
        for p in pipes:
            if applies(cfg, p, pair):
                out += [(p, pair, s) for s in seeds]
    return out


def cmd_check_data(args):
    from mgw import crossspecies as cs
    from mgw.bench import prepare
    cfg, ser, root = _context(args)
    miss = prepare.missing_files(ser, args.tier)
    hesta, mosta = prepare.data_dirs()
    log(f"HESTA {hesta}, MOSTA {mosta}; {len(ser.all_sections(args.tier))} sections in tier {args.tier}")
    if not miss:
        log("all present")
        return
    print(f"{len(miss)} missing:")
    for ds, s, p in miss:
        print(f"curl -L -C - -o {p} {cs.DOWNLOAD_URL[ds].format(s)}")
    sys.exit(1)


def cmd_prepare(args):
    from mgw import orthologs
    from mgw.bench import prepare
    cfg, ser, root = _context(args)
    seeds = _list(args.seeds, int) or list(cfg.raw["seeds"][args.tier])
    pids = _list(args.pairs) or ser.tier(args.tier)
    ortho = orthologs.load_one2one()
    scores = prepare.laterality_table()
    secs = []
    for pid in pids:
        for e in ser.pairs[pid].embryos:
            secs += [(e.dataset, s) for s in e.sections if (e.dataset, s) not in secs]
    log(f"prepare: {len(secs)} sections x seeds {seeds} -> {root}")
    for seed in seeds:
        for n, (ds, s) in enumerate(secs):
            t0 = time.perf_counter()
            prepare.prepare_section(ds, s, cfg, seed, root, ortho, scores, force=args.force)
            log(f"  seed {seed} [{n + 1}/{len(secs)}] {s} ({time.perf_counter() - t0:.0f} s)")
        stacks = set()
        for pid in pids:
            pair = ser.pairs[pid]
            prepare.pair_features(pair, cfg, seed, root, ortho)
            log(f"  seed {seed} features {pid}")
            if pair.kind == "cross":
                for e in pair.embryos:
                    if e.name not in stacks:
                        prepare.stack_features(e, cfg, seed, root)
                        stacks.add(e.name)
    log("prepare done")


def cmd_plan(args):
    cfg, ser, root = _context(args)
    tasks = _select(cfg, ser, args, args.tier)
    out = Path(args.out or root / "tasks" / f"{cfg.raw['name']}_{args.tier}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for i, (p, pair, s) in enumerate(tasks):
            fh.write(json.dumps(dict(i=i, pipeline=p, pair=pair.id, seed=s, config=str(cfg.path),
                                     overrides=list(cfg.overrides))) + "\n")
    log(f"{len(tasks)} tasks -> {out}  (array: --array=0-{len(tasks) - 1})")
    print(str(out))


def cmd_run(args):
    from mgw.bench import config, series
    from mgw.bench.pipelines import run_task
    cfg, ser, root = _context(args)
    if args.task is not None:
        lines = Path(args.tasks).read_text().splitlines()
        t = json.loads(lines[args.task])
        cfg = config.load(t["config"], t["overrides"] + list(args.set or []))
        ser = series.load(cfg.series_path())
        jobs = [(t["pipeline"], ser.pairs[t["pair"]], int(t["seed"]))]
    else:
        jobs = _select(cfg, ser, args, args.tier)
    log(f"{len(jobs)} task(s), root {root}")
    failed = []
    for p, pair, s in jobs:
        try:
            run_task(cfg, p, pair, s, root, force=args.force, log=log)
        except Exception as e:   # noqa: BLE001
            log(f"[failed] {p} {pair.id} seed {s}: {type(e).__name__}: {e}")
            failed.append((p, pair.id, s))
            if args.task is not None or args.stop_on_error:
                raise
    if failed:
        log(f"{len(failed)} failed: {failed}")
        sys.exit(1)


def cmd_report(args):
    from mgw.bench.report import build
    cfg, ser, root = _context(args)
    out = build.report(cfg, ser, root, tier=args.tier, name=args.name, seeds=_list(args.seeds, int),
                       figures_on=not args.no_figures, log=log)
    print(str(out))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mgw.bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="YAML config (path, or name in bench/configs); default bench.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    ap.add_argument("--root", default=None, help="cache and results root (default $MGW_BENCH_DIR or <data>/bench)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def tier_arg(p, default="dev"):
        p.add_argument("--tier", default=default, choices=["smoke", "dev", "screen", "full"])

    p = sub.add_parser("check-data"); tier_arg(p); p.set_defaults(fn=cmd_check_data)
    p = sub.add_parser("prepare"); tier_arg(p)
    p.add_argument("--seeds"); p.add_argument("--pairs"); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("plan"); tier_arg(p)
    p.add_argument("--seeds"); p.add_argument("--pairs"); p.add_argument("--pipelines"); p.add_argument("--out")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("run"); tier_arg(p)
    p.add_argument("--seeds"); p.add_argument("--pairs"); p.add_argument("--pipelines")
    p.add_argument("--tasks", help="task list from `plan`"); p.add_argument("--task", type=int, help="line of --tasks")
    p.add_argument("--force", action="store_true"); p.add_argument("--stop-on-error", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report"); tier_arg(p)
    p.add_argument("--name", help="report folder under reports/ (default: the tier)")
    p.add_argument("--seeds"); p.add_argument("--no-figures", action="store_true")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
