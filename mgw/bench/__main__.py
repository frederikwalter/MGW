"""python -m mgw.bench [--config step1.yaml] [--set key=value ...] [--root DIR] <command> ...

Commands (in the order of a run):
  check-data  list the tier's sections without a raw file (with curl commands)
  prepare     reduce sections, cache organ counts and ortholog counts, compute the features (data server, CPU)
  plan        write the task list (one line per pipeline x pair x seed) for a job array
  run         run tasks: --task i of a task list (SLURM array), or everything matching the filters
  noise       the noise margin from the Dev seeds
  report      the Go table, guardrails, viewers and PNG sheets
  view        one viewer (pair, seed)
  promote     make a pipeline the reference (the old one goes to the history)
  gates       g1 | g2: the within-species gates for the field (doc §3; config gates.yaml)

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


def cmd_noise(args):
    from mgw.bench.report import table
    cfg, ser, root = _context(args)
    res = table.noise_margin(cfg, ser, root, tier=args.tier)
    print(json.dumps(res, indent=1))


def cmd_report(args):
    from mgw.bench.report import build
    cfg, ser, root = _context(args)
    out = build.report(cfg, ser, root, tier=args.tier, name=args.name, reference=args.reference,
                       viewers=not args.no_viewer, pngs=not args.no_figures, seeds=_list(args.seeds, int),
                       sweep=args.sweep, sweep_values=_list(args.sweep_values), log=log)
    print(str(out))


def cmd_view(args):
    from mgw.bench.report import viewer
    cfg, ser, root = _context(args)
    out = viewer.build(cfg, ser.pairs[args.pair], args.seed, root, Path(args.out) if args.out else None)
    print(str(out))


def cmd_gates(args):
    """G1 / G2 on the gate embryos (doc §3): unit rows, the summary and the pass table under results/gates/."""
    import pandas as pd
    from mgw.bench import config, gates, prepare, series
    cfg = config.load(args.config or "gates.yaml", args.set)
    ser = series.load(cfg.series_path())
    root = prepare.bench_root(args.root)
    seeds = _list(args.seeds, int)
    embryos = _list(args.embryos)
    log(f"gate {args.gate} ({cfg.path.name}): embryos {embryos or list(cfg.raw['gates']['embryos'])}, "
        f"seeds {seeds if seeds is not None else cfg.raw['seeds']['dev']}, root {root}")
    rows, summ, sel = gates.run(cfg, ser, root, args.gate, embryos=embryos, seeds=seeds, grid=not args.no_grid,
                                force=args.force, log=log)
    out = root / "results" / "gates" / f"{cfg.raw['name']}_{args.gate}"
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        if args.gate == "g1":
            print("\nG1 per embryo (mean r_heldout of each predictor; pass: field > nearest + margin, >= McCann - "
                  "margin):")
            print(summ.to_string(index=False) if len(summ) else "no results")
            print("\nG1 settings, best first (score: mean r_heldout of the field over the embryos, label-free):")
            print(sel.to_string(index=False) if sel is not None and len(sel) else "no results")
        else:
            print("\nG2 per embryo and variant (errors in mean section spacings; pass: med3d < 0.5, lock <= 1.5 x "
                  "chance, field < point cloud):")
            print(summ.to_string(index=False) if len(summ) else "no results")
    log(f"tables -> {out}")


def cmd_promote(args):
    import yaml
    cfg, ser, root = _context(args)
    path = cfg.reference_path()
    ref = yaml.safe_load(path.read_text())
    if args.pipeline not in cfg.raw["pipelines"]:
        raise SystemExit(f"{args.pipeline} is not a pipeline of {cfg.path}")
    old = dict(ref["current"], hash=cfg.pipeline_hash(ref["current"]["pipeline"]) if ref["current"]["pipeline"] in
               cfg.raw["pipelines"] else None, until=time.strftime("%Y-%m-%d"))
    ref["history"] = (ref.get("history") or []) + [old]
    ref["current"] = dict(pipeline=args.pipeline, config=cfg.path.name, overrides=list(cfg.overrides),
                          hash=cfg.pipeline_hash(args.pipeline), since=time.strftime("%Y-%m-%d"), note=args.note or "")
    path.write_text("# The reference pipeline that Δκ is measured against; see `promote` in mgw/bench/__main__.py.\n"
                    + yaml.safe_dump(ref, sort_keys=False, allow_unicode=True))
    log(f"reference: {args.pipeline} ({ref['current']['hash']}); history {len(ref['history'])} entries")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mgw.bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="YAML config (path, or name in bench/configs); default step1.yaml, gates.yaml for `gates`")
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
    p = sub.add_parser("noise"); tier_arg(p); p.set_defaults(fn=cmd_noise)
    p = sub.add_parser("report"); tier_arg(p, "screen")
    p.add_argument("--name"); p.add_argument("--reference", help="a pipeline name, or history:<i>")
    p.add_argument("--seeds"); p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--sweep", help="a config key varied across result folders, e.g. reduce.max_obs")
    p.add_argument("--sweep-values", help="its values, e.g. 1000,2500,5000,10000")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("view"); p.add_argument("--pair", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out"); p.set_defaults(fn=cmd_view)
    p = sub.add_parser("promote"); p.add_argument("pipeline"); p.add_argument("--note")
    p.set_defaults(fn=cmd_promote)
    p = sub.add_parser("gates", help="G1 / G2 (default config: gates.yaml)")
    p.add_argument("gate", choices=["g1", "g2"])
    p.add_argument("--embryos", help="default: gates.embryos of the config"); p.add_argument("--seeds")
    p.add_argument("--no-grid", action="store_true", help="G1: the config's own setting only")
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_gates)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
