"""How a much larger field behaves: train the fields of one pair at a few sizes, then score them (bench/README.md).

    python bench/scripts/field_scale.py train --size L [--embryo CS18_E1 | --array-index i] [--set field.tf32=true]
    python bench/scripts/field_scale.py score --sizes current,L,XL

`train` fits one full field per embryo of the pair (cached under the size's field hash in cache/fields/, like every
other field), so it runs anywhere: sim14, or one SLURM array task per embryo (--array-index 0 | 1 = side a | b).
`score` uses the cached fields of every size that has both and writes reports/field_scale/<pair>_seed<s>.csv:
- fit_<embryo>: R^2 of f at the embryo's own real spots (mean over sections; variance about each section's mean);
- std_ratio_<embryo>: std(f) / std(measured features) at those spots (< 1: the field smooths);
- kappa_pts_field: kappa_bal of stage 2 (unbalanced FGW) on the stack points of pts2pts_s-euc_f-euc_ub, with the
  field's features;
- kappa_vol: kappa_bal on the field's sampled volume, as vol2vol_s-euc_f-euc_ub.
kappa_bal is for information only; no setting here is chosen by labels.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mgw.bench import config, field, prepare, series, volume  # noqa: E402
from mgw.bench.pipelines import stacked  # noqa: E402
from mgw.bench.pipelines.common import Task  # noqa: E402

# field settings per size, on top of bench.yaml (current: bench.yaml itself)
SIZES = {
    "current": [],
    "L": ["field.width=512", "field.depth=6", "field.n_freq=256", "field.residual=true", "field.steps=100000",
          "field.batch=8192", "field.lr=1.0e-3"],
    # L without the smoothness penalty and with a 10x weaker depth loss: do the regularizers cap the fit?
    "L_noreg": ["field.width=512", "field.depth=6", "field.n_freq=256", "field.residual=true", "field.steps=100000",
                "field.batch=8192", "field.lr=1.0e-3", "field.lambda_smooth=0", "field.lambda_depth=0.1"],
    "XL": ["field.width=1024", "field.depth=8", "field.n_freq=512", "field.residual=true", "field.layernorm=true",
           "field.steps=200000", "field.batch=16384", "field.lr=5.0e-4"],
}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def r2(pred, true):
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean(0)) ** 2).sum())


def context(args, size):
    cfg = config.load(args.config, SIZES[size] + list(args.set))
    pair = series.load(cfg.series_path()).pairs[args.pair]
    root = prepare.bench_root(args.root)
    return cfg, pair, root, Task(cfg, "vol2vol_s-euc_f-euc_ub", pair, args.seed, root, log=lambda *a: None)


def sides(task):
    """{a | b: (embryo, stack)}: side a is the target embryo of a_stack."""
    out = {}
    for key, direction in (("a", "a_stack"), ("b", "b_stack")):
        out[key] = (task.target_embryo(direction), stacked.get_stack(task, direction))
    return out


def cmd_train(args):
    cfg, pair, root, task = context(args, args.size)
    todo = list(sides(task).values())
    if args.array_index is not None:
        todo = [todo[args.array_index]]
    elif args.embryo:
        todo = [t for t in todo if t[0].name == args.embryo]
    log(f"size {args.size} (field hash {cfg.field_hash()}): {', '.join(SIZES[args.size]) or 'bench.yaml'}; "
        f"embryos {[e.name for e, _ in todo]}, root {root}")
    for emb, ST in todo:
        fld = field.get_field(cfg, root, args.seed, pair.id, emb.name, ST, ST["emb"]["X_ortho"], log=log)
        n = sum(p.numel() for p in fld.net.parameters())
        log(f"  {emb.name}: {n / 1e6:.2f}M parameters -> {field.field_path(root, cfg, args.seed, pair.id, emb.name)}")


def kappa(task, sd, ms, p):
    """Pair-level kappa_bal (mean of both directions) and the mean rejected mass of one stage-2 solve."""
    Ca, Cb = volume.intra_cost("fgw", ms["a"], task.cfg), volume.intra_cost("fgw", ms["b"], task.cfg)
    P, _, _ = volume.stage2(Ca, Cb, ms["a"]["F"], ms["b"]["F"], task.scale, task.fgw, a=ms["a"]["mass"],
                            b=ms["b"]["mass"], tau=p["tau"], epsilon=p["epsilon"],
                            unbalanced_correction=p["unbalanced_correction"])
    rej = volume.rejected(P, ms["a"]["mass"], ms["b"]["mass"])
    out = []
    for direction, qk, tk, Pd in (("b_stack", "a", "b", P), ("a_stack", "b", "a", P.T)):
        STq, STt = sd[qk][1], sd[tk][1]
        qs, _ = task.sides(direction)
        T = np.asarray(volume.target_pool(ms[tk], STt, p["k_target"]) @ STt["comp"])
        out.append(np.nanmean([task.score(direction, qs[n], "x", T,
                                          np.asarray(volume.query_map(ms[qk], STq, s, p["k_query"]) @ Pd),
                                          role="selected", keep=False)["kappa_bal"]
                               for s, n in enumerate(STq["names"])]))
    return float(np.mean(out)), float(np.mean(rej[:2]))


def cmd_score(args):
    rows = []
    for size in args.sizes.split(","):
        cfg, pair, root, task = context(args, size)
        sd = sides(task)
        missing = [e.name for e, _ in sd.values()
                   if not field.field_path(root, cfg, args.seed, pair.id, e.name).exists()]
        if missing:
            log(f"[skip] {size}: no field for {missing} (field hash {cfg.field_hash()})")
            continue
        p = task.spec["params"]
        row = dict(size=size, field_hash=cfg.field_hash(), settings=" ".join(SIZES[size]))
        flds = {}
        for key, (emb, ST) in sd.items():
            fld = field.get_field(cfg, root, args.seed, pair.id, emb.name, ST, ST["emb"]["X_ortho"], log=log)
            flds[key] = fld
            off, F = ST["offsets"], ST["emb"]["X_ortho"]
            fit, ratio = [], []
            for k in range(len(ST["names"])):
                X, Y = ST["xyz"][off[k]:off[k + 1]], F[off[k]:off[k + 1]]
                P_ = fld.features(X)
                fit.append(r2(P_, Y))
                ratio.append(P_.std(0).mean() / Y.std(0).mean())
            row.update({f"fit_{emb.name}": float(np.mean(fit)), f"std_ratio_{emb.name}": float(np.mean(ratio)),
                        f"params_M_{emb.name}": sum(q.numel() for q in fld.net.parameters()) / 1e6,
                        f"train_s_{emb.name}": fld.meta["history"][-1].get("seconds", np.nan)})
        row["fit"] = float(np.mean([row[f"fit_{e.name}"] for e, _ in sd.values()]))
        pts = {k: volume.measure(cfg, root, args.seed, pair.id, e.name, ST, "points", p["n_samples"])
               for k, (e, ST) in sd.items()}
        pts = {k: dict(m, F=flds[k].features(m["X"])) for k, m in pts.items()}
        row["kappa_pts_field"], row["rej_pts_field"] = kappa(task, sd, pts, p)
        vol = {k: volume.measure(cfg, root, args.seed, pair.id, e.name, ST, "field", p["n_samples"])
               for k, (e, ST) in sd.items()}
        row["kappa_vol"], row["rej_vol"] = kappa(task, sd, vol, p)
        log(f"  {size}: fit {row['fit']:.3f} | kappa_bal points + field {row['kappa_pts_field']:.3f}, "
            f"volume {row['kappa_vol']:.3f}")
        rows.append(row)
    if not rows:
        return
    out = prepare.bench_root(args.root) / "reports" / "field_scale" / f"{args.pair}_seed{args.seed}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():                       # keep the rows of sizes not scored now
        old = pd.read_csv(out)
        df = pd.concat([old[~old["size"].isin(df["size"])], df], ignore_index=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.drop(columns=["settings", "field_hash"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log(f"-> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="bench.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="extra config overrides")
    ap.add_argument("--root", default=None, help="bench root (default $MGW_BENCH_DIR or <data>/bench)")
    ap.add_argument("--pair", default="X4")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--size", required=True, choices=list(SIZES))
    t.add_argument("--embryo", help="one embryo of the pair (default: both)")
    t.add_argument("--array-index", type=int, choices=[0, 1], help="0: side a (human), 1: side b (mouse)")
    t.set_defaults(fn=cmd_train)
    s = sub.add_parser("score")
    s.add_argument("--sizes", default="current,L,XL")
    s.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
