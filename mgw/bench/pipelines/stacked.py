"""The registered stack of an embryo: stage 1 of the volume approach (docs/st_3d_alignment_strategies.md §1b).

The stack (order, registration, gaps; `bench.stack`) is built once per (embryo, seed, settings) and cached in
cache/stacks/. Its points then carry the pair's features: the ortholog embedding, the organ compositions and the
held-out genes of that side.
"""
import numpy as np

from mgw import stack3d as s3
from mgw.bench import prepare, stack as bstack, store
from mgw.bench.solvers import centred


def stack_file(task, embryo):
    return (task.root / "cache" / "stacks" / embryo.name /
            f"seed{task.seed}_{task.cfg.stack_hash()}.npz")


def get_stack(task, direction):
    """The target embryo of a direction as a stack, with the pair's features attached (stack order)."""
    embryo = task.target_embryo(direction)
    _, ts = task.sides(direction)
    f = stack_file(task, embryo)
    if f.exists():
        ST = bstack.from_arrays(np.load(f))
    else:
        feats = prepare.load_stack_features(embryo, task.cfg, task.seed, task.root)
        names = list(embryo.sections)
        scale = float(np.median([s3.rms_radius(ts[s].xy) for s in names]))
        xy_raw = [centred(ts[s].xy, scale) for s in names]
        with task.time("stack"):
            ST, diag = bstack.build_stack(names, xy_raw, [feats[s] for s in names], [ts[s].label for s in names],
                                          bstack.StackParams.from_mapping(task.cfg.raw["stack"]),
                                          dataset=embryo.dataset, fgw_alpha=task.fgw.alpha, seed=task.seed)
        task.log(f"  stack {embryo.name}: order {ST['names']} ({diag['mode']}), z {np.round(ST['z'], 3).tolist()}")
        store.savez(f, **bstack.to_arrays(ST, diag))
        ST = bstack.from_arrays(np.load(f))
    ST["emb"]["X_ortho"] = np.vstack([ts[s].emb for s in ST["names"]])
    ST["comp"] = np.vstack([ts[s].comp for s in ST["names"]])
    ST["Y"] = np.vstack([ts[s].Y for s in ST["names"]])
    return ST
