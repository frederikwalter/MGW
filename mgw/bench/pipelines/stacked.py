"""The registered stack of an embryo: stage 1 of the volume approach (docs/st_3d_alignment_strategies.md §1b).

The stack (order, registration, gaps; `bench.stack`) is built once per (embryo, seed, settings) and cached in
cache/stacks/. Its points then carry the pair's features: the ortholog embedding, the organ compositions and the
held-out genes of that side.
"""
import contextlib
from pathlib import Path

import numpy as np

from mgw import stack3d as s3
from mgw.bench import prepare, stack as bstack, store
from mgw.bench.solvers import centred


def stack_path(root, cfg, seed, embryo):
    return Path(root) / "cache" / "stacks" / embryo.name / f"seed{seed}_{cfg.stack_hash()}.npz"


def stack_file(task, embryo):
    return stack_path(task.root, task.cfg, task.seed, embryo)


def load_stack(cfg, root, seed, embryo, secs, log=print, timer=None):
    """The stack of `embryo` (built and cached on the first call), with the features of `secs` ({name: Section},
    e.g. one side of a pair) attached in stack order: emb["X_ortho"], comp, Y, and L per point."""
    f = stack_path(root, cfg, seed, embryo)
    if f.exists():
        ST = bstack.from_arrays(np.load(f))
    else:
        feats = prepare.load_stack_features(embryo, cfg, seed, root)
        names = list(embryo.sections)
        scale = float(np.median([s3.rms_radius(secs[s].xy) for s in names]))
        xy_raw = [centred(secs[s].xy, scale) for s in names]
        with (timer or contextlib.nullcontext)():
            ST, diag = bstack.build_stack(names, xy_raw, [feats[s] for s in names], [secs[s].label for s in names],
                                          bstack.StackParams.from_mapping(cfg.raw["stack"]),
                                          dataset=embryo.dataset, fgw_alpha=cfg.raw["fgw"]["alpha"], seed=seed)
        log(f"  stack {embryo.name}: order {ST['names']} ({diag['mode']}), z {np.round(ST['z'], 3).tolist()}")
        store.savez(f, **bstack.to_arrays(ST, diag))
        ST = bstack.from_arrays(np.load(f))
    ST["emb"]["X_ortho"] = np.vstack([secs[s].emb for s in ST["names"]])
    ST["comp"] = np.vstack([secs[s].comp for s in ST["names"]])
    ST["Y"] = np.vstack([secs[s].Y for s in ST["names"]])
    ST["L"] = np.concatenate([np.full(secs[s].n, secs[s].L) for s in ST["names"]])
    return ST


def get_stack(task, direction):
    """The target embryo of a direction as a stack, with the pair's features attached (stack order)."""
    _, ts = task.sides(direction)
    return load_stack(task.cfg, task.root, task.seed, task.target_embryo(direction), ts, log=task.log,
                      timer=lambda: task.time("stack"))
