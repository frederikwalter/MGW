"""Result layout, manifests and loaders.

results/<pipeline>-<hash8>/<pair>/seed<s>/
  manifest.json     config (resolved), hashes, commit, host, devices, versions, timings, status
  scores.csv        one row per (direction, query, target, role): kappa_bal, K, objective, seconds
  transfer.npz      per selected row "<direction>|<query>": the transferred organ of every query spot (uint8, the
                    index into metrics.ORGANS), for the label-transfer figure
"""
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mgw.bench.config import REPO, canonical

SCORE_ID = ["pipeline", "hash", "pair", "seed", "direction", "direction_label", "query", "target", "role"]


def safe(name):
    return str(name).replace(":", "_").replace("/", "_")


def result_dir(root, cfg, pipeline, pair_id, seed):
    return Path(root) / "results" / cfg.pipeline_dir(pipeline) / safe(pair_id) / f"seed{seed}"


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def env_info():
    info = dict(host=socket.gethostname(), python=platform.python_version(), commit=git_commit(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
                cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        import jax
        import ott
        info.update(jax=jax.__version__, ott=ott.__version__, devices=[str(d) for d in jax.devices()])
    except Exception as e:   # noqa: BLE001
        info["devices"] = f"unavailable: {e}"
    return info


def manifest_path(d):
    return Path(d) / "manifest.json"


def is_done(d, cfg, pipeline):
    m = manifest_path(d)
    if not m.exists():
        return False
    try:
        man = json.loads(m.read_text())
    except json.JSONDecodeError:
        return False
    return man.get("status") == "done" and man.get("hash") == cfg.pipeline_hash(pipeline)


def write_manifest(d, cfg, pipeline, pair, seed, status, started, extra=None):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    man = dict(status=status, pipeline=pipeline, hash=cfg.pipeline_hash(pipeline), pair=pair.id,
               seed=seed, started=started, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
               config_file=str(cfg.path) if cfg.path else None, overrides=list(cfg.overrides),
               pipeline_key=json.loads(canonical(cfg.pipeline_key(pipeline))), env=env_info(), **(extra or {}))
    tmp = manifest_path(d).with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=1, default=str))
    os.replace(tmp, manifest_path(d))


def write_scores(d, rows):
    df = pd.DataFrame(rows)
    front = [c for c in SCORE_ID if c in df]
    df = df[front + [c for c in df if c not in front]]
    tmp = Path(d) / "scores.tmp.csv"
    df.to_csv(tmp, index=False)
    os.replace(tmp, Path(d) / "scores.csv")
    return df


def savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def key(direction, query):
    """The key of a query's transferred labels in transfer.npz."""
    return f"{direction}|{query}"


# ---------- loading results
def iter_result_dirs(root, pipelines=None):
    base = Path(root) / "results"
    if not base.exists():
        return
    for pdir in sorted(base.iterdir()):
        if not pdir.is_dir():
            continue
        name = pdir.name.rsplit("-", 1)[0]
        if pipelines and name not in pipelines and pdir.name not in pipelines:
            continue
        for pair_dir in sorted(pdir.iterdir()):
            for seed_dir in sorted(pair_dir.glob("seed*")):
                if manifest_path(seed_dir).exists():
                    yield seed_dir


def load_scores(root, pipelines=None, status="done"):
    """All scores.csv under root (optionally only some pipelines or <pipeline>-<hash> folders)."""
    frames = []
    for d in iter_result_dirs(root, pipelines):
        man = json.loads(manifest_path(d).read_text())
        if status and man.get("status") != status or not (d / "scores.csv").exists():
            continue
        df = pd.read_csv(d / "scores.csv")
        df["result_dir"] = str(d)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
