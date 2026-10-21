"""Pipelines: a strategy (how targets are formed and chosen) with an aligner. One task = (pipeline, pair, seed)."""
import time
import traceback

from mgw.bench import store
from mgw.bench.pipelines import real2d, volume as volume_strategy
from mgw.bench.pipelines.common import Task

STRATEGY = {
    "real2d": real2d.run,
    "volume": volume_strategy.run,
}


def _peak_device_bytes():
    """Peak memory of the first JAX device, if the backend reports it (GPUs do; None otherwise)."""
    try:
        import jax
        stats = jax.devices()[0].memory_stats() or {}
        return stats.get("peak_bytes_in_use")
    except Exception:   # noqa: BLE001
        return None


def applies(cfg, pipeline, pair):
    return pair.kind in cfg.raw["pipelines"][pipeline]["kinds"]


def run_task(cfg, pipeline, pair, seed, root, force=False, log=print):
    """Run one task unless its results exist with the same hash. Returns the result folder."""
    out = store.result_dir(root, cfg, pipeline, pair.id, seed)
    if not force and store.is_done(out, cfg, pipeline):
        log(f"[skip] {pipeline} {pair.id} seed {seed}: done ({out})")
        return out
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.perf_counter()
    log(f"[run] {pipeline} ({cfg.pipeline_hash(pipeline)}) {pair.id} seed {seed} -> {out}")
    store.write_manifest(out, cfg, pipeline, pair, seed, "running", started)
    try:
        task = Task(cfg, pipeline, pair, seed, root, log=log)
        STRATEGY[cfg.raw["pipelines"][pipeline]["strategy"]](task)
        df = task.save()
    except Exception:
        store.write_manifest(out, cfg, pipeline, pair, seed, "failed", started,
                             extra=dict(error=traceback.format_exc()))
        raise
    task.timings["total"] = time.perf_counter() - t0
    store.write_manifest(out, cfg, pipeline, pair, seed, "done", started,
                         extra=dict(timings=task.timings, n_rows=len(df), cost_scale=task.scale,
                                    peak_device_bytes=_peak_device_bytes()))
    sel = df[df.role == "selected"]
    if len(sel):
        log(f"[done] {pipeline} {pair.id} seed {seed}: kappa_bal (selected) "
            + ", ".join(f"{d} {g.kappa_bal.mean():.3f}" for d, g in sel.groupby("direction_label"))
            + f"; {task.timings['total']:.0f} s")
    return out
