# Cross-species alignment benchmark (`mgw.bench`)

Step 1 of [`docs/st_3d_alignment_strategies.md`](../docs/st_3d_alignment_strategies.md) §6: the metric harness (κ_bal with OMS and Cohen's κ, guardrails), the series of §4, the 2D ↔ 2D reference pipeline (A with FGW, both directions), its nulls and ceilings, and a report with an interactive alignment viewer. The volume pipeline of §1 (steps 2–6) plugs into the same harness.

## Commands

```bash
python -m mgw.bench [--config step1.yaml] [--set key=value ...] [--root DIR] <command>
```

| Command | Where | What |
| --- | --- | --- |
| `check-data --tier screen` | data server | sections without a raw file, with `curl` commands |
| `prepare --tier screen [--seeds 0-4]` | data server (CPU, RAM) | reduce sections, cache organ counts and ortholog counts, compute the features |
| `plan --tier screen` | anywhere | task list `tasks/<name>_<tier>.jsonl`, one line per pipeline × pair × seed |
| `run --tasks FILE --task $SLURM_ARRAY_TASK_ID` | GPU | one task of the list (job arrays) |
| `run --tier dev [--pipelines A_fgw] [--pairs X4] [--seeds 0,1]` | GPU | every matching task, in sequence (screen session) |
| `noise --tier dev` | anywhere | noise margin: 2 × the SD over seeds of the pair-level κ_bal |
| `report --tier screen [--reference history:0]` | anywhere | the Go table, ceilings, ranking, viewers, PNG sheets |
| `report --tier dev --sweep reduce.max_obs --sweep-values 1000,2500,5000,10000` | anywhere | κ_bal and solve time against one setting |
| `promote <pipeline>` | locally, then commit | make a pipeline the reference; the old one goes to `history` |

The root (`--root`, `$MGW_BENCH_DIR`, or `<MGW data dir>/bench`) holds `cache/`, `results/`, `tasks/` and `reports/`. Raw sections come from `$MGW_HESTA_DIR` / `$MGW_MOSTA_DIR`, or `<data dir>/hesta`, `/mosta`.

Run `prepare` before `run`. `run` never reads the h5ad files, so only `cache/` has to be copied to a GPU cluster.

## Step 1 runbook

1. **LNT, data server.** `check-data`, then `prepare --tier dev --seeds 0-4` and `prepare --tier screen`.
2. **LNT, GPU boxes.** Run Dev (`run --tier dev --seeds 0,1`, `--seeds 2,3`, … in parallel screen sessions), then `noise --tier dev`.
3. **MAX_OBS sweep.** Prepare and run with `--set reduce.max_obs=1000`, `2500` and `10000`, plus the default 5000, for `--pairs X4,W:CS18_E1,W:E12.5_E1 --pipelines A_fgw --seeds 0`. 10000 runs on LRZ.
4. **LRZ.** rsync `cache/` (sections and features of seed 0) to the LRZ root, then `plan --tier screen` and a job array over the task list, with 1 GPU per task.
5. **Results.** rsync `results/` back, then `report --tier screen`; `reports/step1_screen/index.html` holds the Go table.

## Comparability across steps

- **Hashes.** Every setting that changes a result enters a hash (see `mgw/bench/config.py`), and results live in `results/<pipeline>-<hash>/<pair>/seed<s>/`. Runs with other settings never overwrite each other, and an unchanged task is skipped.
- **Shared schema.** `scores.csv` has one row per (direction, query, target, role), with the same columns for every pipeline. New pipelines add rows, never columns that old reports need.
- **Recomputable metrics.** Every candidate keeps its exact organ × organ matrix C and the top-64 entries per row and column. The finals (selected, oracle) keep the dense P. Most new label metrics can be computed from what is stored.
- **Moving reference.** [`configs/reference.yaml`](configs/reference.yaml) names the current reference and keeps the history. `report --reference history:<i>` recomputes Δκ against any earlier one.
- **Frozen series.** [`configs/series.yaml`](configs/series.yaml) fixes the sections of every embryo, and each step uses the same pairs.

## Adding the volume pipeline (steps 2–6)

A strategy is a function `run(task)` in `mgw/bench/pipelines/`, registered in `pipelines/__init__.py:STRATEGY` and allowed in `config.STRATEGIES` with its default parameters. It forms targets, solves, and calls `task.score(...)` for every coupling with a role (`candidate`, `selected`, `null:*`). Scoring, storage, the report and the viewer then work unchanged. Add the pipeline to a config file (e.g. `configs/step3.yaml`) and run the same commands.

The volume pipeline builds both stacks with `pipelines/stacked.get_stack` (stage 1), fits and samples the fields, solves 3D ↔ 3D (stage 2), and scores every real query section against the target volume: `P` = the query section's spots × the target's volume samples (the query spots carried to the query's samples), `xt` = the samples' 3D coordinates, `T_full` = the target bins carried to the samples. `task.score` accepts a 3D `xt` (fold-overs in 3D) and takes K from the whole target embryo, so the rows compare directly with A's. Store the top-k only (`dense: []`): a dense section × volume P is large.

## Notes

- **Seeds.** Seed s reduces with Voronoi seed 42 + s and seeds the randomized PCAs with s, so seed 0 reproduces the 3D notebook's preprocessing.
- **Cross-cost scale.** One per pair (the 0.99 quantile of all a × b feature distances), so a balanced FGW solve of (x, y) is the transpose of (y, x), and each pair of sections is solved once (`tests/test_solvers.py`). The notebook scales per query instead.
- **No mirror solve in A.** FGW on Euclidean distances is invariant to reflections (`tests/test_solvers.py`); `orientation` reports the sign.
- **Fold-overs.** The share of query points whose local Jacobian, fitted over 16 neighbours, flips against the majority orientation (2D targets), or whose image normal points against the mean normal (3D targets).
- **K.** `metrics.presence: embryo` (default) scores the query organs present anywhere in the target embryo, so a target section that lacks an organ scores it 0 instead of dropping it. `presence: target` gives the first harness's K (each target alone). The setting enters the pipeline hash.
- **Feature z-scoring.** `features.zscore_by: section` (default) z-scores genes within each section; `embryo` within each embryo, which keeps the depth offsets that a 3D field needs. G1 decides between them on a label-free score, and A then runs on the chosen features (doc §3, §6 step 2).
- **Tests.** `pytest` runs the metric, solver and config tests. On a machine with the four smoke sections, it also runs end-to-end checks of the cache and the compact storage.
