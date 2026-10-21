# Cross-species alignment benchmark (`mgw.bench`)

Steps 1–3 of [`docs/st_3d_alignment_strategies.md`](../docs/st_3d_alignment_strategies.md) §6:
- step 1: the metric harness (κ_bal with OMS and Cohen's κ, guardrails), the series of §4, the 2D ↔ 2D reference pipeline (A with FGW, both directions), its nulls and ceilings, and a report with an interactive alignment viewer;
- step 2: the field of §1c–§1d (`mgw/bench/field.py`) and the within-species gates G1 and G2 (`mgw/bench/gates.py`, `configs/gates.yaml`);
- step 3: the volume pipelines V_fgw, V_linear, V_gw and the point-cloud ablation PC_fgw (`mgw/bench/pipelines/volume.py`, `configs/step3.yaml`).

Steps 2 and 3 are implemented and tested on a synthetic embryo pair (`mgw/bench/synthetic.py`); they have not yet been run on HESTA/MOSTA.

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
| `gates g1 [--embryos CS18_E1,...] [--seeds 0-2] [--no-grid]` | GPU (or CPU, slower) | G1, held-out section, over the grid of `gates.yaml` (the default config of `gates`) |
| `gates g2 [--embryos ...] [--seeds ...]` | GPU | G2, the odd/even split, with field and point cloud |

The root (`--root`, `$MGW_BENCH_DIR`, or `<MGW data dir>/bench`) holds `cache/`, `results/`, `tasks/` and `reports/`. Raw sections come from `$MGW_HESTA_DIR` / `$MGW_MOSTA_DIR`, or `<data dir>/hesta`, `/mosta`. Steps 2–3 add `cache/stacks/<embryo>/*_skip.npz` (the direct couplings of sections two apart), `cache/fields/<pair>/<embryo>/` (trained fields, `.pt`), `cache/volumes/<pair>/<embryo>/` (sampled volumes) and `results/gates/`.

Run `prepare` before `run`. `run` never reads the h5ad files, so only `cache/` has to be copied to a GPU cluster.

## Step 1 runbook

1. **LNT, data server.** `check-data`, then `prepare --tier dev --seeds 0-4` and `prepare --tier screen`.
2. **LNT, GPU boxes.** Run Dev (`run --tier dev --seeds 0,1`, `--seeds 2,3`, … in parallel screen sessions), then `noise --tier dev`.
3. **MAX_OBS sweep.** Prepare and run with `--set reduce.max_obs=1000`, `2500` and `10000`, plus the default 5000, for `--pairs X4,W:CS18_E1,W:E12.5_E1 --pipelines A_fgw --seeds 0`. 10000 runs on LRZ.
4. **LRZ.** rsync `cache/` (sections and features of seed 0) to the LRZ root, then `plan --tier screen` and a job array over the task list, with 1 GPU per task.
5. **Results.** rsync `results/` back, then `report --tier screen`; `reports/step1_screen/index.html` holds the Go table.

## Steps 2–3 runbook (gates, then volumes)

All commands run from the repository root with the environment active (`uv run` or `.venv`). Add `--root DIR` if `$MGW_BENCH_DIR` is not set. The field trains on CUDA when torch sees a GPU; `$MGW_FIELD_DEVICE=cpu|cuda` overrides that. The device is not part of any hash.

1. **Features, data server.** The gates and the volumes use the same section, feature and stack caches as A, so `prepare --tier dev --seeds 0-2` from step 1 is enough for `zscore_by: section`. The G1 grid also compares `embryo`, so run:
   ```bash
   python -m mgw.bench --config gates.yaml prepare --tier dev --seeds 0-2 --set features.zscore_by=embryo
   ```
   With other features the stack is a new cache entry and is rebuilt once per embryo and seed.
2. **G1, GPU.** Start with the grid on one seed, then run the best settings on all seeds to get the noise margin:
   ```bash
   python -m mgw.bench gates g1 --seeds 0                        # grid: zscore_by x field.sigma_z (gates.yaml)
   python -m mgw.bench --set features.zscore_by=embryo --set field.sigma_z=1.0 gates g1 --seeds 0-2 --no-grid
   ```
   - Output: `results/gates/gates_g1/rows.csv` (one row per setting, embryo, seed, section and predictor), `summary.csv` (per embryo: mean r_heldout of each predictor, the deltas, the margin and `passed`), `selection.csv` (settings ranked by the label-free score) and `best.json`. A short table is printed.
   - Units already done are skipped, keyed by the gate hash. `--force` recomputes them.
3. **G2, GPU.**
   ```bash
   python -m mgw.bench --set features.zscore_by=<chosen> --set field.sigma_z=<chosen> gates g2 --seeds 0-2
   ```
   - Output: `results/gates/gates_g2/rows.csv` and `summary.csv` (per embryo and variant, field against point cloud). Embryos with < 4 sections are skipped.
4. **Rerun A on the chosen features, then the volumes on Dev (X1, X4, both directions), GPU.**
   ```bash
   S="--set features.zscore_by=<chosen> --set field.sigma_z=<chosen>"
   python -m mgw.bench --config step3.yaml $S run --tier dev --pairs X1,X4 --pipelines A_fgw --seeds 0
   python -m mgw.bench --config step3.yaml $S run --tier dev --pairs X1,X4 --pipelines V_fgw,PC_fgw,V_linear,V_gw --seeds 0
   python -m mgw.bench --config step3.yaml $S report --tier dev
   ```
   - `reports/step3_dev/` holds the Go table with the columns V, PC, V_feat_OT, V_GW and dV / dPC (κ_bal minus A's), and `volumes.csv` with rejected mass, the L and depth reached, and the parallel-section consistency.
   - N, τ, ε and k are pipeline parameters, e.g. `--set pipelines.V_fgw.params.n_samples=10000` or `...params.tau=0.9`. Every override enters only that pipeline's hash.
   - The field and the volume are cached per pair, embryo, seed and field hash, so V_fgw, V_linear and V_gw share them.
   - To see the depth of the matched mass in X4, look at `L_reached_rel` and `rejected_query` per direction in `volumes.csv`.
5. **Rough cost per seed** (estimates, not measured on the data):
   - A field (3000 steps, batch 2048) takes about 1 min on a GPU and 4–5 min on CPU.
   - G1 trains one field per interior section, 15 for the four Dev embryos: about 15–20 min per setting and seed on a GPU. The default grid (2 × 3) on one seed takes about 2 h, plus the skip couplings once per embryo (moscot + OT-ICP, a few minutes).
   - G2: 4 fields and 4 solves of 3000 × 3000 per embryo and variant, about 10–20 min per embryo.
   - V_fgw on a pair: 2 fields, then one 6000 × 6000 FGW solve, a few minutes on a GPU (A's 5000 × 5000 solves take about 2 min each), then scoring.
   - GPU memory for that solve: a few 6000² float32 matrices, < 2 GB. Host memory while scoring: one query section × N in float64, about 250 MB, with a peak of about 1.5 GB.
   - N = 10000 roughly triples the solve time and memory.

## Comparability across steps

- **Hashes.** Every setting that changes a result enters a hash (see `mgw/bench/config.py`), and results live in `results/<pipeline>-<hash>/<pair>/seed<s>/`. Runs with other settings never overwrite each other, and an unchanged task is skipped.
- **Shared schema.** `scores.csv` has one row per (direction, query, target, role), with the same columns for every pipeline. New pipelines add rows, never columns that old reports need.
- **Recomputable metrics.** Every candidate keeps its exact organ × organ matrix C and the top-64 entries per row and column. The finals (selected, oracle) keep the dense P. Most new label metrics can be computed from what is stored.
- **Moving reference.** [`configs/reference.yaml`](configs/reference.yaml) names the current reference and keeps the history. `report --reference history:<i>` recomputes Δκ against any earlier one.
- **Frozen series.** [`configs/series.yaml`](configs/series.yaml) fixes the sections of every embryo, and each step uses the same pairs.

## Adding pipelines (steps 4–6)

A strategy is a function `run(task)` in `mgw/bench/pipelines/`, registered in `pipelines/__init__.py:STRATEGY` and allowed in `config.STRATEGIES` with its default parameters. It forms targets, solves, and calls `task.score(...)` for every coupling with a role (`candidate`, `selected`, `null:*`). Scoring, storage, the report and the viewer then work unchanged. Add the pipeline to a config file (e.g. `configs/step3.yaml`) and run the same commands.

The volume strategy (`pipelines/volume.py`) works in three stages:
1. It builds both stacks with `pipelines/stacked.get_stack` (stage 1), then fits and samples the fields (`field.get_field` / `get_volume`, cached), or takes the stack points (`representation: points`).
2. It runs one 3D ↔ 3D solve per pair (`volume.stage2`, stage 2).
3. It scores every real query section against the target volume with `task.score`:
   - `P` = A_q P_vol, the query section's spots × the target's samples. A_q spreads each spot over its k nearest own samples.
   - `xt` = the samples' 3D coordinates.
   - `T_full` / `Yt` = the target bins / held-out genes pooled from the k nearest real target spots.

Rows have role `selected`, so they compare directly with A's. Only the top-k is stored (`dense: []`).

Step 4 adds aligners through the pipeline's `aligner` key: `volume.stage2` dispatches on it, and `config.ALIGNERS` lists the allowed names. For example, a Riemannian FGW would replace the Euclidean `dist_norm` of the samples with shortest paths. MGW would use the field itself (`Field.net.feat`) as φ.

## The synthetic pair

`mgw/bench/synthetic.py` writes, under any root, what `prepare` writes on the data server:
- a 3D body with organs and smooth gene fields, some of which vary only in depth;
- two homologous embryos cut into 6 and 4 rotated parallel sections, with partial depth overlap;
- their section and feature caches, and a series `synthetic.yaml` with the cross pair `SX`.

`write_stacks` builds their stacks with the real stage 1 but the true gaps. Then every command runs on it unchanged, e.g. `--root <dir> --set series=<dir>/synthetic.yaml --set features.pca_comp=8`. The tests (`tests/test_volume.py`) use it for end-to-end runs of V_fgw, PC_fgw, V_linear, the report, G1 and the G2 CLI. Its numbers say nothing about the method.

## Notes

- **Seeds.** Seed s reduces with Voronoi seed 42 + s and seeds the randomized PCAs with s, so seed 0 reproduces the 3D notebook's preprocessing.
- **Cross-cost scale.** One per pair (the 0.99 quantile of all a × b feature distances), so a balanced FGW solve of (x, y) is the transpose of (y, x), and each pair of sections is solved once (`tests/test_solvers.py`). The notebook scales per query instead.
- **No mirror solve in A.** FGW on Euclidean distances is invariant to reflections (`tests/test_solvers.py`); `orientation` reports the sign.
- **Fold-overs.** The share of query points whose local Jacobian, fitted over 16 neighbours, flips against the majority orientation (2D targets), or whose image normal points against the mean normal (3D targets).
- **K.** `metrics.presence: embryo` (default) scores the query organs present anywhere in the target embryo, so a target section that lacks an organ scores it 0 instead of dropping it. `presence: target` gives the first harness's K (each target alone). The setting enters the pipeline hash.
- **Feature z-scoring.** `features.zscore_by: section` (default) z-scores genes within each section; `embryo` within each embryo, which keeps the depth offsets that a 3D field needs. G1 decides between them on a label-free score, and A then runs on the chosen features (doc §3, §6 step 2).
- **Unbalanced solves.** OTT's unbalanced GW with `gw_unbalanced_correction` returns NaN at ε = 5e-3 and τ ≤ 0.9 in float32. `solvers.fgw` / `gw` then retry once without the correction and record `ugw_fallback` in the row.
- **Tests.** `pytest` runs the following:
  - the metric, solver and config tests;
  - the field and measure tests, on a stack built from synthetic ground truth;
  - the end-to-end tests of steps 2–3 on the synthetic pair, about 40 s on CPU.

  On a machine with the four smoke sections, it also runs end-to-end checks of the cache and the compact storage.
