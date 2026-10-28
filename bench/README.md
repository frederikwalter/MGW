# Cross-species alignment benchmark (`mgw.bench`)

The benchmark answers two questions about aligning human (HESTA) and mouse (MOSTA) embryos:

1. **2D ↔ 2D baseline.** Which tool aligns real sections best?
2. **3D ↔ 3D.** Which tool is best for each of the two couplings, and how much better is 3D than the best 2D tool?
   - **Within embryo:** coupling neighbouring sections into a stack.
   - **Between embryos:** volume ↔ volume.

**One metric: organ-balanced Cohen's κ (κ_bal).** C is the soft organ × organ matrix of a coupling: the mass sent from each query organ to each target organ, at full resolution. κ_bal is Cohen's κ of C after every query organ is given the same mass. 0 is chance, and 1 means every organ's tissue lands on the same organ in the other embryo. Every score is the mean of both directions (`mgw/bench/metrics.py`).

**One figure:** for each pair and direction, every query section's annotated organs next to the organs transferred along the best 2D and the best 3D alignment.

## Pipelines

All live in [`configs/bench.yaml`](configs/bench.yaml); the pairs are in [`configs/series.yaml`](configs/series.yaml) (Dev: X1, X4; Screen: X1–X5; Full: + X6).

Names follow `sides_space_features_marginals` (the full scheme is in [`docs/cross_species_alignment_results.md`](../docs/cross_species_alignment_results.md#run-names)):
- **sides:** `sec2sec` (2D section → every 2D target section), `sec2nbr` (2D section → its neighbour in the same embryo, stage 1), `pts2pts` (3D stack points ↔ 3D stack points), `vol2vol` (3D field volume ↔ 3D field volume).
- **space** (the GW term): `s-euc` (Euclidean distances), `s-knn` (shortest paths in a kNN graph), `s-pb` (geodesics under a pull-back metric, MGW), `s-none`.
- **features** (the cross cost): `f-euc` (Euclidean), `f-cos` (cosine), `f-lab` (organ labels, a ceiling), `f-none`.
- **marginals:** `bal` (balanced), `ub` (both sides relaxed, τ = `tau`), `semi0.7` (query balanced, target τ_b = 0.7), `nn` (soft nearest neighbour). For `sec2nbr` this slot is the fit: `icp` or `procrustes`.

| Question | Pipeline | Before 2026-10-28 | What it does |
| --- | --- | --- | --- |
| 1 | `sec2sec_s-euc_f-euc_bal` | `A_fgw` | Every query section × every real target section, FGW on Euclidean distances (OTT-JAX). The target with the lowest objective is selected |
| 1 | `sec2sec_s-pb_f-euc_bal` | `A_mgw` | The same, but the intra-section cost is MGW's geodesics under the pull-back metric of φ (φ trained once per section) |
| 2a | `sec2nbr_s-euc_f-euc_icp` | `S_moscot_icp` | Neighbours coupled by moscot FGW, refined by OT-ICP (the current stage 1) |
| 2a | `sec2nbr_s-euc_f-euc_procrustes` | `S_moscot_procrustes` | The same moscot coupling, one Procrustes fit and no OT-ICP. Shows what OT-ICP adds |
| 2a | `sec2nbr_s-pb_f-euc_icp` | `S_mgw_icp` | Neighbours coupled by FGW on MGW geodesics, refined by OT-ICP |
| 2b | `vol2vol_s-euc_f-euc_ub` | `V_fgw` | A neural field per stack, sampled volumes, unbalanced FGW on Euclidean distances |
| 2b | `pts2pts_s-euc_f-euc_ub` | `PC_fgw` | The registered stack points (no field), slab masses, unbalanced FGW |
| 2b | `vol2vol_s-knn_f-euc_ub` | `V_rfgw` | Field volumes, FGW on shortest paths in a kNN graph (SciPy Dijkstra; aligner `rfgw` in the config) |
| 2b | `vol2vol_s-pb_f-euc_ub` | `V_mgw` | Field volumes, FGW on geodesics under the pull-back metric of the field's features |

How each question is scored:
- **2D pipelines:** κ_bal of the selected coupling of every query section.
- **`sec2nbr_*` pipelines:** κ_bal of the kept coupling of every pair of neighbouring sections, both ways. The score is per embryo.
- **3D pipelines:** stack both embryos with `stack.coupling`, then score κ_bal on the real query spots of every section, so the rows compare one-to-one with the 2D rows.

## Runbook

```bash
python -m mgw.bench [--set key=value ...] [--root DIR] <command>     # config: bench/configs/bench.yaml

python -m mgw.bench check-data --tier dev            # missing raw sections, with curl commands
python -m mgw.bench prepare --tier dev --seeds 0-2   # data server: sections, features (CPU, RAM)
python -m mgw.bench run --tier dev --pipelines sec2sec_s-euc_f-euc_bal,sec2sec_s-pb_f-euc_bal          # question 1
python -m mgw.bench run --tier dev --pipelines \
    sec2nbr_s-euc_f-euc_icp,sec2nbr_s-euc_f-euc_procrustes,sec2nbr_s-pb_f-euc_icp                      # question 2a
#   set stack.coupling in bench.yaml to the coupling of the best sec2nbr_* tool (or pass --set stack.coupling=...)
python -m mgw.bench run --tier dev --pipelines \
    vol2vol_s-euc_f-euc_ub,pts2pts_s-euc_f-euc_ub,vol2vol_s-knn_f-euc_ub,vol2vol_s-pb_f-euc_ub         # question 2b
python -m mgw.bench report --tier dev
```

- `run` takes `--pairs X4` and `--seeds 0` to narrow the run down. Finished tasks are skipped, and `--force` reruns them.
- For job arrays, `plan --tier screen` writes one task per line, and `run --tasks FILE --task $SLURM_ARRAY_TASK_ID` runs one of them.
- `run` never reads the h5ad files, so only `cache/` has to be copied to a GPU machine.
- MGW and the fields train on CUDA when torch sees a GPU.

## Field size test

`bench/scripts/field_scale.py` trains the fields of one pair at larger sizes (`L`: 512×6 residual, 256 Fourier
frequencies, 100k steps; `XL`: 1024×8 residual with LayerNorm, 512 frequencies, 200k steps) and scores them: fit R²
at the spots, and κ_bal of stage 2 with the field's features on the stack points and on the sampled volume.

```bash
python bench/scripts/field_scale.py train --size L                      # both embryos of X4 (sim14: hours)
python bench/scripts/field_scale.py score --sizes current,L,XL          # reports/field_scale/X4_seed0.csv
```

XL runs on LRZ as a job array (`bench/scripts/lrz_field_scale.sbatch`; its header lists the caches to copy).
`field.tf32=true` speeds up A100 / H100 and enters no hash.

## Output

The root (`--root`, `$MGW_BENCH_DIR`, or `<MGW data dir>/bench`) holds `cache/`, `results/` and `reports/`.

- `results/<pipeline>-<hash>/<pair>/seed<s>/`:
  - `scores.csv`: one row per (direction, query, target, role), with `kappa_bal`, `K`, `objective` and `seconds`.
  - `transfer.npz`: the transferred organ of every query spot of the selected couplings.
  - `manifest.json`: the config, the environment and the timings.
- `reports/<tier>/`:
  - `index.html` and `table.md`: the three κ_bal tables (2D tools; within-embryo tools; 3D tools with Δ against the best 2D tool) and the figures.
  - `2d.csv`, `within.csv`, `3d.csv`: the tables.
  - `png/transfer_<pair>_<direction>_seed<s>.png`: each query section annotated | best 2D | best 3D.
- Values are means over seeds, ± the SD where there are several. ★ marks the best tool: the highest median over the pairs, or the highest mean over the embryos for question 2a.

## Caches and hashes

- Every setting that changes a result enters a hash (`mgw/bench/config.py`). Runs with other settings never overwrite each other, and an unchanged task is skipped.
- The caches:
  - sections: `cache/sections/`, keyed by `reduce`;
  - features: `cache/features/`, keyed by `reduce` and `features`;
  - stacks: `cache/stacks/<embryo>/`, keyed by the above plus `stack`;
  - fields and volumes: `cache/fields/`, `cache/volumes/`, keyed by the stack plus `field`.
- `stack.coupling` enters the stack key only when it is not `moscot_icp`, so the first stacks keep their names. Each `sec2nbr_*` tool builds and caches its own stacks. The 3D pipelines use the stacks of `stack.coupling`, and so do their fields.
- The `mgw` block enters only the MGW pipelines (`s-pb`), and `rfgw` only `vol2vol_s-knn_f-euc_ub`. `field.device` enters no hash.

## Notes

- **Seeds.** Seed s reduces with Voronoi seed 42 + s and seeds the PCAs, φ and the fields with s.
- **Cross-cost scale.** There is one per pair: the 0.99 quantile of all a × b feature distances. A balanced FGW solve of (x, y) is then the transpose of (y, x), so each pair of sections is solved once.
- **No mirror solve in 2D.** FGW on intra-section distances is invariant to reflections.
- **K.** `metrics.presence: embryo` scores the query organs present anywhere in the target embryo. A target section that lacks an organ therefore scores 0 on it instead of dropping it. The `sec2nbr_*` tools use the organs of both neighbouring sections.
- **Unbalanced solves.** OTT's unbalanced GW with `gw_unbalanced_correction` can return NaN at ε = 5e-3 and τ ≤ 0.9 in float32. The solve then retries once without the correction and logs `ugw fallback`.
- **Cost.**
  - `sec2sec_s-pb_f-euc_bal` trains one φ per section (20k steps), then runs the same FGW grid as `sec2sec_s-euc_f-euc_bal`.
  - `vol2vol_s-knn_f-euc_ub` and `vol2vol_s-pb_f-euc_ub` run an all-pairs Dijkstra on the 6000 samples per embryo (CPU), before the same FGW solve as `vol2vol_s-euc_f-euc_ub`.
- **Tests.** `pytest` runs the unit tests of κ_bal, the config hashes and the solver. The pipelines are checked by real runs on the LNT servers.
