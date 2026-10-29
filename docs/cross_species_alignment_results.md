# Cross-species alignment: features, space and the target

Oct 27, 2026 · updated Oct 29 (all six pairs, three seeds; depth, the hard pairs and where space helps; 3D FGW at higher resolution) · @Frederik

*Pair ids follow the stage-ordered numbering of 2026-10-29 ([`series.yaml`](../bench/configs/series.yaml)): X4 → X5, X5 → X9, X6 → X8. The notebooks' stored outputs still show the old ids.*

**What aligns the samples best so far:** each section matched on its features against all spots of the other embryo, with cosine distance and a relaxed target marginal (`sec2emb_s-none_f-cos_semi0.7`). On all six cross pairs and three seeds, it reaches κ\_bal 0.468, against 0.402 for 3D FGW on the registered stacks (`pts2pts_s-euc_f-euc_ub`). That is 40–63 % of what perfect labels allow. 3D FGW only draws level when it uses every spot and cosine features: 0.467 against 0.469 on seed 0 (section 8).

**What the results say about the problem:**
- **The cross-species signal is in the features.** Space alone is at chance in 2D. In 3D, the GW term adds a small but consistent +0.013 κ\_bal, most where one embryo's volume lies inside the other's. It acts as a general regulariser there, not as a depth cue: the gain is spread over all sections and moves no mass in depth. More points don't change it: the gain stays at +0.015 from 6000 points to every spot (section 8).
- **Access to every target spot matters more than the stack:** one target section caps any cost, and the stack, slab masses and joint solve add nothing measurable. Resolution helps 3D FGW three times as much as its GW term does (section 8).
- **The early pairs are hardest** (X1, X2: about 40 % of the ceiling). Their organs are there to be found, but they are less alike between the species, even in genes the embedding never sees (r 0.51–0.57 against 0.74–0.76 on X5, X8, X9). Within each pair, the least conserved organs fail most.
- **κ\_bal can't see depth, and nothing places sections in depth:** X8, which has no depth overlap, scores highest of all pairs. No run sends a section's mass closer in laterality L than chance, not even the label ceiling. L itself is only a weak reference between the species.

## Questions and setup

1. **Which choices align human and mouse best?** Space (distances within each embryo, the GW term), the features (the shared ortholog PCA), the target (one section, the stack or the whole embryo), the marginals, the resolution.
2. **What do the results say about the data?** Where the signal is, which pairs are hard and why, and what each score measures. Also how far each alignment is from what the labels would allow.

- **2D first** (X5, CS18\_E1 ↔ E12.5\_E1, seed 0). Each section is aligned with the section of the other embryo whose laterality L is closest. This removes every stage-1 uncertainty (stacking, gaps, field).
- **Then three pairs** (X1, X5, X9, seed 0): 3D FGW on the registered stack points, taken apart one change at a time, next to feature-only matching on the whole target embryo.
- **Then all six cross pairs** (X1–X3, X5, X8, X9) **and three seeds** (0, 1, 2): the comparisons that mattered, for the seed noise and the harder pairs. X2 and X3 have the mouse inside the human, and X8 has no depth overlap.
- **Then 3D FGW at higher resolution** (X1–X3, X5, X8, X9, seed 0): 6000, 20 000 and all points per embryo, with and without the GW term, on 80–94 GB GPUs (LRZ).
- **Two scores** per query section, both directions:
  - **κ\_bal** (0 = chance, 1 = every organ to itself);
  - **gene r**: the mean Pearson r of 500 held-out ortholog genes between each query spot and what the coupling transports to it, minus the same with the target spots shuffled. No run sees these genes.

## Run names

Every run, pipeline or diagnostic, is named `sides_space_features_marginals`. Each slot answers one question:
- **sides:** what is matched to what, in which dimension, and in how many solves;
- **space:** the GW term, prefixed `s-`;
- **features:** the cross cost, prefixed `f-`;
- **marginals:** how the masses are constrained.

Every slot is always filled (`s-none` and `f-none` mark a missing term), so a name always says whether a Euclidean distance is used. A value is appended to a token only when the run differs from the config default (`s-euc0.9`: GW weight α = 0.9 instead of 0.5).

Example: `pts2pts_s-euc_f-euc_ub` matches the 3D stack points to the 3D stack points. Its GW term uses Euclidean 3D distances, its cross cost Euclidean feature distances, and both sides are relaxed. This is 3D FGW, the default pipeline of `mgw.bench`, formerly `PC_fgw`.

| slot | token | meaning | tool |
| --- | --- | --- | --- |
| **sides** | `sec2sec` | 2D → 2D. Each query section against every real target section, one solve each; the target with the lowest objective is kept | OTT-JAX (`solvers.fgw`) |
| | `sec2part` | 2D → 2D. Each query section against its laterality partner section only | OTT-JAX |
| | `sec2emb` | 2D → whole embryo. Each query section against every spot of all target sections at once (no stack, no target coordinates), **one solve per query section**, uniform masses | OTT-JAX Sinkhorn |
| | `pts2pts` | 3D ↔ 3D. The registered stack points of both embryos (stage 1): 6000 per embryo, slab masses, **one joint solve** for all sections, mapped back to the real spots by kNN. `pts2pts20k`: 20 000 points; `pts2ptsall`: every spot (up to 50 000) | stage 1 + OTT-JAX |
| | `vol2vol` | 3D ↔ 3D. The same, on volumes sampled from a neural field fitted to each stack | field (PyTorch) + OTT-JAX |
| | `sec2nbr` | Stage 1, 2D → 2D within one embryo. Each section against its neighbour, to build the stack | moscot or MGW, then `stack3d` |
| **space** (GW term, weight α = 0.5; 0.8 in stage 1) | `s-none` | no GW term: pure Wasserstein on the cross cost | |
| | `s-euc` | Euclidean distances of the coordinates (2D or 3D) | OTT-JAX GW; moscot in `sec2nbr` |
| | `s-knn` | shortest paths in a kNN graph of the points (formerly "Riemannian FGW", aligner `rfgw`) | SciPy Dijkstra (`solvers.knn_geodesic`) |
| | `s-pb` | geodesics under the pull-back metric of φ or of the field's features (MGW) | MGW (`solvers.pullback_geodesic`; φ per section, PyTorch) |
| **features** (cross cost) | `f-none` | no cross cost: space alone (pure GW) | |
| | `f-euc` | Euclidean distance of the shared ortholog PCA (30 PCs) | |
| | `f-cos` | cosine distance of the same features | |
| | `f-lab` | organ labels (0 same organ, 1 different): a ceiling, not a method | |
| **marginals** | `bal` | both sides balanced | |
| | `ub` | both sides relaxed, τ = 0.95 (`tau` in bench.yaml) | OTT-JAX unbalanced |
| | `semi0.7` | query side balanced, target side relaxed to τ\_b = 0.7 (always with its value) | |
| | `nn` | τ\_b → 0: each query spot spreads its mass over its nearest target spots in the cost (softmax at ε), with no constraint on the target | |
| **fit** (in place of the marginals, `sec2nbr` only) | `icp`, `procrustes` | the coupling refined by OT-ICP, or turned into one weighted Procrustes fit | `stack3d` |

The runs in this doc, with the labels they have in the notebooks' outputs (the notebooks executed before 2026-10-28 show the old labels):

| run | what it is | old label |
| --- | --- | --- |
| `sec2part_s-euc_f-euc_bal` | 2D FGW on the partner section (α = 0.5): the cost of the `mgw.bench` pipeline `sec2sec_s-euc_f-euc_bal`, on one target section | `fgw` |
| `sec2part_s-euc0.9_f-euc_bal`, `…_s-euc0.99_…` | the same with more weight on space | `fgw@0.9`, `fgw@0.99` |
| `sec2part_s-none_f-euc_bal` | features only | `w` |
| `sec2part_s-none_f-euc_semi0.7` | features only, target side relaxed | `w_semi@0.7` |
| `sec2part_s-euc_f-none_bal` | space only | `gw` |
| `sec2part_s-none_f-lab_bal`, `sec2part_s-euc_f-lab_bal` | labels as the cross cost, without and with space | `label_w`, `label_fgw@0.5` |
| `sec2part_s-none_f-lab_nn` | label ceiling on the partner section | `label_nn` |
| `pts2pts_s-euc_f-euc_ub` | 3D FGW, the default pipeline of `mgw.bench` | `PC_fgw` |
| `pts2pts_s-none_f-euc_ub` | the same solve without the GW term: same points, masses and marginals | `PC_w` |
| `pts2pts_s-none_f-euc_semiτ` | the same with the query side balanced and the target side at τ\_b | `PC_w_semi@τ_b` |
| `pts2pts20k_s-none_f-euc_ub` | the same on 20 000 points per embryo | `PC_w@20k` |
| `pts2pts{,20k,all}_s-{euc,none}_f-{euc,cos}_ub` | 3D FGW and the same solve without the GW term, Euclidean or cosine features, at 6000, 20 000 or all points per embryo (section 8) | |
| `sec2emb_s-none_f-euc_bal`, `…_semiτ`, `…_nn` | whole embryo, Euclidean features | `E_w`, `E_w_semi@τ`, `E_w_nn` |
| `sec2emb_s-none_f-cos_semiτ`, `…_nn` | whole embryo, cosine features | `cos_w_semi@τ`, `cos_w_nn` |
| `sec2emb_s-none_f-lab_nn` | label ceiling on the whole embryo | `E_label_nn`, `label_nn` (whole embryo) |

## 1. In 2D, space never helps (X5)

*Notebooks: [`section_cost_ablation.ipynb`](../demos/section_cost_ablation.ipynb), "Quick follow-ups" (features only, the α sweep, the GW energy); [`section_cost_ablation_run1.ipynb`](../demos/section_cost_ablation_run1.ipynb), "Solve and score" (space only, the mirror flips, labels with space at α = 0.99); [`label_alignment.ipynb`](../demos/label_alignment.ipynb) (labels without and with space, at α = 0.5: X5, balanced, the partner pairs).*

| run (partner sections, balanced) | κ\_bal | gene r |
| --- | --- | --- |
| `sec2part_s-none_f-euc_bal` (features only) | 0.295 | 0.361 |
| `sec2part_s-euc_f-euc_bal` (α = 0.5) | 0.293 | 0.350 |
| `sec2part_s-euc0.9_f-euc_bal` | 0.188 | 0.341 |
| `sec2part_s-euc0.99_f-euc_bal` | 0.068 | 0.118 |
| `sec2part_s-euc_f-none_bal` (space only) | −0.021 | −0.028 |

- **Space alone is at chance.** Its couplings are also mirror images of `sec2part_s-euc_f-euc_bal`'s on 6 of 11 sections, which is what a near-random coupling gives.
- **The GW term is active but does not help.** The GW energy measures how far a coupling distorts the distances within the sections (0 = none). At α = 0.5 it halves the GW energy of `sec2part_s-none_f-euc_bal` (0.035 against 0.062; a random coupling has 0.074) for almost no extra feature cost (+0.005), yet κ\_bal and gene r do not move. With more weight on space, both fall.
- **Labels with geometry are capped too.** Organ labels as the cross cost reach κ\_bal 0.50 at α = 0.5 (`sec2part_s-euc_f-lab_bal`) and 0.17 at α = 0.99.

## 2. In 2D, the partner section is the limit (X5)

*Notebooks: [`section_cost_ablation.ipynb`](../demos/section_cost_ablation.ipynb), "Quick follow-ups" and "The whole target embryo"; [`label_alignment.ipynb`](../demos/label_alignment.ipynb) (`sec2part_s-none_f-lab_bal`).*

- **The partner section caps any cost.** Even with perfect labels, `sec2part_s-none_f-lab_nn` reaches only 0.624, because organs are missing from the partner sections. Against the whole embryo, `sec2emb_s-none_f-lab_nn` reaches 0.892.
- **The balanced marginal costs about 0.05.** With the query side balanced and the target side relaxed, features only rise from 0.295 (`sec2part_s-none_f-euc_bal`) to 0.338–0.340 (`…_semi0.9`, `…_semi0.7`). With perfect labels, balance costs 0.13 (`sec2part_s-none_f-lab_bal` 0.498 against `sec2part_s-none_f-lab_nn` 0.624).
- **The features do not follow laterality.** On the whole embryo, the mass goes to sections as far in L as an even spread would. Either L is not comparable between the species or organ coverage matters more than depth, so pairing sections by laterality is doubtful. Section 7: on all pairs, L agrees only loosely with organ content across species, and even the label ceiling ignores it.

## 3. Taking the 3D FGW pipeline apart (X1, X5, X9, seed 0)

*Notebook: [`whole_embryo_tests.ipynb`](../demos/whole_embryo_tests.ipynb) as of commit `3044b4f`, sections 1 (solver tolerance), 2 and 3 (the runs and their results).*

κ\_bal, mean over the query sections of both directions. The Sinkhorn threshold is 1e-5 for all runs (see below). One seed: section 4 repeats the key rows on all pairs and three seeds.

| run | X1 | X5 | X9 | mean |
| --- | --- | --- | --- | --- |
| `pts2pts_s-euc_f-euc_ub` (3D FGW) | 0.351 | 0.440 | 0.435 | 0.409 |
| `pts2pts_s-none_f-euc_ub` (no GW term) | 0.347 | 0.432 | 0.421 | 0.400 |
| `pts2pts20k_s-none_f-euc_ub` (20 000 points) | 0.373 | 0.469 | 0.467 | 0.436 |
| `sec2emb_s-none_f-euc_bal` | 0.339 | 0.396 | 0.415 | 0.383 |
| `sec2emb_s-none_f-euc_semi0.9` | 0.366 | 0.465 | 0.471 | 0.434 |
| `sec2emb_s-none_f-euc_semi0.7` | 0.388 | 0.475 | 0.482 | 0.448 |
| `sec2emb_s-none_f-lab_nn` (ceiling) | 0.973 | 0.892 | 0.859 | 0.908 |

- **The GW term adds little:** `pts2pts_s-euc_f-euc_ub` − `pts2pts_s-none_f-euc_ub` (same points, masses and marginals; only the GW term differs) is +0.004, +0.008 and +0.014 in κ\_bal, and nothing in gene r (0.307 against 0.308). Over six pairs and three seeds the gain turns out small but consistent (section 4).
- **Resolution matters more:** 20 000 points per embryo instead of 6000 gain +0.026 to +0.047 (`pts2pts20k_s-none_f-euc_ub`), more than the GW term on every pair. Section 8 finds the same with the GW term, on all six pairs.
- **The whole embryo, per section, does as well:** `sec2emb_s-none_f-euc_semi0.9` ties `pts2pts20k_s-none_f-euc_ub` (0.434 against 0.436) and is above 3D FGW on 3 of 3 pairs (+0.015 to +0.036). So the stack, the slab masses and the joint solve add nothing measurable either; resolution and access to every target section do.
- **Relaxing the target helps, but only on the whole embryo:** from balanced (0.383) to τ\_b = 0.7 (0.448). Below 0.7 the curve is flat (0.450 at 0.5, 0.447 for `sec2emb_s-none_f-euc_nn`). On the 6000 stack points it does nothing: `pts2pts_s-none_f-euc_semi0.7` reaches 0.404 against 0.400 for `pts2pts_s-none_f-euc_ub`.
- **The solver threshold can be loosened.** At 1e-5 instead of 1e-7, every score is unchanged (largest change 0.0000), and the solves run 7–13 × faster (`pts2pts_s-euc_f-euc_ub` 95 s → 7 s). At 1e-7, it reproduces the `mgw.bench` scores exactly.

## 4. All six pairs, three seeds

*Notebook: [`whole_embryo_tests.ipynb`](../demos/whole_embryo_tests.ipynb) as of commit `e7284e3`, sections 2 (results across pairs) and 3 (seed stability).*

κ\_bal, mean over the query sections of both directions, then over seeds 0, 1 and 2:

| run | X1 | X2 | X3 | X5 | X8 | X9 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pts2pts_s-euc_f-euc_ub` (3D FGW) | 0.354 | 0.327 | 0.376 | 0.441 | 0.481 | 0.436 | 0.402 |
| `pts2pts_s-none_f-euc_ub` (no GW term) | 0.350 | 0.305 | 0.360 | 0.431 | 0.471 | 0.422 | 0.390 |
| `sec2emb_s-none_f-euc_semi0.7` | 0.387 | 0.345 | 0.398 | 0.476 | 0.531 | 0.483 | 0.437 |
| `sec2emb_s-none_f-cos_semi0.95` | 0.363 | 0.329 | 0.409 | 0.480 | 0.518 | 0.486 | 0.431 |
| `sec2emb_s-none_f-cos_semi0.9` | 0.379 | 0.343 | 0.419 | 0.502 | 0.539 | 0.509 | 0.448 |
| `sec2emb_s-none_f-cos_semi0.7` (whole-embryo match) | 0.409 | 0.364 | 0.429 | 0.524 | 0.557 | 0.523 | 0.468 |
| `sec2emb_s-none_f-cos_semi0.5` | 0.420 | 0.370 | 0.431 | 0.527 | 0.561 | 0.524 | 0.472 |
| `sec2emb_s-none_f-cos_nn` | 0.427 | 0.375 | 0.431 | 0.529 | 0.563 | 0.522 | 0.475 |
| `sec2emb_s-none_f-lab_nn` (ceiling) | 0.973 | 0.913 | 0.891 | 0.892 | 0.882 | 0.858 | 0.902 |
| whole-embryo match, share of the ceiling | 0.42 | 0.40 | 0.48 | 0.59 | 0.63 | 0.61 | |

Gene r, same averaging:

| run | X1 | X2 | X3 | X5 | X8 | X9 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pts2pts_s-euc_f-euc_ub` (3D FGW) | 0.074 | 0.086 | 0.245 | 0.346 | 0.481 | 0.500 | 0.289 |
| `sec2emb_s-none_f-cos_semi0.95` | 0.119 | 0.147 | 0.346 | 0.435 | 0.545 | 0.601 | 0.366 |
| `sec2emb_s-none_f-cos_semi0.7` (whole-embryo match) | 0.092 | 0.125 | 0.319 | 0.404 | 0.535 | 0.573 | 0.341 |
| `sec2emb_s-none_f-cos_nn` | 0.079 | 0.106 | 0.270 | 0.354 | 0.491 | 0.516 | 0.303 |
| `sec2emb_s-none_f-lab_nn` (ceiling) | 0.114 | 0.094 | 0.217 | 0.335 | 0.415 | 0.491 | 0.277 |

**Alignment:**
- **The whole-embryo match scores higher than 3D FGW in every (pair, seed):** `sec2emb_s-none_f-cos_semi0.7` − `pts2pts_s-euc_f-euc_ub` is positive in 18 of 18, at +0.055 (X1), +0.037 (X2), +0.053 (X3), +0.083 (X5), +0.087 (X9) and +0.075 (X8), mean +0.065. Taking the mean of the two direction means instead gives 0.469 against 0.404, the same picture.
- **Cosine scores higher than Euclidean in every cell:** +0.031 on average at τ\_b = 0.7 (+0.019 on X2 to +0.048 on X5), 18 of 18.
- **τ\_b at 0.7 or below.** The best κ\_bal comes from `sec2emb_s-none_f-cos_nn` in 14 of 18 cells, from τ\_b = 0.5 in 3 and from 0.7 in 1; never from above 0.7. Fixing τ\_b at 0.7 gives up at most 0.019 (X1), at most 0.007 on X3, X5, X8, X9, and keeps gene r at 0.341 against 0.303 for `nn`.

**The data:**
- **Seed noise is small.** The median SD of a score over the seeds is 0.002, the largest 0.013. The 3D runs are the noisy ones (SD up to 0.013 on X3 and 0.009 on X2), because their stack and their 6000-point sample change with the seed; the whole-embryo runs stay at or below 0.006. Paired differences are steadier still. This replaces the "0.01–0.02" guess: a paired difference of 0.01 is readable.
- **Space carries some signal in 3D, mainly when one volume sits inside the other.** `pts2pts_s-euc_f-euc_ub` − `pts2pts_s-none_f-euc_ub` is positive in 18 of 18 cells, mean +0.013: +0.003 on X1 (within noise), +0.022 on X2, +0.015 on X3, +0.010 to +0.014 on X5, X8, X9. Gene r does not move (0.289 for both). The gain is largest on X2 and X3, where the mouse lies inside the human, and there the whole-embryo match gains least over 3D FGW. The geometry may help to place the smaller volume within the larger one. Section 7 does not bear this out: the gain is spread over all sections and moves no mass in depth. Section 8: it does not grow at higher resolution either.
- **The early pairs are hardest.** The whole-embryo match reaches 0.40–0.42 of the label ceiling on X1 and X2, 0.48 on X3 and 0.59–0.63 on X5, X8, X9. Gene r agrees: 0.09–0.13 on X1 and X2, 0.40–0.57 on X5, X8, X9. The ceilings are highest on the early pairs (0.97, 0.91), so the organs are there to be found; the features do not find them. Section 7: the early organs are also less alike between the species in their expression.
- **κ\_bal cannot see depth.** X8 has no depth overlap, yet every feature run scores best on it (whole-embryo match 0.557, 3D FGW 0.481). κ\_bal counts organs, and most organs exist at every depth, so a section matched to the wrong depth can still score well. Whether a method places sections at a plausible depth needs its own score. Nothing in the whole-embryo match steers a section to the right depth, since it uses no coordinates.

## 5. The features

*Notebooks: [`section_cost_ablation.ipynb`](../demos/section_cost_ablation.ipynb), "The features: what each PC weighs in the cross cost" and "Does the cost need all the PCs?" (X5); [`whole_embryo_tests.ipynb`](../demos/whole_embryo_tests.ipynb) as of commit `3044b4f`, sections 5 (variants) and 6 (cosine with the τ\_b sweep).*

**What the 30 PCs carry (X5).** The embedding is mostly shared between the species. PCs 1–3 carry 63 % of the cross cost and order the organs the same way in both species (r 0.94, 0.97, 0.71); 85 % of the cost is on PCs with organ r > 0.5. Two smaller defects: PCs 8 and 9 order the organs differently in the two species (r −0.62, −0.30; 4 % of the cost), and the mouse spots spread about 1.4 × more on almost every PC. More PCs would not help: `sec2part_s-none_f-euc_bal` levels off at 20 PCs (κ\_bal 0.293 at 20, 0.295 at 30).

**Variants** (of `sec2emb_s-none_f-euc_semi0.9`; change in κ\_bal against the embedding as it is):

| variant | X1 | X5 | X9 | mean |
| --- | --- | --- | --- | --- |
| cosine instead of Euclidean distance (`f-cos`) | +0.014 | +0.038 | +0.036 | +0.029 |
| PCs rescaled per species | +0.003 | +0.004 | +0.007 | +0.005 |
| without the contradicting PCs (uses labels) | +0.025 | −0.001 | −0.004 | +0.007 |
| first 20 PCs whitened | −0.099 | −0.082 | −0.083 | −0.088 |

- **Cosine distance helps on all three pairs**, with gene r unchanged. It ignores the length of each spot's feature vector, which likely varies with depth or tissue rather than identity.
- **Whitening hurts** both scores: the low-variance PCs are mostly noise, so weighting the PCs by their variance is right.
- The other two are within noise.

**Cosine with the τ\_b sweep** (`sec2emb_s-none_f-euc_*` against `sec2emb_s-none_f-cos_*`):

κ\_bal (gene r in brackets), mean over the query sections:

| marginals | `f-euc` | `f-cos` | `f-cos` − 3D FGW (X1 / X5 / X9) |
| --- | --- | --- | --- |
| `semi0.95` | 0.421 (0.391) | 0.443 (0.389) | +0.013 / +0.042 / +0.049 |
| `semi0.9` | 0.434 (0.385) | 0.463 (0.384) | +0.029 / +0.063 / +0.072 |
| `semi0.7` | 0.448 (0.359) | 0.486 (0.358) | +0.061 / +0.084 / +0.087 |
| `semi0.5` | 0.450 (0.338) | 0.492 (0.341) | +0.072 / +0.088 / +0.090 |
| `nn` | 0.447 (0.309) | 0.495 (0.317) | +0.080 / +0.087 / +0.092 |
| `pts2pts_s-euc_f-euc_ub` (3D FGW) | 0.409 (0.307) | | |

- **The two gains add up.** With cosine distance, κ\_bal keeps rising as the target is relaxed, up to 0.495 for `sec2emb_s-none_f-cos_nn`, +0.08 to +0.09 over 3D FGW on every pair. Cosine adds more the more the target is relaxed (+0.022 at τ\_b 0.95, +0.048 for `nn`).
- **Gene r does not follow.** Cosine leaves gene r unchanged at every τ\_b, and gene r falls with relaxation as before (0.389 at 0.95, 0.317 for `nn`).
- **τ\_b = 0.7 is the compromise:** `sec2emb_s-none_f-cos_semi0.7` reaches κ\_bal 0.486 (2 % below the best) with gene r 0.358, above 3D FGW on both scores. Six pairs and three seeds confirm this (section 4).
- **For the organs, the transport constraint barely matters.** The best κ\_bal comes with no constraint on the target at all, i.e. matching each spot to its nearest spots in the feature space. The marginal mainly keeps the coupling spread, which helps gene r.

## 6. τ\_b cannot be chosen by gene r

*Notebook: [`whole_embryo_tests.ipynb`](../demos/whole_embryo_tests.ipynb) as of commit `3044b4f`, section 4 ("Choosing τ\_b without labels").*

Labels may score a result but not choose a setting, so τ\_b would ideally be chosen by gene r on half of the held-out genes. It does not work: gene r picks τ\_b = 0.95 (κ\_bal 0.420 against 0.445 at the best τ\_b), with a loss above 0.05 for 2 of 6 pair-directions. Gene r falls steadily as the target is relaxed and favours spread-out couplings: averaging over many target spots smooths the transported expression and raises the correlation. The label oracle `sec2emb_s-none_f-lab_nn` has a lower gene r (0.331) than the balanced `sec2emb_s-none_f-euc_bal` (0.381). Gene r stays a useful score, but it should not choose settings that change how spread the coupling is.

## 7. Depth, the hard pairs and where space helps (X1–X3, X5, X8, X9, seed 0)

*Notebook: [`depth_and_hard_pairs.ipynb`](../demos/depth_and_hard_pairs.ipynb) as of commit `90b37a9`, sections 1 (depth score), 2 (is L comparable), 3 (the hard pairs) and 4 (where space helps).*

Five runs of section 4, recomputed on seed 0 to keep their couplings: `pts2pts_s-euc_f-euc_ub`, `pts2pts_s-none_f-euc_ub`, `sec2emb_s-none_f-cos_semi0.7`, `sec2emb_s-none_f-cos_nn` and `sec2emb_s-none_f-lab_nn`. Their κ\_bal and gene r match section 4's seed-0 scores to 5e-6.

**Depth.** Per query section, the coupling's mass is followed to the target sections that receive it. Three scores come from the target sections' laterality L:
- **depth skill**, from the mass-weighted distance in L between the query section and the target sections: 0 = the mass spread like the target tissue, 1 = all of it on the target section with the closest L;
- **depth order ρ**: the Spearman correlation between the query's L and the mass-weighted target L over the query sections. It needs only the order of the sections to agree, not the scale of L;
- **end share** (X8 only): the share of the mass on the end section nearest the other embryo.

| run | depth skill | depth order ρ | X8 end share (human → mouse / mouse → human) |
| --- | --- | --- | --- |
| `pts2pts_s-euc_f-euc_ub` (3D FGW) | 0.04 | 0.19 | 0.21 / 0.16 |
| `pts2pts_s-none_f-euc_ub` (no GW term) | 0.04 | 0.13 | 0.21 / 0.17 |
| `sec2emb_s-none_f-cos_semi0.7` (whole-embryo match) | 0.06 | 0.29 | 0.22 / 0.25 |
| `sec2emb_s-none_f-cos_nn` | 0.09 | 0.28 | 0.21 / 0.24 |
| `sec2emb_s-none_f-lab_nn` (ceiling) | 0.05 | 0.38 | 0.29 / 0.36 |
| chance | 0 | 0 | 0.25 / 0.22 |

Depth skill and ρ are means over the six pairs and both directions. The ρ of one (pair, direction) rests on 2–10 query sections, so it is noisy.

- **No run places sections in depth, not even the label ceiling.** The mean depth skill is 0.04–0.09 for every run, and the mass goes where an even spread over the target would put it. A single (pair, direction) reaches at most 0.34 (X2, mouse queries, `sec2emb_s-none_f-cos_nn`). The GW term barely changes either score (depth skill 0.04 for both 3D runs, ρ 0.19 against 0.13).
- **On X8, only the labels lean towards the near end:** 0.29 and 0.36 of the mass, against 0.25 and 0.22 by chance. The feature runs stay at chance. X8 still scores highest in κ\_bal (section 4), so on this pair κ\_bal only says that the organs match.
- **L is a weak reference across species.** Sections closer in L hold more similar organs (organ composition, cosine) on X2 (ρ −0.67) and X5 (−0.53), but barely on X1 (−0.06), X9 (−0.12) and X3 (−0.25); pooled −0.23 over 188 section pairs. L is derived from organ classes. Organ content follows it only loosely between the species, and the labels themselves place sections at chance (table above).
- **So the depth-aware score of the next steps can't use L as its reference.** On L, every method would score near chance.

**Why the early pairs are hard.** Per-organ recall is the share of a query organ's mass that lands on the same organ, pooled over the query sections of both directions. The organ conservation scores come from labels only:
- **r\_same**: the Pearson r between an organ's human and mouse mean profiles, on the 500 held-out genes, centred per gene over the shared organs;
- **own best**: the share of organs whose best-matching organ in the other species is themselves;
- **f\_same** and **f\_margin**: the same comparison in the shared embedding (cosine of the organ centroids), i.e. what the alignment sees. f\_margin is the cosine to the organ itself minus the highest cosine to any other organ.

| pair | organs | r\_same | own best | f\_same | f\_margin | recall (whole-embryo match) |
| --- | --- | --- | --- | --- | --- | --- |
| X1 | 10 | 0.51 | 0.60 | 0.42 | 0.09 | 0.43 |
| X2 | 12 | 0.57 | 0.42 | 0.57 | 0.15 | 0.39 |
| X3 | 14 | 0.64 | 0.71 | 0.58 | 0.10 | 0.40 |
| X5 | 13 | 0.76 | 1.00 | 0.69 | 0.14 | 0.53 |
| X8 | 12 | 0.74 | 0.67 | 0.69 | 0.13 | 0.57 |
| X9 | 13 | 0.76 | 0.85 | 0.72 | 0.18 | 0.55 |

- **The early organs are less alike between the species.** r\_same is 0.51–0.57 on X1 and X2, against 0.74–0.76 on X5, X8, X9. This holds in the expression of genes the embedding never sees, so it is not the features alone. Section 4 found that the ceiling is high on the early pairs (the organs are there); they are just harder to recognise across the species.
- **Recall follows conservation within each pair.** The Spearman of per-organ recall with r\_same is 0.56–0.87 on all pairs but X9 (0.12), and 0.69 pooled. On X1, recall follows the feature margin most closely (ρ 0.92).
- **Which organs fail.** Heart and liver reach 0.77–0.98 on every pair, and brain 0.70–0.93 from X3 on. On X1 and X2, brain (15 % and 11 % of the query bins) drops to 0.58 and 0.42. Mesenchyme, epidermis, craniofacial and gut sit at 0.12–0.39, and X2's urogenital and ganglion at 0.08. Gut, urogenital and ganglion are low on X1 and X2 and higher on the later pairs. One odd case is not early: spinal cord reaches only 0.06 on X3 and 0.12 on X9, despite a high r\_same.
- **Gene r is low within every organ on X1 and X2** (0.02–0.08, against 0.09–0.50 on X5, X8, X9), so no single organ carries the early pairs' low gene r.
- **The stage pairing is close to right.** Mean organ r\_same over every human × mouse embryo of the pairs, on 2000 shared 1:1 orthologs:
  - 4 of 6 human embryos match their paired mouse embryo best (X3, X5, X8) or within 0.006 (X1: CS12-13 matches E10.5 0.440 and its pair E9.5 0.434).
  - CS14-15 (X2) matches E11.5 (0.49) and E12.5 (0.47) a little better than its pair E10.5 (0.46).
  - CS23 (X9) matches E13.5 (0.675) a little better than E14.5 (0.658).
  - Each cell averages over the organs its two embryos share (10–14), so the differences are small and partly a matter of organ sets. Part of X2's difficulty may be a stage mismatch; most is the lower conservation of the early organs.

**Where space helps.** κ\_bal gain of the GW term per query section (`pts2pts_s-euc_f-euc_ub` − `pts2pts_s-none_f-euc_ub`), by whether the query's L lies inside the target's L range:

| pairs | human → mouse, beyond the range | human → mouse, inside | mouse → human, beyond | mouse → human, inside |
| --- | --- | --- | --- | --- |
| X2, X3 (mouse inside human) | +0.018 (13) | +0.021 (7) | – | +0.021 (7) |
| X1, X5, X8, X9 | +0.008 (9) | +0.010 (17) | +0.008 (11) | +0.010 (7) |

Number of query sections in brackets.

- **The gain is everywhere, not at the ends of the overlap.** It is positive on 70 of 71 query sections. It is about the same inside and beyond the target range, and at the ends of a stack and inside it. It is twice as large on X2 and X3 as elsewhere, on every kind of section.
- **It does not move mass in depth.** The mass-weighted distance in L changes by at most 0.005 on every pair. The mass profiles over the target sections, with and without the GW term, nearly coincide on the slices.
- **On the slices, the gained spots are scattered.** For each spot, the slices of X2, X3, X5 compare the organ that the GW term transfers with the one transferred without it. The term gains 1.4–5.9 % of the scored spots and loses 0.6–5.4 %, spread over the section rather than in whole organs. Section 4 suggested that the geometry places the smaller volume within the larger one. That is not borne out: the GW term acts as a general regulariser of the coupling, stronger when one volume lies inside the other, not as a depth cue.

## 8. 3D FGW at higher resolution (X1–X3, X5, X8, X9, seed 0)

*Script: [`bench/scripts/fgw_scale.py`](../bench/scripts/fgw_scale.py) as of commit `7fdbbda`, run on the LRZ AI Systems (one H100 or A100 per pair). The tables are also in `<root>/diag/fgw_scale/report/table.md`.*

3D FGW and the same solve without the GW term, at 6000 points per embryo (the `mgw.bench` setting), 20 000 and every spot, with Euclidean and cosine features. Points, slab masses, τ and ε are those of `pts2pts_s-euc_f-euc_ub`. The cosine runs scale their cost by the 0.99 quantile of the cosine distances. Every spot means up to 50 000 × 25 000 points (X2, X3). X8's embryos have fewer than 20 000 spots each, so its `pts2ptsall` runs are its `pts2pts20k` runs. Scored as in section 4.

κ\_bal, mean over the query sections of both directions:

| run | X1 | X2 | X3 | X5 | X8 | X9 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pts2pts_s-euc_f-euc_ub` (3D FGW) | 0.351 | 0.335 | 0.389 | 0.440 | 0.481 | 0.435 | 0.405 |
| `pts2pts_s-none_f-euc_ub` | 0.346 | 0.313 | 0.371 | 0.432 | 0.471 | 0.421 | 0.392 |
| `pts2pts_s-euc_f-cos_ub` | 0.345 | 0.343 | 0.400 | 0.466 | 0.498 | 0.458 | 0.419 |
| `pts2pts_s-none_f-cos_ub` | 0.342 | 0.320 | 0.376 | 0.456 | 0.484 | 0.438 | 0.403 |
| `pts2pts20k_s-euc_f-euc_ub` | 0.377 | 0.361 | 0.421 | 0.480 | 0.522 | 0.484 | 0.441 |
| `pts2pts20k_s-none_f-euc_ub` | 0.373 | 0.335 | 0.401 | 0.469 | 0.512 | 0.467 | 0.426 |
| `pts2pts20k_s-euc_f-cos_ub` | 0.374 | 0.366 | 0.434 | 0.509 | 0.538 | 0.503 | 0.454 |
| `pts2pts20k_s-none_f-cos_ub` | 0.370 | 0.339 | 0.408 | 0.495 | 0.526 | 0.480 | 0.436 |
| `pts2ptsall_s-euc_f-euc_ub` | 0.383 | 0.374 | 0.441 | 0.492 | 0.522 | 0.504 | 0.452 |
| `pts2ptsall_s-none_f-euc_ub` | 0.378 | 0.347 | 0.421 | 0.482 | 0.512 | 0.486 | 0.438 |
| `pts2ptsall_s-euc_f-cos_ub` | 0.379 | 0.382 | 0.454 | 0.523 | 0.538 | 0.525 | 0.467 |
| `pts2ptsall_s-none_f-cos_ub` | 0.375 | 0.351 | 0.427 | 0.509 | 0.526 | 0.502 | 0.448 |
| `sec2emb_s-none_f-cos_semi0.7` (whole-embryo match, seed 0) | 0.412 | 0.366 | 0.428 | 0.524 | 0.559 | 0.522 | 0.469 |

The whole-embryo row is 3D FGW at 6000 points plus the seed-0 difference that section 4's notebook prints. Its seed-0 scores themselves are not stored with the doc. The 6000-point runs here reproduce those of section 4: within 0.001 in κ\_bal on X1, X5 and X9, and within 0.001 in the GW gain on all six pairs.

The GW term, κ\_bal of `s-euc` − `s-none` (same points, masses and marginals):

| points per embryo | features | X1 | X2 | X3 | X5 | X8 | X9 | mean | positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6000 | `f-euc` | +0.004 | +0.022 | +0.018 | +0.008 | +0.011 | +0.014 | +0.013 | 6 of 6 |
| 6000 | `f-cos` | +0.003 | +0.023 | +0.025 | +0.011 | +0.014 | +0.020 | +0.016 | 6 of 6 |
| 20 000 | `f-euc` | +0.004 | +0.026 | +0.020 | +0.011 | +0.010 | +0.017 | +0.015 | 6 of 6 |
| 20 000 | `f-cos` | +0.004 | +0.028 | +0.026 | +0.014 | +0.011 | +0.023 | +0.018 | 6 of 6 |
| all | `f-euc` | +0.004 | +0.026 | +0.019 | +0.011 | +0.010 | +0.018 | +0.015 | 6 of 6 |
| all | `f-cos` | +0.004 | +0.031 | +0.026 | +0.014 | +0.011 | +0.023 | +0.018 | 6 of 6 |

- **The GW term doesn't gain from resolution.** From 6000 points to every spot, its gain goes from +0.013 to +0.015 with Euclidean features, and from +0.016 to +0.018 with cosine. It is positive on every pair at every resolution, with section 4's pattern: largest on X2 and X3 (+0.018 to +0.031), smallest on X1 (+0.003 to +0.004). It lowers gene r by at most 0.004.
- **Resolution helps with and without the GW term, by the same amount.** From 6000 points to every spot, κ\_bal rises by +0.047 for 3D FGW and +0.046 without the GW term, three times the GW term's gain. The rise is +0.032 on X1 and up to +0.069 on X9; three quarters of it comes by 20 000 points. Gene r rises from 0.289 to 0.343.
- **Cosine helps in 3D too, but half as much as on the whole embryo.** For 3D FGW it adds +0.014 at 6000 points and +0.015 at every spot, against +0.031 for the whole-embryo match (section 4). It is slightly negative on X1 (−0.003 to −0.006) and largest on X5 (+0.026 to +0.031). It lowers gene r by about 0.01. Cosine also changes the transported mass. The unbalanced solve rejects 35–68 % of each side's mass with cosine, against 64–80 % with Euclidean features, so part of the difference may come from the marginals rather than the distance (see Caveats).
- **On every spot with cosine features, 3D FGW is level with the whole-embryo match.** `pts2ptsall_s-euc_f-cos_ub` reaches 0.467, against 0.469 for the whole-embryo match on the same seed.
  - By pair, it is ahead on X2 and X3 (+0.016, +0.026), where the mouse lies inside the human, level on X5 and X9, and behind on X1 and X8 (−0.033, −0.021).
  - Without the GW term it reaches 0.448, so the GW term is what brings it level.
  - Its gene r is 0.331, against 0.341 for the whole-embryo match (section 4, three seeds).
  - The whole-embryo match needs no stack and fits on a smaller GPU. The 3D run needs stage 1 and an 80 GB GPU.
- **Cost.** The largest FGW solve took 21 s at 6000 points, 3 min at 20 000 and 8 min at every spot (X5 on an A100). A whole pair took 29–72 min with at most 72 GB of host RAM, and every solve converged. The peak GPU memory was not recorded.

## Caveats

- **Every alignment here is scored on organs, not on depth.** A section matched to the wrong depth can still score well (X8 scores highest of all pairs). Section 7's depth scores use laterality L, which is derived from organ classes and agrees only loosely with organ content across species. Near-chance depth scores may reflect L as much as the methods.
- **Section 7 rests on seed 0.** Its κ\_bal gains of the GW term (+0.008 to +0.021 per group) are above the paired seed noise of section 4. The organ conservation scores come from the 500 held-out genes of each pair's seed-0 features.
- **Three seeds.** The seed changes the tiles, the PCAs, the stack and the point sample, not the embryos; an SD from three values is a rough margin.
- **Sections 3 and 5 rest on one seed and three pairs.** With a seed SD of about 0.002 for the whole-embryo runs, their paired differences of 0.01 or more are likely real, but the feature variants were not repeated.
- **The 2D results rest on one pair** and few distinct section pairs: five of six human sections pair with the same mouse section.
- **Section 8 rests on seed 0 and ran on the LRZ AI Systems** (X1–X3 and X8 on H100s, X5 and X9 on A100s). Its caches were rebuilt there from the raw data. Its 6000-point runs reproduce sim13's to within 0.001, so the caches agree. Its whole-embryo row is inferred from section 4's seed-0 differences, not recomputed.
- **Euclidean and cosine 3D runs transport different amounts of mass** (section 8: 64–80 % against 35–68 % rejected at τ = 0.95), for the reason given in the next caveat. Their κ\_bal difference mixes the distance and the marginals.
- **The unbalanced mass above 1** seen in the first 2D ablation is not a bug: with a KL penalty on the marginals, the transported mass scales with the cost, so the same τ means different things for different costs. The `semi` runs avoid this.

## Next steps

**Understanding the data and the problem**

*Notebook: [`depth_and_hard_pairs.ipynb`](../demos/depth_and_hard_pairs.ipynb) (commit `90b37a9`, executed on tueilnt-sim15) answers the first three items (section 7).*

- [x] **Depth.** Per query section, compute the mass-weighted laterality L of the target sections that receive its mass, for the whole-embryo match and 3D FGW, on all pairs. Does any method place sections at a plausible depth (on X8, human sections at the lateral end of the mouse embryo)? Is L comparable between the species at all? Section 2 found that the features ignore L. *Done (section 7): no run, not even the label ceiling, places sections in L better than chance; L agrees only loosely with organ content across species.*
- [x] **Why the early pairs are hard.** Compare per-organ recall and gene r on X1 and X2 against X5, X8, X9. Which organs fail, and are they the ones whose ortholog expression differs most? Does the stage match explain part of it (X3 is in the ambiguous window)? *Done (section 7): the early organs are less conserved, and recall follows conservation; brain, mesenchyme, epidermis, craniofacial and gut fail most. The stage pairing is close to right; X2 may fit E11.5 slightly better.*
- [x] **Where space carries information.** The GW gain is largest on X2 and X3, where the mouse lies inside the human. Check per section whether the gain sits at the ends of the overlap, i.e. whether the geometry resolves which part of the larger volume is meant. *Done (section 7): the gain is spread over all sections, inside and beyond the overlap, and moves no mass in depth.*
- [ ] **A depth reference.** κ\_bal is blind to depth, and L is too weak a reference for a depth score: even the label ceiling scores at chance on it (section 7). A depth score needs a reference that organ content follows across species. Until there is one, compare methods on κ\_bal and gene r, and report the depth placement only as a diagnostic.
- [ ] **The stage pairing of X2.** CS14-15 matches E11.5 slightly better than its pair E10.5 (section 7). Run the whole-embryo match on CS14-15 ↔ E11.5 and check whether κ\_bal and gene r rise.

**Aligning the samples** (on X1 and X5, plus X2 and X3)
- [ ] **Add the whole-embryo match to `mgw.bench`** as a pipeline (cosine, all spots, `semi0.7`), so it is scored the same way as the other methods, including the supporting metrics (per-organ recall, coverage).
- [ ] **Features.** They limit every method, most of all on the early pairs (X1, X2: 0.40–0.42 of the ceiling, gene r about 0.1). Try integration across species (Harmony, scVI) and other gene sets. Part of the gap is biological (section 7: the early organs are less conserved in expression). On X1, recall follows the organs' margin in the feature space (ρ 0.92), so a better embedding should still help there.
- [ ] **Space as an addition, where it helps.** Two tests, scored on κ\_bal and gene r, with the depth placement as a diagnostic (no depth reference yet, see above):
  - the whole-embryo match with a GW term: the query section in 2D against the target's spots in stack coordinates (`sec2emb_s-euc_f-cos_semi0.7`). GW compares distances only within each side, so 2D against 3D works;
  - 3D FGW with cosine features (`pts2pts_s-euc_f-cos_ub` against `pts2pts_s-none_f-cos_ub`). *Done for κ\_bal and gene r (section 8): the GW term adds +0.016 to +0.018, a little more than with Euclidean features, at every resolution. Its depth placement was not scored.*
  Together they show whether geometry adds anything once the features are as good as they currently get, and whether it places sections at the right depth.
- [ ] **3D FGW on every spot as a second reference.** With cosine features it is level with the whole-embryo match and ahead on X2 and X3 (section 8). To keep it in `mgw.bench`, `pts2pts` needs `f-cos` and an option for every spot. It runs only on an 80 GB GPU (LRZ, `bench/scripts/fgw_scale.py`).
- [ ] **Set `fgw.inner_tol` to 1e-5 in `bench.yaml`** (still 1e-7): the same scores, 7–13 × faster.
- [ ] **Keep the field paused** for the cross-species step.

Notebooks, executed on tueilnt-sim13 unless noted: `demos/section_cost_ablation.ipynb` (2D, X5: the quick follow-ups and the features) and `demos/section_cost_ablation_run1.ipynb` (2D, X5: the full ablation loop, which the current `section_cost_ablation.ipynb` holds unexecuted); `demos/whole_embryo_tests.ipynb`. That notebook changed between rounds. The first round (X1, X5, X9, seed 0; sections 3, 5 and 6) is in commit `3044b4f` (`git show 3044b4f:demos/whole_embryo_tests.ipynb`). The six pairs and three seeds (section 4) are in commit `e7284e3`, on branch `sim/whole-embryo-all-pairs-seeds-20261027`; that version drops the tests the first round answered. `demos/depth_and_hard_pairs.ipynb` (section 7) is in commit `90b37a9`, executed on tueilnt-sim15. Its results are cached in `<root>/diag/depth_organs/`. Section 8 comes from `bench/scripts/fgw_scale.py` (commit `7fdbbda`), run on the LRZ AI Systems (array job 5884771). Its results are in `<root>/diag/fgw_scale/` and on LRZ in `$DSS/ai-runs/MGW/fgw-scale/`. All of these are merged into `main`. `demos/label_alignment.ipynb` (2D, every mouse section against every human section of one pair, organ labels as the cost without and with space, any marginals) now holds the label runs of `section_cost_ablation.ipynb`. It has not been executed yet: the label values in sections 1–2 come from the ablation runs, and its check against the doc must reproduce them on X5. Its scores are saved to `<root>/diag/label_alignment/`.
