# Cross-species alignment: features, space and the target

Oct 27, 2026 · @Frederik

The features carry the cross-species signal; space does not. In 2D, matching on space alone is at chance, and giving space any weight next to the features never helps. In 3D, the GW term adds at most 0.014 κ\_bal. What helps is the target: matching each section, on features alone, against all spots of the other embryo beats the 3D benchmark pipeline `pts2pts_s-euc_f-euc_ub` on all three pairs tested. With cosine distance and a relaxed target marginal (`sec2emb_s-none_f-cos_semi0.7`), it reaches κ\_bal 0.486 against 0.409 (+0.06 to +0.09 on every pair), about half of what perfect labels allow.

## Question and setup

Which part of the stage-2 cost helps a human ↔ mouse alignment: **space** (distances within each embryo, the GW term), the **features** (the shared ortholog PCA), or the choice of **target**? And how far is each from what the labels would allow?

- **2D first** (X4, CS18\_E1 ↔ E12.5\_E1, seed 0). Each section is aligned with the section of the other embryo whose laterality L is closest. This removes every stage-1 uncertainty (stacking, gaps, field).
- **Then three pairs** (X1, X4, X5, seed 0): the 3D pipeline on the registered stack points against feature-only matching on the whole target embryo.
- **Two scores** per query section, both directions:
  - **κ\_bal**, the benchmark metric (0 = chance, 1 = every organ to itself).
  - **gene r**: the mean Pearson r of 500 held-out ortholog genes between each query spot and what the coupling transports to it, minus the same with the target spots shuffled. No run sees these genes.

## Run names

Every run, benchmark pipeline or diagnostic, is named `sides_space_features_marginals`. Each slot answers one question:
- **sides:** what is matched to what, in which dimension, and in how many solves;
- **space:** the GW term, prefixed `s-`;
- **features:** the cross cost, prefixed `f-`;
- **marginals:** how the masses are constrained.

Every slot is always filled (`s-none` and `f-none` mark a missing term), so a name always says whether a Euclidean distance is used. A value is appended to a token only when the run differs from the config default (`s-euc0.9`: GW weight α = 0.9 instead of 0.5).

Example: `pts2pts_s-euc_f-euc_ub` matches the 3D stack points to the 3D stack points. Its GW term uses Euclidean 3D distances, its cross cost Euclidean feature distances, and both sides are relaxed. This is the benchmark pipeline that was called `PC_fgw`.

| slot | token | meaning | tool |
| --- | --- | --- | --- |
| **sides** | `sec2sec` | 2D → 2D. Each query section against every real target section, one solve each; the target with the lowest objective is kept | OTT-JAX (`solvers.fgw`) |
| | `sec2part` | 2D → 2D. Each query section against its laterality partner section only | OTT-JAX |
| | `sec2emb` | 2D → whole embryo. Each query section against every spot of all target sections at once (no stack, no target coordinates), **one solve per query section**, uniform masses | OTT-JAX Sinkhorn |
| | `pts2pts` | 3D ↔ 3D. The registered stack points of both embryos (stage 1): 6000 per embryo, slab masses, **one joint solve** for all sections, mapped back to the real spots by kNN. `pts2pts20k`: 20 000 points | stage 1 + OTT-JAX |
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
| `sec2part_s-euc_f-euc_bal` | 2D FGW on the partner section (α = 0.5): the cost of the benchmark's `sec2sec_s-euc_f-euc_bal`, on one target section | `fgw` |
| `sec2part_s-euc0.9_f-euc_bal`, `…_s-euc0.99_…` | the same with more weight on space | `fgw@0.9`, `fgw@0.99` |
| `sec2part_s-none_f-euc_bal` | features only | `w` |
| `sec2part_s-none_f-euc_semi0.7` | features only, target side relaxed | `w_semi@0.7` |
| `sec2part_s-euc_f-none_bal` | space only | `gw` |
| `sec2part_s-none_f-lab_bal`, `sec2part_s-euc_f-lab_bal` | labels as the cross cost, without and with space | `label_w`, `label_fgw@0.5` |
| `sec2part_s-none_f-lab_nn` | label ceiling on the partner section | `label_nn` |
| `pts2pts_s-euc_f-euc_ub` | the 3D benchmark pipeline | `PC_fgw` |
| `pts2pts_s-none_f-euc_ub` | the same solve without the GW term: same points, masses and marginals | `PC_w` |
| `pts2pts20k_s-none_f-euc_ub` | the same on 20 000 points per embryo | `PC_w@20k` |
| `sec2emb_s-none_f-euc_bal`, `…_semiτ`, `…_nn` | whole embryo, Euclidean features | `E_w`, `E_w_semi@τ`, `E_w_nn` |
| `sec2emb_s-none_f-cos_semiτ`, `…_nn` | whole embryo, cosine features | `cos_w_semi@τ`, `cos_w_nn` |
| `sec2emb_s-none_f-lab_nn` | label ceiling on the whole embryo | `E_label_nn`, `label_nn` (whole embryo) |

## 1. In 2D, space never helps (X4)

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

## 2. In 2D, the partner section is the limit (X4)

- **The partner section caps any cost.** Even with perfect labels, `sec2part_s-none_f-lab_nn` reaches only 0.624, because organs are missing from the partner sections. Against the whole embryo, `sec2emb_s-none_f-lab_nn` reaches 0.892.
- **The balanced marginal costs about 0.05.** With the query side balanced and the target side relaxed, features only rise from 0.295 to 0.340 (`sec2part_s-none_f-euc_semi0.7` to `…_semi0.9`). With perfect labels, balance costs 0.13 (`sec2part_s-none_f-lab_bal` 0.498 against `sec2part_s-none_f-lab_nn` 0.624).
- **The features do not follow laterality.** On the whole embryo, the mass goes to sections as far in L as an even spread would. Either L is not comparable between the species or organ coverage matters more than depth, so pairing sections by laterality is doubtful.

## 3. Across pairs: the whole embryo beats the 3D pipeline

κ\_bal, mean over the query sections of both directions. The Sinkhorn threshold is 1e-5 for all runs (see below).

| run | X1 | X4 | X5 | mean |
| --- | --- | --- | --- | --- |
| `pts2pts_s-euc_f-euc_ub` (benchmark) | 0.351 | 0.440 | 0.435 | 0.409 |
| `pts2pts_s-none_f-euc_ub` (no GW term) | 0.347 | 0.432 | 0.421 | 0.400 |
| `pts2pts20k_s-none_f-euc_ub` (20 000 points) | 0.373 | 0.469 | 0.467 | 0.436 |
| `sec2emb_s-none_f-euc_bal` | 0.339 | 0.396 | 0.415 | 0.383 |
| `sec2emb_s-none_f-euc_semi0.9` | 0.366 | 0.465 | 0.471 | 0.434 |
| `sec2emb_s-none_f-euc_semi0.7` | 0.388 | 0.475 | 0.482 | 0.448 |
| `sec2emb_s-none_f-lab_nn` (ceiling) | 0.973 | 0.892 | 0.859 | 0.908 |

- **The GW term adds little:** `pts2pts_s-euc_f-euc_ub` − `pts2pts_s-none_f-euc_ub` (same points, masses and marginals; only the GW term differs) is +0.004, +0.008 and +0.014 in κ\_bal, and nothing in gene r (0.307 against 0.308).
- **Resolution matters more:** 20 000 points per embryo instead of 6000 gain +0.026 to +0.047 (`pts2pts20k_s-none_f-euc_ub`), and beat the benchmark on every pair without any GW term.
- **The whole embryo, per section, does as well:** `sec2emb_s-none_f-euc_semi0.9` ties `pts2pts20k_s-none_f-euc_ub` (0.434 against 0.436) and beats the benchmark on 3 of 3 pairs (+0.015 to +0.036). So the stack, the slab masses and the joint solve add nothing measurable either; resolution and access to every target section do.
- **Relaxing the target helps:** from balanced (0.383) to τ\_b = 0.7 (0.448). Below 0.7 the curve is flat (0.450 at 0.5, 0.447 for `sec2emb_s-none_f-euc_nn`).
- **The solver threshold can be loosened.** At 1e-5 instead of 1e-7, every score is unchanged (largest change 0.0000), and the solves run 7–13 × faster (`pts2pts_s-euc_f-euc_ub` 95 s → 7 s). At 1e-7, it reproduces the benchmark exactly.

## 4. The features

**What the 30 PCs carry (X4).** The embedding is mostly shared between the species. PCs 1–3 carry 63 % of the cross cost and order the organs the same way in both species (r 0.94, 0.97, 0.71); 85 % of the cost is on PCs with organ r > 0.5. Two smaller defects: PCs 8 and 9 order the organs differently in the two species (r −0.62, −0.30; 4 % of the cost), and the mouse spots spread about 1.4 × more on almost every PC. More PCs would not help: `sec2part_s-none_f-euc_bal` levels off at 20 PCs (κ\_bal 0.293 at 20, 0.295 at 30).

**Variants** (of `sec2emb_s-none_f-euc_semi0.9`; change in κ\_bal against the embedding as it is):

| variant | X1 | X4 | X5 | mean |
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

| marginals | `f-euc` | `f-cos` | `f-cos` − `pts2pts_s-euc_f-euc_ub` (X1 / X4 / X5) |
| --- | --- | --- | --- |
| `semi0.95` | 0.421 (0.391) | 0.443 (0.389) | +0.013 / +0.042 / +0.049 |
| `semi0.9` | 0.434 (0.385) | 0.463 (0.384) | +0.029 / +0.063 / +0.072 |
| `semi0.7` | 0.448 (0.359) | 0.486 (0.358) | +0.061 / +0.084 / +0.087 |
| `semi0.5` | 0.450 (0.338) | 0.492 (0.341) | +0.072 / +0.088 / +0.090 |
| `nn` | 0.447 (0.309) | 0.495 (0.317) | +0.080 / +0.087 / +0.092 |
| `pts2pts_s-euc_f-euc_ub` | 0.409 (0.307) | | |

- **The two gains add up.** With cosine distance, κ\_bal keeps rising as the target is relaxed, up to 0.495 for `sec2emb_s-none_f-cos_nn`, +0.08 to +0.09 over the benchmark on every pair. Cosine adds more the more the target is relaxed (+0.022 at τ\_b 0.95, +0.048 for `nn`).
- **Gene r does not follow.** Cosine leaves gene r unchanged at every τ\_b, and gene r falls with relaxation as before (0.389 at 0.95, 0.317 for `nn`).
- **τ\_b = 0.7 is the compromise:** `sec2emb_s-none_f-cos_semi0.7` reaches κ\_bal 0.486 (2 % below the best) with gene r 0.358, still above the benchmark on both scores.
- **For the organs, the transport constraint barely matters.** The best κ\_bal comes with no constraint on the target at all, i.e. matching each spot to its nearest spots in the feature space. The marginal mainly keeps the coupling spread, which helps gene r.

## 5. τ\_b cannot be chosen by gene r

The benchmark lets labels score a result but not choose a setting, so τ\_b would ideally be chosen by gene r on half of the held-out genes. It does not work: gene r picks τ\_b = 0.95 (κ\_bal 0.420 against 0.445 at the best τ\_b), with a regret above 0.05 for 2 of 6 pair-directions. Gene r falls steadily as the target is relaxed and favours spread-out couplings: averaging over many target spots smooths the transported expression and raises the correlation. The label oracle `sec2emb_s-none_f-lab_nn` has a lower gene r (0.331) than the balanced `sec2emb_s-none_f-euc_bal` (0.381). Gene r stays a useful score, but it should not choose settings that change how spread the coupling is.

## Caveats

- **One seed and three pairs.** Differences below about 0.01–0.02 are not readable. X2 and X3 (mouse inside human) and X6 (no depth overlap) are not tested yet.
- **The 2D results rest on one pair** and few distinct section pairs: five of six human sections pair with the same mouse section.
- **`pts2pts_s-euc_f-euc_ub` was run in its benchmark configuration** (6000 points). FGW on 20 000 points would need distance matrices too large for the GPU, so whether the GW term helps at higher resolution is open.
- **The unbalanced mass above 1** seen in the first 2D ablation is not a bug: with a KL penalty on the marginals, the transported mass scales with the cost, so the same τ means different things for different costs. The `semi` runs avoid this.

## Next steps

- [ ] Make `sec2emb_s-none_f-cos_semi0.7` a benchmark pipeline (cosine features, whole embryo, every spot, query side balanced, τ\_b = 0.7) and run it on X1–X5 with the Dev seeds (0, 1, 2), for the noise margin and the harder pairs X2 and X3.
- [ ] Fix τ\_b on the Dev pairs (X1, X4) and check it on the others, as the strategy doc allows for thresholds. 0.7 is the current compromise between κ\_bal and gene r.
- [ ] Set `fgw.inner_tol` to 1e-5 in `bench.yaml`: the same scores at a fraction of the runtime.
- [ ] Pause work on the field for the cross-species step. Stage 1 may still matter within an embryo.
- [ ] Work on the features: they are now the limit, at about half of the label ceiling. Integration across species (Harmony, scVI) and other gene sets are the next candidates.

Notebooks: `demos/section_cost_ablation.ipynb` (2D, X4) and `demos/whole_embryo_tests.ipynb` (X1, X4, X5), on branch `sim/whole-embryo-tests-20261027`, executed on tueilnt-sim13.
