# Cross-species alignment: features, space and the target

Oct 27, 2026 · @Frederik

The features carry the cross-species signal; space does not. In 2D, matching on space alone is at chance, and giving space any weight next to the features never helps. In 3D, the GW term adds at most 0.014 κ\_bal. What helps is the target: matching each section, on features alone, against all spots of the other embryo beats the 3D pipeline `PC_fgw` on all three pairs tested. With cosine distance and a relaxed target marginal, it reaches κ\_bal 0.486 against 0.409 for `PC_fgw` (+0.06 to +0.09 on every pair), about half of what perfect labels allow.

## Question and setup

Which part of the stage-2 cost helps a human ↔ mouse alignment: **space** (distances within each embryo, the GW term), the **features** (the shared ortholog PCA), or the choice of **target**? And how far is each from what the labels would allow?

- **2D first** (X4, CS18\_E1 ↔ E12.5\_E1, seed 0). Each section is aligned with the section of the other embryo whose laterality L is closest. This removes every stage-1 uncertainty (stacking, gaps, field).
- **Then three pairs** (X1, X4, X5, seed 0): the 3D pipeline on the registered stack points against feature-only matching on the whole target embryo.
- **Two scores** per query section, both directions:
  - **κ\_bal**, the benchmark metric (0 = chance, 1 = every organ to itself).
  - **gene r**: the mean Pearson r of 500 held-out ortholog genes between each query spot and what the coupling transports to it, minus the same with the target spots shuffled. No run sees these genes.

| run | space | features | target |
| --- | --- | --- | --- |
| `fgw` (= `A_fgw`) | 2D distances, α = 0.5 | yes | partner section |
| `gw` | 2D distances | no | partner section |
| `w` | no | yes | partner section |
| `PC_fgw` (benchmark) | 3D distances on the stack points, α = 0.5 | yes | whole embryo, 6000 points, τ = 0.95 both sides |
| `PC_w` | no | yes | as `PC_fgw` |
| `E_w_semi@τ_b` | no | yes | whole embryo, every spot, per query section; query side balanced, target side relaxed to τ\_b |
| `label_nn` | no | organ labels | ceiling: each spot to the target spots of its own organ |

## 1. In 2D, space never helps (X4)

| run (partner sections, balanced) | κ\_bal | gene r |
| --- | --- | --- |
| `w` (features only) | 0.295 | 0.361 |
| `fgw` (α = 0.5) | 0.293 | 0.350 |
| `fgw` (α = 0.9) | 0.188 | 0.341 |
| `fgw` (α = 0.99) | 0.068 | 0.118 |
| `gw` (space only) | −0.021 | −0.028 |

- **Space alone is at chance.** Its couplings are also mirror images of `fgw`'s on 6 of 11 sections, which is what a near-random coupling gives.
- **The GW term is active but does not help.** The GW energy measures how far a coupling distorts the distances within the sections (0 = none). At α = 0.5 it halves the GW energy of `w` (0.035 against 0.062; a random coupling has 0.074) for almost no extra feature cost (+0.005), yet κ\_bal and gene r do not move. With more weight on space, both fall.
- **Labels with geometry are capped too.** Organ labels as the cross cost reach κ\_bal 0.50 at α = 0.5 and 0.17 at α = 0.99.

## 2. In 2D, the partner section is the limit (X4)

- **The partner section caps any cost.** Even with perfect labels, `label_nn` reaches only 0.624 on the partner sections, because organs are missing from them. Against the whole embryo it reaches 0.892.
- **The balanced marginal costs about 0.05.** With the query side balanced and the target side relaxed, `w` rises from 0.295 to 0.340 (τ\_b 0.7–0.9). With perfect labels, balance costs 0.13 (`label_w` 0.498 against `label_nn` 0.624).
- **The features do not follow laterality.** On the whole embryo, the mass goes to sections as far in L as an even spread would. Either L is not comparable between the species or organ coverage matters more than depth, so pairing sections by laterality is doubtful.

## 3. Across pairs: the whole embryo beats the 3D pipeline

κ\_bal, mean over the query sections of both directions. The Sinkhorn threshold is 1e-5 for all runs (see below).

| run | X1 | X4 | X5 | mean |
| --- | --- | --- | --- | --- |
| `PC_fgw` (benchmark) | 0.351 | 0.440 | 0.435 | 0.409 |
| `PC_w` (no GW term) | 0.347 | 0.432 | 0.421 | 0.400 |
| `PC_w` on 20 000 points | 0.373 | 0.469 | 0.467 | 0.436 |
| `E_w` (balanced) | 0.339 | 0.396 | 0.415 | 0.383 |
| `E_w_semi@0.9` | 0.366 | 0.465 | 0.471 | 0.434 |
| `E_w_semi@0.7` | 0.388 | 0.475 | 0.482 | 0.448 |
| `label_nn`, whole embryo (ceiling) | 0.973 | 0.892 | 0.859 | 0.908 |

- **3D geometry adds little:** `PC_fgw` − `PC_w` is +0.004, +0.008 and +0.014 in κ\_bal, and nothing in gene r (0.307 against 0.308).
- **Resolution matters more:** `PC_w` on 20 000 points instead of 6000 gains +0.026 to +0.047, and beats `PC_fgw` on every pair without any geometry.
- **The whole embryo, per section, does as well:** `E_w_semi@0.9` ties `PC_w` on 20 000 points (0.434 against 0.436) and beats `PC_fgw` on 3 of 3 pairs (+0.015 to +0.036). So the stack, the slab masses and the joint solve add nothing measurable either; resolution and access to every target section do.
- **Relaxing the target helps:** from balanced (0.383) to τ\_b = 0.7 (0.448). Below 0.7 the curve is flat (0.450 at 0.5, 0.447 for the nearest neighbour).
- **The solver threshold can be loosened.** At 1e-5 instead of 1e-7, every score is unchanged (largest change 0.0000), and the solves run 7–13 × faster (`PC_fgw` 95 s → 7 s). At 1e-7, `PC_fgw` reproduces the benchmark exactly.

## 4. The features

**What the 30 PCs carry (X4).** The embedding is mostly shared between the species. PCs 1–3 carry 63 % of the cross cost and order the organs the same way in both species (r 0.94, 0.97, 0.71); 85 % of the cost is on PCs with organ r > 0.5. Two smaller defects: PCs 8 and 9 order the organs differently in the two species (r −0.62, −0.30; 4 % of the cost), and the mouse spots spread about 1.4 × more on almost every PC. More PCs would not help: `w` levels off at 20 PCs (κ\_bal 0.293 at 20, 0.295 at 30).

**Variants** (`E_w_semi@0.9`, change in κ\_bal against the embedding as it is):

| variant | X1 | X4 | X5 | mean |
| --- | --- | --- | --- | --- |
| cosine instead of Euclidean distance | +0.014 | +0.038 | +0.036 | +0.029 |
| PCs rescaled per species | +0.003 | +0.004 | +0.007 | +0.005 |
| without the contradicting PCs (uses labels) | +0.025 | −0.001 | −0.004 | +0.007 |
| first 20 PCs whitened | −0.099 | −0.082 | −0.083 | −0.088 |

- **Cosine distance helps on all three pairs**, with gene r unchanged. It ignores the length of each spot's feature vector, which likely varies with depth or tissue rather than identity.
- **Whitening hurts** both scores: the low-variance PCs are mostly noise, so weighting the PCs by their variance is right.
- The other two are within noise.

**Cosine with the τ\_b sweep:**

κ\_bal (gene r in brackets), mean over the query sections:

| τ\_b | Euclidean | cosine | cosine − `PC_fgw` (X1 / X4 / X5) |
| --- | --- | --- | --- |
| 0.95 | 0.421 (0.391) | 0.443 (0.389) | +0.013 / +0.042 / +0.049 |
| 0.9 | 0.434 (0.385) | 0.463 (0.384) | +0.029 / +0.063 / +0.072 |
| 0.7 | 0.448 (0.359) | 0.486 (0.358) | +0.061 / +0.084 / +0.087 |
| 0.5 | 0.450 (0.338) | 0.492 (0.341) | +0.072 / +0.088 / +0.090 |
| 0 (nearest neighbour) | 0.447 (0.309) | 0.495 (0.317) | +0.080 / +0.087 / +0.092 |
| `PC_fgw` | 0.409 (0.307) | | |

- **The two gains add up.** With cosine distance, κ\_bal keeps rising as the target is relaxed, up to 0.495 for the soft nearest neighbour, +0.08 to +0.09 over `PC_fgw` on every pair. Cosine adds more the more the target is relaxed (+0.022 at τ\_b 0.95, +0.048 for the nearest neighbour).
- **Gene r does not follow.** Cosine leaves gene r unchanged at every τ\_b, and gene r falls with relaxation as before (0.389 at 0.95, 0.317 for the nearest neighbour).
- **τ\_b = 0.7 is the compromise:** κ\_bal 0.486 (2 % below the best) with gene r 0.358, still above `PC_fgw` on both scores.
- **For the organs, the transport constraint barely matters.** The best κ\_bal comes with no constraint on the target at all, i.e. matching each spot to its nearest spots in the feature space. The marginal mainly keeps the coupling spread, which helps gene r.

## 5. τ\_b cannot be chosen by gene r

The benchmark lets labels score a result but not choose a setting, so τ\_b would ideally be chosen by gene r on half of the held-out genes. It does not work: gene r picks τ\_b = 0.95 (κ\_bal 0.420 against 0.445 at the best τ\_b), with a regret above 0.05 for 2 of 6 pair-directions. Gene r falls steadily as the target is relaxed and favours spread-out couplings: averaging over many target spots smooths the transported expression and raises the correlation. The label oracle `label_nn` has a lower gene r (0.331) than balanced `E_w` (0.381). Gene r stays a useful score, but it should not choose settings that change how spread the coupling is.

## Caveats

- **One seed and three pairs.** Differences below about 0.01–0.02 are not readable. X2 and X3 (mouse inside human) and X6 (no depth overlap) are not tested yet.
- **The 2D results rest on one pair** and few distinct section pairs: five of six human sections pair with the same mouse section.
- **`PC_fgw` was run in its benchmark configuration** (6000 points). FGW on 20 000 points would need distance matrices too large for the GPU, so whether geometry helps at higher resolution is open.
- **The unbalanced mass above 1** seen in the first 2D ablation is not a bug: with a KL penalty on the marginals, the transported mass scales with the cost, so the same τ means different things for different costs. The semi-relaxed runs avoid this.

## Next steps

- [ ] Make `E_w` a benchmark pipeline (cosine features, whole embryo, every spot, query side balanced, τ\_b = 0.7) and run it on X1–X5 with the Dev seeds (0, 1, 2), for the noise margin and the harder pairs X2 and X3.
- [ ] Fix τ\_b on the Dev pairs (X1, X4) and check it on the others, as the strategy doc allows for thresholds. 0.7 is the current compromise between κ\_bal and gene r.
- [ ] Set `fgw.inner_tol` to 1e-5 in `bench.yaml`: the same scores at a fraction of the runtime.
- [ ] Pause work on the field for the cross-species step. Stage 1 may still matter within an embryo.
- [ ] Work on the features: they are now the limit, at about half of the label ceiling. Integration across species (Harmony, scVI) and other gene sets are the next candidates.

Notebooks: `demos/section_cost_ablation.ipynb` (2D, X4) and `demos/whole_embryo_tests.ipynb` (X1, X4, X5), on branch `sim/whole-embryo-tests-20261027`, executed on tueilnt-sim13.
