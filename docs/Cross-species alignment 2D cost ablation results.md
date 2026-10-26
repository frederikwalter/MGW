# Cross-species alignment: 2D cost ablation results

Oct 26, 2026 · @Frederik

In 2D, the features carry all of the cross-species signal: aligning on features alone matches the full fused GW cost (κ\_bal 0.295 vs 0.293), and space alone is at chance (−0.021).

## Question and setup

The question is which part of the stage-2 cost helps a human ↔ mouse alignment and which part hurts: space, features or organ labels. It is tested on real 2D sections first, to remove the stage-1 uncertainties of the 3D pipeline (stacking, z gaps, field, volume sampling), since the field volume was found to do worse than the real stack points.

- **Pair:** X4, CS18\_E1 (human, 6 sections) ↔ E12.5\_E1 (mouse, 5 sections), seed 0, settings of `A_fgw` in `bench.yaml` (ε = 5e-3).
- **Pairing:** every section is aligned with the section of the other embryo whose laterality L is closest. The pairing uses no labels and is the same for every run, so the choice of target is not part of the comparison. Both directions are scored: 11 query sections over 9 unique section pairs.
- **Only the cost changes between runs:**

| run | within each section | between sections | role |
| --- | --- | --- | --- |
| `fgw` | Euclidean distances | features (ortholog PCA) | reference, α = 0.5 (= `A_fgw`) |
| `gw` | Euclidean distances | none | space only |
| `label_fgw@α` | Euclidean distances | organ labels | ceiling; α = 0.5, 0.9, 0.99 |
| `w` | none | features | features only (Sinkhorn) |

Every run is solved balanced (τ = 1) and unbalanced (τ = 0.95). Two scores per query section:

- **κ\_bal**, the benchmark metric. It is circular for `label_fgw`, which sees the labels it is scored on.
- **gene r**: the mean Pearson r of the 500 held-out ortholog genes between each query spot and what P transports to it, minus the same with the target spots shuffled. No run sees these genes, so this score is fair for all of them.

## Results

Balanced `fgw` reproduces the benchmark's `A_fgw` scores for the same section pairs exactly (max κ\_bal difference 0.0000), so the reference is sound.

&#91;embedded content: demos/section\_cost\_ablation.ipynb · X4, seed 0, run on tueilnt-sim12, 2026-10-23\]

The unbalanced solves (τ = 0.95) and the mirror flips are below. The mass moved differs strongly between costs, so the unbalanced rows are not comparable across runs (see Caveats).

| run | κ\_bal, τ = 0.95 | gene r, τ = 0.95 | mass moved, τ = 0.95 | flips vs `fgw` (balanced, of 11) |
| --- | --- | --- | --- | --- |
| `fgw` | 0.343 | 0.373 | 0.287 | 0 |
| `w` | 0.340 | 0.372 | 0.309 | 0 |
| `gw` | −0.021 | −0.027 | 1.447 | 6 |
| `label_fgw`, α 0.5 | 0.614 | 0.276 | 0.751 | 1 |
| `label_fgw`, α 0.9 | 0.517 | 0.264 | 1.281 | 1 |
| `label_fgw`, α 0.99 | 0.161 | 0.164 | 1.430 | 1 |

## What the results suggest

The features contribute; in 2D, space neither helps nor hurts once features are in the cost.

- **Space alone is at chance.** `gw` scores about 0 on both measures. In 6 of 11 query sections, the rigid map fitted to its coupling is a reflection where `fgw`'s is not, so GW's mirror ambiguity is part of why it fails.
- **The GW term adds nothing at α = 0.5.** `w` (features only) matches or slightly beats `fgw` on both scores: κ\_bal 0.295 vs 0.293, gene r 0.361 vs 0.350.
- **The geometry fights the labels.** With organ labels as the cross cost, κ\_bal falls from 0.50 (α = 0.5) to 0.47 (α = 0.9) to 0.17 (α = 0.99). Even the most label-weighted run reaches only 0.50, so the label ceiling for these section pairs is low.
- **Unbalanced helps a little.** τ = 0.95 raises κ\_bal by about 0.05 for `fgw`, `w` and `label_fgw@0.5`, consistent with tissue that has no counterpart in the partner section.

## Caveats

These are results from one pair and one seed, resting on few distinct section pairs.

- **Thin pairing.** Human S2–S6 (L 0.56–0.67) all pair with mouse S5 (L 0.667), and four of the five mouse sections pair with human S6. Human S1 ↔ mouse S3 has a large depth gap (L 0.33 vs 0.10).
- **Unbalanced rows are not comparable across costs.** The unbalanced feature runs transport only 0.29–0.31 of the mass, while unbalanced `gw` and `label_fgw@0.9/0.99` transport 1.28–1.45. More than the total mass should not happen; OTT's unbalanced GW correction probably distorts the marginals. The balanced rows are the clean comparison.
- **`label_fgw` has no feature term.** Its lower gene r reflects the missing features, not harm done by the labels.
- **Runtime.** The 108 solves took about 4 hours on one RTX 4080 Super (19–35 min per section pair). A multi-pair or 3D follow-up needs to plan for that.

## Next steps

Before building on these results, the unbalanced mass anomaly and the thin pairing need fixing.

- [ ] Find out why unbalanced GW transports more than the total mass (OTT's `gw_unbalanced_correction`), or compare costs balanced only.
- [ ] Repeat on more pairs (X1, X5) and seeds, so each conclusion rests on more than a handful of section pairs.
- [ ] Run the same cost ablation in 3D on the real-points measure (`PC_fgw`), not the field, and check whether space starts to help once depth is modelled.
- [ ] Test `w` as a cheap baseline in the benchmark: it matches `fgw` in 2D and needs no GW solve.

The notebook is `demos/section_cost_ablation.ipynb` on branch `sim/section-cost-ablation-20261022` (executed version: commit `84b8389`).
