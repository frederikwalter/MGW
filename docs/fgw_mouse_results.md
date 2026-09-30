# FGW on the mouse embryo pair (E9.5 → E10.5): results

This note analyses the fused Gromov–Wasserstein (FGW) run in `demos/mouse_align.ipynb`, which
implements [fgw_joint_embedding_plan.md](fgw_joint_embedding_plan.md). All numbers come from the
stored notebook outputs (commit `ed24846`, CUDA run, seed 1).

**Short version.** The FGW code works. Every setting gives a proper rotation, and the objective
terms move with α as expected. But on this pair, the CCA-based fused term does **not** improve
label transfer. AMI falls steadily as the fused term gets more weight. The drop is larger with
all three CCA components than with component 0 alone. The unimodal joint-PCA baseline shows that
FGW *can* improve things a lot (AMI +0.08, ARI +0.22, ACC +0.18). So the limit is the quality of
the CCA joint space, not the solver.

## 1. What was run

| Step | Setting |
|---|---|
| Slices | E9.5 (A, n = 5 913 spots) → E10.5 (B, m = 18 408 spots), 24 040 shared genes |
| Features | joint log-PCA (30 comps) → spatial-only GW "feeler" (E10.5 downsampled to 8 000) → CCA with 3 comps |
| CCA canonical correlations ρ_k | 0.942, 0.869, 0.633 → ρ² weights 0.887, 0.755, 0.401 |
| Neural fields | φ (E9.5): final loss ≈ 0.078; ψ (E10.5): final loss ≈ 0.49 |
| Baseline | pure GW on the pullback-geodesic costs `C_M`, `C_N` (`out = mgw_align_core(...)`) |
| FGW sweep | `mgw_resolve(out, pre, use_fgw=True, fgw_alpha=α, fused_comps=k)` for α ∈ {0.9, 0.7, 0.5, 0.3} and k ∈ {1, 3}, with `fused_source="field"` and `fused_weight="rho2"` (the defaults) |
| Upper bound | FGW with α = 0.5 and `M` taken from the joint gene PCA (not modality-agnostic) |

`mgw_resolve` reuses φ, ψ, `C_M` and `C_N` from the baseline, so all rows share the same
geometry. Only the coupling changes.

## 2. How to read the table

Each row is one coupling `P` (5 913 × 18 408). The columns are:

| Column | Meaning |
|---|---|
| `orientation_det` | Each E9.5 spot is mapped to its barycentric image in E10.5, `Y = P·xs2 / P·1`. A Procrustes fit is then run between `xs` and `Y`. The column is the sign of the determinant of the resulting rotation. **+1** means a proper rotation. **−1** means the coupling mirrors the tissue, which is the failure mode that pure GW can't rule out. |
| `AMI` | **Adjusted Mutual Information** between the *projected* labels and the *true* E9.5 labels. It is 0 for chance and 1 for identical partitions. It ignores label names and is corrected for chance. It rewards a mapping where each true class lands consistently in one predicted class, and it gives small classes a fair say. This is the main metric. |
| `ARI` | **Adjusted Rand Index**, a pair-counting agreement score. For every pair of spots it asks whether both labelings agree on "same class / different class". It is chance-corrected (0 = chance, 1 = perfect). Large classes dominate it. |
| `ACC` | Plain **accuracy**: the fraction of E9.5 spots whose projected label equals their true label *by name*. Large classes also dominate it. |
| `half_GW` | The GW term of the objective that OTT minimises, evaluated at `P`: `0.5 · Σ (C_M[i,k] − C_N[j,l])² P_ij P_kl`. This is the geometric distortion: lower means the coupling preserves intra-slice geodesic distances better. The ½ is OTT's convention. It is `NaN` for the GW row because the notebook only records objective terms for FGW runs. |
| `lin` | The linear (Wasserstein) term, `fused_penalty · ⟨M, P⟩`, with `fused_penalty = (1−α)/(2α)`. `M_ij` is the ρ²-weighted squared distance between the CCA variates that φ predicts at E9.5 spot *i* and ψ predicts at E10.5 spot *j*. It is scaled to [0, 1] by its 99th percentile. |

`half_GW + lin` is the objective OTT minimises (before entropy). Multiplying it by 2α gives the
POT-style objective `α·GW + (1−α)·⟨M, P⟩`.

**How the label projection works** (`metrics.ami_on_projected_labels(P.T, B_labels, A_labels)`):
each E9.5 spot *j* takes its column of `Pᵀ`, normalised to sum to 1. This gives the distribution
over E10.5 spots that *j* is sent to. Summing that mass per E10.5 annotation gives
`p(label | j)`, and the argmax is the projected label. Only spots whose annotation occurs in
*both* slices are scored. The metrics therefore measure: "if I annotate E9.5 by transferring
E10.5's annotation through `P`, how well do I recover E9.5's own annotation?"

## 3. Results

Derived columns: `fused_penalty = (1−α)/(2α)`, `⟨M,P⟩ = lin / fused_penalty`,
linear share `= lin / (half_GW + lin)`, and Δ values relative to the pure-GW row.

| Setting | det | AMI | ARI | ACC | half_GW | lin | penalty | ⟨M,P⟩ | lin share | ΔAMI | ΔARI | ΔACC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GW** | +1 | **0.402** | **0.247** | **0.537** | – | – | – | – | – | – | – | – |
| FGW α=0.9 k=1 | +1 | 0.381 | 0.219 | 0.514 | 0.00256 | 0.00028 | 0.056 | 0.0051 | 10 % | −0.021 | −0.028 | −0.023 |
| FGW α=0.7 k=1 | +1 | 0.354 | 0.208 | 0.497 | 0.00280 | 0.00053 | 0.214 | 0.0025 | 16 % | −0.048 | −0.039 | −0.040 |
| FGW α=0.5 k=1 | +1 | 0.351 | 0.211 | 0.500 | 0.00303 | 0.00080 | 0.500 | 0.0016 | 21 % | −0.050 | −0.037 | −0.037 |
| FGW α=0.3 k=1 | +1 | 0.346 | 0.217 | 0.500 | 0.00331 | 0.00133 | 1.167 | 0.0011 | 29 % | −0.056 | −0.030 | −0.037 |
| FGW α=0.9 k=3 | +1 | 0.333 | 0.220 | 0.529 | 0.00270 | 0.00159 | 0.056 | 0.0287 | 37 % | −0.068 | −0.027 | −0.008 |
| FGW α=0.7 k=3 | +1 | 0.318 | 0.215 | 0.529 | 0.00318 | 0.00499 | 0.214 | 0.0233 | 61 % | −0.084 | −0.032 | −0.008 |
| FGW α=0.5 k=3 | +1 | 0.299 | 0.208 | 0.522 | 0.00367 | 0.01073 | 0.500 | 0.0215 | 75 % | −0.103 | −0.040 | −0.015 |
| FGW α=0.3 k=3 | +1 | 0.285 | 0.209 | 0.518 | 0.00428 | 0.02370 | 1.167 | 0.0203 | 85 % | −0.116 | −0.038 | −0.019 |
| *FGW α=0.5 joint-PCA (unimodal)* | +1 | *0.484* | *0.463* | *0.718* | 0.00451 | 0.08359 | 0.500 | 0.1672 | 95 % | *+0.082* | *+0.216* | *+0.181* |

## 4. Findings

1. **Orientation is +1 everywhere.** Neither the fused term nor its weight flips the map.
   The pure-GW baseline was *already* correctly oriented for this seed, though. So the main
   reason for FGW in the plan (breaking mirror/rotation symmetry) was never tested here: the
   sweep measures what the fused term costs on a run that didn't need fixing.

2. **The solver behaves as designed.** As α falls, the linear share rises steadily
   (10 → 29 % for k=1, 37 → 85 % for k=3). `⟨M,P⟩` falls and `half_GW` rises: the solver trades
   geometric distortion for feature agreement. At α = 0.3, k=3, the GW term is 67 % above its
   α = 0.9 value.

3. **The CCA fused term makes label transfer worse, and the effect grows with its weight.**
   AMI falls steadily in α for both k. The best FGW row (α = 0.9, k = 1) is still 0.021 below GW.
   ARI drops by about 0.03–0.04 in every FGW row, with little dependence on α.

4. **comps=1 vs comps=3.**
   - **k = 1** builds `M` from CCA variate 0 alone (ρ = 0.94). `M_ij = w₀(u_i − v_j)²` is a 1-D
     cost: every E10.5 spot on the same feat0 contour costs the same. Such a cost is easy to
     satisfy. A coupling only has to match feat0 *levels*, which leaves almost all of the
     geometric freedom intact. That is why `⟨M,P⟩` goes almost to zero (0.0011 at α = 0.3), the
     linear share stays below 30 %, and AMI drops by at most 0.056. It acts like a weak "don't
     cross the main expression gradient" prior.
   - **k = 3** adds variates 1 and 2 (ρ = 0.87, 0.63; weights 0.76, 0.40). Three independent
     constraints can't all be met by one geometry-preserving map. `⟨M,P⟩` stays about 20× higher
     than for k = 1, and barely falls as the penalty rises 20× (0.029 → 0.020). So the linear
     term keeps pulling against the GW term, and it dominates the objective from α ≤ 0.7. This
     is where the larger AMI loss (up to −0.116) comes from.
   - **Where accuracy goes:** k = 3 keeps ACC close to GW (−0.01 to −0.02), but its AMI is the
     worst. So the big classes still land correctly, while the smaller classes get scrambled.
     AMI penalises this, and ACC and ARI barely notice it. k = 1 loses more ACC (−0.04) but less
     AMI, which suggests it shifts whole regions a little rather than fragmenting them.

5. **The joint-PCA upper bound shows FGW can help.** With `M` built from 30 joint gene-PCA
   dimensions, the fused term dominates the objective (95 %). The geometry is distorted the most
   of any run (`half_GW` 0.0045). Yet all three label metrics improve sharply. So a *good*
   cross-slice cost is worth more than a small loss in geometric fidelity, and the CCA variates
   aren't yet that cost. One caveat: the MOSTA annotations were themselves derived from gene
   expression, so an expression-based `M` is scored on its own terms. The gain is an upper bound
   for the unimodal case, not a fair target for ST↔MSI.

## 5. The projected labels

The notebook plots each coupling twice, side by side. **Left: E10.5 in its own coordinates,
coloured by its true annotation.** (The panel title says "A: annotated", but the data is slice
B = E10.5.) **Right: E9.5 in its own coordinates, coloured by the E10.5 labels projected
through `P`.** The true E9.5 annotation isn't shown in these panels. The right panel shows what
the coupling *claims* about E9.5, and the AMI/ARI/ACC above say how well that matches the truth.
`conf_thresh=0.5` has no effect, because the confidences aren't passed to
`plot_projected_labels`. Every spot is coloured, however low its confidence.

**GW (baseline).** The projection has large, contiguous, smoothly bounded regions. The coarse
anatomy transfers well:
- the outer tissue ring (green) follows the whole boundary;
- the thick olive band follows the dorsal curve from the head into the trunk;
- a compact block of organ labels (red, brown, pink, orange, light cyan) sits in the upper centre,
  as in E10.5;
- the lower trunk/tail is mostly light blue with a purple core.

The weaknesses are the ones GW is known for. Regions come out *too* blocky: thin structures such
as the dark-blue and cyan layers along the rim in E10.5 appear only as fragments, and the organ
block has straight, sharp borders rather than the interleaved pattern of the true annotation.
GW moves whole neighbourhoods together, so it gets the layout right but loses fine structure.
That explains the moderate AMI (0.40) and accuracy (54 %).

**FGW α = 0.5, comps = 1.** The overall layout is the same as GW (the ring, the olive band, the
organ block in the upper centre). But the regions start to fray:
- orange, which is a compact blob under the brown region in GW, now runs as a streak down
  into the trunk;
- in the lower half, the light-blue region is broken up by green and olive patches;
- purple splits into several islands;
- dark-blue and cyan speckle appears along the rim.

This is what a 1-D fused cost produces. Spots that lie on the same feat0 contour are
interchangeable to `M`. Where the GW optimum is flat, the linear term tips individual spots
toward partners that lie elsewhere in E10.5 but have the same feat0 value. Where that value is a
smooth gradient (trunk ↔ tail), the result is streaks along the contour. The damage is
moderate (AMI −0.05).

**FGW α = 0.5, comps = 3.** The projection becomes visibly *salt-and-pepper*:
- the organ block is broken up by scattered red and dark-blue spots;
- brown no longer forms one coherent blob;
- purple, cyan and red dots are scattered through the whole lower trunk.

The tissue outline and the olive band survive, because those follow both the geometry and the
dominant expression gradient. At this setting 75 % of the objective is the linear term. Each
E9.5 spot is largely placed by its own (noisy) 3-D CCA value rather than by its neighbours, so
spatial coherence falls apart. The big classes stay mostly right (ACC −0.015), but the class
structure is scrambled (AMI −0.10).

**In short.** More weight on the CCA cost makes the projected labels less spatially coherent,
not more precise. FGW does *not* recover the fine structure that GW misses. It replaces GW's
over-smoothing with noise. Fixing GW's blockiness would take a cross-cost that distinguishes
tissue types locally, which the joint-PCA run shows is possible when one exists.

## 6. Why the CCA fused cost is weak here

- **Few dimensions, mostly spatial gradients.** Three CCA variates can't separate the dozen or so
  annotation classes. They were fitted to correlate across pairs from a *spatial-only* Euclidean
  GW feeler. Between E9.5 and E10.5 the embryo grows and bends, so those pairs are only roughly
  right. The variates that survive are the broad gradients (head ↔ trunk ↔ tail, visible in the
  feat2 field plots), not tissue-specific programs.
- **Circularity.** Those gradients encode roughly what the feeler, and therefore GW, already
  knows. The fused term adds little new information, but enough noise to pull `P` away from
  the geometric optimum.
- **Inflated ρ.** ρ_k is computed between E9.5 spots and *barycentric averages* of E10.5 spots
  (`Z_bary_on_X` in `util.project_informative_features`). Averaging removes noise, so ρ
  overstates how well a single E10.5 spot's variate agrees. The ρ² weights (0.89/0.76/0.40)
  therefore overweight components 1–2.
- **Noisy ψ.** ψ reaches a fit loss of only about 0.49 on standardised variates, so it explains
  roughly half their variance. The E10.5 side of `M` ("field" source) is a blurred version of
  the variates. Blurring removes the sharp tissue boundaries that label transfer needs.

## 7. Caveats of this evaluation

- **One seed, one direction.** All rows use seed 1 and score only E10.5 → E9.5 label transfer.
  Differences of about 0.02 (e.g. α = 0.9, k = 1 vs GW) may be within seed-to-seed variation.
- **The GW row has no objective terms.** Without `half_GW` for the baseline, and without
  `⟨M, P_GW⟩`, we can't say how much the pure-GW plan already agrees with `M`.
- **Regression check incomplete.** The plan's check against the paper map (distance 0.072) was
  skipped because `PAPER_P` doesn't exist on the machine that ran the notebook. Only
  `orientation det = +1` was confirmed.
- **The sanity checks (M = 0 ⇒ GW, α → 0 ⇒ entropic OT) and the ST↔MSI check** (plan steps 2
  and 4) are not part of this notebook.

## 8. Recommended next steps

1. **Log the baseline terms.** Add `half_GW = 0.5·mgw._gw_energy_lowmem(C_M, C_N, P)` and
   `⟨M_k, P_GW⟩` for k = 1, 3 to the GW row. If `⟨M, P_GW⟩` is already close to the FGW values,
   `M` carries no information beyond the geometry.
2. **Test the intended use case: symmetry breaking.** Run several seeds (or `n_restarts`), find
   ones where pure GW gives `orientation_det = −1` or a poor AMI, and check whether a
   *light* fused term (α ∈ {0.95, 0.99}, k = 1) fixes them. That is where the k = 1 cost should
   pay off.
3. **Use `fused_source="rep"`** to avoid the ψ blur, and compare it with `"field"`.
4. **Phase 2 CCA refinement.** Refit the CCA on barycentric pairs from the MGW coupling instead
   of the spatial-only feeler. Compute ρ on single-spot pairs (sampled from `P`) rather than on
   barycentric averages. Then rebuild `M`. More CCA components (5–10), still ρ²-weighted, are
   worth trying once ρ is honest.
5. **Better diagnostics in the notebook.** Plot the true E9.5 annotation next to each
   projection. Pass `B_conf` to `plot_projected_labels` so that low-confidence spots are greyed
   out. Report the symmetric AMI: `metrics.evaluate_coupling` already averages both directions.
6. **Run the multimodal check** (`demo_mgw_y7.ipynb`, ST↔MSI). There, no joint-PCA bound exists,
   and even a weak cross-cost may matter more.
