# Cross-species alignment in 3D: volumes from serial sections

Discussion notes. The goal is a spot-level coupling between human (HESTA) and mouse (MOSTA) embryos that is good enough to serve as a benchmark in a methods paper. The only ground truth we trust is the organ labels. The notes build on [`demos/human_mouse_align_3d.ipynb`](../demos/human_mouse_align_3d.ipynb), [`mgw/stack3d.py`](../mgw/stack3d.py) and the benchmark harness in [`mgw/bench`](../mgw/bench) ([`bench/README.md`](../bench/README.md)).

**The approach.** Each embryo is treated as a continuous volume, not as a pile of 2D sections:
1. **Stage 1, within species.** Register the serial sections into a stack. Fit a neural field over the stack (features and tissue occupancy at every 3D point), and sample points evenly through the volume.
2. **Stage 2, across species.** Align the two sampled volumes 3D ↔ 3D.
3. **Evaluation.** Carry the coupling back to the real spots, score it against the organ labels, and compare it with the best 2D ↔ 2D alignment of real sections.

**Decisions so far:**
- **The representation is a volume:** a field for the model, sampled points for the computation (§1a).
- **Stage mismatch is handled by choosing pairs** (§4). Interpolating between stages comes later.
- **No supervision beyond the evaluation labels.** Fields, methods and parameters are chosen by label-free scores.
- **Results are symmetric.** Every pipeline runs in both directions, and every score averages the two.
- **The reference is 2D ↔ 2D.** Strategy A aligns each query section with the best real target section, chosen by the FGW objective. The volumes must beat it (§2a).
- **One headline metric,** organ-balanced Cohen's κ, ranks pipelines (§2a).
- **External tools are adapted only if they earn it** (§5c).

---

## 0. The problem and where we stand

**Input:** 3–13 sagittal sections per embryo, at unknown spacing. **Output:** a coupling P between the spots of two embryos.

**Why it is hard:**
- *Depth.* Two sections are cut at different mediolateral positions, which looks like a deformation in 2D.
- *A coarse stack.* There are few sections, and many gap estimates are only lower bounds (3 of 5 in CS18_E1, 2 of 4 in E12.5_E1).
- *Species mismatch.* Size, proportions and timing differ; the label sets are only harmonised.
- *Unsupervised selection.* Choices are made by an objective; the labels only score the result.

**Lessons from the 3D notebook:**
- **Between sections there is only a prior.** Interpolating a held-out section from its neighbours (McCann) beat copying the nearer neighbour by just +0.00 to +0.04 in label agreement. The field must prove it is a better prior (§3, G1).
- **The objective and the labels can disagree.** A lower FGW objective (0.172 vs 0.179) came with worse label accuracy (0.53 vs 0.69). Every unsupervised choice gets a regret check (§2e).
- **z is the least reliable axis,** and the z-scale is an explicit parameter (§1e).

**Status (2026-10-26).** The harness has been cut down to two questions, one metric (κ_bal) and one label-transfer figure ([`bench/README.md`](../bench/README.md)):
1. Which tool aligns real sections best in 2D?
2. Which tools are best for the within-embryo and the between-embryo coupling in 3D, and by how much does 3D beat 2D?

The gates, nulls, ceilings, guardrails and the synthetic pair are gone from the code, so §2b–§2g and §3 describe plans, not code. The first real results are on X4, seed 0 (§7).
- The stack points beat 2D clearly (κ_bal 0.439 against 0.303).
- The field volumes only beat 2D once the field's smoothness penalty and depth loss are relaxed (0.317).

Implementation choices are in §6a.

---

## 1. The approach

### 1a. Representation: a field for the model, points for the computation

A raw stack puts all mass on a few planes, and its geodesics can cross between sections along only a few edges. A volume fixes both:
- **The model is a field** over all of 3D. It can be evaluated anywhere and differentiated, which MGW needs (§1e).
- **The computation stays on points,** because OT and GW couple measures. Points are sampled evenly through the field's volume, with equal masses and the same number for both species.

The field knows nothing beyond the sections. Between them it is a smoothness prior, so it must beat simple priors on held-out sections before it is used (§3).

### 1b. Stage 1: the stack (within species)

Built by `mgw.bench.stack` and cached by `pipelines/stacked.get_stack`. Each step and its method:

| Step | Method now | Could replace it later |
| --- | --- | --- |
| 1. Section order | File order (HESTA); inferred from FGW objectives (MOSTA) | Metadata or the laterality score L |
| 2a. Coupling of neighbours | FGW (moscot, spatial weight 0.8) | PASTE2 (partial FGW), MGW |
| 2b. Placement of each section | OT-ICP: rigid, i.e. rotation or reflection plus translation | Affine or non-rigid: Spateo, MOSAICField, STalign (§5a) |
| 2c. Common frame | Compose into the middle section's frame, then 2 global sweeps | — |
| 3. z | Gaps from the in-plane variogram | \|ΔL\|, metadata, a z-stretch fitted in stage 2 (§1e) |
| 4. Couplings kept | OT-ICP's final coupling, as displacement pairs (the peak of each row and column) | Whatever 2a/2b produce |

Every alternative is scored on G3 (§3). All steps use a within-species embedding of the embryo's own genes, not the cross-species embedding of §1c.

**Inferring the order (step 1).** MOSTA files do not record the order of the sections. For every pair of sections (400 points each), solve FGW and evaluate its objective. A low objective means two sections look alike, so it serves as a dissimilarity. Neighbouring sections should be the most alike, so the order is the shortest path that visits every section once. It is solved exactly (Held–Karp, `stack3d.infer_order`) and oriented to agree with the file order.

**OT-ICP (step 2b).** ICP (iterative closest point) aligns two point clouds by repeating two steps: match each point to its closest partner, then fit the best rigid transform to the matches. OT-ICP replaces the closest-point matching with an OT coupling. Starting from the FGW coupling, it alternates:
1. a weighted Procrustes fit: the rotation (or reflection) and translation that best fit the current coupling;
2. a new linear OT, with cost λ·‖x_a − (x_b R + t)‖² + (1 − λ)·‖f_a − f_b‖² (λ = 0.5, each term median-scaled).

It runs once as a rotation and once as a reflection, and keeps the lower cost (to catch a section mounted face down). Once the sections overlap, a point can only pair with nearby points of similar expression.

**Why not FGW alone?** FGW returns a coupling but no transform. Its GW term only compares distances within each section, so it cannot see rotations or reflections. Without OT-ICP:
- each section would still need a transform, at best one Procrustes fit on the FGW coupling (OT-ICP stopped after its first step);
- that coupling is entropic and blurry, and on a nearly symmetric section it can mix the true match with its mirror image, which pulls the fit towards an average;
- nothing would decide between rotation and reflection (the harness already logs FGW's implied orientation, `fgw_det`, next to OT-ICP's choice);
- the kept couplings would be non-local. Displacement pairs would then join distant points, and the field's depth loss (§1c) would learn a wrong derivative along z.

**Tested on X4 (§7b):** without OT-ICP (`stack.coupling: moscot_procrustes`: one weighted Procrustes fit on moscot's coupling), the neighbour couplings score lower on 17 of 18 neighbour pairs. Mean κ_bal drops from 0.581 to 0.530.

**Can MGW be used here?** Yes, as the coupling of step 2a: MGW aligns 2D sections and returns a coupling, which would start OT-ICP in place of moscot's. It still needs step 2b, because MGW gives no transform either. It is a low-priority candidate:
- neighbouring sections of one embryo share shape and size, which is where Euclidean FGW already works; MGW's learned metric is meant for differences in proportions;
- it trains one network per section (20k iterations, 3 restarts by default), for every neighbouring pair.

It could pay off where sections are torn or folded, since geodesics follow the tissue.

**Tested on X4 (§7b):** MGW + OT-ICP (`mgw_icp`) gives the same couplings as moscot + OT-ICP, equal to within 1e-4 per neighbour pair. OT-ICP converges to the same fixed point from either start. MGW costs about 45 min per embryo instead of seconds, so moscot stays.

### 1c. Stage 1: the field

One field per embryo, pair and seed, in the stack's frame:

$$x = (x, y, z) \;\mapsto\; \big(f(x),\, o(x)\big)$$

- f is the pair's shared ortholog embedding, the features that the cross cost uses.
- o is the tissue occupancy, in [0, 1].

**The shared embedding** (`orthologs.shared_embedding`, via `prepare.pair_features`). It is computed once per pair and seed, over all spots of both embryos:
1. map genes through the 1:1 ortholog table;
2. keep the union of the 2000 most variable orthologs of each embryo, minus the 500 held-out evaluation genes;
3. z-score every gene within each section (or each embryo, see below);
4. run one joint PCA on the stacked spots of both embryos, keeping 30 components.

Both species therefore share one coordinate system. The cross cost is the Euclidean distance in this space, divided by its 0.99 quantile.

**Training uses the real spots only.** Interpolated points are never feature targets, or the field would learn the McCann interpolant back. Held-out genes and labels are never used either.

**Losses:**
1. **Features:** ‖f(x_i) − e_i‖² at the real spots.
2. **Depth.** For a displacement pair (i in section k, j in k + 1) and a point x(t) on the segment between them, the derivative of f along the segment must match the finite difference:

   $$\|J_f(x(t))\,(x_j - x_i) - (e_j - e_i)\|^2$$

   This is the only depth information in the data. Without it, f would be either flat or oscillating along z.
3. **Occupancy.** Positives are the spots and points near the segments, so each section's outline carries across the gap. Negatives are in-plane points outside the tissue and gap points far from every segment.
4. **Smoothness:** a penalty on ‖J_f‖, with a lower Fourier bandwidth in z than in-plane.

**The losses set how well f fits the spots (§7c).** With smoothness and the full depth loss, f explains only 50–70 % of the feature variance at its own training spots, whatever the size of the net (up to 512×6). Dropping the smoothness penalty and weakening the depth loss tenfold raises this to 86–93 %. The field then violates the depth constraint more, and whether it still predicts missing sections is untested.

**The domain is clipped to [z_first, z_last].** There are no data beyond the outermost sections. The two species then cover different depth ranges, which stage 2 must allow (§1e).

**Per-section or per-embryo z-scoring.** Z-scoring per section removes batch effects between sections, but also real depth trends: a gene that rises towards lateral sections becomes flat. Z-scoring per embryo (`features.zscore_by: embryo`) keeps those trends, and the batch effects with them. G1 picks one (§3). A uses the same features, so that Δκ compares representations, not features.

### 1d. Stage 1: sampling the volume

- **Points:** quasi-uniform (Sobol) in the clipped box, kept where o(x) > 0.5, until N points. The density is the same along z as in-plane.
- **N:** the same for both embryos of a pair; a first guess is 5–10k (1–2 × `MAX_OBS`), so one solve costs about one large 2D solve.
- **Masses** are equal, and the **features** are f at each point.

### 1e. Stage 2: aligning the volumes (across species)

One 3D ↔ 3D solve per pair, with the cross cost from the field features:
- **Unbalanced on both sides, with the same τ.** Depth ranges overlap only partly in most pairs (§4), and a balanced coupling would squeeze one range onto the other. Equal τ keeps the solve symmetric, so P and Pᵀ serve both directions. Example: in X4 the mouse stack has no sections between L 0.12 and 0.75, exactly where the human stack lies.
- **The z-scale.** Distances are max-normalised, so size does not matter, but the ratio of z to in-plane does. The first runs use the variogram z. A scan over a z-stretch s of one side, chosen by the objective, comes later, with its own regret check (§2e).
- **The aligner ladder** (§5b): FGW on Euclidean distances; Riemannian FGW on shortest paths in a 3D kNN graph; MGW with the pull-back metric of f. The field already is the φ that MGW would otherwise train. On X4 the three score the same on the field volume (0.288–0.290, §7d): the representation limits them, not the aligner.
- **The unbalanced solve.** In every 3D run so far, OTT's unbalanced correction returned NaN, and the solve was retried without it. That retry rejects 46–63 % of the mass of each side, far more than τ = 0.95 implies. This is unresolved.

**Point-cloud ablation.** The same stage 2 on the raw stack points, with real features and masses proportional to each section's slab (half the gap to each neighbour). It isolates what the field adds over weighting the real points by volume.

### 1f. Back to the real spots

Labels exist only on the real spots, so scoring happens there:
- **Query side (A_q):** each real query spot spreads over its k nearest samples in its own volume.
- **Target side (B_t):** each target sample takes the organ composition and held-out genes of its k nearest real target spots.
- **Score** A_q P_vol exactly like a 2D coupling (`Task.score` with 3D target coordinates), one row per (direction, query section), so the rows compare one-to-one with A's.

A target sample in a gap takes the labels of the nearest real spots. This is used for evaluation only; the field never sees labels.

---

## 2. Success metrics

- Selection never sees labels.
- Every score is the mean of the two directions.
- Every score is reported between a floor (§2g) and a same-species ceiling (§2f), per query section, per pair, and pooled.

### 2a. The headline metric: organ-balanced Cohen's κ

**In one sentence:** *averaged over organs, how far the alignment moves from chance (0) towards sending every organ's tissue to the same organ in the other embryo (1).*

**The agreement table C.** C_ab is the mass of P that goes from query organ a to target organ b, at full resolution. It is a soft confusion matrix.

**Three numbers from C:**
1. **Organ match score (OMS):** the mean over the K shared organs of $m_o = C_{oo} / \sum_b C_{ob}$, the share of an organ's mass that lands on the same organ.
2. **Organ-balanced κ (headline):** $\kappa_{bal} = (\mathrm{OMS} - 1/K)\,/\,(1 - 1/K)$. This is Cohen's κ after giving every query organ equal mass, i.e. the soft version of chance-adjusted balanced accuracy.
3. **Standard Cohen's κ** on the unscaled C: reported, not ranked.

**Which organs count (K):** the organs of the query section present anywhere in the *target embryo* (`metrics.presence: embryo`). A 2D target section that lacks an organ scores 0 on it. Otherwise A could score higher by picking sections that leave hard organs out.

**Why this metric:** 0 is chance and 1 is perfect for any K; every organ counts equally (standard κ is dominated by large organs); there is no arg-max, so a blurred coupling loses score; and every pipeline outputs a P on real query spots.

**How it ranks pipelines,** paired against the reference (A with FGW, on the same features):
- **Δκ:** the candidate's κ_bal minus the reference's, per pair; summarised as the median over X1–X5.
- **Wins:** pairs where Δκ exceeds the noise margin (2 × the SD of κ_bal over seeds on Dev).

**A candidate advances** when its median Δκ > 0, it wins on ≥ 3 of 5 pairs, and it passes the guardrails:
- *selection works:* regret ≤ 0.05 for every choice made by an objective (§2e);
- *a plausible map:* ≤ 5 % fold-overs (points whose surface normal flips against the mean);
- *the negative control X6* (no depth overlap): human queries land at the lateral end of the mouse embryo, mouse queries at the medial end of the human embryo.

For the paper, report the ceiling share κ_bal / κ_bal(ceiling) next to Δκ.

### 2b. Supporting metrics (need labels)

| Metric | Measures | Column |
| --- | --- | --- |
| Label transfer accuracy and balanced accuracy | Spot-level correspondence | `acc_hard`, `bal_acc_hard` |
| Per-organ recall m_o | Which organs fail | `recall` |
| Organ coverage (≥ 10 % of mass reaches the counterpart) | Whether whole organs go missing | `coverage` |
| Boundary-tolerant accuracy (label within radius r) | Organ boundaries that shift across species | new |
| Held-out ortholog co-expression r, vs a shuffled baseline | Expression agreement outside the embedding | `coexpr_r`, `coexpr_r0` |

### 2c. Plausibility of the map (label-free)

- **Smoothness:** fold-overs, local Jacobian condition number, kNN preservation.
- **GW distortion** of P_vol.
- **Sharpness:** effective number of partners per spot, exp(row entropy).
- **Rejected mass** of the unbalanced solve, and where in depth. In X4 it should sit where the other embryo has no sections.

### 2d. Consistency (label-free)

- **Parallel sections:** a query embryo's sections should map to images that share a normal and keep their order and spacing. The volume pipeline gets this almost by construction, so the check catches a stage 2 that bends the volume.
- **Direction symmetry:** with equal τ, P(H→M) = P(M→H)ᵀ by construction; report it whenever τ differs.
- **Cycle consistency** (H→M→H) and **stability** over seeds, N, τ and field settings.

### 2e. Does selection pick well?

The volume pipeline chooses only τ, ε and the z-stretch s; A also picks a target section per query. For every choice made by an objective, report the Spearman correlation between objective and κ_bal over the candidates, and the **regret**: the κ_bal of the label-best candidate minus that of the objective-best one. A choice with regret > 0.05 is fixed by hand.

### 2f. Ceilings: same species

- **W (within embryo):** each section against the other sections of its own embryo, for A.
- **S (odd/even split):** one embryo's odd against its even sections, as two volumes (G2). The true map is known.
- **Between embryos of one stage:** H1–H3 and M3 serve the volume pipelines; M1 and M2 have one query section and serve A only.

### 2g. Floors and nulls

- **For A:** a random real section, the laterality partner (closest L), rigid placement only, feature-only OT, geometry-only GW, SCOT, the organ-frequency prior (κ_bal = 0), and the label oracle (headroom).
- **For the volumes:** feature-only OT and geometry-only GW between the volumes, and the point-cloud ablation, which the field must beat.

**A possible claim for the paper.** Across the series, aligning field-sampled volumes raises κ_bal over the objective-chosen 2D alignment on most pairs, closes a stated share of the gap to the same-species ceilings, and passes the guardrails.

---

## 3. Within-species gates for the field

*Not in the code since the slim harness (§6a). Kept as the design for the held-out checks of §7c.*

Before any cross-species run, the field is tested within one species, which isolates the representation from the species mismatch. Selection uses label-free scores only; label agreement is reported.

**G1. Held-out section.** For every interior section k, build the stack and field without k, and predict k at its own spots.
- **Compared with:** McCann interpolation between k − 1 and k + 1; copying the nearer neighbour; copying it through the coupling.
- **Selection score:** held-out-gene r at k's spots. Feature error and label agreement are reported.
- **Pass:** beats the nearer-neighbour copy by more than the seed noise, and is no worse than McCann.
- G1 also picks the z-scoring (section or embryo) and the field's hyperparameters.

**G2. Odd/even split.** Fit a field and a volume to the odd and to the even sections of one stack, and align them with stage 2. The true map is the identity.
- **Scores:** median 3D and z error of the barycentric map, in section spacings; *plane locking*, the share of mapped z within ±10 % of a spacing of the other half's sections (≈ 20 % by chance).
- **Variants:** *common z* (both halves keep the full stack's z: the representation alone) and *re-estimated z* (each half estimates its own, larger gaps).
- **Pass:** median 3D error < 0.5 spacings, locking ≤ 1.5 × chance, and better than the point-cloud ablation.
- Long stacks: CS12-13_E2 (8), CS14-15_E2 (10), CS17_E1 (10), CS18_E1 (6), CS23_E2 (8; S2 and S4 excluded, no raw counts), and E16.5_E2 (13), the only long mouse stack. The other mouse stacks have ≤ 5 sections.

**G3. Stack building.** Stage 1 is scored on W, G1 and G2. This is where the registration methods of §1b and §5a compete.

The thresholds are first guesses, to be revisited once the noise on Dev is known.

---

## 4. Benchmark series

Frozen in [`bench/configs/series.yaml`](../bench/configs/series.yaml). It spans stage (early, mid, late), depth overlap (good, partial, none), stack shape, and level (within embryo, between embryos, across species). L is the laterality score from [`laterality/laterality_scores.csv`](../laterality/laterality_scores.csv) (`L_pair` for HESTA, raw L for MOSTA); stage windows follow [`hesta_mosta_slide_pairs.md`](hesta_mosta_slide_pairs.md).

### 4a. Cross-species pairs

| # | Human (sections; L range) | Mouse (sections; L range) | Stage | Depth overlap | Why it is in the set |
| --- | --- | --- | --- | --- | --- |
| X1 | CS12-13_E2 (8; 0.33–0.53) | E9.5_E2 (4; 0.48–0.59) | early | partial | Smallest sections (5–12k spots): the fast development pair |
| X2 | CS14-15_E2 (10; 0.05–0.79) | E10.5_E1 (3; 0.07–0.35) | early | mouse inside human | The human stack spans midline to lateral; the only good midline anchor |
| X3 | CS17_E1 (10; 0.47–1.00) | E11.5_E1 (4; 0.71–0.77) | mid (ambiguous window) | mouse inside human | A thin mouse volume (L range 0.06) |
| X4 | CS18_E1 (6; 0.33–0.67) | E12.5_E1 (5; 0.10–0.81) | mid | human inside mouse, but mouse has a gap 0.12–0.75 | The development pair; the mouse volume in the gap is mostly prior |
| X5 | CS23_E2 (8; 0.39–0.79) | E14.5_E1 (5; 0.47–0.83) | late | good | Late stage, largest sections (≈ 0.7M spots), the best depth match |
| X6 | CS20_E2 (4; 0.27–0.76) | E13.5_E1 (4; 0.04–0.21) | late | none | Negative control: the guardrail of 2a |

### 4b. Same-species ceilings

| # | Pair | Level | Matches | For |
| --- | --- | --- | --- | --- |
| W | Leave-one-out of every stack in X1–X6 | within embryo | all | A |
| S | Odd/even split of the long stacks (§3, G2) | within embryo | X1–X5 | volumes |
| H1 | CS12-13_E1 (4) ↔ CS12-13_E2 (8) | between embryos, human | X1 | A, volumes |
| H2 | CS14-15_E3 (4) ↔ CS14-15_E2 (10) | between embryos, human | X2 | A, volumes |
| H3 | CS23_E1 (3) ↔ CS23_E2 (8) | between embryos, human | X5 | A, volumes |
| M1 | E9.5_E1S1 → E9.5_E2 | between embryos, mouse | X1 | A (one query section) |
| M2 | E12.5_E2S1 → E12.5_E1 | between embryos, mouse | X4 | A (one query section) |
| M3 | E14.5_E2 (2) → E14.5_E1 | between embryos, mouse | X5 | A, volumes (check the organ labels first) |

Optional: E16.5_E1 ↔ E16.5_E2 (5 vs 13 sections), the only mouse volume ↔ volume ceiling, with no human partner.

### 4c. Optional timing controls

CS17_E1 ↔ CS18_E1 and E11.5_E1 ↔ E12.5_E1 (same species, adjacent stages). With X3 and X4 they separate timing from species. Diagnostics only.

### 4d. Tiers and compute

| Tier | Pairs | Use |
| --- | --- | --- |
| Dev | X1, X4 (+ W and S of their embryos) | Every code change, the noise margin, the gates |
| Screen | X1–X5 (ranked), X6 (guardrail), H1, H2, M2, W, S | Every pipeline that passes Dev |
| Full | Screen + H3, M1, M3, timing controls | Finalists and the paper's numbers |

A needs n_q × n_t 2D solves per pair (30 for X4, about 1 h in both directions). The volume pipeline needs one field per embryo and seed (minutes on a GPU) and one 3D solve per pair and setting, so it should cost no more than A.

---

## 5. Methods for the two stages

### 5a. Stage 1: registration within an embryo

"Rigid" means each section moves only by a rotation (or a reflection) and a translation: no scaling, shearing or warping, so distances within a section are preserved.

| Method | Transform | Exists | Note |
| --- | --- | --- | --- |
| moscot FGW + OT-ICP | rigid | `bench.stack` (current) | The baseline (§1b) |
| FGW + one Procrustes fit | rigid | `stack.coupling: moscot_procrustes` | Ablation: what OT-ICP adds (§1b). X4: 0.530 against 0.581 |
| MGW + OT-ICP | rigid | `stack.coupling: mgw_icp` | Learned-metric coupling (§1b). X4: same as moscot + OT-ICP |
| PASTE2 | rigid | no | Partial FGW, for partly overlapping neighbours |
| Spateo (morpho alignment) | non-rigid | no | Probabilistic, uses expression; builds 3D models |
| MOSAICField | affine, then a neural deformation field | no | Designed for serial slices; preliminary code, GPU |
| STalign | non-rigid (LDDMM) | no | Needs features rasterised into image channels |

All are scored on G3. Non-rigid methods also change the couplings that the depth loss uses, so a method can win on W and still lose on G1.

### 5b. Stage 2: aligning the volumes (the ladder)

| Method | Intra-volume cost | Exists |
| --- | --- | --- |
| **FGW** | Euclidean distance | `V_fgw` (`solvers.fgw`, any dimension; τ for unbalanced) |
| **Riemannian FGW** | Shortest path in a 3D kNN graph | `V_rfgw` (`solvers.knn_geodesic`) |
| **MGW** | Shortest path under the pull-back metric of f | `V_mgw` (`solvers.pullback_geodesic` on the field's feature head) |

- FGW → Riemannian FGW tests whether *following the tissue* helps (tolerance to bending and posture).
- Riemannian FGW → MGW tests whether *the learned metric* helps (tolerance to differences in organ size).
- External: Spateo and STalign in 3D, only after §5c.

### 5c. Adopting external tools

An external tool gets adaptation work only if it passes three gates:
1. **Triage on Dev, with minimal conversion:** the tool's own preprocessing and defaults, our ortholog features (or the first PCs rasterised, for image tools), no tuning.
2. **Performance:** within the noise margin of the best in-house method in its role (G3 for stage 1, median Δκ for stage 2), or clearly better on one Dev pair.
3. **Pipeline fit:** installable and maintained, runs on our hardware within the budget of §4d, and consumes and returns what the pipeline passes around.

A tool that fails triage is reported with its setup and not pursued.

---

## 6. Proposed order of experiments

The harness was cut down in October 2026 to the two questions of the status note (§0). All pipelines live in [`bench/configs/bench.yaml`](../bench/configs/bench.yaml):

| Question | Pipelines |
| --- | --- |
| 2D tools | `A_fgw`, `A_mgw` |
| Within-embryo tools | `S_moscot_icp`, `S_moscot_procrustes`, `S_mgw_icp` |
| Between-embryo tools | `V_fgw`, `PC_fgw`, `V_rfgw`, `V_mgw` |

1. **All pipelines on X4, seed 0 (done, §7a–§7b, §7d).**
   - 2D: FGW = MGW.
   - Within the embryo: moscot + OT-ICP is the best, and the cheapest.
   - Between the embryos: the stack points beat 2D by +0.136. The field volumes lose to 2D.
2. **Why the field volumes lose (done, §7c).** The field's features, not the points between the sections, cost most of the score. A grid over size, bandwidth and steps, and a 512×6 field, raise the fit only a little. Relaxing the smoothness and depth losses raises it a lot and lifts the volume above 2D.
3. **Next: the field's regularization.**
   - Held-out sections for the relaxed field (L_noreg): does it still predict missing sections?
   - Smoothness and depth loss separately: which of the two sets the cap?
   - Only then XL on LRZ (`bench/scripts/lrz_field_scale.sbatch`).
4. **Next: the unbalanced solve.** Find ε and τ at which OTT's unbalanced correction does not return NaN, and check where the rejected mass lies in X4.
5. **Dev tier** (X1 and X4, seeds 0–2) with the chosen field, for the seed noise. Then Screen (X1–X5).
6. **Full tier:** the best stage 1 with the best stage 2. These are the benchmark numbers.

### 6a. Implementation choices for steps 2–3

First choices, recorded so results can be read against them. Details are in the module docstrings.

**Field (`field.py`).**
- MLP on anisotropic Fourier features (64 frequencies; σ_z = 1 cycle per stack radius) plus raw coordinates; two heads (f, occupancy logit).
  - First version: width 128, depth 3, σ_xy = 4, 3000 steps.
  - Since the grid on X4 (§7c): width 256, depth 4, σ_xy = 8, 20 000 steps. This is the best held-out R² of 12 settings, the label-free choice.
  - Optional residual blocks (`field.residual`, `field.layernorm`) for deeper nets, and `field.tf32` for A100 and H100.
- Coordinates centred and divided by the stack's in-plane RMS radius (one scale for all axes); features divided by their RMS.
- Loss weights: features 1, depth 1, occupancy 0.5, smoothness 1e-3. Displacement pairs are sampled by mass, with equal total weight per gap.
- Occupancy: negatives farther than 2 spot spacings from tissue or segments; positives jittered in-plane by 0.5 spacings.

**Sampling.** Scrambled Sobol in the in-plane bounding box + 2 spacings, times [z_first, z_last]. z and xy share one unit, so the density is isotropic. Sample spacing h = (box volume × acceptance / N)^{1/3}.

**Stage 2 (`volume.py`).**
- Cross cost: `Task.cross_cost`, scaled from the *real* features.
- τ = 0.95 on both sides (ρ ≈ 0.1 at ε = 5e-3). If OTT's unbalanced correction returns NaN (small ε, τ ≤ 0.9), the solve drops the correction and records `ugw_fallback`.
- New aligners dispatch on the pipeline's `aligner` key.

**Back to the spots.**
- A_q: 8 nearest samples, Gaussian weights at the sample spacing. B_t: 3 nearest real target spots, 1/3 each.
- Point cloud: sections subsampled in proportion to their spots if needed; slab masses floored at 5 % of the mean (end sections get their inner half only).
- Only κ_bal is stored (`scores.csv`), plus the transferred organ per query spot (`transfer.npz`) for the figure. The rejected mass is logged per solve.

**Removed with the slim harness (October 2026).**
- **G1 and G2** (`gates.py`), the extra columns of the volume pipelines and the synthetic pair (`synthetic.py`). They remain in the git history (commit `c6a8011` and earlier).
- **The held-out check survives as a diagnostic.** `bench/scripts/field_scale.py` and the field grid (§7c) fit fields without one interior section k: the other sections keep their full-stack positions, and k − 1, k + 1 are joined by their cached direct coupling. They then score R² of f at k's spots, against copying the nearer neighbour section.

---

## 7. Results so far (X4, seed 0)

All numbers are from one pair (X4: CS18_E1 ↔ E12.5_E1) and one seed, run on tueilnt-sim14 (RTX 4080 Super), October 22–24, 2026. They carry no seed noise yet, so differences below about 0.01–0.02 should not be read. κ_bal is the mean of both directions. Reports and tables:
- `data/bench/reports/dev/` (the benchmark);
- `data/bench/diag/field_grid_1022.csv` (the field grid);
- `data/bench/reports/field_scale/X4_seed0.csv` (the field sizes).

### 7a. 2D ↔ 2D

| Tool | κ_bal | human stack | mouse stack |
| --- | --- | --- | --- |
| FGW (`A_fgw`) | **0.303** | 0.287 | 0.319 |
| MGW (`A_mgw`) | 0.302 | 0.286 | 0.317 |

- The two tie. MGW costs twice the time: one φ per section, about 2.5 min each.
- For all six human query sections, FGW selects the same mouse section, E12.5_E1S3.

### 7b. 3D, within the embryo (coupling neighbouring sections)

| Tool | CS18_E1 | E12.5_E1 | Mean |
| --- | --- | --- | --- |
| moscot FGW + OT-ICP (`moscot_icp`) | 0.561 | 0.601 | **0.581** |
| moscot FGW + one Procrustes fit (`moscot_procrustes`) | 0.479 | 0.581 | 0.530 |
| MGW + OT-ICP (`mgw_icp`) | 0.561 | 0.601 | 0.581 |

- OT-ICP is worth about 0.05. MGW's start changes nothing after OT-ICP.
- `stack.coupling` stays `moscot_icp`. The report marks `mgw_icp` as best, but only by a 1e-5 difference.

### 7c. Why the field volumes lose: the field's features

The first runs (§7d) put the field volumes below 2D (0.268 against 0.303), but the stack points far above (0.439). The diagnostics separate the representation's parts:

1. **Same points, other features.** Stage 2 on the stack points with the field's f in place of the measured features drops κ_bal from 0.439 to 0.295. The field's features cost about 0.14 of the 0.17 gap. Sampling the gaps costs only about 0.03.
2. **The first field underfits its own spots.** It explains 37–67 % of the variance per section, and its output keeps 43–62 % of the measured spread.
3. **As a prior across gaps it works.** On every interior held-out section (7 of 7), the field beats copying the nearer neighbour section. Held-out R² is 0.08–0.45 against −0.64 to 0.43.
4. **Size, bandwidth and steps help little.** The grid covered 12 settings: 3000 or 20 000 steps × 128×3 or 256×4 × σ_xy 4, 8 or 16.
   - Fit R² 0.46 → 0.60, volume 0.243 → 0.293.
   - Held-out R² stays flat at 0.280–0.296.
   - At 3000 steps a wider bandwidth hurts; it needs more training. The winner by held-out R² (256×4, σ_xy 8, 20 000 steps) is now the default.
5. **A much larger net hits a plateau.** L is a 512×6 residual net with 256 frequencies, 100 000 steps and 1.86M parameters, trained in 2.3 h per field.
   - Fit 0.662; κ_bal on the points 0.384, on the volume 0.292.
   - The feature loss stalls (0.40 → 0.39 between 50k and 100k steps).
6. **The regularizers set the cap.** L_noreg is the same net with no smoothness penalty and a depth loss of 0.1 instead of 1. It trains in 52 min per field.

| Field | Fit R² (CS18 / E12.5) | Spread kept | κ_bal points + field | κ_bal volume |
| --- | --- | --- | --- | --- |
| first (128×3, 3000 steps) | 0.466 / 0.567 | 0.47 / 0.55 | 0.295 | 0.268 |
| grid winner (256×4, σ_xy 8, 20k) | 0.543 / 0.646 | 0.52 / 0.60 | 0.349 | 0.290 |
| L (512×6 residual, 100k) | 0.603 / 0.722 | 0.56 / 0.66 | 0.384 | 0.292 |
| **L_noreg** (no smoothness, depth 0.1) | **0.863 / 0.931** | **0.76 / 0.85** | **0.453** | **0.317** |

Without the regularizers:
- The fit jumps to about 0.9.
- On the stack points, the field's features now beat the measured ones (0.453 against 0.439), perhaps because f averages out noise between neighbouring spots.
- The volume beats 2D for the first time (0.317 against 0.303).

The price:
- The depth loss rises from 0.36 to 1.29 (CS18_E1) and from 0.26 to 0.89 (E12.5_E1), so f between the sections is less constrained.
- The solve rejects more mass (0.56–0.63).

Held-out R² of L_noreg is not measured yet. It decides whether the gain is real or only memorization of the measured sections.

### 7d. 3D, between the embryos (volume ↔ volume)

| Tool | First field | Grid-winner field | Δ against 2D (grid-winner) |
| --- | --- | --- | --- |
| FGW on the stack points (`PC_fgw`, no field) | **0.439** | **0.439** | **+0.136** |
| FGW on the field volume (`V_fgw`) | 0.268 | 0.290 | −0.014 |
| Riemannian FGW (`V_rfgw`) | 0.267 | 0.290 | −0.014 |
| MGW on the field's metric (`V_mgw`) | 0.266 | 0.288 | −0.015 |
| *FGW on the volume of L_noreg (§7c)* | | *0.317* | *+0.014* |

- The three aligners tie on every field, so the field limits them, not the aligner.
- Every 3D solve fell back to the unbalanced solve without OTT's correction (§1e). The rejected mass is 46–48 % (field volumes) and 70 % (stack points).

**What this means so far:**
- Aligning whole embryos in 3D pays off on X4 when the measured spots are aligned directly: +0.136 over the best 2D alignment.
- A field-sampled volume helps only once the field fits the data closely. Even then it stays well below the stack points (0.317 against 0.439).
- Before the field approach earns its cost, it must show that it still predicts missing sections with the regularizers relaxed (§6, step 3).

---

## 8. Open questions

- **Beyond the outermost sections.** Extend the volume by half a gap? Is there a principled outer boundary, such as the midline for medial stacks?
- **The z-scale.** Is a z-stretch chosen by the stage-2 objective trustworthy, or should z come from |ΔL| or metadata?
- **The depth loss.** It and the smoothness penalty cap how well the field fits the spots (§7c). How weak can they be before the field stops predicting missing sections? Which of the two matters? Do entropic couplings or their peaks make better pairs?
- **Evaluation in the gaps.** Is nearest-spot labelling fair to the volumes in stacks with large gaps (X4)?
- **N and τ.** Both trade resolution and robustness to partial overlap against cost. With OTT's correction failing, the effective rejection is far above τ = 0.95 (§1e).
- **Field or points?** On X4 the stack points beat every field volume (§7d). Does the field add anything beyond filling the gaps, and is filling them worth it?
- **Non-rigidity.** With MGW's metric and a non-rigid stage 1, when is a good κ_bal the warp's doing rather than the correspondence's?
- **Thresholds.** Should regret 0.05, 5 % fold-overs, ≥ 3 of 5 wins and the gate thresholds be set from Dev, like the noise margin?
- **Unit of evaluation.** Labels are noisy at boundaries and organs are not exact homologues. Is organ-level κ_bal enough?
