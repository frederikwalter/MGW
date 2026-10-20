# Cross-species alignment in 3D: volumes from serial sections

Discussion notes. The goal is a spot-level coupling between human (HESTA) and mouse (MOSTA) embryos that is good enough to serve as a benchmark in a methods paper. The only ground truth we trust is the organ labels. The notes build on [`demos/human_mouse_align_3d.ipynb`](../demos/human_mouse_align_3d.ipynb), [`mgw/stack3d.py`](../mgw/stack3d.py) and the benchmark harness in [`mgw/bench`](../mgw/bench) ([`bench/README.md`](../bench/README.md)).

**The approach in one paragraph.** Each embryo is treated as a continuous volume, not as a pile of 2D sections:
1. **Stage 1, within species.** The serial sections are registered into a stack. A neural field is then fitted over the stack: features and tissue occupancy at every point in 3D. Points are sampled evenly through the volume, and the field gives their features.
2. **Stage 2, across species.** The two sampled volumes are aligned 3D ↔ 3D.
3. **Evaluation.** The coupling is carried back to the real spots and scored against the organ labels. It is compared with the best 2D ↔ 2D alignment of real sections.

The notes run in this order:
- §0 states the problem and the facts that shape the approach;
- §1 describes the approach stage by stage;
- §2 defines success;
- §3 sets the within-species gates that the field must pass first;
- §4 chooses a representative series of pairs;
- §5 assigns alignment methods to the two stages;
- §6 proposes an order of experiments;
- §7 lists open questions.

**Decisions so far:**
- **The representation is a volume.** It is modelled as a field and computed on as sampled points (§1a).
- **Stage mismatch is handled by choosing pairs** (§4). Interpolating between stages comes later.
- **No supervision beyond the evaluation labels.** Fields, methods and parameters are chosen without labels, landmarks or organ matches. The within-species gates (§3) select on label-free scores.
- **Results are symmetric.** Every pipeline runs in both directions, and every score averages the two.
- **The 2D ↔ 2D alignment is the reference.** Strategy A (each query section against the best real target section, chosen by the FGW objective) is what the volumes must beat (§2a).
- **External tools are adapted only if they earn it** (§5c).
- **One headline metric,** organ-balanced Cohen's κ, ranks pipelines (§2a).

---

## 0. The problem and where we stand

**Inputs.** Each embryo has 3–13 sagittal sections at unknown spacing. Both full databases are available.

**Output.** A coupling P between the spots of the two embryos. Where one embryo sits inside the other is a means to that end, not the goal.

**Why it is hard:**
1. *Depth.* Two sections are cut at different mediolateral positions. In 2D, the difference shows up as a deformation.
2. *A coarse stack.* There are only a few sections, with gaps estimated from the in-plane variogram. Of those gap estimates, 3 of 5 in CS18_E1 and 2 of 4 in E12.5_E1 are saturated, so they are only lower bounds.
3. *Species mismatch.* Proportions, size and developmental timing differ. The stage correspondences are windows, not matched stages. The label sets are only harmonised, not identical.
4. *Unsupervised selection.* Methods and parameters must be picked by an objective. The labels only score the result.

**Facts from the 3D notebook runs that shape the approach:**
- **Nothing lies between the sections except a prior.** In a leave-one-out test, interpolating a held-out section from its two neighbours along their coupling (McCann displacement interpolation) beat copying the nearer neighbour by only +0.00 to +0.04 in label agreement. A field cannot add information between the sections either. It can only be a better prior, so it has to prove that on held-out sections (§3, G1).
- **The objective and the labels can disagree.** Searching harder over an objective that ranks candidates differently from the labels made things worse: a lower FGW objective (0.172 vs 0.179) came with a worse label accuracy (0.53 vs 0.69). Every unsupervised choice therefore gets a regret guardrail (§2e).
- **z is the least reliable axis.** The gaps are lower bounds, and a field inherits any spacing error. Euclidean distances are not invariant to stretching one axis, so the z-scale is an explicit parameter (§1e).

**Status.** Step 1 of §6, the harness, is implemented in `mgw.bench`: metrics, series, the 2D reference A with its nulls and ceilings, and the report and viewer. This revision changes it as follows:
- K now counts the organs of the whole target embryo (§2a);
- fold-overs work for 3D targets;
- genes can be z-scored per embryo instead of per section (§1c);
- the pipelines of earlier strategies are gone.

---

## 1. The approach

### 1a. Representation: a field for the model, points for the computation

GW sees a 3D representation in only two ways:
1. **The measure:** where the mass sits. In a raw stack, all mass sits on a few planes, at depths that differ between the two embryos.
2. **The path domain:** which routes geodesic distances can take. In a stack, they can only cross between sections along a few edges.

A volume fixes both, but only on the computational side:
- **The model is a field:** expression and occupancy over all of 3D. It can be evaluated anywhere, at any density, and it can be differentiated, which MGW needs (§1e).
- **The computation stays on points.** OT and GW couple measures. So points are sampled evenly through the field's volume, with equal masses, and both species get the same number of points.

The field does not know more than the sections. Between them it is a smoothness prior, which suits a large organ that changes smoothly (the liver), and organs smaller than a gap stay invisible to any representation. That is why the field must beat the simple priors on held-out sections before it is used (§3).

### 1b. Stage 1: the stack (within species)

This is the existing stack building (`mgw.bench.stack`, cached by `pipelines/stacked.get_stack`):
1. **Order:** the file order for HESTA. For MOSTA, the order is inferred from the FGW objective between all pairs of sections.
2. **Registration:** FGW between neighbouring sections (moscot), refined by OT-ICP (rotation or reflection), composed into the frame of the middle section and refitted in global sweeps.
3. **z:** the gaps from the in-plane variogram.
4. **Couplings:** the coupling between each pair of neighbouring sections, kept as displacement pairs: the peak of each row and column, weighted by mass.

The couplings are *not* thrown away. They are the only data on how tissue continues from one section to the next, and the field uses them twice (§1c). Other registration methods can replace step 2 (§5a).

### 1c. Stage 1: the field

One field per embryo and seed, in the stack's frame:

$$x = (x, y, z) \;\mapsto\; \big(f(x),\, o(x)\big)$$

- f is the shared ortholog embedding: the same features that the cross cost uses.
- o is the tissue occupancy, in [0, 1].

**Training data: the real spots only.** Training never uses interpolated points as targets, or the field would just learn the McCann interpolant back. It never uses the held-out genes, and never the labels.

**Losses:**
1. **Features at the spots:** ‖f(x_i) − e_i‖² over minibatches of real spots.
2. **Depth from the couplings.** For a displacement pair (i in section k, j in k + 1) and a point x(t) = (1 − t)x_i + t x_j on its segment, the derivative of f along the segment must match the finite difference:

   $$\|J_f(x(t))\,(x_j - x_i) - (e_j - e_i)\|^2$$

   This is the only depth information the data hold. Without it, the z-part of f is unconstrained: spectral bias makes f nearly flat in z, so the depth collapses, while high-frequency encodings let it oscillate.
3. **Occupancy.**
   - *Positives:* the real spots, and points near the segments of the displacement pairs. The couplings carry each section's outline across the gap.
   - *Negatives:* in-plane points outside each section's tissue, and gap points far from every segment.
4. **Smoothness:** a penalty on ‖J_f‖, with a lower Fourier-feature bandwidth in z than in-plane, because the data are much coarser along z.

**The domain is clipped to [z_first, z_last].** Beyond the outermost sections there are no data, so there is no volume. The two species then cover different depth ranges, which stage 2 must allow (§1e).

**Feature z-scoring.** The harness z-scores every gene within each section. That removes offsets between sections, including the real ones: a gene that rises towards lateral sections becomes flat across the stack, and the field loses that depth signal. `features.zscore_by: embryo` keeps the offsets. Section effects of the technology are the risk. G1 decides between the two (§3), and the 2D reference A runs on the same features as the volume, so that Δκ compares representations, not features.

### 1d. Stage 1: sampling the volume

- **Points:** quasi-uniform points (Sobol, or Poisson-disk) in the clipped box, kept where o(x) > 0.5, until N points.
  - N is the same for both embryos of a pair.
  - Sampling is roughly as dense in z as in-plane, so a plain 3D kNN graph has diagonal edges (§1e).
- **Masses:** equal, so the measure approximates a uniform distribution over the tissue volume.
- **Features:** f at each point.
- **Points per embryo:** a first guess is N ≈ `MAX_OBS` to 2 × `MAX_OBS`, about 5–10k. One dense GW solve per pair is then the size of one large 2D solve.

### 1e. Stage 2: aligning the volumes (across species)

Stage 2 is one 3D ↔ 3D solve per pair, between the two sampled volumes, with the cross cost from the field features (scaled once per pair, as in the harness).
- **Unbalanced on both sides,** with the same τ. Depth ranges overlap only partly in most pairs (§4), and a balanced coupling would squeeze one embryo's range onto the other's.
  - Example: in X4 the mouse stack has no sections between L 0.12 and 0.75, which is exactly where the human stack lies. The mouse volume there is mostly prior.
  - With the same τ on both sides, the solve is symmetric: P and its transpose serve both directions, as in A.
- **The z-scale.** Distances are max-normalised, so the overall size does not matter, but the ratio between z and in-plane does. The first runs use the variogram z. A scan over a z-stretch s of one side, chosen by the objective, is the planned extension. It needs its own regret check (§2e).
- **The aligner ladder.** Each step changes one thing (§5b):
  - FGW on Euclidean distances;
  - Riemannian FGW on shortest paths in a 3D kNN graph of the samples (no `stack_graph` needed, since the samples are dense in z);
  - MGW with the pull-back metric of the field f itself. The field already is the φ that MGW would otherwise train.

**Ablation without a field (point cloud).** The same stage 2 is run on the raw stack points instead of the samples:
- each section's points get masses proportional to the thickness of its slab (half the gap to each neighbour);
- the features are the real ones.

This isolates what the field adds over simply weighting the real points by volume.

### 1f. Back to the real spots

Labels exist only on the real spots, so every score is computed there:
- **Query side.** Every real spot of a query section spreads over its k nearest samples in its own embryo's volume, with Gaussian weights at the sample spacing (rows sum to 1). This gives A_q.
- **Target side.** Every target sample takes the bins of its k nearest real target spots. This gives B_t: the samples' organ composition B_t T and held-out genes B_t Y.
- **The coupling scored** for a query section is A_q P_vol, against the target samples. It is scored exactly like a 2D coupling (`Task.score` with 3D target coordinates), one row per (direction, query section). The rows therefore compare one-to-one with A's.

A target sample in a gap takes the labels of the nearest real spots. This is the nearest-section copy, used for *evaluation* only. The field never sees labels.

---

## 2. Success metrics

Four principles:
- Selection never sees labels.
- Every score is symmetric: the mean of the two directions.
- Every score is reported between a floor (a null or simple baseline) and a ceiling (same species, §2f).
- Scores are reported per query section, per embryo pair, and pooled over the series.

### 2a. The headline metric: organ-balanced Cohen's κ

**In one sentence:** *averaged over organs, how far the alignment moves from chance (0) towards sending every organ's tissue to the same organ in the other embryo (1).*

**The agreement table.** For one query section and its target (a real target section in A, the target volume in the volume pipelines):
- C_ab is the mass of P that goes from query organ a to target organ b.
- It is computed at full resolution: every bin carries the row of P of its point.
- C is a "soft" confusion matrix. Arg-max label transfer would turn it into the usual hard one.

**Three numbers come out of C:**
1. **Organ match score (OMS),** the raw number. Each organ o present on both sides gets $m_o = C_{oo} / \sum_b C_{ob}$, the share of its mass that lands on the same organ. OMS is the mean of $m_o$ over the K shared organs.
2. **Organ-balanced κ, the headline:**
   - Rescale every row of C so that each query organ has the same total mass, then apply Cohen's formula.
   - With C restricted to the K shared organs, observed agreement then equals OMS, and chance agreement equals exactly 1/K. Hence $\kappa_{bal} = (\mathrm{OMS} - 1/K)\,/\,(1 - 1/K)$.
   - It is the soft version of chance-adjusted balanced accuracy (`balanced_accuracy_score(adjusted=True)` in scikit-learn). For two organs it is Youden's J.
3. **Standard Cohen's κ** on the unscaled C. It is reported for readers who expect it, but not used for ranking.

**Which organs count (K).** These are the organs of the query section present anywhere in the *target embryo* (`metrics.presence: embryo`), whatever the target is.
- A 2D target section that lacks an organ therefore scores it 0 for that query, instead of dropping it from K.
- Without this rule, A could score higher just by choosing sections that leave hard organs out, and A and the volumes would be scored on different K.

**Per embryo pair:** the mean over all query sections and both directions.

**Why this metric:**
- **Easy to read.** 0 is chance and 1 is perfect, whatever the pair and whatever K.
- **Balanced.** Every organ counts equally. Standard κ is dominated by large organs, and it drops when one class dominates even when agreement is high (the "kappa paradox").
- **No arg-max.** A blurred coupling loses score even when its peak happens to be right.
- **Symmetric,** and computable for every pipeline, because all of them output a P on real query spots.

The Landis–Koch bands are a convention, not a calibration. The yardstick is the same-species ceiling (2f).

**How it ranks pipelines.** The comparison is paired, against the reference pipeline: A with FGW, on the same features.
1. **Δκ:** a candidate's κ_bal minus the reference's on each pair. The summary is the median over the Screen pairs X1–X5.
2. **Wins:** the number of pairs on which Δκ exceeds the noise margin, 2 × the SD of κ_bal over seeds on the Dev tier.

**A candidate advances** when:
- its median Δκ is > 0;
- it wins on ≥ 3 of the 5 pairs;
- it passes the guardrails.

**Guardrails (pass or fail, not ranked):**
- *Selection works:* regret ≤ 0.05 in κ_bal over every choice made by an objective (2e).
- *A plausible map:* ≤ 5 % fold-overs. A query section's image in the target volume is a surface, and a point folds over when its normal flips against the mean normal.
- *The negative control stays honest* (X6, whose depth ranges do not overlap):
  - the mass of each query section must land in the matching end of the other embryo's depth range, measured as the mass-weighted L of the target spots it reaches;
  - human queries must land at the lateral end of the mouse embryo, and mouse queries at the medial end of the human embryo.

**For the paper,** the ceiling share κ_bal / κ_bal(ceiling) is reported next to Δκ.

### 2b. Supporting metrics: correspondence (need labels)

| Metric | What it measures | Exists |
| --- | --- | --- |
| Label transfer accuracy and balanced accuracy, at reduced points and at full resolution | Spot-level correspondence | `metrics.label_scores` (`acc_hard`, `bal_acc_hard`) |
| Per-organ recall $m_o$ | *Which* organs fail, e.g. small or species-specific ones | `recall` column |
| Organ × organ matrix C | Coarse correspondence; the basis of κ_bal | stored per candidate |
| Organ coverage: share of query organs whose counterpart receives ≥ 10 % of their mass | Whether whole organs go missing | `coverage` |
| Boundary-tolerant accuracy: correct if the label occurs within radius r of the target point | Boundaries that sit in slightly different places across species | new |
| Held-out ortholog co-expression r, against the shuffled baseline | Expression agreement on genes outside the embedding | `coexpr_r`, `coexpr_r0` |

### 2c. Plausibility of the map (label-free)

- **Smoothness:** fold-overs and the condition number of the local Jacobian (2D or 3D targets), and the share of each spot's k nearest neighbours that stay neighbours after mapping.
- **GW distortion** of P_vol, reported on its own.
- **Sharpness:** the effective number of partners per spot, exp(entropy of the row of P).
- **Rejected mass** of the unbalanced solve: how much is left unmatched, and where in depth. In X4 it should sit in the depth ranges that the other embryo lacks.

### 2d. Consistency (label-free)

- **Parallel sections.** A query embryo's sections were cut in parallel, so their images in the target volume should share one normal, keep their order (|ρ| against the query's own z) and keep their spacing up to one scale.
  - Unlike a per-section placement, the volume pipeline gets this almost by construction, since the query's sections are part of its own volume. The check catches a stage 2 that bends the volume.
- **Direction symmetry:** with equal τ on both sides, P(H→M) = P(M→H)ᵀ by construction. Report it whenever τ differs.
- **Cycle consistency:** after H→M→H, how far a spot lands from where it started, in units of the section radius.
- **Stability:** variance over seeds, N, τ and the field's hyperparameters.

### 2e. Does selection pick well?

The volume pipeline makes fewer choices than A, which picks a target section for every query. Its choices are the stage-2 parameters (τ, ε, the z-stretch s). For every choice made by an objective, report:
- the Spearman correlation between the objective and κ_bal over the candidates;
- the **regret:** the κ_bal of the label-best candidate minus that of the objective-best one.

For A, the candidates are the real target sections of each query (already in the report). A choice with regret > 0.05 is fixed by hand, not left to the objective.

### 2f. Ceilings: same species

- **Within an embryo (W).** For A: each section against the other sections of its own embryo. It measures the 2D machinery when nothing differs but the section.
- **Odd/even split (S).** One embryo's odd sections against its even sections, as two volumes (§3, G2). This is the volume pipeline's within-embryo ceiling. Its true map is known.
- **Between embryos.** Two embryos of one stage. Only embryos with ≥ 2 sections form a volume, so H1–H3 (and M3) serve the volume pipelines. M1 and M2 have a one-section query and remain A's ceilings.
- **The cross-species score as a share of the matching ceiling.**

### 2g. Floors and nulls

For the 2D reference A (implemented):
- a random real section (the expectation over the candidates);
- the laterality partner (closest L);
- rigid placement only (OT-ICP from the selected coupling);
- feature-only OT, geometry-only GW and SCOT;
- the organ-frequency prior (κ_bal = 0 by construction);
- the label oracle, which shows the headroom.

For the volume pipelines:
- the same stage 2 with features only (linear OT between the volumes) and with geometry only (GW without features);
- the point-cloud ablation (1e), which is the floor that the field must beat.

**A possible claim for the paper.** Across the series in §4, aligning field-sampled volumes raises κ_bal over the objective-chosen 2D alignment of real sections on most pairs. It closes a stated share of the gap to the same-species ceilings and passes the guardrails.

---

## 3. Within-species gates for the field

The field is tested within one species before any cross-species run. These tests isolate the representation from the species mismatch. Selection uses only the label-free scores. Label agreement is reported, but not used to choose.

**G1. Held-out section.**
- For every interior section k, build the stack and the field without k. Then predict k at its own spot positions (z from the full stack).
- Compare four predictions:
  - the field;
  - McCann interpolation between k − 1 and k + 1 (`bench.stack.loo_check`);
  - copying the nearer neighbour;
  - copying the nearer neighbour, matched through the coupling.
- **Scores:**
  - held-out-gene r at the section's spots (label-free, *the selection score*);
  - feature reconstruction error;
  - label agreement (reported only).
- **Pass:** the field beats copying the nearest section on held-out-gene r by more than the seed noise, and is no worse than McCann.
- G1 also decides the feature z-scoring (section or embryo) and the field's hyperparameters.

**G2. Odd/even split.**
- Split one stack into its odd and its even sections, and fit a field and a volume to each half.
- Align the two volumes with stage 2. The true map is the identity in 3D.
- **Scores (label-free):**
  - median 3D error of the barycentric map, in units of the mean section spacing;
  - median z error;
  - *plane locking:* the share of mapped z within ±10 % of a spacing of the other half's sections, against the 20 % expected without locking.
- **Two variants:**
  - *common z:* both halves keep the z of the full stack, which tests the representation alone;
  - *re-estimated z:* each half re-estimates its gaps, which are twice as large and saturate more. This tests the gap estimates too.
- Run the point-cloud ablation alongside. G2 shows directly whether raw stacks lock onto their planes and whether the field fixes that.
- **Pass:** the median 3D error is below half a section spacing, there is no plane locking (share ≤ 1.5 × chance), and the field does better than the point cloud.
- Embryos with ≥ 6 sections: CS12-13_E2 (8), CS14-15_E2 (10), CS17_E1 (10), CS18_E1 (6), CS23_E2 (10). Mouse stacks in the series have ≤ 5 sections (a 3/2 split). E16.5_E2 (13) is the only long mouse stack.

**G3. Stack building.** The registration of stage 1 is scored on W (A's within-embryo ceiling) and on G1 and G2. This is where alternative registration methods compete (§5a).

The thresholds are first guesses, to be revisited once the noise on the Dev tier is known.

---

## 4. Benchmark series: a representative subset of HESTA and MOSTA

The series is frozen in [`bench/configs/series.yaml`](../bench/configs/series.yaml). It covers four axes:
- **Stage:** early, mid and late.
- **Depth overlap:** the L ranges of the two stacks overlap well, partly, or not at all.
- **Stack shape:** many against few sections, and a wide against a narrow L range.
- **Level:** within embryo, between embryos of one species, and across species.

L is the laterality score from [`laterality/laterality_scores.csv`](../laterality/laterality_scores.csv): `L_pair` for HESTA, raw L for MOSTA. Stage windows follow [`hesta_mosta_slide_pairs.md`](hesta_mosta_slide_pairs.md). Every embryo of the cross-species pairs has ≥ 3 sections, so every cross-species pair forms two volumes.

### 4a. Cross-species pairs (both embryos as volumes, both directions)

| # | Human (sections; L range) | Mouse (sections; L range) | Stage | Depth overlap | Why it is in the set |
| --- | --- | --- | --- | --- | --- |
| X1 | CS12-13_E2 (8; 0.33–0.53) | E9.5_E2 (4; 0.48–0.59) | early | partial | Smallest sections (5–12k spots): the fast development pair |
| X2 | CS14-15_E2 (10; 0.05–0.79) | E10.5_E1 (3; 0.07–0.35) | early | mouse inside human | The human stack spans midline to lateral; the only good midline anchor |
| X3 | CS17_E1 (10; 0.47–1.00) | E11.5_E1 (4; 0.71–0.77) | mid (ambiguous window) | mouse inside human | A thin mouse volume (L range 0.06) |
| X4 | CS18_E1 (6; 0.33–0.67) | E12.5_E1 (5; 0.10–0.81) | mid | human inside mouse, but mouse has a gap 0.12–0.75 | The development pair; the mouse volume in the gap is mostly prior |
| X5 | CS23_E2 (10; 0.39–0.79) | E14.5_E1 (5; 0.47–0.83) | late | good | Late stage, largest sections (≈ 0.7M spots), the best depth match |
| X6 | CS20_E2 (4; 0.27–0.76) | E13.5_E1 (4; 0.04–0.21) | late | none | Negative control: the guardrail of 2a |

### 4b. Same-species ceilings

| # | Pair | Level | Matches | For |
| --- | --- | --- | --- | --- |
| W | Leave-one-out of every stack in X1–X6 | within embryo | all | A |
| S | Odd/even split of the long stacks (§3, G2) | within embryo | X1–X5 | volumes |
| H1 | CS12-13_E1 (4) ↔ CS12-13_E2 (8) | between embryos, human | X1 | A, volumes |
| H2 | CS14-15_E3 (4) ↔ CS14-15_E2 (10) | between embryos, human | X2 | A, volumes |
| H3 | CS23_E1 (3) ↔ CS23_E2 (10) | between embryos, human | X5 | A, volumes |
| M1 | E9.5_E1S1 → E9.5_E2 | between embryos, mouse | X1 | A (one query section) |
| M2 | E12.5_E2S1 → E12.5_E1 | between embryos, mouse | X4 | A (one query section) |
| M3 | E14.5_E2 (2) → E14.5_E1 | between embryos, mouse | X5 | A, volumes (check the organ labels first) |

E16.5_E1 ↔ E16.5_E2 (5 vs 13 sections) is the only mouse volume ↔ volume ceiling with two real stacks. It has no human partner, so it is optional.

### 4c. Optional timing controls

CS17_E1 ↔ CS18_E1 and E11.5_E1 ↔ E12.5_E1 (same species, adjacent stages). Together with X3 and X4, they separate the effect of timing from the effect of species. They are diagnostics only.

### 4d. Tiers and compute

| Tier | Pairs | Use |
| --- | --- | --- |
| Dev | X1, X4 (+ W and S of their embryos) | Every code change, the noise margin, the gates |
| Screen | X1–X5 (ranked), X6 (guardrail), H1/H2, M2, W, S | Every pipeline that passes Dev |
| Full | Screen + H3, M1, M3, timing controls | Finalists only, and the numbers for the paper |

**Cost.**
- A needs n_q × n_t 2D solves per pair (30 for X4), and the current FGW run takes about 1 h for X4 in both directions.
- The volume pipeline needs one field per embryo and seed (minutes on a GPU) and one 3D ↔ 3D solve per pair and stage-2 setting, which serves both directions. It should cost no more than A.
- Riemannian FGW adds an all-pairs shortest-path computation on N samples; MGW adds the field's Jacobians at the samples.

---

## 5. Methods for the two stages

The aligner is chosen separately for each role:
- **stage 1:** registering neighbouring sections of one embryo;
- **stage 2:** aligning two volumes;
- **reference:** 2D ↔ 2D on real sections.

### 5a. Stage 1: registration within an embryo

| Method | Kind | Exists | Note |
| --- | --- | --- | --- |
| moscot FGW + OT-ICP | rigid (rotation or reflection) | `bench.stack`, current | The baseline |
| PASTE2 | partial FGW (overlap fraction s) | no | Neighbouring sections overlap only partly |
| Spateo (morpho alignment) | non-rigid, probabilistic, with expression | no | Builds 3D models from serial sections |
| MOSAICField | affine OT, then a neural deformation field | no | Designed for serial slices; preliminary code, GPU |
| STalign | LDDMM on rasterised images | no | Needs features rasterised into channels |

They are scored on G3 (W, G1, G2). Non-rigid registration also changes the couplings that the field's depth loss uses, so a method can win on W and still lose on G1.

### 5b. Stage 2: aligning the volumes (the ladder)

| Method | Intra-volume cost | Learned? | Exists |
| --- | --- | --- | --- |
| **FGW** | Euclidean distance | no | `solvers.fgw` (dense, any dimension; τ for unbalanced) |
| **Riemannian FGW** | Shortest path in a 3D kNN graph of the samples | no | `geometry.knn_graph` + Dijkstra; a small wrapper |
| **MGW** | Shortest path under the pull-back metric of the field f | the field | `geometry.pullback_metric_field` + `geodesic_distances_fast` (check 3D); a wrapper that passes f as φ |

- FGW → Riemannian FGW tests whether *following the tissue* helps. Geodesics stay inside the volume and tolerate differences in bending and posture.
- Riemannian FGW → MGW tests whether *the learned metric* helps. Distances measured as expression change may tolerate species differences in organ size.
- **Nulls:** feature-only OT and geometry-only GW between the volumes (2g).
- **External:** Spateo (3D ↔ 3D) and STalign (volume ↔ volume, on rasterised fields). They only enter after the gates of 5c.

### 5c. Adopting external tools

An external tool gets adaptation work only if it seems to perform well *and* could become part of the pipeline. So it passes three gates:

1. **Triage on the Dev tier, with minimal input conversion.**
   - Use the tool's own preprocessing and default parameters, in its role (5a or 5b).
   - Give it the ortholog embedding or the ortholog genes as features, or a generic rasterisation of the first PCs for image tools.
   - No tool-specific tuning.
2. **Performance gate.** In its role, the tool must:
   - come within the noise margin of the best in-house method (on G3 for stage 1, on median Δκ for stage 2); or
   - beat it clearly on at least one Dev pair.
3. **Pipeline gate.** The tool must also:
   - be installable and maintained;
   - run on our hardware;
   - fit the compute budget of 4d;
   - consume and return what the pipeline passes around (coordinates, features, a coupling or a deformation).

A tool that fails triage is reported as run, with its minimal setup, and is not pursued.

---

## 6. Proposed order of experiments

Each step says what tells us to continue.

1. **Harness, reference, ceilings (implemented).**
   - `mgw.bench`: κ_bal with OMS and Cohen's κ, the guardrails, the series, A with FGW in both directions, its nulls, W and the between-embryo ceilings, the report and the viewer.
   - This revision: K over the target embryo, fold-overs in 3D, the per-embryo z-scoring option, and no pipelines from earlier strategies.
   - Run A on Dev (5 seeds) for the noise margin, then on Screen.
   - *Go:* one table with κ_bal, oracle, nulls and ceilings for every pair.
2. **Gates G1 and G2 on the Dev embryos** (CS18_E1, CS12-13_E2, E12.5_E1, E9.5_E2).
   - Fit the field, choose its hyperparameters and the feature z-scoring by the label-free G1 score.
   - Rerun A on the chosen features, which then becomes the reference.
   - *Go:* the field passes G1 and G2. If it fails G1 but the point cloud passes G2, continue with the point-cloud ablation alone.
3. **The volume pipeline with FGW on Dev (X1, X4), both directions.**
   - Compare with A, with the point-cloud ablation and with the volume nulls.
   - Check rejected mass and the depth of the matched mass in X4.
   - *Go:* the rule of 2a on Dev, then on Screen.
4. **Stage-2 ladder on fixed volumes:** Riemannian FGW and MGW, and the z-stretch scan with its regret check. External 3D tools enter here only after 5c.
5. **Stage-1 registration screen** on G3 (W, G1, G2): OT-ICP against the methods of 5a that passed triage, then the downstream Δκ of the best one.
6. **Full tier.** Cross the best stage-1 method with the best stage-2 method. These numbers are the benchmark: Δκ, wins, ceiling share and the guardrails.

---

## 7. Open questions

- **Beyond the outermost sections.** The volume is clipped at the first and last section. Should it extend by half a gap? Is there a principled outer boundary in z, such as the midline for medial stacks?
- **The z-scale.** The gaps are lower bounds. Is a z-stretch chosen by the stage-2 objective trustworthy, or should z come from |ΔL| or the metadata where those exist?
- **The depth loss.** Supervising f along the displacement pairs ties the field to the registration's couplings. How strongly should it be weighted against smoothness? Does an entropic coupling or its peaks make the better pairs?
- **Evaluation in the gaps.** Target samples in a gap are scored with the labels of the nearest real spots. Is that fair to the volume pipeline in stacks with large gaps (X4)?
- **N and τ.** How many samples per embryo, and how much unbalancedness? Both trade resolution and robustness to partial overlap against cost.
- **Non-rigidity.** MGW's metric and a non-rigid stage 1 add freedom. At what point is a good κ_bal the warp's doing rather than the correspondence's?
- **Thresholds.** The regret of 0.05, the 5 % fold-overs, the ≥ 3 of 5 wins and the gate thresholds are first guesses. Should they be set from the Dev tier, as the noise margin is?
- **Unit of evaluation.** Labels are noisy at boundaries, and organs are not exact homologues across species. Is κ_bal at the organ level enough?
