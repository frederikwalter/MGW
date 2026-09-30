# Plan: Fused Gromov–Wasserstein (FGW) in the MGW pipeline

## Context

`mgw_align_core` (`mgw/mgw.py:119`) currently solves pure GW between the two pullback-geodesic
cost matrices `C_M` (n×n) and `C_N` (m×m) via `solve_gw_ott` (`mgw/gw.py:102`). Pure GW only
compares distances *within* each slice, so:
- it can't tell mirror images or rotations apart (the notebook checks `orientation det` for this reason),
- it has many near-equivalent local optima on nearly symmetric tissue.

The goal is a **fused** objective, `GW(C_M, C_N) + λ·⟨M, P⟩`, where `M_ij` is a cross-slice
distance in a **joint embedding** of both slices. The method must stay **modality-agnostic**
(it has to work for ST↔MSI, as in `demos/demo_mgw_y7.ipynb`). So the shared-gene joint PCA
(`X_feat`/`Z_feat`) can only serve as a unimodal baseline, not as the main method.

## Feasibility and choice of embedding

**Solver:** fully supported. `ott-jax 0.6.0`'s `QuadraticProblem` already accepts
`geom_xy` + `fused_penalty` and adds `fused_penalty · M` to every linearization
(`quadratic_problem.py:83,324`). No new solver code is needed. The extra memory is one dense
n×m matrix (≈435 MB float32 for mouse, which is small next to `C_N`).

**Joint space: the CCA canonical variates already are one.** `util.project_informative_features`
(`mgw/util.py:26`) fits a CCA whose `x_weights_`/`y_weights_` map each slice's features into
a common k-dim space where corresponding spots correlate. Sign and ordering are consistent
across slices by construction. This works for any pair of modalities, so it's the right basis.

**Only CCA component 0 vs. all components.** Using component 0 alone is feasible, but it's a weak
cost on its own:
- With one component, `M_ij = (u_i − v_j)²` is a 1-D cost. Every spot on the same iso-contour of
  feat0 costs the same, so M only separates spots "along one axis". It can fix a coarse
  orientation, but it can't localise matches.
- Components 1–2 add independent, if weaker, evidence. The right fix is to **weight each
  component by its canonical correlation ρ_k** (e.g. `w_k = ρ_k²`), not to drop them. Weak
  components then contribute little. "Only component 0" stays available as `fused_comps=1`,
  so both can be compared.
- ρ_k is not stored at the moment. It has to be computed on the feeler pairs inside
  `project_informative_features`.

**Raw variates vs. neural-field-smoothed variates.** The ψ fit loss is ≈0.49 (vs 0.07 for φ) and
the SM feature plots are very noisy. Recommended default: build M from **φ(xs), ψ(xs2)**. These
are the fields' denoised predictions of the same CCA variates, and they are consistent with the
geometry that defines C_M/C_N. Raw `X_rep/Z_rep` stays available as an option.

**Should a new feature space be learned?** Not as a first step:
- **Circularity caveat:** the CCA is fitted on pairs from a *spatial-only* GW feeler. So M partly
  encodes that feeler's matching. This is fine for breaking symmetry, but it biases P toward the
  feeler. Keep λ moderate and report how much of the objective each term contributes.
- **Cheap "learned" upgrade (phase 2, optional):** iterative CCA refinement. Run MGW/FGW → refit
  the CCA on barycentric pairs from the new, better P → rebuild M → re-solve FGW (1–2 rounds).
  This reuses the existing CCA code and stays linear, interpretable and modality-agnostic.
- **Deep joint encoders** (contrastive/OT-trained, or a shared head on φ/ψ) are a much bigger
  change. They have the same circularity plus a risk of collapse, and would change MGW itself.
  Only worth it if phases 1–2 fall short.

## Changes

### 1. `mgw/util.py` — `project_informative_features`
- After `cca.fit`, compute `rho_k = corrcoef(X_std @ xw[:,k], Z_std @ yw[:,k])` for each component.
- Return `cca_corr=rho` in the dict (and in `meta`).

### 2. `mgw/gw.py` — `solve_gw_ott`
- Add `M=None, fused_penalty=1.0`. When `M` is given, build
  `geom_xy = geometry.Geometry(cost_matrix=jnp.asarray(M, float32))` and pass
  `geom_xy=..., fused_penalty=...` to `QuadraticProblem`.
- `M=None` keeps today's behaviour bit-for-bit.

### 3. `mgw/mgw.py`
- `mgw_preprocess`: store `pre["cca_corr"] = feeler["cca_corr"]` (None when there is no feeler).
- New helper `_fused_cost(E_A, E_B, weights, n_comps)` → z-score each variate per slice, keep
  the first `n_comps` columns, scale by `sqrt(w_k)`, compute the squared Euclidean `cdist`, then
  divide by the 99th percentile and clip to `[0,1]`. This matches how `C_M/C_N` are scaled, so
  λ is interpretable.
- `mgw_align_core` new kwargs:
  - `fgw_alpha: Optional[float] = None`: `None`/`1.0` gives pure GW (current behaviour). Otherwise
    use the POT convention `α·GW + (1−α)·W`, mapped to OTT as `fused_penalty = (1−α)/α`.
  - `fused_source: str = "field"`: `"field"` uses `phi(xs)`, `psi(xs2)`
    (via `plotting.predict_on_model`, `mgw/plotting.py:143`). `"rep"` uses `X_rep`/`Z_rep`.
  - `fused_comps: Optional[int] = None`: number of CCA components to use (None = all, 1 = feat0).
  - `fused_weight: str = "rho2"`: one of `"rho2"`, `"rho"` or `"uniform"`.
  - Build `M` after the cost matrices (after `mgw.py:238`) and pass it to `solve_gw_ott`.
  - Guard: if `fgw_alpha < 1` and `use_cca_feeler=False`, raise an error. PCA features of two
    different slices are not a joint space. With `"uniform"` and a unimodal user override, the
    per-slice PCA would silently be wrong.
  - Return `M`, and add `fgw_alpha`/`fused_*` to `config`.
- Restart selection: replace `_gw_objective` with an `_fgw_objective` that adds
  `fused_penalty · ⟨M, P⟩` (and pass `M` through the recursive restart call). Check the scale
  factor against OTT's convention on a small problem. OTT's linearization uses L(P) and not
  2L(P), so the GW part may need a factor ½ to match what OTT optimises.
- `mgw_align` wrapper: pass the new kwargs through.

### 4. `demos/mouse_align.ipynb`
- Update the intro markdown: FGW now uses a CCA-derived joint space, so it stays modality-agnostic.
- Add a `FGW_ALPHA` parameter (start at 0.5; sweep 0.9/0.7/0.5/0.3), and `fused_comps` 1 vs 3.
- Add a comparison cell: GW vs FGW(feat0) vs FGW(all, ρ²-weighted). Report
  `orientation det`, `metrics.ami_on_projected_labels` (`mgw/metrics.py:110`), the GW and linear
  term magnitudes, and the existing label-projection plot.
- Optional unimodal upper bound: FGW with M from the joint PCA `X_feat/Z_feat`, clearly labelled
  as not modality-agnostic.

### 5. `docs/mgw_align_core.md`
- Add a short section on the fused term, the α↔`fused_penalty` mapping and how M is built.

### Phase 2 (optional, after results)
- `mgw.refine_cca(pre, P, CCA_comp)`: barycentric pairs from P → refit CCA → new `X_rep/Z_rep`
  and ρ. Then call `mgw_align_core` again with `fused_source="rep"` or retrained fields.

## Verification
1. **Regression:** run `mouse_align.ipynb` with `fgw_alpha=None`. P must equal the current run
   (distance to the paper map stays 0.072, orientation det +1).
2. **Solver sanity:** on a small random problem, `M=0` gives the same result as GW. Large λ
   (α→0) gives roughly the plain entropic-OT plan on M.
3. **FGW runs:** α ∈ {0.9, 0.7, 0.5, 0.3} × `fused_comps` ∈ {1, 3}. Compare the AMI of projected
   annotations, the orientation, and the relative contribution of the GW and linear terms.
4. **Multimodal check:** run the same FGW settings on `demos/demo_mgw_y7.ipynb` (ST↔MSI) to
   confirm it works without shared features.
