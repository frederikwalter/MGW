# HESTA alignment: why the SM features look like noise

Oct 1, 2026 · @Frederik

The noise in the SM (slice B) features most likely comes from the CCA feeler step, not from normalization, log1p or PCA. This is a code-reading diagnosis. It has not been checked on the data yet.

## Symptom

In `demos/hesta_align.ipynb` (CS17_E1S8 → CS18_E1S3), the `plot_net` figures show the neural-field targets for each slice:

- **ST (slice A, `X_rep`)**: clear anatomy, e.g. the heart and liver show up as bright blobs. φ fits it well.
- **SM (slice B, `Z_rep`)**: per-bin salt-and-pepper noise with no visible anatomy. ψ can only fit a blurry, faceted field.

The training losses match this:

| Network | Slice | Final loss (step 20,000) |
| --- | --- | --- |
| φ | ST, CS17_E1S8 | 0.158 |
| ψ | SM, CS18_E1S3 | 0.870 |

The ψ loss barely moves from its starting value of 0.97, which is what you would expect when the targets are close to spatially unstructured noise. The pullback metric for slice B, and therefore the GW cost C_N, is then mostly meaningless.

## Root cause: the CCA feeler treats the two slices differently

The SM weights are fit on a smoothed copy of slice B but applied to the raw slice B. The ST weights are fit and applied on the same raw data. That mismatch always lands on the SM side.

The steps in `project_informative_features` (`mgw/util.py:127-143`):

1. A spatial-only GW couples 8,000 random bins of A with 8,000 of B (`P`, entropic ε = 1e-3).
2. `Z_bary_on_X = P @ Zpca_gw` maps B's PCs onto A's bins. Each row is a weighted average over many B bins, so it is a heavily **smoothed** version of B.
3. The CCA is fit on `X_std` (A's **raw** per-bin PCs) and `Z_std` (the **smoothed** B values). `Z_scaler` is also fit on the smoothed values.
4. `Z_cca_full = Z_scaler.transform(Z_pca) @ cca.y_weights_` applies the weights to B's **raw**, unsmoothed per-bin PCs.

Why that produces noise:

- Averaging shrinks the spatially incoherent, low-variance PCs (the higher PCs) much more than the coherent, high-variance ones (the top PCs).
- CCA whitens each set, so those shrunken PCs are scaled back up to unit variance and can win large canonical weights.
- In the raw data, those same PCs carry mostly per-bin noise. So the raw projection is dominated by noise even when the fit looked fine on the smoothed data.
- If the feeler coupling is diffuse, `Z_bary_on_X` is nearly constant. Its small leftover variation is itself sampling noise, so the y-weights end up fit to noise.

Slice A avoids this because its weights are fit and applied on the same kind of values. The final z-scoring (`rep_norm="zscore"`) only rescales each column, so it can't undo a bad projection direction.

## Why mouse works and HESTA does not

`mouse_align.ipynb` runs the same pipeline with the same feeler, so the flaw is always there. On HESTA, several things likely make it much worse:

| Factor | Mouse (MOSTA E9.5 → E10.5) | HESTA (CS17_E1S8 → CS18_E1S3) |
| --- | --- | --- |
| Bins per slice | Small slices, no `max_obs` subsampling | 66,586 and 132,277 bins, randomly subsampled to `MAX_OBS` |
| Feeler GW input | Slices of similar shape | Embryos at different stages, B about 2× the bins of A |
| Spatial-only matching | Reasonably well-posed | Very ambiguous: near-symmetric blob shapes, so the coupling is likely diffuse |
| Per-bin signal | Enough for the raw and smoothed values to stay similar | Sparse Stereo-seq bins, so raw and smoothed values differ a lot |

A more diffuse feeler coupling means a smoother `Z_bary_on_X`, which means a bigger raw-vs-smoothed gap and more noise in `Z_rep`. Which factor dominates is still open; the checks below separate them.

## How to confirm

All of these run on the GPU machine after cell 4. They use the `pre["feeler"]` dict, which already returns `P_feeler`, `Z_bary_on_X`, `Z_pca`, `X_pca`, `cca` and `meta["idxZ_feeler"]`.

| Check | Expected if the diagnosis is right |
| --- | --- |
| Scatter B's first 3 joint PCs (`pre["B_"].obsm["X_pca"][:, :3]`) over `xs2` | Clear anatomy, so the data and PCA are fine |
| Per PC: `Z_bary_on_X.std(0) / Z_pca[idxZ].std(0)` | Ratio well below 1, smallest for the higher PCs |
| Per PC: `abs(cca.y_weights_)` vs `abs(cca.x_weights_)` | y-weights put much more mass on higher PCs than the x-weights |
| Canonical correlations on the feeler set | Low, or high on the smoothed data but low when recomputed with raw `Zpca_gw` |
| Row entropy of `P_feeler` vs `log(8000)` | Close to `log(8000)`, i.e. a diffuse coupling |

If the first check shows noise too, the problem is upstream (loading, counts layer, or bin size), and the rest of this doc does not apply.

## Possible fixes (not implemented)

The recommended fix is to fit and apply the CCA on only the top PCs. Nothing has been changed in the code.

1. **Restrict the CCA to the top PCs (recommended).** Add a `cca_pcs` option to `project_informative_features`, passed through `mgw_preprocess` and `mgw_align`. When set, use only the first k PCs on both sides, e.g. k = 10.
   - The top joint PCs are spatially coherent on both slices, so any combination of them stays structured, even if the feeler coupling is weak.
   - With `cca_pcs=None` as the default, the mouse and R114 results stay exactly as they are.
2. **Fit `Z_scaler` on raw data.** Fit it on `Zpca_gw` instead of `Z_bary_on_X`, so slice B is standardized against the data the weights are applied to. On its own this changes little, because CCA directions don't depend on per-column scaling. It is only worth doing together with fix 1.
3. **Skip the CCA for HESTA.** Set `use_cca_feeler=False`. The joint PCA already puts both slices on the same axes, so the top 3 PCs with `rep_norm="zscore"` are valid targets. This is the fallback if the checks show that the feeler coupling is close to uniform.
4. **Reduce per-bin noise.** Smooth each slice's PCs over spatial neighbours (or aggregate to larger bins) before subsampling. This helps both slices, but it changes the input data, not just the feeler.

## Other notes

- **Stale outputs.** The saved notebook outputs show 20,000 bins per slice, but `MAX_OBS` is now 14,000. The figures come from an earlier run.
- **Not verified on data.** The HESTA files are only on the GPU machine, so this analysis comes from reading `mgw/util.py`, `mgw/mgw.py` and the saved plots.
- **GW out-of-memory warnings.** The final GW solve logged allocator warnings. They are separate from the feature problem.
