# `mgw.mgw_align_core`: steps, iterations and solvers

This note walks through `mgw_align_core` (`mgw/mgw.py:120`). It covers every loop the
function runs, from the restarts down to single Sinkhorn updates, and explains how the
OTT-JAX `GromovWasserstein` and `Sinkhorn` solvers used in `mgw/gw.py` iterate.
The solver details follow the installed version, **ott-jax 0.6.0**.

The function takes the output of `mgw_preprocess`:

| key            | shape      | meaning                                                          |
|----------------|------------|------------------------------------------------------------------|
| `xs`, `xs2`    | `(n, 2)`, `(m, 2)` | spatial coordinates of slices A and B, rescaled to the unit square |
| `X_rep`, `Z_rep` | `(n, f_M)`, `(m, f_N)` | per-spot features (CCA or PCA, normalised) that the neural fields are fitted to |

It returns the coupling `P` (`n × m`) and all intermediate objects (networks, metric
tensors, cost matrices, config).

---

## 1. Overview of the nested loops

```
mgw_align_core
└── restart loop                     k = 1 … n_restarts            (default 3)
    ├── train φ   (Adam)             t = 1 … niter                 (default 20 000)
    ├── train ψ   (Adam)             t = 1 … niter
    ├── pullback metric              one vectorised pass, no iterations
    ├── Dijkstra all-pairs           once per source node (n and m times)
    └── solve_gw_ott
        └── GW outer loop            ℓ = 1 … outer_maxit           (default 3000)
            └── Sinkhorn inner loop  s = 1 … inner_maxit           (default 3000)
                                     convergence is checked every 10 steps
```

In the worst case the GW solve runs `outer_maxit × inner_maxit = 9·10⁶` Sinkhorn
updates, each costing $O(nm)$. In practice both loops stop early once their tolerance
is reached.

---

## 2. Step by step

### Step 0: set up (`mgw.py:177-192`)

* Reads `xs, xs2, X_rep, Z_rep` from `pre`.
* Picks `cuda` if it is available and sets the torch default dtype (`float64` by default).
* `suffix` is `_<tag>` when `save_dir` and `tag` are given. It is used in the cache
  file names `phi{suffix}.pt`, `psi{suffix}.pt` and `P{suffix}.npy`.

### Step 1: restart loop (`mgw.py:193-213`)

This runs if `n_restarts > 1` and no cached `phi{suffix}.pt` exists yet:

```python
for k in range(n_restarts):
    run = mgw_align_core(pre, ..., seed=seed + k, n_restarts=1)   # recursive, no save_dir
    obj = _gw_objective(run["C_M"], run["C_N"], run["P"], device)
    keep run with the smallest obj
```

* Each restart is a **full, independent pipeline**. It retrains φ and ψ with a different
  seed, rebuilds the geodesic costs and re-solves GW. Different seeds give different
  neural fields, so each restart has **different cost matrices** `C_M, C_N`, not just a
  different GW initialisation. The GW solver itself is deterministic and always starts
  from the same point (see §3.3).
* `save_dir` is not passed to the child calls, so they cache nothing. Only the winning
  run's `phi`, `psi` and `P` are saved afterwards.
* The selection criterion `_gw_objective` (`mgw.py:407`) is the **unregularised** GW loss
  with squared-difference cost:

  $$
  \mathcal{E}(P)=\sum_{i,i',j,j'} (C^M_{ii'}-C^N_{jj'})^2 P_{ij}P_{i'j'}
  = p^\top (C^M\odot C^M) p + q^\top (C^N\odot C^N) q - 2\langle C^M P,\; P C^N\rangle,
  $$

  where $p=P\mathbb 1$ and $q=P^\top\mathbb 1$. It is computed with the expanded form, so
  no $n^2m^2$ tensor is ever formed.

If a cached `phi{suffix}.pt` exists, the restart loop is skipped. The single-run path
below loads φ and ψ from disk, but it **re-solves GW** rather than loading `P{suffix}.npy`.

### Step 2: neural fields φ and ψ (`mgw.py:215-248`)

* Seeds torch, CUDA and numpy. With `deterministic=True` it also enables deterministic
  kernels and uses fused Adam.
* `φ: ℝ² → ℝ^{f_M}` and `ψ: ℝ² → ℝ^{f_N}` are `PhiModel` MLPs
  (`mgw/models.py:7`). The layer widths default to `(128, 256, 256, 128)` and each hidden
  layer uses Softplus, so the Jacobians are smooth.
* **Iterations:** `models.train_phi` (`mgw/models.py:22`) runs **`niter` full-batch Adam
  steps** on the MSE loss

  $$\min_\theta \tfrac1{n f}\,\|\varphi_\theta(x_s) - X_{\text{rep}}\|_F^2 .$$

  There are no mini-batches and no early stopping, so it always runs exactly `niter` steps
  for φ and then again for ψ. The loss is printed every `print_every` steps.
* When `save_dir` is set, the trained weights are cached, or loaded from the cache if
  they already exist.

The idea is that φ is a smooth interpolation of the expression landscape over the tissue.
Its derivative tells you how fast the features change as you move in space.

### Step 3: pullback metric (`mgw.py:260-262`, `geometry.py:12`)

For every spot $x$, `pullback_metric_field` computes the Jacobian
$J(x)=\partial\varphi/\partial x \in \mathbb R^{f\times 2}$ with `vmap(jacrev(...))`, in a
single batched pass with no iterations. It then forms

$$
g(x) = \frac{J(x)^\top J(x)}{\alpha} + \varepsilon_{\text{geo}} I_2,
\qquad
\alpha = \operatorname{median}_x \tfrac12\operatorname{tr}\!\big(J(x)^\top J(x)\big).
$$

* $J^\top J$ is the metric that φ pulls back onto the plane. Moving a small step $dx$
  changes the features by $\|J\,dx\|$, so distances become large where expression changes
  quickly, for example across tissue boundaries.
* Dividing by $\alpha$ gives the metric a typical eigenvalue of about 1. Adding
  $\varepsilon_{\text{geo}}I$ (`geodesic_eps`, default `1e-2`) keeps $g$ positive definite
  in flat regions, so some Euclidean distance always remains.

### Step 4: kNN graph and geodesic distances (`mgw.py:264-267`, `geometry.py:39,80`)

1. `knn_graph` builds a symmetric k-nearest-neighbour graph on the spatial coordinates
   (`knn_k`, default 12).
2. `geodesic_distances_fast` gives every edge $(i,j)$ its Riemannian length under the
   midpoint metric:

   $$
   w_{ij} = \sqrt{\Delta x^\top \tfrac12(g_i+g_j)\,\Delta x},\qquad \Delta x = x_j-x_i .
   $$

3. **Iterations:** SciPy `dijkstra` computes **all-pairs shortest paths**, running one
   Dijkstra search from each source node. That is $n$ searches for slice A and $m$ for
   slice B, each costing roughly $O(E\log n)$. The result `D_M` (`n×n`) and `D_N` (`m×m`)
   approximates the geodesic distance on the surface $\varphi(\text{tissue})$.

### Step 5: cost matrices (`mgw.py:269-278`)

```python
C = (_norm_geod(D) ** cost_p);  C /= C.max()
```

* `_norm_geod` clips negative values, zeroes the diagonal and divides by the 99th
  percentile of the off-diagonal distances. This is a robust rescaling that ignores a few
  very long paths.
* `cost_p ∈ {1, 2}` chooses between plain distances and squared distances (the default is 2).
* The final division by the maximum puts both `C_M` and `C_N` in $[0,1]$, so the two slices
  are on a comparable scale. This matters because GW compares $C^M_{ii'}$ directly with
  $C^N_{jj'}$.

### Step 6: Gromov–Wasserstein (`mgw.py:280-286`, `gw.py:102`)

The default parameters are:

```python
gw_params = dict(inner_maxit=3000, outer_maxit=3000,
                 inner_tol=1e-7,   outer_tol=1e-7,   epsilon=1e-4)
```

`solve_gw_ott` does the following:

1. Casts `C_M`, `C_N` and the uniform marginals $a=\mathbb 1_n/n$, $b=\mathbb 1_m/m$ to
   `float32` JAX arrays.
2. Wraps each cost matrix in a `Geometry`. The `epsilon` given to these two geometries is
   never used by GW. Only the ε of the linearised problem (below) matters.
3. Builds a `QuadraticProblem(geom_x, geom_y, a, b, tau_a=1, tau_b=1)`. With `tau = 1`
   the problem is **balanced**, so both marginals are enforced exactly.
4. Creates `Sinkhorn(max_iterations=inner_maxit, threshold=inner_tol)` as the inner solver
   and `GromovWasserstein(linear_solver=..., epsilon, max_iterations=outer_maxit,
   threshold=outer_tol)` as the outer solver.
5. JIT-compiles the solver and runs it. It returns `P = out.matrix` as a numpy array.

(The `verbose` key in `gw_params` is accepted but ignored, because the diagnostic print
in `solve_gw_ott` is commented out.)

The two solvers are explained in §3.

### Step 7: return / cache (`mgw.py:288-308`)

The function saves `P{suffix}.npy` if `save_dir` is set and returns a dict with `P`, the
coordinates, the features, `phi`, `psi`, `G_M`, `G_N`, `C_M`, `C_N`, the feeler output and
the merged config.

---

## 3. How the solvers work

### 3.1 The problem being solved

This is entropic Gromov–Wasserstein with squared loss:

$$
\min_{P\in\Pi(a,b)}\;
\underbrace{\sum_{i,i',j,j'} (C^M_{ii'}-C^N_{jj'})^2\,P_{ij}P_{i'j'}}_{\mathcal E(P)}
\;-\;\varepsilon H(P),
\qquad
\Pi(a,b)=\{P\ge 0:\;P\mathbb 1=a,\;P^\top\mathbb 1=b\}.
$$

Here $H(P) = -\sum_{ij} P_{ij}(\log P_{ij}-1)$ is the entropy.
GW does not compare points across the two slices directly. It looks for a matching
$P$ that **preserves pairwise distances**: if $i\leftrightarrow j$ and $i'\leftrightarrow j'$
are matched, then $C^M_{ii'}$ should be close to $C^N_{jj'}$. $\mathcal E$ is quadratic
and non-convex in $P$, so GW is solved by repeatedly linearising it (outer loop) and
solving each linear problem with Sinkhorn (inner loop).

### 3.2 Linearisation (Peyré, Cuturi & Solomon 2016)

For the squared loss, expanding $(c-c')^2 = c^2 + c'^2 - 2cc'$ gives the gradient
$\nabla\mathcal E(P) = 2\,L(P)$, where

$$
L(P) \;=\; \underbrace{(C^M\odot C^M)\,p\,\mathbb 1_m^\top \;+\; \mathbb 1_n\,\big((C^N\odot C^N)\,q\big)^\top}_{\text{marginal term}}
\;-\; 2\,C^M P\, C^N ,
\qquad p = P\mathbb 1,\; q = P^\top\mathbb 1 .
$$

In OTT this is `QuadraticProblem.update_linearization`. The marginal term is
`marginal_dependent_cost` and the last term uses `h1(x)=x`, `h2(y)=2y` from
`make_square_loss`. Because the problem is balanced, $p=a$ and $q=b$ throughout, so the
marginal term is **constant** and only $-2C^MPC^N$ changes from one iteration to the next.
Computing it costs two dense matrix products, about $O(n^2m + nm^2)$.

### 3.3 GW outer loop (`GromovWasserstein`, `ott/solvers/quadratic/gromov_wasserstein.py`)

**Initialisation.** The default `QuadraticInitializer` linearises around the independent
coupling $P^{(0)} = ab^\top$, so the first cost is $L(ab^\top)$. This start point is
deterministic. The same cost matrices always produce the same `P`.

**Iteration** $\ell = 0,1,2,\dots$:

1. **Linearise.** Build the linear OT problem with cost $L^{(\ell)} = L(P^{(\ell)})$ and
   regularisation $\varepsilon$ (`epsilon=1e-4`, absolute, not relative to the cost scale).
2. **Solve it with Sinkhorn:**

   $$
   P^{(\ell+1)} = \arg\min_{P\in\Pi(a,b)} \langle L^{(\ell)}, P\rangle - \varepsilon H(P).
   $$

   The solution has the Gibbs form
   $P^{(\ell+1)}_{ij} = \exp\!\big((f_i + g_j - L^{(\ell)}_{ij})/\varepsilon\big)$.
   Up to the factor 2 in the gradient, this is a **mirror-descent / proximal-point step**
   with KL geometry on $\mathcal E$. Each outer iteration takes the current coupling,
   computes which pairs look cheap given it, and moves to the entropic OT plan for that
   cost.
3. **Record** the regularised linear cost
   $c_\ell = \langle L^{(\ell)},P^{(\ell+1)}\rangle - \varepsilon H(P^{(\ell+1)})$
   (`linear_sol.reg_ot_cost`) and whether that Sinkhorn call converged.

Each Sinkhorn call starts **cold**, from zero potentials, because `warm_start` defaults to
`False` and is not changed in `gw.py`.

**Stopping** (`ott/solvers/was_solver.py`, `WassersteinSolver._converged/_continue`):

* It always runs at least `min_iterations = 5` outer iterations (the OTT default).
* It stops when two consecutive recorded costs are close,
  `jnp.isclose(c[ℓ-2], c[ℓ-1], rtol=outer_tol)`. That test means
  $|c_{\ell-2}-c_{\ell-1}| \le 10^{-8} + \texttt{outer\_tol}\cdot|c_{\ell-1}|$.
  Note that `jnp.isclose` also applies its default absolute tolerance of `1e-8`.
* It also stops if a cost becomes NaN or inf (divergence), or after `outer_maxit` iterations.
* `out.converged` is true only if the loop stopped before `outer_maxit` **and every inner
  Sinkhorn call converged**. `out.n_iters` gives the number of outer iterations actually
  run.

Because $\mathcal E$ is non-convex, the result is a **local** optimum that depends on the
initialisation and on ε. A small ε (1e-4) gives sharp, nearly deterministic couplings,
but each Sinkhorn solve then needs more iterations, and the outer loop can switch between
nearby matchings before it settles.

### 3.4 Sinkhorn inner loop (`Sinkhorn`, `ott/solvers/linear/sinkhorn.py`)

This solves one entropic linear OT problem with cost $C$ (here $C=L^{(\ell)}$):

$$
\min_{P\in\Pi(a,b)} \langle C,P\rangle - \varepsilon H(P)
\quad\Longleftrightarrow\quad
\max_{f,g}\; \langle f,a\rangle + \langle g,b\rangle - \varepsilon\sum_{ij} e^{(f_i+g_j-C_{ij})/\varepsilon}.
$$

The defaults used are `lse_mode=True`, `inner_iterations=10`, `norm_error=1`, no
momentum, no Anderson acceleration and sequential updates.

**Initialisation:** $f = 0$ and $g = 0$.

**One iteration** $s$ (log-domain, `Sinkhorn.lse_step`) alternates two exact block
coordinate-ascent steps on the dual:

$$
\begin{aligned}
g_j &\leftarrow \varepsilon\log b_j \;-\; \varepsilon\log\sum_i \exp\!\big((f_i - C_{ij})/\varepsilon\big)
&&\text{(column marginals become exactly } b)\\
f_i &\leftarrow \varepsilon\log a_i \;-\; \varepsilon\log\sum_j \exp\!\big((g_j - C_{ij})/\varepsilon\big)
&&\text{(row marginals become exactly } a)
\end{aligned}
$$

The primal coupling is $P_{ij} = \exp\big((f_i+g_j-C_{ij})/\varepsilon\big)$. In scaling
form this is the classic $u \leftarrow a / (Kv)$, $v \leftarrow b / (K^\top u)$ with
$K = e^{-C/\varepsilon}$. OTT uses the log-sum-exp form because with $\varepsilon=10^{-4}$
the kernel $e^{-C/\varepsilon}$ would underflow to 0 in `float32`. Each iteration costs
$O(nm)$.

**Error and stopping.**

* The error is only computed on every 10th iteration (`inner_iterations=10`). The loop runs
  in blocks of 10 updates, and each block ends with a convergence check.
* Just after an $f$-update the row marginals are exact. The error is therefore the column
  marginal violation, $\text{err} = \|P^\top\mathbb 1 - b\|_1$.
* Sinkhorn stops when `err < inner_tol` (1e-7), when err is not finite, or after
  `inner_maxit` updates. It runs at most `ceil(3000/10) = 300` checks.
* When it stops at the cap without meeting the tolerance, the call is marked not converged.
  That flag feeds into the GW `out.converged` flag above.

Note that `inner_tol = 1e-7` is an L1 marginal error in `float32`, and the marginals have
entries of size $1/n$. Reaching that tolerance is hard, so the inner loop often runs to the
full `inner_maxit`. The small ε makes this worse, because Sinkhorn converges more slowly as
ε decreases (roughly linearly, at a rate that worsens as $\|C\|_\infty/\varepsilon$ grows).
If the GW solve is slow, loosening `inner_tol` or using a larger ε are the first settings
to try.

---

## 4. Fused term (optional FGW)

Pure GW only compares distances *within* each slice, so it can't tell mirror images apart
and has many near-equivalent optima on nearly symmetric tissue. With `use_fgw=True`,
`mgw_align_core` adds a linear cross-slice term. It is **off by default**, and with
`use_fgw=False` the pipeline is exactly the pure-GW pipeline described above.

**Objective (POT convention).**

$$
\min_{P\in\Pi(a,b)}\; \alpha\,\mathcal E(P) + (1-\alpha)\,\langle M, P\rangle \;-\;\varepsilon' H(P),
\qquad \alpha = \texttt{fgw\_alpha}\in(0,1].
$$

**Mapping to OTT.** `QuadraticProblem(..., geom_xy=M, fused_penalty=λ)` adds $\lambda M$ to
every linearisation, which is $L(P)+\lambda M$. Since $\nabla\mathcal E = 2L$ (§3.2), OTT's
fixed points are stationary points of $\tfrac12\mathcal E(P) + \lambda\langle M,P\rangle$.
Matching this to the POT objective up to a constant factor gives

$$
\lambda = \texttt{fused\_penalty} = \frac{1-\alpha}{2\alpha}.
$$

**Building M** (`_build_fused_cost`, `_joint_sqdist` in `mgw/mgw.py`).

1. Take a per-spot embedding in the **CCA joint space** from the feeler in `mgw_preprocess`.
   The CCA weights map both slices into a common space for any pair of modalities, so FGW
   stays modality-agnostic.
   * `fused_source="field"` (default): use $\varphi(x_s)$ and $\psi(x_t)$, the fields'
     denoised predictions of the CCA variates.
   * `fused_source="rep"`: use the raw `X_rep`, `Z_rep`.
2. Keep the first `fused_comps` components (None = all, 1 = feat0 only). Z-score each
   component per slice and weight component $k$ by $w_k$:
   * `fused_weight="rho2"` (default): $w_k=\rho_k^2$, where $\rho_k$ is the canonical
     correlation on the feeler pairs (`pre["cca_corr"]`),
   * `"rho"`: $w_k=|\rho_k|$,
   * `"uniform"`: $w_k=1$.
3. Set $M_{ij}=\sum_k w_k (u_{ik}-v_{jk})^2$. Divide by its 99th percentile and clip to
   $[0,1]$, which is the same scale as `C_M`/`C_N`. M is stored as a dense `float32` $n\times m$
   matrix.

`use_fgw=True` requires `use_cca_feeler=True`, because per-slice PCA features are not a joint
space. `fused_cost=` bypasses this check with a precomputed $n\times m$ cost, for example a
unimodal joint-PCA baseline.

**Caveat.** The CCA is fitted on pairs from a spatial-only GW feeler, so M partly encodes that
feeler's matching. Keep α moderate and check the reported term magnitudes.

**Restarts and outputs.**
* With FGW on, restarts are ranked by $\tfrac12\mathcal E(P) + \lambda\langle M,P\rangle$,
  which is what OTT optimises.
* The result contains `M` and `fgw_obj_terms = {"gw": ½E, "lin": λ⟨M,P⟩}`. Both are `None`
  when FGW is off.
* With `save_dir`, the coupling is saved as `P{suffix}_fgw{α}[_k{comps}].npy`, so it does not
  overwrite the GW `P{suffix}.npy`. The cached φ/ψ are shared.

**Sweeps.** `mgw.mgw_resolve(out, pre, fgw_alpha=..., fused_comps=...)` re-solves the coupling
of a finished run. It reuses that run's φ, ψ, `C_M` and `C_N`, so no fields are retrained and
no geodesics recomputed. Only the (F)GW solve is repeated.

---

## 5. Quick parameter reference

| parameter | where | effect |
|---|---|---|
| `n_restarts` | restart loop | number of independent full pipelines; the one with the lowest $\mathcal E(P)$ is kept |
| `seed` | restarts / training | restart `k` uses `seed + k` |
| `niter`, `lr`, `widths` | φ/ψ training | full-batch Adam steps, step size, MLP shape |
| `geodesic_eps` | pullback metric | isotropic floor added to $g(x)$; larger values make distances closer to Euclidean |
| `knn_k` | graph | graph connectivity; too small can disconnect the graph and give infinite distances |
| `cost_p` | costs | 1 = distances, 2 = squared distances |
| `epsilon` | GW + Sinkhorn | entropic regularisation; smaller values give sharper `P` and slower convergence |
| `outer_maxit`, `outer_tol` | GW outer loop | cap and relative-change tolerance on the linearised cost |
| `inner_maxit`, `inner_tol` | Sinkhorn | cap and L1 marginal-error tolerance, checked every 10 steps |
| `use_fgw` | GW | switches the fused linear term on (default `False` = pure GW) |
| `fgw_alpha` | FGW | GW weight α in `α·GW + (1−α)·⟨M,P⟩`; OTT `fused_penalty = (1−α)/(2α)` |
| `fused_source` | FGW | `"field"` (φ/ψ predictions) or `"rep"` (raw CCA variates) for M |
| `fused_comps` | FGW | number of CCA components in M (None = all, 1 = feat0) |
| `fused_weight` | FGW | per-component weight: `"rho2"`, `"rho"` or `"uniform"` |
| `fused_cost` | FGW | precomputed n×m cross cost (bypasses the CCA-feeler check) |
