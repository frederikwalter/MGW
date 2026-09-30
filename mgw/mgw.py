# mgw.py
from typing import Optional, Dict, Any, Tuple
import numpy as np, torch, os, scipy.sparse as sp
from mgw import util, models, geometry, plotting
from mgw.gw import solve_gw_ott
from sklearn.preprocessing import normalize

def mgw_preprocess(
    A, B,
    *,
    # feature config
    PCA_comp: int = 30,
    CCA_comp: int = 3,
    use_cca_feeler: bool = True,
    feeler_downsample: Optional[int] = 8000,
    log1p_X: bool = True,
    log1p_Z: bool = True,
    use_pca_X: bool = True,
    use_pca_Z: bool = True,
    # sampling
    max_obs_A: Optional[int] = None,
    max_obs_B: Optional[int] = None,
    # feeler flavors
    spatial_only: bool = True,
    feature_only: bool = False,
    # normalization
    rep_norm: str = "l2",
    # misc
    verbose: bool = True,
) -> Dict[str, Any]:
    """Preprocess A,B into unit-square coords + representation features.

    Parameters
    ----------
    rep_norm : str, default "l2"
        How to normalize the CCA / feature representations fed to the
        neural fields.  ``"l2"`` applies sklearn L2 row-normalisation
        (existing default).  ``"zscore"`` applies per-column zero-mean
        unit-variance scaling (matches the original experiment in
        ``mgw_mouse_embryo.ipynb`` which used ``util.normalize_range``).
    """
    A_ = _maybe_subsample(A, max_obs_A)
    B_ = _maybe_subsample(B, max_obs_B)

    xs  = _to_unit_square(A_.obsm["spatial"])
    xs2 = _to_unit_square(B_.obsm["spatial"])

    def _get_pca(adata, n_comps, log1p):
        if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= n_comps:
            return np.asarray(adata.obsm["X_pca"][:, :n_comps], dtype=np.float64)
        return util.pca_from(adata, n_comps=n_comps, log1p=log1p)

    # Base PCA or raw features
    if use_pca_X:
        X_pca = _get_pca(A_, n_comps=PCA_comp, log1p=log1p_X)
    else:
        X_pca = A_.X.toarray() if sp.issparse(A_.X) else np.asarray(A_.X)
        X_pca = X_pca.astype(np.float64)
        if log1p_X: X_pca = np.log1p(X_pca)

    if use_pca_Z:
        Z_pca = _get_pca(B_, n_comps=PCA_comp, log1p=log1p_Z)
    else:
        Z_pca = B_.X.toarray() if sp.issparse(B_.X) else np.asarray(B_.X)
        Z_pca = Z_pca.astype(np.float64)
        if log1p_Z: Z_pca = np.log1p(Z_pca)

    if verbose:
        print(f"[mgw.pre] PCA/raw shapes: X={X_pca.shape}  Z={Z_pca.shape}")

    # Optional CCA feeler
    feeler = None
    if use_cca_feeler:
        if verbose:
            print("[mgw.pre] CCA feeler: PCA → small GW (fused) → barycentric → CCA")
        feeler = util.project_informative_features(
            path_X=None, path_Z=None,
            adata_X=A_, adata_Z=B_,
            PCA_comp=PCA_comp, CCA_comp=CCA_comp,
            n_downsample=feeler_downsample,
            log1p_X=log1p_X, log1p_Z=log1p_Z,
            verbose=verbose,
            spatial_only=spatial_only,
            feature_only=feature_only,
        )
        X_rep = feeler["X_cca_full"]
        Z_rep = feeler["Z_cca_full"]
    else:
        X_rep, Z_rep = X_pca, Z_pca

    # L2 row-norm for model targets
    X_pca = normalize(X_pca)
    Z_pca = normalize(Z_pca)
    
    # Normalize representation features for neural-field targets
    if rep_norm == "zscore":
        X_rep = util.normalize_range_np(X_rep)
        Z_rep = util.normalize_range_np(Z_rep)
    else:  # default: "l2"
        X_rep = normalize(X_rep)
        Z_rep = normalize(Z_rep)
    
    return dict(
        A_=A_, B_=B_,
        xs=xs, xs2=xs2,
        X_feat=X_pca, Z_feat=Z_pca,
        X_rep=X_rep, Z_rep=Z_rep,
        feeler=feeler,
        cca_corr=None if feeler is None else feeler["cca_corr"],
        config=dict(
            PCA_comp=PCA_comp, CCA_comp=CCA_comp, use_cca_feeler=use_cca_feeler,
            feeler_downsample=feeler_downsample, log1p_X=log1p_X, log1p_Z=log1p_Z,
            use_pca_X=use_pca_X, use_pca_Z=use_pca_Z,
            spatial_only=spatial_only, feature_only=feature_only,
            rep_norm=rep_norm,
            max_obs_A=max_obs_A, max_obs_B=max_obs_B,
        ),
    )

def mgw_align_core(
    pre: Dict[str, Any],
    *,
    # neural fields
    widths: tuple = (128, 256, 256, 128),
    lr: float = 1e-3,
    niter: int = 20_000,
    print_every: int = 1_000,
    # metric + graph
    knn_k: int = 12,
    geodesic_eps: float = 1e-2,
    # GW solver
    gw_params: Optional[Dict[str, Any]] = None,
    cost_p: int = 2,
    seed: Optional[int] = 0,
    deterministic: bool = False,
    n_restarts: int = 3,
    # fused GW (optional linear term in the CCA joint space; off by default)
    use_fgw: bool = False,
    fgw_alpha: float = 0.5,
    fused_source: str = "field",
    fused_comps: Optional[int] = None,
    fused_weight: str = "rho2",
    fused_cost: Optional[np.ndarray] = None,
    # device / dtype
    device: Optional[str] = None,
    torch_default_dtype: torch.dtype = torch.float64,
    # caching
    save_dir: Optional[str] = None,
    tag: Optional[str] = None,
    # misc
    verbose: bool = True,
    plot_net: bool = False,
) -> Dict[str, Any]:
    """Core: learn φ,ψ; pullback metrics; GW; return coupling + intermediates.

    Fused GW
    --------
    use_fgw : bool, default False
        Switch the linear (Wasserstein) term on.  When False the pipeline is
        pure GW exactly as before.  When True the objective is
        ``α·GW + (1−α)·<M, P>`` (POT convention), where ``M`` is a cross-slice
        cost in the CCA joint space.
    fgw_alpha : float in (0, 1], default 0.5
        GW weight α.  Mapped to OTT as ``fused_penalty = (1−α)/(2α)`` because
        OTT optimizes ``0.5·GW + fused_penalty·<M, P>``.
    fused_source : {"field", "rep"}, default "field"
        Build ``M`` from the neural-field predictions ``φ(xs), ψ(xs2)`` or from
        the raw representations ``X_rep, Z_rep``.
    fused_comps : int, optional
        Number of CCA components to use (None = all, 1 = feat0 only).
    fused_weight : {"rho2", "rho", "uniform"}, default "rho2"
        Per-component weight, from the canonical correlations ρ_k.
    fused_cost : ndarray (n, m), optional
        Precomputed cross cost; bypasses the CCA check (e.g. for a unimodal
        joint-PCA baseline).
    """
    xs, xs2 = pre["xs"], pre["xs2"]
    X_rep, Z_rep = pre["X_rep"], pre["Z_rep"]

    fused_penalty = _check_fgw_args(pre, fgw_alpha, fused_source, fused_weight, fused_cost) if use_fgw else None

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_dtype(torch_default_dtype)
    if verbose: print(f"[mgw.core] device={device} dtype={torch.get_default_dtype()}")

    suffix = "" if not save_dir else ("" if tag is None else f"_{tag}")
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    P_file = f"P{suffix}{_fgw_file_tag(fgw_alpha, fused_comps, fused_cost) if use_fgw else ''}.npy"
    fgw_kwargs = dict(use_fgw=use_fgw, fgw_alpha=fgw_alpha, fused_source=fused_source,
                      fused_comps=fused_comps, fused_weight=fused_weight, fused_cost=fused_cost)

    if n_restarts > 1 and not (save_dir and os.path.isfile(os.path.join(save_dir, f"phi{suffix}.pt"))):
        best, best_obj = None, np.inf
        for k in range(n_restarts):
            run = mgw_align_core(pre, widths=widths, lr=lr, niter=niter, print_every=print_every, knn_k=knn_k,
                                 geodesic_eps=geodesic_eps, gw_params=gw_params, cost_p=cost_p,
                                 seed=None if seed is None else seed + k, deterministic=deterministic, n_restarts=1,
                                 **fgw_kwargs,
                                 device=device, torch_default_dtype=torch_default_dtype, verbose=verbose, plot_net=plot_net)
            if use_fgw:
                obj = run["fgw_obj_terms"]["gw"] + run["fgw_obj_terms"]["lin"]
                if verbose: print(f"[mgw.core] restart {k + 1}/{n_restarts}: FGW objective={obj:.6e}")
            else:
                obj = _gw_objective(run["C_M"], run["C_N"], run["P"], device)
                if verbose: print(f"[mgw.core] restart {k + 1}/{n_restarts}: GW objective={obj:.6e}")
            if obj < best_obj: best, best_obj = run, obj
        best["config"].update(n_restarts=n_restarts, save_dir=save_dir, tag=tag)
        if save_dir:
            torch.save(best["phi"].state_dict(), os.path.join(save_dir, f"phi{suffix}.pt"))
            torch.save(best["psi"].state_dict(), os.path.join(save_dir, f"psi{suffix}.pt"))
            np.save(os.path.join(save_dir, P_file), best["P"])
        return best

    xs_t, xs2_t = torch.from_numpy(xs).to(device), torch.from_numpy(xs2).to(device)
    ys_t, ys2_t = torch.from_numpy(X_rep).to(device), torch.from_numpy(Z_rep).to(device)
    dim_e, dim_f_M, dim_f_N = 2, ys_t.shape[1], ys2_t.shape[1]
    if verbose: print(f"[mgw.core] dims: E=2, F_M={dim_f_M}, F_N={dim_f_N}")

    # φ, ψ
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        if verbose: print(f"[mgw.core] seed={seed}")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if verbose: print("[mgw.core] deterministic training (fused Adam, deterministic kernels)")
    phi = models.PhiModel(2, dim_f_M, widths=widths).to(device)
    psi = models.PhiModel(2, dim_f_N, widths=widths).to(device)

    phi_path = psi_path = None
    if save_dir:
        phi_path = os.path.join(save_dir, f"phi{suffix}.pt")
        psi_path = os.path.join(save_dir, f"psi{suffix}.pt")

    if phi_path and os.path.isfile(phi_path) and os.path.isfile(psi_path):
        if verbose: print("[mgw.core] loading cached φ, ψ")
        phi.load_state_dict(torch.load(phi_path, map_location=device))
        psi.load_state_dict(torch.load(psi_path, map_location=device))
        phi.eval(); psi.eval()
    else:
        if verbose: print("[mgw.core] training φ, ψ")
        phi = models.train_phi(phi, xs_t,  ys_t,  lr=lr, niter=niter, print_every=print_every, device=device, fused=deterministic)
        psi = models.train_phi(psi, xs2_t, ys2_t, lr=lr, niter=niter, print_every=print_every, device=device, fused=deterministic)
        phi.eval(); psi.eval()
        if save_dir:
            torch.save(phi.state_dict(), phi_path)
            torch.save(psi.state_dict(), psi_path)

    if plot_net:
        rng = np.random.default_rng(0)
        for k in rng.choice(dim_f_M, size=min(5, dim_f_M), replace=False):
            X_pred = plotting.predict_on_model(phi, xs)
            plotting.plot_fit_on_cloud(xs, ys_t[:,k].cpu().numpy(), X_pred[:,k],
                                       title_true=f'ST feat{k} (true)', title_pred=f'φ(xs) feat{k} (pred)')
            Z_pred = plotting.predict_on_model(psi, xs2)
            plotting.plot_fit_on_cloud(xs2, ys2_t[:,k].cpu().numpy(), Z_pred[:,k],
                                       title_true=f'SM feat{k} (true)', title_pred=f'ψ(xs2) feat{k} (pred)')

    if verbose: print("[mgw.core] pullback metrics & geodesics")
    G_M = geometry.pullback_metric_field(phi, torch.from_numpy(xs).to(device),  eps=geodesic_eps).cpu()
    G_N = geometry.pullback_metric_field(psi, torch.from_numpy(xs2).to(device), eps=geodesic_eps).cpu()

    Gs = geometry.knn_graph(xs,  k=knn_k)
    Gt = geometry.knn_graph(xs2, k=knn_k)
    D_M = geometry.geodesic_distances_fast(xs,  G_M, Gs)
    D_N = geometry.geodesic_distances_fast(xs2, G_N, Gt)

    def _norm_geod(D):
        D = np.maximum(D, 0.0); np.fill_diagonal(D, 0.0)
        q = np.quantile(D[np.triu_indices_from(D, 1)], 0.99)
        return D / (q + 1e-12)

    if cost_p not in (1, 2):
        raise ValueError(f"cost_p must be 1 or 2, got {cost_p}")
    if verbose: print(f"[mgw.core] cost matrices: d^{cost_p} (cost_p={cost_p})")
    C_M = _norm_geod(D_M)**cost_p; C_M /= (C_M.max() + 1e-12)
    C_N = _norm_geod(D_N)**cost_p; C_N /= (C_N.max() + 1e-12)

    if gw_params is None:
        gw_params = dict(verbose=True, inner_maxit=3000, outer_maxit=3000,
                         inner_tol=1e-7,   outer_tol=1e-7,   epsilon=1e-4)
    M = None
    if use_fgw:
        M = _build_fused_cost(pre, phi, psi, fused_source, fused_comps, fused_weight, fused_cost, verbose)
    P, fgw_obj_terms = _solve_coupling(C_M, C_N, gw_params, M, fused_penalty, fgw_alpha, device, verbose)

    if save_dir:
        np.save(os.path.join(save_dir, P_file), P)

    return dict(
        P=P, xs=xs, xs2=xs2,
        X_rep=X_rep, Z_rep=Z_rep,
        X_feat=pre.get("X_feat"), Z_feat=pre.get("Z_feat"),
        phi=phi, psi=psi,
        G_M=G_M, G_N=G_N,
        C_M=C_M, C_N=C_N,
        M=M, fgw_obj_terms=fgw_obj_terms,
        feeler=pre.get("feeler"),
        config=dict(
            **pre["config"],
            widths=widths, lr=lr, niter=niter,
            knn_k=knn_k, geodesic_eps=geodesic_eps,
            gw_params=gw_params, cost_p=cost_p, seed=seed, deterministic=deterministic, n_restarts=n_restarts, device=device,
            save_dir=save_dir, tag=tag,
            **_fgw_config(use_fgw, fgw_alpha, fused_penalty, fused_source, fused_comps, fused_weight, fused_cost),
        )
    )

def mgw_resolve(
    out: Dict[str, Any],
    pre: Dict[str, Any],
    *,
    gw_params: Optional[Dict[str, Any]] = None,
    use_fgw: bool = True,
    fgw_alpha: float = 0.5,
    fused_source: str = "field",
    fused_comps: Optional[int] = None,
    fused_weight: str = "rho2",
    fused_cost: Optional[np.ndarray] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Re-solve the coupling of a finished ``mgw_align_core`` run with new (F)GW settings.

    Reuses ``out``'s φ, ψ and cost matrices ``C_M, C_N`` (no retraining, no
    geodesics), so sweeping ``fgw_alpha`` / ``fused_comps`` only costs the GW
    solves.  Returns a shallow copy of ``out`` with ``P``, ``M``,
    ``fgw_obj_terms`` and ``config`` replaced.  Nothing is written to disk.
    """
    fused_penalty = _check_fgw_args(pre, fgw_alpha, fused_source, fused_weight, fused_cost) if use_fgw else None
    if device is None:
        device = out["config"].get("device") or ('cuda' if torch.cuda.is_available() else 'cpu')
    if gw_params is None:
        gw_params = out["config"]["gw_params"]

    M = None
    if use_fgw:
        M = _build_fused_cost(pre, out["phi"], out["psi"], fused_source, fused_comps, fused_weight, fused_cost, verbose)
    P, fgw_obj_terms = _solve_coupling(out["C_M"], out["C_N"], gw_params, M, fused_penalty, fgw_alpha, device, verbose)

    res = dict(out)
    res.update(P=P, M=M, fgw_obj_terms=fgw_obj_terms)
    res["config"] = dict(out["config"], gw_params=gw_params,
                         **_fgw_config(use_fgw, fgw_alpha, fused_penalty, fused_source, fused_comps, fused_weight, fused_cost))
    return res

def mgw_align(
    A, B,
    *,
    # preprocess knobs
    PCA_comp: int = 30, CCA_comp: int = 3, use_cca_feeler: bool = True,
    feeler_downsample: Optional[int] = 8000,
    log1p_X: bool = True, log1p_Z: bool = True,
    use_pca_X: bool = True, use_pca_Z: bool = True,
    max_obs_A: Optional[int] = None, max_obs_B: Optional[int] = None,
    spatial_only: bool = True, feature_only: bool = False,
    rep_norm: str = "l2",
    # core knobs
    widths: tuple = (128, 256, 256, 128), lr: float = 1e-3,
    niter: int = 20_000, print_every: int = 1_000,
    knn_k: int = 12, geodesic_eps: float = 1e-2,
    gw_params: Optional[Dict[str, Any]] = None,
    cost_p: int = 2, seed: Optional[int] = 0, deterministic: bool = False, n_restarts: int = 3,
    use_fgw: bool = False, fgw_alpha: float = 0.5, fused_source: str = "field",
    fused_comps: Optional[int] = None, fused_weight: str = "rho2",
    fused_cost: Optional[np.ndarray] = None,
    device: Optional[str] = None, torch_default_dtype: torch.dtype = torch.float64,
    save_dir: Optional[str] = None, tag: Optional[str] = None,
    verbose: bool = True, plot_net: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper: preprocess → core."""
    pre = mgw_preprocess(
        A, B,
        PCA_comp=PCA_comp, CCA_comp=CCA_comp, use_cca_feeler=use_cca_feeler,
        feeler_downsample=feeler_downsample,
        log1p_X=log1p_X, log1p_Z=log1p_Z,
        use_pca_X=use_pca_X, use_pca_Z=use_pca_Z,
        max_obs_A=max_obs_A, max_obs_B=max_obs_B,
        spatial_only=spatial_only, feature_only=feature_only,
        rep_norm=rep_norm,
        verbose=verbose,
    )
    return mgw_align_core(
        pre,
        widths=widths, lr=lr, niter=niter, print_every=print_every,
        knn_k=knn_k, geodesic_eps=geodesic_eps,
        gw_params=gw_params, cost_p=cost_p, seed=seed, deterministic=deterministic, n_restarts=n_restarts,
        use_fgw=use_fgw, fgw_alpha=fgw_alpha, fused_source=fused_source,
        fused_comps=fused_comps, fused_weight=fused_weight, fused_cost=fused_cost,
        device=device, torch_default_dtype=torch_default_dtype,
        save_dir=save_dir, tag=tag,
        verbose=verbose, plot_net=plot_net,
    )

def transfer(adata_a, adata_b, P, tag_a="A", tag_b="B", eps=1e-12):
    """Barycentrically project features from adata_b onto adata_a's coordinate space.

    P must be a (n_a, n_b) coupling matrix as returned by mgw_align.
    To project in the reverse direction call transfer(adata_b, adata_a, P.T).

    Returns an AnnData on adata_a's spatial grid whose .X concatenates
    adata_a's original features with the projected adata_b features.
    """
    return util.bary_proj(adata_a, adata_b, P, first_tag=tag_a, second_tag=tag_b, eps=eps)

def _gw_objective(C1, C2, P, device) -> float:
    C1, C2, P = (torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device) for a in (C1, C2, P))
    p, q = P.sum(1), P.sum(0)
    return float(p @ (C1 * C1) @ p + q @ (C2 * C2) @ q - 2 * ((C1 @ P) * (P @ C2)).sum())

def _gw_energy_lowmem(C1, C2, P, block: int = 1024) -> float:
    """Same value as `_gw_objective`, computed in row/column blocks.

    Avoids the dense float64 temporaries (C2*C2 is m×m; P, C1@P, P@C2 are n×m) that
    `_gw_objective` builds, which at full slice size (~6k × 18k) add ~5 GB.
    Constant terms are exact in float64; the cross term uses float32 products.
    """
    p, q = P.sum(1, dtype=np.float64), P.sum(0, dtype=np.float64)
    const = sum(float(p[i:i + block] @ np.square(C1[i:i + block]) @ p) for i in range(0, len(p), block))
    const += sum(float(q[j:j + block] @ np.square(C2[j:j + block]) @ q) for j in range(0, len(q), block))
    P32 = np.asarray(P, dtype=np.float32)
    CP = np.asarray(C1, dtype=np.float32) @ P32              # n×m float32
    cross = 0.0
    for j in range(0, P32.shape[1], block):                  # <C1 P, P C2>, C2 symmetric
        PC = P32 @ np.asarray(C2[j:j + block], dtype=np.float32).T
        cross += float(np.einsum("ij,ij->", CP[:, j:j + block], PC, dtype=np.float64))
    return const - 2.0 * cross

def _fgw_objective(C1, C2, P, M, fused_penalty, device=None) -> Tuple[float, float]:
    """(GW, linear) terms of the objective OTT optimizes: 0.5·GW(P) + fused_penalty·<M, P>."""
    lin = sum(float(np.einsum("ij,ij->", M[i:i + 1024], P[i:i + 1024], dtype=np.float64)) for i in range(0, len(P), 1024))
    return 0.5 * _gw_energy_lowmem(C1, C2, P), float(fused_penalty * lin)

def _solve_coupling(C_M, C_N, gw_params, M, fused_penalty, fgw_alpha, device, verbose):
    """Solve GW (M is None) or FGW; return P and, for FGW, the objective terms."""
    if M is None:
        if verbose: print(f"[mgw.core] solving GW with {gw_params}")
        P = solve_gw_ott(C_M, C_N, **gw_params)
        if verbose: print(f"[mgw.core] coupling: shape={P.shape}, mass={P.sum():.6f}")
        return P, None
    if verbose: print(f"[mgw.core] solving FGW (alpha={fgw_alpha}, fused_penalty={fused_penalty:.4g}) with {gw_params}")
    P = solve_gw_ott(C_M, C_N, M=M, fused_penalty=fused_penalty, **gw_params)
    gw_term, lin_term = _fgw_objective(C_M, C_N, P, M, fused_penalty, device)
    if verbose:
        print(f"[mgw.core] coupling: shape={P.shape}, mass={P.sum():.6f}")
        print(f"[mgw.core] FGW terms: 0.5*GW={gw_term:.4e}  penalty*<M,P>={lin_term:.4e}  "
              f"(linear share {lin_term / (gw_term + lin_term + 1e-30):.1%})")
    return P, dict(gw=gw_term, lin=lin_term)

def _check_fgw_args(pre, fgw_alpha, fused_source, fused_weight, fused_cost) -> float:
    """Validate the FGW settings and return OTT's fused_penalty = (1−α)/(2α)."""
    if not (0.0 < fgw_alpha <= 1.0):
        raise ValueError(f"fgw_alpha must be in (0, 1], got {fgw_alpha}")
    if fused_source not in ("field", "rep"):
        raise ValueError(f"fused_source must be 'field' or 'rep', got {fused_source!r}")
    if fused_weight not in ("rho2", "rho", "uniform"):
        raise ValueError(f"fused_weight must be 'rho2', 'rho' or 'uniform', got {fused_weight!r}")
    if fused_cost is None:
        if not pre["config"].get("use_cca_feeler", False):
            raise ValueError("use_fgw=True needs a joint feature space: run mgw_preprocess with use_cca_feeler=True "
                             "(per-slice PCA features are not comparable across slices), or pass fused_cost=.")
    else:
        shape = (pre["xs"].shape[0], pre["xs2"].shape[0])
        if np.shape(fused_cost) != shape:
            raise ValueError(f"fused_cost must have shape {shape}, got {np.shape(fused_cost)}")
    return (1.0 - fgw_alpha) / (2.0 * fgw_alpha)

def _fgw_file_tag(fgw_alpha, fused_comps, fused_cost) -> str:
    tag = f"_fgw{fgw_alpha:g}"
    if fused_cost is not None: return tag + "_custom"
    return tag + ("" if fused_comps is None else f"_k{fused_comps}")

def _fgw_config(use_fgw, fgw_alpha, fused_penalty, fused_source, fused_comps, fused_weight, fused_cost) -> Dict[str, Any]:
    if not use_fgw:
        return dict(use_fgw=False)
    return dict(use_fgw=True, fgw_alpha=fgw_alpha, fused_penalty=fused_penalty,
                fused_source="custom" if fused_cost is not None else fused_source,
                fused_comps=fused_comps, fused_weight=fused_weight)

def _fused_weights(cca_corr, n: int, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(n)
    if cca_corr is None:
        raise ValueError(f"fused_weight={mode!r} needs the CCA canonical correlations, which this `pre` does not "
                         "contain; rerun mgw_preprocess or use fused_weight='uniform'.")
    rho = np.abs(np.asarray(cca_corr, dtype=np.float64)[:n])
    return rho**2 if mode == "rho2" else rho

def _joint_sqdist(E_A, E_B, weights) -> np.ndarray:
    """Weighted squared distance between per-slice z-scored embeddings, scaled to [0, 1] (float32)."""
    def _z(E):
        E = np.asarray(E, dtype=np.float64)
        return (E - E.mean(0)) / (E.std(0) + 1e-8)
    s = np.sqrt(np.asarray(weights, dtype=np.float64))
    U = (_z(E_A) * s).astype(np.float32)
    V = (_z(E_B) * s).astype(np.float32)
    M = (U * U).sum(1)[:, None] + (V * V).sum(1)[None, :] - 2.0 * (U @ V.T)
    np.maximum(M, 0.0, out=M)
    q = np.quantile(M, 0.99)
    M /= (q + 1e-12)
    np.clip(M, 0.0, 1.0, out=M)
    return M

def _build_fused_cost(pre, phi, psi, fused_source, fused_comps, fused_weight, fused_cost, verbose) -> np.ndarray:
    """Cross-slice cost M (n×m) in the CCA joint space, or the user-supplied fused_cost."""
    if fused_cost is not None:
        if verbose: print("[mgw.core] FGW cross cost: user-supplied fused_cost")
        return np.asarray(fused_cost, dtype=np.float32)
    if fused_source == "field":
        E_A = plotting.predict_on_model(phi, pre["xs"])
        E_B = plotting.predict_on_model(psi, pre["xs2"])
    else:
        E_A, E_B = pre["X_rep"], pre["Z_rep"]
    n = min(E_A.shape[1], E_B.shape[1])
    if fused_comps is not None:
        n = min(n, int(fused_comps))
    cca_corr = pre.get("cca_corr")
    if cca_corr is None and pre.get("feeler") is not None:
        cca_corr = pre["feeler"].get("cca_corr")
    w = _fused_weights(cca_corr, n, fused_weight)
    if verbose:
        print(f"[mgw.core] FGW cross cost: source={fused_source}, comps={n}, weights={np.round(w, 3)}")
    return _joint_sqdist(E_A[:, :n], E_B[:, :n], w)

def _to_unit_square(x: np.ndarray) -> np.ndarray:
    return util.normalize_coords_to_unit_square(np.asarray(x, dtype=float))

def _norm_geod(D: np.ndarray) -> np.ndarray:
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    q = np.quantile(D[np.triu_indices_from(D, 1)], 0.99)
    return D / (q + 1e-12)

def _maybe_subsample(adata, max_obs: Optional[int], seed: int = 42):
    if max_obs is None or adata.n_obs <= max_obs:
        return adata.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=max_obs, replace=False)
    return adata[idx].copy()

