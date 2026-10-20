"""Strategy B1: one plane per query through a 3D stack (cells 5, 22 and 24 of demos/human_mouse_align_3d.ipynb).

A query is a dict(name, xy, emb, labels, Y, full_obs, L[, Y_full]); the stack is the dict of `bench.stack`, with
`emb` holding the features the query is compared with, `Y` the evaluation genes and `L` {section: L_pair}.

1. Coarse: a direct 2D -> 3D FGW of the query into the stack (relaxed stack marginal); a plane fitted to its image.
2. Real sections: a 2D alignment to each; z from the parabola through the best one and its neighbours.
3. Fine: a z scan over virtual sections, a tilt grid at the best z0, a z refinement.
4. The final alignment at the best plane, with the comparisons (laterality partner, best real section).
"""
import gc
import json
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from mgw import crossspecies as cs, stack3d as s3
from mgw.bench import solvers
from mgw.bench.solvers import FGWParams, fgw_objective


@dataclass(frozen=True)
class MGWParams:
    phi_arc: tuple = (128, 256, 256, 128)
    knn_k: int = 12
    geodesic_eps: float = 1e-1
    lr: float = 1e-3
    n_iter: int = 20_000
    fused_alpha: float = 0.1
    gw_params: dict = field(default_factory=lambda: dict(verbose=False, inner_maxit=3000, outer_maxit=3000,
                                                          inner_tol=1e-7, outer_tol=1e-7, epsilon=1e-4))
    seed: int = 1


@dataclass(frozen=True)
class PlaceParams:
    method: str = "fgw"            # final alignments: "fgw" or "mgw"
    fgw: FGWParams = FGWParams()
    coarse_alpha: float = 1.0      # direct 2D -> 3D FGW: weight of the cross cost (OTT fused_penalty)
    coarse_eps: float = 5e-3
    tau_stack: float = 0.1         # relaxation of the stack marginal in the coarse step
    z_window: object = None        # half-width of the z0 scan around the coarse z0; None: the whole stack
    n_z: int = 15
    tilt_max: float = 15.0
    n_tilt: int = 3
    min_coverage: float = 0.6
    virtual_features: str = "rescaled"   # "rescaled", "cost" or "linear"
    z_select: str = "scan"         # "scan" or "sections"
    mirror_test: bool = True
    max_obs_query: int = 1000      # tiles of a virtual section
    mgw: MGWParams = MGWParams()

    @classmethod
    def from_mapping(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in names})


@dataclass
class PlotContext:
    palette: dict
    z_stretch: float = 1.0
    stack_name: str = ""


METHOD_NAME = {"mgw": "MGW", "fgw": "FGW"}


# ---------- 2D alignments
def solve_2d(xa, fa, xb, fb, M, p: PlaceParams, method="fgw", tag=""):
    """Coupling (n_a x n_b) of two point sets with the cross cost M; fa, fb are the features MGW fits phi, psi to."""
    if method == "fgw":
        return solvers.fgw(xa, xb, M, p.fgw)[0]
    import torch
    from mgw import mgw
    g = p.mgw
    pre = mgw.mgw_preprocess(solvers.light(xa, fa), solvers.light(xb, fb), PCA_comp=fa.shape[1], use_cca_feeler=False,
                             cross_key="X_emb", coords="isotropic", rep_norm="zscore", verbose=False)
    pre["M"] = np.asarray(M, dtype=np.float32)
    out = mgw.mgw_align_core(pre, widths=g.phi_arc, lr=g.lr, niter=g.n_iter, knn_k=g.knn_k,
                             geodesic_eps=g.geodesic_eps, gw_params=g.gw_params, seed=g.seed, deterministic=True,
                             n_restarts=1, fused_alpha=g.fused_alpha, verbose=False, tag=tag)
    P = np.asarray(out["P"])
    del out, pre
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return P


def align_2d(q, xb, fb, M, p: PlaceParams, method="fgw", tag=""):
    """Align a query to a section; with mirror_test also its mirror image. Returns the better by the objective."""
    best = None
    for mirrored in ((False, True) if p.mirror_test else (False,)):
        xa = q["xy"] * [-1, 1] if mirrored else q["xy"]
        P = solve_2d(xa, q["emb"], xb, fb, M, p, method, tag=tag)
        obj = fgw_objective(xa, xb, M, P, p.fgw.alpha)
        if best is None or obj < best["obj"]:
            best = dict(P=P, obj=obj, mirrored=mirrored, det=cs.orientation_det(xa, xb, P) * (-1 if mirrored else 1))
    return best


def evaluate(P, q, labels_b, Y_b, full=True):
    """Label transfer and held-out gene co-expression (on the points, and on every bin of the query if full)."""
    lt = cs.label_transfer(P, q["labels"], labels_b)
    r, r0 = cs.coexpression(P, q["Y"], Y_b)
    out = dict(label_acc=lt["acc"], label_bal_acc=lt["bal_acc"], coexpr_r=float(np.nanmedian(r)),
               coexpr_r_shuffled=float(np.nanmedian(r0)), pred=lt["pred"], r=r, r0=r0)
    if full and q.get("Y_full") is not None:
        fs = cs.full_scores(P, q["full_obs"], labels_b, lt["pred"], Y_b, q["Y_full"])
        out.update(label_acc_full=fs["acc"], label_bal_acc_full=fs["bal_acc"], coexpr_r_full=fs["median_r"],
                   coexpr_r_shuffled_full=fs["median_r0"])
    return out


# ---------- placement in the stack
def coarse_place(q, ST, emb_key, p: PlaceParams):
    M = cdist(q["emb"], ST["emb"][emb_key])
    M /= np.quantile(M, 0.99)
    if p.method == "mgw":
        from mgw import mgw
        g = p.mgw
        S = solvers.light(ST["xyz"], ST["emb"][emb_key])
        S.obsm["spatial3d"] = ST["xyz"]
        pre = mgw.mgw_preprocess(solvers.light(q["xy"], q["emb"]), S, PCA_comp=q["emb"].shape[1], use_cca_feeler=False,
                                 cross_key="X_emb", coords="isotropic", rep_norm="zscore", spatial_key_B="spatial3d",
                                 verbose=False)
        pre["graph_N"] = s3.stack_graph(pre["xs2"], ST["section"], ST["pairs"], k_in=g.knn_k)
        out = mgw.mgw_align_core(pre, widths=g.phi_arc, lr=g.lr, niter=g.n_iter, knn_k=g.knn_k,
                                 geodesic_eps=g.geodesic_eps,
                                 gw_params={**g.gw_params, "epsilon": p.coarse_eps, "tau_b": p.tau_stack},
                                 seed=g.seed, deterministic=True, n_restarts=1, fused_alpha=p.coarse_alpha,
                                 verbose=False)
        P = np.asarray(out["P"])
        del out, pre
    else:
        from mgw.gw import solve_gw_ott
        Cs = cdist(ST["xyz"], ST["xyz"])
        s = Cs.max()
        P = solve_gw_ott(cdist(q["xy"], q["xy"]) / s, Cs / s, M=M, alpha=p.coarse_alpha, epsilon=p.coarse_eps,
                         tau_a=1.0, tau_b=p.tau_stack, inner_maxit=3000, outer_maxit=200, verbose=False)
        del Cs
    gc.collect()
    fit = s3.plane_from_coupling(ST["xyz"], P, ST["frame"], section=ST["section"])
    xy_img, _ = s3.plane_coords(fit["bary"], fit["plane"], ST["frame"])
    fit["det"] = cs.orientation_det(q["xy"], xy_img, np.eye(len(xy_img)))
    fit["P"] = P
    return fit


def virtual(q, ST, pl, emb_key, M_scale, p: PlaceParams):
    """Virtual section of a plane, its interpolated features and the query's cross cost (None if too small)."""
    vs = s3.VirtualSection(ST["xyz"], ST["pairs"], pl, ST["frame"], max_obs=p.max_obs_query)
    if vs.n_obs < 50 or vs.coverage < p.min_coverage:
        return vs, None, None
    F = ST["emb"][emb_key]
    if p.virtual_features == "cost":
        return vs, vs.interp(F), vs.interp_cost(q["emb"], F) / M_scale
    fv = vs.interp(F) if p.virtual_features == "linear" else vs.interp_rescaled(F)
    return vs, fv, cdist(q["emb"], fv) / M_scale


def scan(q, ST, emb_key, planes, M_scale, p: PlaceParams, mirror=None):
    """Align q to the virtual section of each plane (FGW); one row per plane."""
    rows = []
    for pl in planes:
        vs, fv, M = virtual(q, ST, pl, emb_key, M_scale, p)
        row = dict(**pl, coverage=vs.coverage, n_virtual=vs.n_obs)
        if fv is not None:
            if mirror is None:
                res = align_2d(q, vs.xy, fv, M, p)
            else:
                xa = q["xy"] * [-1, 1] if mirror else q["xy"]
                P = solve_2d(xa, q["emb"], vs.xy, fv, M, p)
                res = dict(P=P, obj=fgw_objective(xa, vs.xy, M, P, p.fgw.alpha), mirrored=mirror,
                           det=cs.orientation_det(xa, vs.xy, P) * (-1 if mirror else 1))
            ev = evaluate(res["P"], q, vs.labels(ST["labels"]), vs.interp(ST["Y"]), full=False)
            row.update(objective=res["obj"], mirrored=res["mirrored"], det=res["det"],
                       label_acc=ev["label_acc"], coexpr_r=ev["coexpr_r"])
        rows.append(row)
    return pd.DataFrame(rows)


def place_query(q, ST, p: PlaceParams, emb_key="X_ortho", final_method=None, compare=True, plots=None):
    """Coarse plane, plane search, final alignment and scores for one query. Returns (summary row, artifacts).

    plots: a PlotContext to draw the per-query figures, or None.
    """
    final_method = final_method or p.method
    name = q["name"]
    M_scale = np.quantile(cdist(q["emb"], ST["emb"][emb_key]), 0.99)
    z_lo, z_hi = ST["z"].min(), ST["z"].max()

    # coarse
    co = coarse_place(q, ST, emb_key, p)
    z_c = co["plane"]["z0"]
    print(f"[{name}] coarse plane z0 {z_c:.3f} (stack {z_lo:.3f}–{z_hi:.3f}), tilt "
          f"{co['plane']['theta_u']:+.1f}°/{co['plane']['theta_v']:+.1f}°, planarity rms {co['rms']:.3f}, det {co['det']:+d}")

    # real sections: one 2D alignment each, and z from the parabola through the best one and its neighbours
    real_obj = {}
    for s in ST["names"]:
        m = ST["section"] == s
        real_obj[s] = align_2d(q, ST["xyz"][m, :2], ST["emb"][emb_key][m],
                               cdist(q["emb"], ST["emb"][emb_key][m]) / M_scale, p)
    z_s, edge_s = s3.z_from_sections(ST["z"], [real_obj[s]["obj"] for s in ST["names"]])

    # fine: z scan, tilt grid, z refine
    lo, hi = (z_lo, z_hi) if p.z_window is None else (max(z_lo, z_c - p.z_window), min(z_hi, z_c + p.z_window))
    zs = np.linspace(lo, hi, p.n_z) if hi > lo else np.array([lo])
    zs = np.unique(np.r_[zs, ST["z"][(ST["z"] >= lo) & (ST["z"] <= hi)]])  # the real sections' z as well
    sz = scan(q, ST, emb_key, [s3.plane(z) for z in zs], M_scale, p).assign(stage="z scan")
    if "objective" not in sz or sz.objective.notna().sum() == 0:
        print(f"[{name}] no virtual section with coverage >= {p.min_coverage}")
        return None, None
    b = sz.loc[sz.objective.idxmin()]
    mirror = bool(b.mirrored) if p.mirror_test else False
    tilts = np.linspace(-p.tilt_max, p.tilt_max, p.n_tilt) if p.n_tilt > 1 else np.array([0.0])
    st = scan(q, ST, emb_key, [s3.plane(b.z0, tu, tv) for tu in tilts for tv in tilts if (tu, tv) != (0, 0)],
              M_scale, p, mirror).assign(stage="tilt grid")
    allr = pd.concat([sz, st], ignore_index=True)
    b = allr.loc[allr.objective.idxmin()]
    dz = (hi - lo) / max(p.n_z - 1, 1) / 2
    if dz > 0:
        sr = scan(q, ST, emb_key, [s3.plane(z, b.theta_u, b.theta_v) for z in (b.z0 - dz, b.z0 + dz) if lo <= z <= hi],
                  M_scale, p, mirror).assign(stage="z refine")
        allr = pd.concat([allr, sr], ignore_index=True)
    b_scan = allr.loc[allr.objective.idxmin()]
    edge_scan = bool(np.isclose(b_scan.z0, z_lo) or np.isclose(b_scan.z0, z_hi))
    b, edge = b_scan, edge_scan
    if p.z_select == "sections":
        bs = scan(q, ST, emb_key, [s3.plane(z_s)], M_scale, p).assign(stage="sections").iloc[0]
        if pd.notna(bs.get("objective")):
            b, edge = bs, edge_s
        else:
            print(f"[{name}] the virtual section at z0 {z_s:.3f} is too small; using the scan")
    best_pl = s3.plane(b.z0, b.theta_u, b.theta_v)

    # final alignment at the best plane
    vs, fv, M = virtual(q, ST, best_pl, emb_key, M_scale, p)
    lab_v, Y_v = vs.labels(ST["labels"]), vs.interp(ST["Y"])
    xa = q["xy"] * [-1, 1] if b.mirrored else q["xy"]
    P = solve_2d(xa, q["emb"], vs.xy, fv, M, p, final_method, tag=f"{name}_3d")
    ev = evaluate(P, q, lab_v, Y_v)
    row = dict(query=name, L_query=q["L"], z0=b.z0, theta_u=b.theta_u, theta_v=b.theta_v, coverage=b.coverage,
               edge=edge, mirrored=bool(b.mirrored), det=int(b.det), objective=fgw_objective(xa, vs.xy, M, P, p.fgw.alpha),
               z0_coarse=z_c, theta_u_coarse=co["plane"]["theta_u"], theta_v_coarse=co["plane"]["theta_v"],
               planarity_rms=co["rms"], det_coarse=co["det"],
               z0_scan=b_scan.z0, theta_u_scan=b_scan.theta_u, theta_v_scan=b_scan.theta_v, edge_scan=edge_scan,
               z0_sections=z_s, edge_sections=edge_s,
               nearest_section=ST["names"][int(np.argmin(np.abs(ST["z"] - b.z0)))],
               **{k: v for k, v in ev.items() if k not in ("pred", "r", "r0")})

    # comparisons: the laterality partner and the best real section (2D, FGW to choose, then final_method)
    comp = {}
    if compare:
        L_ok = {s: l for s, l in ST.get("L", {}).items() if np.isfinite(l)}
        L_partner = min(L_ok, key=lambda s: abs(L_ok[s] - q["L"])) if np.isfinite(q["L"]) and L_ok else None
        best_real = min(real_obj, key=lambda s: real_obj[s]["obj"])
        for kind, s in (("L partner", L_partner), ("best real section", best_real)):
            if s is None:
                continue
            m = ST["section"] == s
            P2 = real_obj[s]["P"]
            if final_method != "fgw":
                xa2 = q["xy"] * [-1, 1] if real_obj[s]["mirrored"] else q["xy"]
                Mr = cdist(q["emb"], ST["emb"][emb_key][m]) / M_scale
                P2 = solve_2d(xa2, q["emb"], ST["xyz"][m, :2], ST["emb"][emb_key][m], Mr, p, final_method,
                              tag=f"{name}_{s}")
            ev2 = evaluate(P2, q, ST["labels"][m], ST["Y"][m])
            comp[kind] = dict(section=s, objective=real_obj[s]["obj"], P=P2,
                              **{k: v for k, v in ev2.items() if k not in ("pred", "r", "r0")})
            row.update({f"{kind.replace(' ', '_')}_{k}": v for k, v in comp[kind].items() if k != "P"})
    row["real_section_objectives"] = json.dumps({s: round(r["obj"], 5) for s, r in real_obj.items()})

    if plots is not None:
        plot_placement(q, ST, plots, allr, b_scan, z_s, z_c, real_obj, co, vs, lab_v, ev, xa, P, row, comp,
                       METHOD_NAME[final_method])
    return row, dict(P=P, plane=best_pl, scan=allr, coarse=co, virtual=vs,
                     comparisons={k.replace(" ", "_"): (c["section"], c["P"]) for k, c in comp.items()})


def drop_section(ST, k, P_loo, Y=None):
    """The stack without section k; its neighbours joined by their direct coupling P_loo (from loo_check)."""
    keep = ST["section"] != ST["names"][k]
    new_idx = np.full(len(keep), -1)
    new_idx[keep] = np.arange(keep.sum())
    pairs = []
    for kk, (i, j, w) in enumerate(ST["pairs"]):
        if kk == k - 1:   # (k-1, k) and (k, k+1) become one pair (k-1, k+1)
            off_a = np.flatnonzero(ST["section"] == ST["names"][k - 1])[0]
            off_b = np.flatnonzero(ST["section"] == ST["names"][k + 1])[0]
            i2, j2, w2 = s3.displacement_pairs(P_loo, off_a, off_b)
            pairs.append((new_idx[i2], new_idx[j2], w2))
        elif kk != k:
            pairs.append((new_idx[i], new_idx[j], w))
    return dict(ST, names=[s for s in ST["names"] if s != ST["names"][k]], z=np.delete(ST["z"], k),
                section=ST["section"][keep], xyz=ST["xyz"][keep], labels=ST["labels"][keep],
                emb={e: x[keep] for e, x in ST["emb"].items()}, Y=None if Y is None else Y[keep], pairs=pairs)


# ---------- plots
def plot_stack_3d(ST, palette, extra=(), title="", z_stretch=1.0):
    """The stack in 3D, one trace per organ class; `extra`: (xyz, colours, name) traces."""
    import plotly.graph_objects as go
    fig = go.Figure()
    coarse = np.where(np.isin(ST["labels"], list(palette)), ST["labels"], "Other")
    for c, col in palette.items():
        m = coarse == c
        if m.any():
            x = ST["xyz"][m]
            fig.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2] * z_stretch, mode="markers", name=c,
                                       marker=dict(size=2, color=col, opacity=0.35 if extra else 0.8)))
    for xyz, colours, name in extra:
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2] * z_stretch, mode="markers", name=name,
                                   marker=dict(size=3.5, color=colours, line=dict(width=0.5, color="black"))))
    fig.update_layout(title=title, height=650, scene=dict(aspectmode="data", zaxis_title=f"z (×{z_stretch:g})"),
                      legend=dict(itemsizing="constant"))
    fig.show()


def plot_placement(q, ST, ctx, allr, b_scan, z_s, z_c, real_obj, co, vs, lab_v, ev, xa, P, row, comp, method_name):
    import matplotlib.pyplot as plt
    name = q["name"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 3.6), gridspec_kw=dict(width_ratios=[2.2, 1, 1.3]))
    g = allr[allr.stage != "tilt grid"].sort_values("z0")
    g0 = g[(g.theta_u == 0) & (g.theta_v == 0)]
    ax[0].plot(g0.z0, g0.objective, "k-o", ms=3, label="objective (virtual, tilt 0)")
    ax[0].plot(ST["z"], [real_obj[s]["obj"] for s in ST["names"]], "s", color="C3", ms=6, label="real sections")
    ax[0].axvline(b_scan.z0, color="C2", lw=1.5, label=f"scan z0 {b_scan.z0:.3f}")
    ax[0].axvline(z_s, color="C3", ls=":", lw=1.5, label=f"sections z0 {z_s:.3f} (parabola)")
    ax[0].axvline(z_c, color="C1", ls="--", lw=1, label=f"coarse z0 {z_c:.3f}")
    ax[0].set_xlabel("z0"); ax[0].set_ylabel("FGW objective")
    a2 = ax[0].twinx()
    a2.plot(g0.z0, g0.label_acc, ":", color="C0", label="label acc (eval)")
    a2.plot(g0.z0, g0.coexpr_r, ":", color="C4", label="median r (eval)")
    a2.set_ylabel("evaluation")
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.12, 1))
    ax[0].set_title(f"{name}: plane search")
    tg = allr[allr.stage == "tilt grid"]
    if len(tg):
        H = pd.concat([tg, g0[np.isclose(g0.z0, tg.z0.iloc[0])]]).pivot_table(index="theta_v", columns="theta_u",
                                                                              values="objective")
        im = ax[1].imshow(H.to_numpy(), cmap="viridis_r", origin="lower")
        ax[1].set_xticks(range(H.shape[1]), [f"{x:g}" for x in H.columns])
        ax[1].set_yticks(range(H.shape[0]), [f"{x:g}" for x in H.index])
        ax[1].set_xlabel("θ_u (°)"); ax[1].set_ylabel("θ_v (°)"); ax[1].set_title(f"tilt at z0 {tg.z0.iloc[0]:.3f}")
        fig.colorbar(im, ax=ax[1])
    else:
        ax[1].axis("off")
    if co["mass_per_section"] is not None:
        ms = co["mass_per_section"].reindex(ST["names"])
        ax[2].bar(range(len(ms)), ms.to_numpy(), color="C1")
        ax[2].set_xticks(range(len(ms)), [f"{s}\nz {z:.2f}" for s, z in zip(ST["names"], ST["z"])], rotation=60,
                         ha="right", fontsize=7)
        ax[2].set_ylabel("share of transported mass"); ax[2].set_title("coarse 2D→3D coupling")
    plt.tight_layout(); plt.show()

    cs.plot_label_panels([(vs.xy, lab_v, f"virtual section z0 {b_scan.z0:.3f}"), (q["xy"], q["labels"], f"{name}: annotated"),
                          (q["xy"], ev["pred"], f"{name}: transferred from the stack")], ctx.palette)
    cs.plot_coupling_lines(xa, vs.xy, P, f"{method_name}: {name} → virtual section", (name, "virtual"))
    q3 = (P @ vs.xyz) / (P.sum(1, keepdims=True) + 1e-30)
    plot_stack_3d(ST, ctx.palette, extra=[(co["bary"], cs.colors_for(q["labels"], ctx.palette), f"{name}: coarse image"),
                                          (q3, cs.colors_for(q["labels"], ctx.palette), f"{name}: final placement")],
                  title=f"{name} in {ctx.stack_name}", z_stretch=ctx.z_stretch)
    print(pd.DataFrame([{k: row[k] for k in ("objective", "label_acc", "label_bal_acc", "coexpr_r", "coexpr_r_shuffled",
                                              "label_acc_full", "coexpr_r_full") if k in row}]
                       + [{k: v for k, v in comp[c].items() if k not in ("section", "P")} for c in comp],
                       index=["3D placement"] + [f"{c} ({comp[c]['section']})" for c in comp]).round(3).to_string())
