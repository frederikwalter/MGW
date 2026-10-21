"""A 3D stack from the serial sections of one embryo (cells 9-17 of demos/human_mouse_align_3d.ipynb).

1. Order: the file order, or inferred from the data (FGW objective between all pairs of sections, then the shortest
   open path; `stack3d.infer_order`).
2. Registration: neighbours matched by FGW (moscot), refined by OT-ICP (rotation or reflection), composed into the
   frame of the middle section and re-fitted to both neighbours (global sweeps).
3. Gaps: from the in-plane variogram (`stack3d.estimate_gap`), or given.
4. The stack: every point with (x, y) in the reference frame and z from the gaps, plus the displacement pairs of the
   neighbour couplings (the edges that virtual sections are cut along).

The stack is a dict (as `ST` in the notebook): names, z, section, xyz, labels, emb {name: features}, pairs, frame,
plus whatever the caller adds (Y, comp, L).
"""
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from mgw import crossspecies as cs, stack3d as s3
from mgw.bench import solvers


@dataclass(frozen=True)
class StackParams:
    order: str = "auto"          # "file", "infer", or "auto" ("file" for HESTA, "infer" for MOSTA)
    order_subsample: int = 400   # points per section in the order inference
    pair_alpha: float = 0.8      # spatial weight of the first FGW between neighbouring sections (moscot)
    pair_eps: float = 1e-2
    icp_lambda: float = 0.5      # OT-ICP: registered spatial distance against expression distance
    icp_iter: int = 15
    icp_eps: float = 5e-3
    section_flip: object = "auto"  # "auto": rotation or reflection by the lower cost; False: rotations only
    global_sweeps: int = 2
    z_gaps: object = None        # None: estimated; or n_sections - 1 gaps in stack units

    @classmethod
    def from_mapping(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in names})


def order_mode(params, dataset):
    return params.order if params.order != "auto" else ("file" if dataset == "HESTA" else "infer")


def subsample(n_obs, n=400, seed=0):
    return np.random.default_rng(seed).choice(n_obs, min(n, n_obs), replace=False)


def infer_section_order(xy_raw, feats, params, fgw_alpha, seed=0):
    """Order of the sections from the FGW objective between every pair (at most order_subsample points each).

    xy_raw, feats: lists in file order. Returns (order, D); the order runs in the file order's direction.
    """
    n = len(xy_raw)
    D = np.zeros((n, n))
    if n <= 2:
        return list(range(n)), D
    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = subsample(len(xy_raw[a]), params.order_subsample, seed), subsample(len(xy_raw[b]), params.order_subsample, seed)
            xa, xb, fa, fb = xy_raw[a][ia], xy_raw[b][ib], feats[a][ia], feats[b][ib]
            P = solvers.moscot_pair(xa, fa, xb, fb, params.pair_alpha, params.pair_eps)
            M = cdist(fa, fb)
            M /= np.quantile(M, 0.99)
            D[a, b] = D[b, a] = solvers.fgw_objective(xa, xb, M, P, fgw_alpha)
    order = s3.infer_order(D)
    if spearmanr(order, range(n))[0] < 0:   # same direction as the file order where they agree
        order = order[::-1]
    return list(order), D


def register_sections(names, xy_raw, feats, params):
    """Rigid registration of neighbouring sections into the frame of the middle one (in stack order).

    Returns dict(T, couplings, reg_rows, flips, reg_xy, ref).
    """
    n = len(names)
    ref = n // 2
    pair_T, couplings, rows = [], [], []
    for k in range(n - 1):
        fa, fb = feats[k], feats[k + 1]
        P0 = solvers.moscot_pair(xy_raw[k], fa, xy_raw[k + 1], fb, params.pair_alpha, params.pair_eps)
        det0 = cs.orientation_det(xy_raw[k + 1], xy_raw[k], P0.T)
        reg = s3.register_pair(xy_raw[k], xy_raw[k + 1], fa, fb, P0, flip=params.section_flip,
                               lam=params.icp_lambda, n_iter=params.icp_iter, epsilon=params.icp_eps)
        pair_T.append(s3.homogeneous(reg["R"], reg["t"]))
        couplings.append(reg["P"])
        rows.append(dict(pair=f"{names[k]} → {names[k + 1]}", fgw_det=det0, reflected=reg["reflected"],
                         angle=np.degrees(np.arctan2(reg["R"][0, 1], reg["R"][0, 0])),
                         shift=float(np.linalg.norm(reg["t"])), rmsd=reg["rmsd"],
                         icp_cost_rotation=reg["cost_rotation"], icp_cost_reflection=reg["cost_reflection"],
                         icp_iters=len(reg["history"])))
    T = s3.compose_to_reference(pair_T, ref) if n > 1 else [np.eye(3)]
    flips = [bool(np.linalg.det(t[:2, :2]) < 0) for t in T]
    if params.global_sweeps and n > 1:
        T = s3.global_sweeps(list(xy_raw), T, couplings, n_sweeps=params.global_sweeps, ref=ref, flips=flips)
    reg_xy = [s3.apply(T[k], xy_raw[k]) for k in range(n)]
    return dict(T=T, couplings=couplings, reg_rows=pd.DataFrame(rows), flips=flips, reg_xy=reg_xy, ref=ref)


def estimate_gaps(names, reg_xy, feats, z_gaps=None, dL=None):
    """z of the sections from the variogram gaps (or z_gaps). Returns dict(z, gaps, gaps_df, vg)."""
    n = len(names)
    if n == 1:
        return dict(z=np.zeros(1), gaps=[], gaps_df=pd.DataFrame(), vg=None)
    vg = s3.inplane_variogram(reg_xy, feats)
    rows = [dict(pair=f"{names[k]} → {names[k + 1]}",
                 **s3.estimate_gap(reg_xy[k], reg_xy[k + 1], feats[k], feats[k + 1], vg),
                 dL_raw=np.nan if dL is None else dL[k])
            for k in range(n - 1)]
    gaps_df = pd.DataFrame(rows)
    gaps = list(z_gaps) if z_gaps is not None else gaps_df.gap.tolist()
    if z_gaps is not None:
        gaps_df["gap_used"] = gaps
    gaps_df["share_gap"] = np.asarray(gaps) / max(sum(gaps), 1e-12)
    gaps_df["share_dL"] = gaps_df.dL_raw / max(gaps_df.dL_raw.sum(), 1e-12)
    return dict(z=np.r_[0.0, np.cumsum(gaps)], gaps=gaps, gaps_df=gaps_df, vg=vg)


def assemble(names, reg_xy, z, labels, feats, couplings, feat_key="X_feat"):
    """The stack dict from registered sections (stack order)."""
    n_obs = [len(x) for x in reg_xy]
    offsets = np.r_[0, np.cumsum(n_obs)]
    ST = dict(
        names=list(names), z=np.asarray(z, dtype=np.float64),
        section=np.concatenate([[s] * n for s, n in zip(names, n_obs)]),
        xyz=np.vstack([np.c_[reg_xy[k], np.full(n_obs[k], z[k])] for k in range(len(names))]),
        labels=np.concatenate([np.asarray(l, dtype=str) for l in labels]),
        emb={feat_key: np.vstack(feats)},
        pairs=[s3.displacement_pairs(couplings[k], offsets[k], offsets[k + 1]) for k in range(len(names) - 1)],
        offsets=offsets,
    )
    c, u, v = s3.principal_axes(ST["xyz"][:, :2])
    ST["frame"] = dict(centre=c, u=u, v=v)
    return ST


def build_stack(names, xy_raw, feats, labels, params, *, dataset, fgw_alpha, dL_raw=None, seed=0):
    """Order, registration, gaps and assembly. Inputs are lists in file order; xy_raw in stack units.

    Returns (ST, diag) with diag = dict(order, mode, D, reg, gaps).
    """
    mode = order_mode(params, dataset)
    if mode == "infer" and len(names) > 2:
        order, D = infer_section_order(xy_raw, feats, params, fgw_alpha, seed=seed)
    else:
        order, D = list(range(len(names))), np.zeros((len(names), len(names)))
    pick = lambda xs: [xs[i] for i in order]
    names, xy_raw, feats, labels = pick(list(names)), pick(list(xy_raw)), pick(list(feats)), pick(list(labels))
    dL = None
    if dL_raw is not None:
        Ls = pick(list(dL_raw))
        dL = [abs(Ls[k + 1] - Ls[k]) for k in range(len(Ls) - 1)]
    reg = register_sections(names, xy_raw, feats, params)
    gp = estimate_gaps(names, reg["reg_xy"], feats, params.z_gaps, dL)
    ST = assemble(names, reg["reg_xy"], gp["z"], labels, feats, reg["couplings"])
    return ST, dict(order=order, mode=mode, D=D, reg=reg, gaps=gp)


def skip_coupling(xa, xb, fa, fb, params):
    """The direct coupling of two registered sections that are not neighbours in the stack (k and k + 2, say):
    FGW (moscot), refined by OT-ICP without reflection. Both already sit in the stack frame, so the coupling only
    supplies the displacement pairs across the wider gap (leave-one-out, odd/even halves)."""
    P0 = solvers.moscot_pair(xa, fa, xb, fb, params.pair_alpha, params.pair_eps)
    return s3.register_pair(xa, xb, fa, fb, P0, flip=False, lam=params.icp_lambda, n_iter=params.icp_iter,
                            epsilon=params.icp_eps)["P"]


def loo_check(ST, reg_xy, feats, Y, params, steps=11, max_obs=1000, gaps=None):
    """Leave-one-out check of the stack (cell 17): for every interior section k, its neighbours are coupled
    directly and virtual sections between them are compared with section k (nearest neighbours in xy).

    Returns (loo DataFrame, loo_pairs {k: coupling of k-1 and k+1}).
    """
    offsets, names, labels = ST["offsets"], ST["names"], ST["labels"]
    rows, loo_pairs = [], {}
    for k in range(1, len(names) - 1):
        a, b = k - 1, k + 1
        loo_pairs[k] = skip_coupling(reg_xy[a], reg_xy[b], feats[a], feats[b], params)
        reg = dict(P=loo_pairs[k])
        # both neighbours already sit in the stack frame; the direct coupling only supplies the pairs
        xyz2 = np.vstack([np.c_[reg_xy[a], np.zeros(len(reg_xy[a]))], np.c_[reg_xy[b], np.ones(len(reg_xy[b]))]])
        pairs2 = [s3.displacement_pairs(reg["P"], 0, len(reg_xy[a]))]
        Y2 = np.vstack([Y[offsets[a]:offsets[a + 1]], Y[offsets[b]:offsets[b + 1]]])
        lab2 = np.r_[labels[offsets[a]:offsets[a + 1]], labels[offsets[b]:offsets[b + 1]]]
        tree = cKDTree(reg_xy[k])
        Yk, labk = Y[offsets[k]:offsets[k + 1]], labels[offsets[k]:offsets[k + 1]]
        for t in np.linspace(0, 1, steps):
            vs = s3.VirtualSection(xyz2, pairs2, s3.plane(t), ST["frame"], max_obs=max_obs)
            nn = tree.query(vs.xyz[:, :2])[1]
            rows.append(dict(section=names[k], t=t, label_agree=float((vs.labels(lab2) == labk[nn]).mean()),
                             median_r=float(np.nanmedian(cs.colwise_r(vs.interp(Y2), Yk[nn]))),
                             t_gaps=(gaps[a] / max(gaps[a] + gaps[k], 1e-12)) if gaps else np.nan))
    return pd.DataFrame(rows), loo_pairs


# ---------- cache
def to_arrays(ST, diag):
    """The stack as flat arrays for np.savez."""
    out = dict(names=np.asarray(ST["names"]), z=ST["z"], section=ST["section"].astype(str), xyz=ST["xyz"],
               labels=ST["labels"].astype(str), offsets=ST["offsets"], order=np.asarray(diag["order"]),
               mode=np.asarray(diag["mode"]), D=diag["D"], transforms=np.stack(diag["reg"]["T"]),
               flips=np.asarray(diag["reg"]["flips"]), centre=ST["frame"]["centre"], u=ST["frame"]["u"],
               v=ST["frame"]["v"], gaps=np.asarray(diag["gaps"]["gaps"], dtype=np.float64))
    for k, (i, j, w) in enumerate(ST["pairs"]):
        out[f"pairs_{k}"] = np.c_[i, j, w]
    for k, P in enumerate(diag["reg"]["couplings"]):
        out[f"coupling_{k}"] = P.astype(np.float32)
    for e, F in ST["emb"].items():
        out[f"emb_{e}"] = F
    if diag["gaps"]["gaps_df"] is not None and len(diag["gaps"]["gaps_df"]):
        out["saturated"] = diag["gaps"]["gaps_df"]["saturated"].to_numpy(bool)
    return out


def from_arrays(z):
    """Inverse of to_arrays (an npz mapping)."""
    names = [str(s) for s in z["names"]]
    n_pairs = len(names) - 1
    ST = dict(names=names, z=np.asarray(z["z"]), section=np.asarray(z["section"]).astype(str),
              xyz=np.asarray(z["xyz"]), labels=np.asarray(z["labels"]).astype(str), offsets=np.asarray(z["offsets"]),
              pairs=[(z[f"pairs_{k}"][:, 0].astype(np.int64), z[f"pairs_{k}"][:, 1].astype(np.int64),
                      z[f"pairs_{k}"][:, 2]) for k in range(n_pairs)],
              frame=dict(centre=np.asarray(z["centre"]), u=np.asarray(z["u"]), v=np.asarray(z["v"])),
              emb={k[4:]: np.asarray(z[k]) for k in z.files if k.startswith("emb_")})
    ST["reg_xy"] = [ST["xyz"][ST["offsets"][k]:ST["offsets"][k + 1], :2] for k in range(len(names))]
    ST["transforms"] = np.asarray(z["transforms"])
    ST["order_mode"] = str(z["mode"])
    return ST
