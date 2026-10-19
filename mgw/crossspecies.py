"""Loading, organ-class harmonization and evaluation for HESTA (human) / MOSTA (mouse) alignments.

The functions are those of `demos/human_mouse_align_merged.ipynb`, with the notebook settings
(folders, MAX_OBS, DROP_LABELS, ...) passed as arguments.
"""
import gc
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from mgw import align_vis, plotting, util

DOWNLOAD_URL = {
    "HESTA": "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000394/stomics/{}_HESTA.h5ad",
    "MOSTA": "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/{}.MOSTA.h5ad",
}

# ---------- shared organ classes
# HESTA (`celltype`) and MOSTA (`annotation`) use different vocabularies; both are mapped to coarse
# organ classes. Blood and cavity are `Excluded`; labels without a clear counterpart map to `Other`.
EXCLUDED = "Excluded"
HARMONIZE = {
    # nervous system
    "brain": "Brain", "meninges": "Brain", "choroid plexus": "Brain", "pituitary": "Brain", "pituitary gland": "Brain",
    "spinal cord": "Spinal cord",
    "notochord": "Notochord",
    "ganglion": "Ganglion", "dorsal root ganglion": "Ganglion", "sympathetic nerve": "Ganglion",
    "facial nerve": "Ganglion", "neural crest": "Ganglion",
    "eye": "Eye", "lens": "Eye", "retina": "Eye", "optic nerve": "Eye",
    "inner ear": "Ear", "otic vesicle": "Ear",
    # organs
    "heart": "Heart",
    "liver": "Liver", "hepatic diverticulum": "Liver", "septum transversum": "Liver",
    "gut": "Gut", "gi tract": "Gut", "primitive gut": "Gut", "primitive gut tube": "Gut", "stomach": "Gut",
    "esophagus": "Gut", "pancreas": "Gut", "pancreas bud": "Gut", "pancreas primordium": "Gut",
    "lung": "Lung", "lung primordium": "Lung", "trachea": "Lung",
    "mesonephron": "Urogenital", "mesonephros": "Urogenital", "pronephron": "Urogenital", "kidney": "Urogenital",
    "metanephros": "Urogenital", "urogenital ridge": "Urogenital", "gonad": "Urogenital", "testis": "Urogenital",
    "ovary": "Urogenital", "adrenal gland": "Urogenital", "bladder": "Urogenital", "urethra": "Urogenital",
    "cloaca": "Urogenital",
    # musculoskeletal
    "somite": "Muscle", "myotome": "Muscle", "dermomyotome": "Muscle", "skeletal muscle": "Muscle", "muscle": "Muscle",
    "sclerotome": "Skeleton", "cartilage": "Skeleton", "cartilage primordium": "Skeleton", "bone": "Skeleton",
    "cranial skeleton": "Skeleton",
    "limb bud": "Limb", "limb": "Limb", "limb ectoderm": "Limb",
    # head and surface
    "branchial arch": "Craniofacial", "pharyngeal": "Craniofacial", "jaw and tooth": "Craniofacial",
    "tongue": "Craniofacial", "submandibular gland": "Craniofacial", "olfactory epithelium": "Craniofacial",
    "thyroid": "Craniofacial",
    "epidermis": "Epidermis", "surface ectoderm": "Epidermis",
    "mesenchyme": "Mesenchyme", "head mesenchyme": "Mesenchyme", "connective tissue": "Mesenchyme",
    "mesentery": "Mesenchyme", "mesothelium": "Mesenchyme", "adipose tissue": "Mesenchyme",
    # no clear counterpart
    "smooth muscle": "Other", "mucosal epithelium": "Other", "thymus": "Other", "spleen": "Other",
    # not scored
    "cavity": EXCLUDED, "extraembryonic tissue": EXCLUDED, "erythroblasts": EXCLUDED,
    "blood vessel": EXCLUDED, "endothelium": EXCLUDED, "agm": EXCLUDED,
}
NOT_SCORED = {"Other", EXCLUDED}


def harmonize(labels):
    low = pd.Series(np.asarray(labels, dtype=str)).str.lower()
    unknown = sorted(set(low) - set(HARMONIZE))
    return low.map(HARMONIZE).fillna("Other").to_numpy(dtype=str), unknown


# ---------- files
def section_path(dataset, sec, hesta_dir, mosta_dir):
    return Path(hesta_dir) / f"{sec}_HESTA.h5ad" if dataset == "HESTA" else Path(mosta_dir) / f"{sec}.MOSTA.h5ad"


def check_files(specs, hesta_dir, mosta_dir):
    """Raise with curl commands for every (dataset, section) whose file is missing."""
    missing = [(ds, s) for ds, s in specs if not section_path(ds, s, hesta_dir, mosta_dir).exists()]
    if missing:
        cmds = "\n".join(f"curl -L -C - -o {section_path(ds, s, hesta_dir, mosta_dir)} {DOWNLOAD_URL[ds].format(s)}"
                         for ds, s in missing)
        raise FileNotFoundError(f"missing sections; download them with\n{cmds}")


def score_key(dataset, sec):
    """Row key of a section in laterality_scores.csv."""
    return f"HESTA:{sec}_HESTA" if dataset == "HESTA" else f"MOSTA:{sec}.MOSTA"


def key_name(k):
    return k.split(":", 1)[1].removesuffix("_HESTA").removesuffix(".MOSTA")


def embryo_sections(scores, dataset, stage, embryo):
    """Section names of one embryo in file order (laterality_scores.csv, indexed by key)."""
    s = scores[(scores.dataset == dataset) & (scores.stage == stage) & (scores.embryo.astype(str) == str(embryo))]
    s = s[s.section.notna()].sort_values("section")
    return [key_name(k) for k in s.index]


def load_section(dataset, sec, path, *, max_obs, subsample="voronoi", agg="sum", lloyd_iter=3, drop_labels=()):
    """Full section (drop_labels removed) and its reduction to max_obs points; full.obs['tile'] maps bins to points."""
    adata = sc.read_h5ad(path)
    counts, label_col = ("counts", "celltype") if dataset == "HESTA" else ("count", "annotation")
    # HESTA X is log-normalized, MOSTA X is something else; both keep raw counts in a layer
    adata.X = adata.layers.pop(counts)
    keep = ~adata.obs[label_col].astype(str).str.lower().isin([l.lower() for l in drop_labels]).to_numpy()
    adata = adata[keep]

    # Keep only what the pipeline needs; coordinates may be stored as uint32
    full = ad.AnnData(sp.csr_matrix(adata.X, dtype=np.float32),
                      obs=pd.DataFrame(index=[f"{sec}_{n}" for n in adata.obs_names]),
                      var=pd.DataFrame(index=adata.var_names.astype(str)))
    full.obsm["spatial"] = np.asarray(adata.obsm["spatial"][:, :2], dtype=np.float64)
    full.obs["label"] = adata.obs[label_col].astype(str).to_numpy()
    full.obs["label_h"], unknown = harmonize(full.obs["label"])
    del adata
    gc.collect()

    method = subsample if max_obs else "random"
    out, tile = util.merge_bins(full, max_obs or full.n_obs, method=method, agg=agg, label_key="label_h",
                                lloyd_iter=lloyd_iter)
    full.obs["tile"] = tile
    if method != "random":
        out.obs_names = f"{sec}_" + out.obs_names
    for a in (out, full):
        a.uns["dataset"], a.uns["section"] = dataset, sec
    nb = out.obs["n_bins"]
    how = method if method == "random" else f"{method}, {agg}"
    print(f"[{sec}] {full.n_obs:,} bins (of {keep.size:,}, {keep.size - keep.sum():,} dropped) x {full.n_vars:,} genes"
          f" -> {out.n_obs:,} points ({how}); bins per point {nb.mean():.1f} (min {nb.min()}, max {nb.max()}),"
          f" mean purity {out.obs['purity'].mean():.3f}")
    if unknown:
        print(f"  labels not in HARMONIZE (scored as Other): {unknown}")
    return out, full


# ---------- expression
def dense(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def gene_var(X):
    mean = np.asarray(X.mean(0)).ravel()
    sq = np.asarray(X.multiply(X).mean(0)).ravel() if sp.issparse(X) else (X ** 2).mean(0)
    return sq - mean ** 2


def lognorm(adata):
    """normalize_total + log1p on a copy."""
    a = adata.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    return a


def full_expression(full, genes):
    """Log-normalized expression of `genes` on every bin of a full section (normalize_total + log1p)."""
    idx = full.var_names.get_indexer(genes)
    assert (idx >= 0).all(), "evaluation genes missing in the full section"
    tot = np.asarray(full.X.sum(1)).ravel()
    target = np.median(tot[tot > 0])  # as sc.pp.normalize_total
    return dense((sp.diags(target / np.where(tot > 0, tot, 1)) @ full.X[:, idx]).log1p())


def load_full_expression(dataset, path, genes, drop_labels=()):
    """Log-normalized expression of `genes` on every bin of a section, with the bins of load_section."""
    adata = sc.read_h5ad(path)
    counts, label_col = ("counts", "celltype") if dataset == "HESTA" else ("count", "annotation")
    keep = ~adata.obs[label_col].astype(str).str.lower().isin([l.lower() for l in drop_labels]).to_numpy()
    full = ad.AnnData(sp.csr_matrix(adata.layers[counts][keep], dtype=np.float32),
                      var=pd.DataFrame(index=adata.var_names.astype(str)))
    del adata
    gc.collect()
    return full_expression(full, genes)


def pick_eval_genes(XA, XB, genes, n):
    """The n most variable matched genes over both matrices (mean variance rank), without mitochondrial genes."""
    ok = ~pd.Index(genes).str.upper().str.startswith("MT-")
    rank = (rankdata(-gene_var(XA)) + rankdata(-gene_var(XB))) / 2
    rank[~ok] = np.inf
    return np.asarray(genes)[np.argsort(rank)[:n]]


# ---------- evaluation
def scored_classes(a_true, b_labels):
    return sorted((set(a_true) & set(b_labels)) - NOT_SCORED)


def label_transfer(P, a_true, b_lab):
    """B's classes moved to A through P (n_A x n_B). Scored: A points whose class also occurs in B."""
    a_true, b_lab = np.asarray(a_true, dtype=str), np.asarray(b_lab, dtype=str)
    pred, _, _ = plotting.project_labels_via_P(np.asarray(P).T, b_lab, direction="A_to_B")
    pred = np.asarray(pred, dtype=str)
    classes = scored_classes(a_true, b_lab)
    mask = np.isin(a_true, classes)
    recall = {c: float((pred[a_true == c] == c).mean()) for c in classes}
    return dict(pred=pred, classes=classes, mask=mask,
                acc=float((pred[mask] == a_true[mask]).mean()) if mask.any() else np.nan,
                bal_acc=float(np.mean(list(recall.values()))) if recall else np.nan, recall=recall)


def colwise_r(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    den = np.sqrt((Xc ** 2).sum(0) * (Yc ** 2).sum(0))
    return np.where(den > 0, (Xc * Yc).sum(0) / np.where(den > 0, den, 1), np.nan)


def coexpression(P, Y_A, Y_B, seed=0):
    """Per-gene Pearson r over A between Y_A and Y_B transported through P; and with B's rows shuffled."""
    P = np.asarray(P)
    rowsum = P.sum(1, keepdims=True) + 1e-30
    r = colwise_r(Y_A, (P @ Y_B) / rowsum)
    perm = np.random.default_rng(seed).permutation(len(Y_B))
    r0 = colwise_r(Y_A, (P @ Y_B[perm]) / rowsum)
    return r, r0


def full_scores(P, A_full, b_lab, pred, Y_B, Y_full, seed=0):
    """Label transfer and gene co-expression on every bin of A's full section (each bin takes its point's values)."""
    P = np.asarray(P)
    tile = A_full.obs["tile"].to_numpy()
    true, pred = A_full.obs["label_h"].to_numpy(str), np.asarray(pred, dtype=str)[tile]
    classes = scored_classes(true, np.asarray(b_lab, dtype=str))
    mask = np.isin(true, classes)
    rowsum = P.sum(1, keepdims=True) + 1e-30
    perm = np.random.default_rng(seed).permutation(len(Y_B))
    r = colwise_r(Y_full, ((P @ Y_B) / rowsum)[tile])
    r0 = colwise_r(Y_full, ((P @ Y_B[perm]) / rowsum)[tile])
    return dict(acc=float((pred[mask] == true[mask]).mean()),
                bal_acc=float(np.mean([(pred[true == c] == c).mean() for c in classes])),
                median_r=float(np.nanmedian(r)), median_r0=float(np.nanmedian(r0)), n_bins=len(tile))


def rigid_scan(xa, xb, a_true, b_lab, Y_A, Y_B, step=5):
    """Nearest-neighbour maps for rotations of A onto B, with and without a reflection."""
    def unit(xy):
        xy = xy - xy.mean(0)
        return xy / np.sqrt((xy ** 2).sum(1).mean())
    a, b = unit(np.asarray(xa, dtype=np.float64)), unit(np.asarray(xb, dtype=np.float64))
    tree = cKDTree(b)
    a_true, b_lab = np.asarray(a_true, dtype=str), np.asarray(b_lab, dtype=str)
    mask = np.isin(a_true, scored_classes(a_true, b_lab))
    rows = []
    for flip in (False, True):
        xy0 = a * [1, -1] if flip else a
        for deg in range(0, 360, step):
            t = np.radians(deg)
            nn = tree.query(xy0 @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]))[1]
            rows.append(dict(map="reflection" if flip else "rotation", angle=deg,
                             acc=(b_lab[nn][mask] == a_true[mask]).mean(),
                             median_r=np.nanmedian(colwise_r(Y_A, Y_B[nn]))))
    return pd.DataFrame(rows)


def mirror_summary(scan):
    """Best rotation and best reflection of a rigid_scan, by labels and by expression."""
    best = []
    for kind, g in scan.groupby("map", sort=False):
        a, e = g.loc[g.acc.idxmax()], g.loc[g.median_r.idxmax()]
        best.append(dict(map=kind, angle_by_labels=a.angle, acc=a.acc, angle_by_expr=e.angle, median_r=e.median_r))
    return pd.DataFrame(best).set_index("map")


def orientation_det(xa, xb, P):
    """Sign of the determinant of the rigid fit of A onto its barycentric image in B (+1 rotation, -1 reflection)."""
    P = np.asarray(P)
    Y = (P @ np.asarray(xb, dtype=P.dtype)) / (P.sum(1, keepdims=True) + 1e-30)
    Xc, Yc = xa - xa.mean(0), Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    return int(np.sign(np.linalg.det(U @ Vt)))


# ---------- plots
def plot_coupling_lines(xa, xb, P, title, names, n_lines=1500, seed=0, ax=None):
    """Draw B in the frame the coupling implies (reflection allowed); join random A points to their peak B target."""
    P = np.asarray(P)
    xb_aligned, _, _, _, _ = align_vis.coupling_procrustes(xa, xb, P, ensure_rotation=False)
    idx = np.random.default_rng(seed).choice(len(xa), min(len(xa), n_lines), replace=False)
    jdx = P[idx].argmax(1)
    shift = np.array([xa[:, 0].max() - xb_aligned[:, 0].min() + 0.3 * np.ptp(xa[:, 0]), 0.0])
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(*xa.T, s=1, c="C0", label=names[0])
    ax.scatter(*(xb_aligned + shift).T, s=1, c="C1", label=f"{names[1]} (coupling frame)")
    ax.add_collection(LineCollection(np.stack([xa[idx], xb_aligned[jdx] + shift], axis=1),
                                     colors="k", alpha=0.08, lw=0.5))
    ax.set_title(title)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(markerscale=8, frameon=False, fontsize=8)
    if show:
        plt.show()


def class_palette(*label_sets, max_classes=10):
    """One colour per organ class (the most frequent `max_classes`, the rest grey as Other)."""
    counts = pd.Series(np.concatenate([np.asarray(l, dtype=str) for l in label_sets])).value_counts()
    classes = [c for c in counts.index if c not in NOT_SCORED][:max_classes]
    palette = dict(zip(classes, align_vis.CATEGORY_COLORS))
    palette["Other"] = "#CBD5E1"
    return palette


def colors_for(labels, palette):
    return [palette.get(l, palette["Other"]) for l in np.asarray(labels, dtype=str)]


def plot_label_panels(panels, palette, title=None):
    """panels: list of (xy, labels, title)."""
    fig, ax = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), constrained_layout=True)
    for a, (xy, lab, t) in zip(np.atleast_1d(ax), panels):
        a.scatter(*np.asarray(xy)[:, :2].T, c=colors_for(lab, palette), s=float(np.clip(28_000 / len(xy), 2, 30)),
                  linewidths=0)
        a.set_title(t); a.set_aspect("equal"); a.axis("off")
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="", color=c, label=k) for k, c in palette.items()],
               loc="outside lower center", ncol=min(len(palette), 8), frameon=False)
    if title:
        fig.suptitle(title)
    plt.show()


def plot_coexpression(r, r0, title):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.linspace(-0.3, 1, 53)
    ax.hist(r0[np.isfinite(r0)], bins=bins, color="#CBD5E1", label=f"shuffled B (median {np.nanmedian(r0):.3f})")
    ax.hist(r[np.isfinite(r)], bins=bins, color="#146C94", alpha=0.8, label=f"coupling (median {np.nanmedian(r):.3f})")
    ax.set_xlabel("Pearson r over A points"); ax.set_ylabel("genes"); ax.set_title(title)
    ax.legend(frameon=False)
    plt.show()
