"""Everything the solvers need, computed once where the raw data are (LNT) and cached as small files.

Per section and seed (`cache/sections/<sec>/seed<s>_<hash>.npz`): the tile coordinates, the majority organ, the
tile x organ bin counts (all that full-resolution kappa_bal needs), L. Next to it, on the data server only
(`cache/counts/...`): the summed counts of the 1:1 ortholog genes per tile and every tile's total count, from which
the features are computed.

Per pair and seed (`cache/features/<pair>/seed<s>_<hash>.npz`): the ortholog embedding of every section of both
embryos (orthologs.shared_embedding over all points of the pair at once, genes z-scored within each section, or each
embryo with features.zscore_by = embryo) and the held-out evaluation genes (the n most variable 1:1 orthologs, never
in the embedding). Same-species pairs use the same gene universe (1:1 orthologs), matched by name. Per embryo
(`cache/features/stack_<embryo>/...`): the within-embryo embedding that registers the stack (stack3d.joint_embedding,
as `X_feat` in the notebook).

Seeds: seed s reduces with Voronoi seed 42 + s and runs the randomized PCAs with random_state s, so seed 0 is the
notebook's setting.
"""
import gc
import os
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from mgw import crossspecies as cs, orthologs, stack3d as s3
from mgw.bench import metrics
from mgw.config import get_data_dir

REPO = Path(__file__).resolve().parents[2]


# ---------- locations
def data_dirs():
    """(HESTA_DIR, MOSTA_DIR) as in the notebook: $MGW_HESTA_DIR / $MGW_MOSTA_DIR, else <data dir>/hesta, /mosta."""
    data = get_data_dir()
    hesta = Path(os.environ.get("MGW_HESTA_DIR") or data / "hesta")
    mosta = os.environ.get("MGW_MOSTA_DIR")
    if mosta is None:
        mosta = next((data / d for d in ("mosta", "mouse_embryo") if (data / d).is_dir()), data / "mosta")
    return hesta.expanduser(), Path(mosta).expanduser()


def bench_root(root=None):
    """Root of caches and results: --root, $MGW_BENCH_DIR, or <data dir>/bench."""
    return Path(root or os.environ.get("MGW_BENCH_DIR") or get_data_dir() / "bench").expanduser()


def laterality_table():
    return pd.read_csv(REPO / "laterality" / "laterality_scores.csv", dtype={"embryo": str}).set_index("key")


def lat(scores, dataset, sec):
    k = cs.score_key(dataset, sec)
    if k not in scores.index:
        return np.nan, np.nan
    return float(scores.loc[k, "L_pair"]), float(scores.loc[k, "L_laterality"])


def section_file(root, sec, seed, h):
    return Path(root) / "cache" / "sections" / sec / f"seed{seed}_{h}.npz"


def counts_file(root, sec, seed, h):
    return Path(root) / "cache" / "counts" / sec / f"seed{seed}_{h}.npz"


def feature_file(root, key, seed, h):
    return Path(root) / "cache" / "features" / key.replace(":", "_") / f"seed{seed}_{h}.npz"


def atomic_savez(path, compressed=True, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


# ---------- sections
def prepare_section(dataset, sec, cfg, seed, root, ortho, scores, force=False):
    """Reduce one section and cache its tiles, organ counts and ortholog counts. Returns the section file."""
    h = cfg.section_hash()
    out, cnt = section_file(root, sec, seed, h), counts_file(root, sec, seed, h)
    if out.exists() and cnt.exists() and not force:
        return out
    r = cfg.raw["reduce"]
    hesta, mosta = data_dirs()
    red, full = cs.load_section(dataset, sec, cs.section_path(dataset, sec, hesta, mosta), max_obs=r["max_obs"],
                                subsample=r["subsample"], agg=r["agg"], lloyd_iter=r["lloyd_iter"],
                                drop_labels=r["drop_labels"], random_state=42 + seed)
    n = red.n_obs
    comp = metrics.composition(full.obs["label_h"].to_numpy(str), full.obs["tile"].to_numpy(), n)
    L_pair, L_raw = lat(scores, dataset, sec)
    atomic_savez(out, xy=np.asarray(red.obsm["spatial"], dtype=np.float64),
                 label=red.obs["label_h"].to_numpy(str), comp=comp.astype(np.float32),
                 n_bins=red.obs["n_bins"].to_numpy(np.int64), purity=red.obs["purity"].to_numpy(np.float32),
                 L_pair=L_pair, L_raw=L_raw, n_full=full.n_obs, identity=bool(n == full.n_obs),
                 organs=np.asarray(metrics.ORGANS), dataset=dataset, section=sec)
    col = "human" if dataset == "HESTA" else "mouse"
    genes = red.var_names.intersection(pd.Index(ortho[col]))
    X = sp.csr_matrix(red[:, genes].X, dtype=np.float32)
    total = np.asarray(red.X.sum(1)).ravel()
    atomic_savez(cnt, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.asarray(X.shape),
                 genes=np.asarray(genes, dtype=str), total=total)
    del red, full
    gc.collect()
    return out


@dataclass
class Section:
    name: str
    dataset: str
    xy: np.ndarray          # tile centroids, pixels
    label: np.ndarray       # majority organ
    comp: np.ndarray        # tile x organ bin counts
    L: float
    n_full: int
    emb: np.ndarray = None  # features of the pair (filled by load_pair)
    Y: np.ndarray = None    # held-out genes of the pair

    @property
    def n(self):
        return len(self.xy)

    @property
    def onehot(self):
        return metrics.onehot(self.label)


def load_section(root, sec, seed, h):
    z = np.load(section_file(root, sec, seed, h), allow_pickle=False)
    return Section(name=sec, dataset=str(z["dataset"]), xy=z["xy"], label=z["label"].astype(str),
                   comp=z["comp"].astype(np.float64), L=float(z["L_pair"]), n_full=int(z["n_full"]))


def load_counts(root, sec, seed, h):
    """The section's ortholog counts, log-normalized with its total counts (normalize_total + log1p)."""
    z = np.load(counts_file(root, sec, seed, h), allow_pickle=False)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    total = z["total"]
    target = np.median(total[total > 0])                                  # as sc.pp.normalize_total
    X = (sp.diags(target / np.where(total > 0, total, 1)) @ X).tocsr()
    X.data = np.log1p(X.data)
    a = ad.AnnData(X.astype(np.float32), var=pd.DataFrame(index=z["genes"].astype(str)))
    a.obs_names = [f"{sec}_{i}" for i in range(a.n_obs)]
    return a


# ---------- features
def _cat(adatas):
    return ad.concat(adatas, join="inner")


def pair_features(pair, cfg, seed, root, ortho):
    """The ortholog embedding and the evaluation genes of every section of a pair. Returns the feature file."""
    h = cfg.feature_hash()
    out = feature_file(root, pair.id, seed, h)
    if out.exists():
        return out
    f, sh = cfg.raw["features"], cfg.section_hash()
    side = lambda e: [load_counts(root, s, seed, sh) for s in e.sections]
    A_list = side(pair.a)
    B_list = A_list if pair.kind == "within" else side(pair.b)
    A, B = _cat(A_list), _cat(B_list)
    # genes are z-scored within each section, or within each embryo (zscore_by: embryo), which keeps the offsets
    # between the sections of a stack: the depth signal that a 3D field over the stack needs
    eb = pair.a if pair.kind == "within" else pair.b
    by_section = f["zscore_by"] == "section"
    gA = np.repeat(pair.a.sections if by_section else [pair.a.name] * len(A_list), [a.n_obs for a in A_list])
    gB = np.repeat(eb.sections if by_section else [eb.name] * len(B_list), [b.n_obs for b in B_list])
    if pair.kind == "cross":
        h2m = dict(zip(ortho.human, ortho.mouse))
        gh, gm = orthologs.restrict_to_orthologs(A.var_names, B.var_names, ortho)
        eval_a = cs.pick_eval_genes(A[:, gh].X, B[:, gm].X, gh, f["n_eval_genes"])
        eval_b = np.array([h2m[g] for g in eval_a])
        EA, EB, emb_genes = orthologs.shared_embedding(A, B, ortho, n_hvg=f["n_hvg"], n_comps=f["pca_comp"],
                                                       exclude=eval_a, groups_A=gA, groups_B=gB, random_state=seed)
    else:
        genes = A.var_names.intersection(B.var_names)
        eval_a = cs.pick_eval_genes(A[:, genes].X, B[:, genes].X, np.asarray(genes), f["n_eval_genes"])
        eval_b = eval_a
        EA, EB, emb_genes = orthologs.shared_embedding(A, B, None, n_hvg=f["n_hvg"], n_comps=f["pca_comp"],
                                                       exclude=eval_a, groups_A=gA, groups_B=gB, random_state=seed)
    arrays = dict(eval_a=np.asarray(eval_a, dtype=str), eval_b=np.asarray(eval_b, dtype=str),
                  emb_genes=np.asarray(emb_genes, dtype=str))
    for side_name, E, lst, ev, emb in (("a", EA, A_list, eval_a, pair.a), ("b", EB, B_list, eval_b,
                                                                          pair.a if pair.kind == "within" else pair.b)):
        off = np.r_[0, np.cumsum([x.n_obs for x in lst])]
        for k, (sec, x) in enumerate(zip(emb.sections, lst)):
            arrays[f"{side_name}/{sec}/emb"] = E[off[k]:off[k + 1]].astype(np.float32)
            arrays[f"{side_name}/{sec}/Y"] = cs.dense(x[:, list(ev)].X).astype(np.float16)
    atomic_savez(out, **arrays)
    del A, B, A_list, B_list
    gc.collect()
    return out


def stack_features(embryo, cfg, seed, root):
    """The within-embryo embedding (X_feat) that registers the stack. Returns the feature file."""
    h = cfg.feature_hash()
    out = feature_file(root, f"stack_{embryo.name}", seed, h)
    if out.exists():
        return out
    f, sh = cfg.raw["features"], cfg.section_hash()
    lst = [load_counts(root, s, seed, sh) for s in embryo.sections]
    E = s3.joint_embedding(lst, n_hvg=f["n_hvg"], n_comps=f["pca_comp"], random_state=seed)
    atomic_savez(out, **{f"{sec}/X_feat": e.astype(np.float32) for sec, e in zip(embryo.sections, E)})
    return out


def load_pair(pair, cfg, seed, root):
    """{"a": {sec: Section}, "b": {sec: Section}} with each section's embedding and evaluation genes."""
    z = np.load(feature_file(root, pair.id, seed, cfg.feature_hash()), allow_pickle=False)
    sh = cfg.section_hash()
    out = {}
    for side, emb in (("a", pair.a), ("b", pair.a if pair.kind == "within" else pair.b)):
        out[side] = {}
        for sec in emb.sections:
            s = load_section(root, sec, seed, sh)
            s.emb = z[f"{side}/{sec}/emb"].astype(np.float64)
            s.Y = z[f"{side}/{sec}/Y"].astype(np.float32)
            out[side][sec] = s
    return out


def load_stack_features(embryo, cfg, seed, root):
    z = np.load(feature_file(root, f"stack_{embryo.name}", seed, cfg.feature_hash()), allow_pickle=False)
    return {sec: z[f"{sec}/X_feat"].astype(np.float64) for sec in embryo.sections}


def missing_files(series, tier):
    """(dataset, section, path) of the tier's sections without a raw file."""
    hesta, mosta = data_dirs()
    return [(ds, s, cs.section_path(ds, s, hesta, mosta)) for ds, s in series.all_sections(tier)
            if not cs.section_path(ds, s, hesta, mosta).exists()]
