"""Per-slice overview of the HESTA and MOSTA h5ad files.

For every slice: number of spots, number of genes in the panel, number of
genes detected in that slice (>= 1 count in at least one spot), and number
of distinct labels present. Writes a CSV and prints a markdown table.

    uv run python data_analysis/dataset_overview.py --jobs 16
    uv run python data_analysis/dataset_overview.py --data-dir /nas/lnt/ga79jin/MGW/data

Only the label column and X's CSR column indices are read (h5py, chunked),
never the full matrix or the count layers; memory stays below ~1 GB per job.
"""
import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mgw.config import get_data_dir

# dataset -> (subdirectory, file glob, label column in obs)
DATASETS = {
    "HESTA": ("hesta", "*_HESTA.h5ad", "celltype"),
    "MOSTA": ("mosta", "*.MOSTA.h5ad", "annotation"),
}
NAME_RE = re.compile(r"^(?P<stage>.+?)_E(?P<embryo>\d+)S(?P<section>\d+)")
CHUNK = 20_000_000  # nonzeros per read when scanning X/indices (~0.4 GB peak per worker)


def detected_genes(path, n_vars):
    """Number of genes with a nonzero entry in at least one spot."""
    with h5py.File(path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Dataset):  # dense
            hit = np.zeros(n_vars, dtype=bool)
            for i in range(0, X.shape[0], 5000):
                hit |= (X[i:i + 5000] != 0).any(axis=0)
            return int(hit.sum())
        enc = X.attrs.get("encoding-type", X.attrs.get("h5sparse_format", "csr"))
        enc = enc.decode() if isinstance(enc, bytes) else str(enc)
        if not enc.startswith("csr"):  # csc: a gene is detected if its column has any stored nonzero
            indptr, data = X["indptr"][:], X["data"]
            nnz_per_col = np.diff(indptr)
            hit = np.zeros(n_vars, dtype=bool)
            for j in np.flatnonzero(nnz_per_col):
                hit[j] = np.any(data[indptr[j]:indptr[j + 1]] != 0)
            return int(hit.sum())
        idx, data = X["indices"], X["data"]
        counts = np.zeros(n_vars, dtype=np.int64)
        for i in range(0, idx.shape[0], CHUNK):
            nz = data[i:i + CHUNK] != 0
            counts += np.bincount(idx[i:i + CHUNK][nz], minlength=n_vars)
        return int((counts > 0).sum())


def _decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def read_labels(f, key):
    """obs[key] as a pandas Series, or None. Handles the current categorical
    encoding (group with codes/categories) and the legacy one (codes dataset
    plus obs/__categories/<key>, as in the MOSTA files)."""
    obs = f["obs"]
    if key not in obs:
        return None
    col = obs[key]
    if isinstance(col, h5py.Group):  # anndata >= 0.8 categorical
        codes, cats = col["codes"][:], _decode(col["categories"][:])
    elif "__categories" in obs and key in obs["__categories"]:  # legacy categorical
        codes, cats = col[:], _decode(obs["__categories"][key][:])
    else:  # plain strings
        return pd.Series(_decode(col[:])).replace("nan", None)
    return pd.Series(pd.Categorical.from_codes(codes, cats))


def summarize(args):
    """Reads only shapes, the label column and X's indices. Never use
    ad.read_h5ad(backed="r") here: it loads layers (full counts) into RAM."""
    dataset, path, label_key = args
    with h5py.File(path, "r") as f:
        X = f["X"]
        n_obs, n_vars = (X.shape if isinstance(X, h5py.Dataset) else tuple(X.attrs["shape"]))
        labels = read_labels(f, label_key)
    m = NAME_RE.match(path.name)
    row = {
        "dataset": dataset,
        "slice": path.name.split(".MOSTA")[0].removesuffix(".h5ad").removesuffix("_HESTA"),
        "file": path.name,
        "stage": m["stage"] if m else "",
        "embryo": int(m["embryo"]) if m else -1,
        "section": int(m["section"]) if m else -1,
        "n_spots": int(n_obs),
        "n_genes_panel": int(n_vars),
        "n_genes_detected": detected_genes(path, int(n_vars)),
        "label_key": label_key if labels is not None else "",
        "n_labels": int(labels.dropna().nunique()) if labels is not None else 0,
        "n_spots_labelled": int(labels.notna().sum()) if labels is not None else 0,
        "labels": "; ".join(sorted(map(str, labels.dropna().unique()))) if labels is not None else "",
    }
    print(f"done {path.name}: {row['n_spots']} spots, {row['n_genes_detected']}/{n_vars} genes, "
          f"{row['n_labels']} labels", file=sys.stderr, flush=True)
    return row


def stage_key(stage):
    """Sort CS12-13 < CS14-15 < CS17 ... and E9.5 < E10.5 < ... numerically."""
    nums = re.findall(r"\d+(?:\.\d+)?", stage)
    return float(nums[0]) if nums else float("inf")


def to_markdown(df):
    cols = ["slice", "n_spots", "n_genes_panel", "n_genes_detected", "n_labels"]
    head = ["Slice", "Spots", "Genes (panel)", "Genes (detected)", "Labels"]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for r in df[cols].itertuples(index=False):
        lines.append(f"| {r.slice} | {r.n_spots:,} | {r.n_genes_panel:,} | {r.n_genes_detected:,} | {r.n_labels} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None, help="contains hesta/ and mosta/ (default: get_data_dir())")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "dataset_overview.csv")
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()

    data_dir = args.data_dir or get_data_dir()
    tasks = []
    for dataset, (sub, pattern, label_key) in DATASETS.items():
        files = sorted((data_dir / sub).glob(pattern))
        print(f"{dataset}: {len(files)} files in {data_dir / sub}", file=sys.stderr)
        tasks += [(dataset, f, label_key) for f in files]

    with ProcessPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(summarize, tasks))

    df = pd.DataFrame(rows)
    df["_s"] = df["stage"].map(stage_key)
    df = df.sort_values(["dataset", "_s", "embryo", "section", "slice"]).drop(columns="_s")
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}", file=sys.stderr)

    for dataset, g in df.groupby("dataset"):
        print(f"\n## {dataset}\n\n{to_markdown(g)}")


if __name__ == "__main__":
    main()
