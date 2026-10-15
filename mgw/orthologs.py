"""One-to-one human-mouse orthologs (Ensembl BioMart), keyed by gene symbol.

The table ships with the package (mgw/resources/human_mouse_one2one.tsv, Ensembl release 116,
built 2026-10-09 with build_one2one_table). Rebuild it with build_one2one_table() to update it.
"""
from pathlib import Path

import pandas as pd

BIOMART_URL = "https://www.ensembl.org/biomart/martservice"
_QUERY = (
    '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Filter name="with_mmusculus_homolog" excluded="0"/>'
    '<Attribute name="external_gene_name"/>'
    '<Attribute name="mmusculus_homolog_associated_gene_name"/>'
    '<Attribute name="mmusculus_homolog_orthology_type"/>'
    '</Dataset></Query>'
)
_MANUAL = (
    "Download it by hand: Ensembl BioMart (https://www.ensembl.org/biomart), dataset 'Human genes', "
    "filter 'Orthologous Mouse Genes: Only', attributes 'Gene name', 'Mouse gene name', "
    "'Mouse homology type'; export as TSV with header and save it to {path}."
)


DEFAULT_PATH = Path(__file__).resolve().parent / "resources" / "human_mouse_one2one.tsv"


def build_one2one_table(path=None, timeout=600):
    """Query BioMart for human-mouse orthologs, keep the one-to-one pairs, and save them as TSV."""
    import requests

    path = Path(path) if path else DEFAULT_PATH
    r = requests.get(BIOMART_URL, params={"query": _QUERY}, timeout=timeout)
    r.raise_for_status()
    if not r.text.startswith("Gene name"):
        raise RuntimeError(f"unexpected BioMart response: {r.text[:200]!r}")
    lines = [l.split("\t") for l in r.text.strip().splitlines()[1:]]
    df = pd.DataFrame(lines, columns=["human", "mouse", "type"])
    df = df[df["type"] == "ortholog_one2one"][["human", "mouse"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def load_one2one(path=None):
    """Return a DataFrame (human, mouse) of strictly one-to-one ortholog symbols.

    Builds the table from BioMart if the file is missing. Accepts both the saved two-column
    file and a raw BioMart export (columns 'Gene name', 'Mouse gene name', 'Mouse homology type').
    """
    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        try:
            build_one2one_table(path)
        except Exception as e:
            raise FileNotFoundError(f"{path} not found and BioMart query failed ({e}). "
                                    + _MANUAL.format(path=path)) from e
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "Mouse homology type" in df:
        df = df[df["Mouse homology type"] == "ortholog_one2one"]
        df = df.rename(columns={"Gene name": "human", "Mouse gene name": "mouse"})
    df = df[["human", "mouse"]].dropna()
    df = df[(df.human != "") & (df.mouse != "")].drop_duplicates()
    # Symbols map to several Ensembl genes now and then; drop them so the map stays 1:1
    df = df[~df.human.duplicated(keep=False) & ~df.mouse.duplicated(keep=False)]
    return df.reset_index(drop=True)


def restrict_to_orthologs(genes_human, genes_mouse, table):
    """Ortholog pairs present in both gene lists, as two aligned arrays (human, mouse)."""
    hit = table[table.human.isin(set(genes_human)) & table.mouse.isin(set(genes_mouse))]
    return hit.human.to_numpy(), hit.mouse.to_numpy()


def shared_embedding(A, B, table=None, n_hvg=2000, n_comps=30, exclude=None):
    """Embed two log-normalized sections in one space spanned by matched genes.

    With `table` (a load_one2one frame) A is human and B mouse, and genes are matched through the
    1:1 orthologs; without it, by name (two sections of one species). The genes used are the union of
    the `n_hvg` most variable matched genes of each section, minus `exclude` (named as in A). Each gene
    is z-scored within each section, which removes offsets between species or sections, and the two
    sections are then projected on a joint PCA.

    Returns (EA, EB, genes): the embeddings (n_A x n_comps, n_B x n_comps) and the genes used (A's names).
    """
    import numpy as np
    import scipy.sparse as sp
    from sklearn.utils.extmath import randomized_svd

    if table is not None:
        ga, gb = restrict_to_orthologs(A.var_names, B.var_names, table)
    else:
        ga = gb = np.intersect1d(A.var_names, B.var_names)
    XA, XB = A[:, ga].X, B[:, gb].X

    def var(X):
        mean = np.asarray(X.mean(0)).ravel()
        sq = np.asarray(X.multiply(X).mean(0)).ravel() if sp.issparse(X) else (np.asarray(X) ** 2).mean(0)
        return sq - mean ** 2

    keep = np.zeros(len(ga), dtype=bool)
    for X in (XA, XB):
        keep[np.argsort(-var(X))[:n_hvg]] = True
    if exclude is not None:
        keep &= ~np.isin(ga, np.asarray(exclude))

    def zscore(X):
        X = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)
        return (X - X.mean(0)) / (X.std(0) + 1e-6)

    J = np.vstack([zscore(XA[:, keep]), zscore(XB[:, keep])])
    J -= J.mean(0)
    U, S, _ = randomized_svd(J, n_components=n_comps, random_state=0)
    E = (U[:, :n_comps] * S[:n_comps]).astype(np.float64)
    return E[:A.n_obs], E[A.n_obs:], np.asarray(ga)[keep]
