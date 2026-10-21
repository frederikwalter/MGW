"""A synthetic embryo pair for tests and dry runs: no h5ad files, the real code path.

The body is an ellipsoid (x anteroposterior, y dorsoventral, z mediolateral) with a few organs (ellipsoidal label
regions, in body-normalized coordinates u = x / semi-axes) and smooth gene fields over u: organ markers at three
widths, gradients along every axis and genes that vary with depth only (a lateral organ, |u_z|, u_z^2). Two
embryos share the template, so they are homologous under scaling:
- `SYN_H` (HESTA): 6 sections at u_z = 0 .. 0.75 (gaps of 1.6 spot spacings), scale 1;
- `SYN_M` (MOSTA): 4 sections at u_z = 0.3 .. 0.75, scale 0.8, organs shifted slightly (a species difference);
their depth ranges overlap partly, as in X1.

Each section is a jittered grid of tiles (spacing `spacing` in body units) on its plane, inside the body, then
rotated by up to 25 degrees, shifted and scaled to pixels, so stage 1 must register it. A tile carries 4 bins at
small offsets (the full-resolution organ counts `comp`), its majority label and L = |u_z| (0 midline, 1 lateral).

`write_bench(root, cfg)` writes what `prepare` would: the section files (section hash), the features of every pair
(feature hash; genes z-scored per section or per embryo as `features.zscore_by`, then one PCA over the pair,
held-out genes never in the embedding), the stack features (per embryo) and a series file `synthetic.yaml` with
the cross pair `SX` in the tiers smoke and dev. Pass `series=<that file>` as an override.
"""
from pathlib import Path

import numpy as np
import yaml

from mgw.bench import metrics, prepare

SEMI = np.array([1.0, 0.6, 1.2])               # body semi-axes (x, y, z); deep, so gaps exceed the spot spacing
# organ: (centre, radii) in body-normalized coordinates
ORGANS = {
    "Brain": ((0.55, 0.2, 0.0), (0.4, 0.55, 0.4)),
    "Spinal cord": ((-0.3, 0.6, 0.0), (0.5, 0.18, 0.15)),
    "Heart": ((0.15, -0.45, 0.2), (0.25, 0.3, 0.25)),
    "Liver": ((-0.2, -0.35, 0.1), (0.3, 0.35, 0.35)),
    "Gut": ((-0.6, -0.05, 0.3), (0.25, 0.3, 0.25)),
    "Limb": ((-0.1, -0.2, 0.75), (0.3, 0.35, 0.25)),     # lateral: only in the outer sections
}
BACKGROUND = "Mesenchyme"
N_EVAL = 12
_u = np.random.default_rng(0).uniform(-1, 1, (20000, 3))
GENE_SD = None                                  # each gene's SD over the body (the noise scale), set below


def organ_of(u, shift=0.0):
    """Organ label of body-normalized points u (n x 3); `shift` moves every organ centre along x."""
    lab = np.full(len(u), BACKGROUND, dtype=object)
    for name, (c, r) in reversed(list(ORGANS.items())):       # the first organ wins where they overlap
        d = (((u - (np.asarray(c) + [shift, 0, 0])) / r) ** 2).sum(1)
        lab[d <= 1] = name
    return lab.astype(str)


def genes(u, shift=0.0):
    """(embedding genes, held-out genes) at body-normalized points u; both smooth in u, the held-out ones other
    functions of the same organs and axes (so they are predictable from the embedding, never part of it)."""
    cols = []
    for c, r in ORGANS.values():
        d = (((u - (np.asarray(c) + [shift, 0, 0])) / r) ** 2).sum(1)
        cols += [np.exp(-0.5 * d * s) for s in (0.15, 0.3, 0.6)]
    x, y, z = u.T
    cols += [x, y, 2 * z, 2 * z ** 2, np.sin(np.pi * x) * np.cos(np.pi * z), x * y, np.sin(4 * z), np.cos(3 * z + x)]
    G = np.stack(cols, 1)
    held = []
    for n, (c, r) in enumerate(ORGANS.values()):
        d = (((u - (np.asarray(c) + [shift, 0, 0])) / r) ** 2).sum(1)
        held += [np.exp(-0.7 * d) * (1 + 0.5 * x), np.exp(-1.5 * d) + 0.3 * z ** 2]
    Y = np.stack(held, 1)[:, :N_EVAL]
    return G, np.log1p(4 * np.clip(Y, 0, None))


GENE_SD = genes(_u[(_u ** 2).sum(1) <= 1])[0].std(0)


def make_embryo(name, dataset, z_rel, *, scale=1.0, shift=0.0, spacing=0.11, seed=0):
    """Sections of one synthetic embryo: {section: dict(xy (pixels), label, comp, L, u (n x 3), G, Y, z)}."""
    rng = np.random.default_rng(seed)
    out = {}
    semi = SEMI * scale
    for k, uz in enumerate(z_rel):
        o = rng.uniform(0, spacing, 2)                         # every section its own grid
        X, Yg = np.meshgrid(np.arange(-semi[0] - o[0], semi[0], spacing), np.arange(-semi[1] - o[1], semi[1], spacing))
        p = np.c_[X.ravel(), Yg.ravel()] + rng.normal(0, 0.15 * spacing, (X.size, 2))
        xyz = np.c_[p, np.full(len(p), uz * semi[2])]
        u = xyz / semi
        xyz, u = xyz[(u ** 2).sum(1) <= 1], u[(u ** 2).sum(1) <= 1]
        # 4 bins per tile, at +-spacing/4: the full-resolution organ counts
        offs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * spacing / 4
        bins = np.concatenate([organ_of(np.c_[(xyz[:, :2] + o) / semi[:2], u[:, 2]], shift) for o in offs])
        comp = metrics.composition(bins, np.tile(np.arange(len(u)), 4), len(u)).astype(np.float32)
        G, Y = genes(u, shift)
        G = G + rng.normal(0, 0.02, G.shape) * GENE_SD
        th = np.radians(rng.uniform(-25, 25))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        xy = (xyz[:, :2] @ R + rng.normal(0, 0.3, 2)) * 100
        out[f"{name}S{k + 1}"] = dict(xy=xy, label=metrics.majority(comp), comp=comp, L=float(abs(uz)), u=u, G=G,
                                      Y=(Y + rng.normal(0, 0.03, Y.shape)).astype(np.float32), z=float(xyz[0, 2]))
    return out


def _zscore(blocks, floor):
    X = np.vstack(blocks)
    return [(b - X.mean(0)) / np.maximum(X.std(0), floor) for b in blocks]


def _embed(groups, n_comp, by_section):
    """One PCA over all sections of `groups` (list of lists of gene matrices), genes z-scored per section or per
    group (embryo). The SD is floored at a quarter of the gene's SD over everything, so a marker of an organ that
    a section lacks is not blown up to unit variance (noise). Returns the same nesting of embeddings."""
    floor = 0.25 * np.vstack([b for g in groups for b in g]).std(0) + 1e-6
    Z = [[z for b in g for z in _zscore([b], floor)] if by_section else _zscore(g, floor) for g in groups]
    J = np.vstack([z for g in Z for z in g])
    mu = J.mean(0)
    _, _, Vt = np.linalg.svd(J - mu, full_matrices=False)
    V = Vt[:min(n_comp, Vt.shape[0])].T
    return [[(z - mu) @ V for z in g] for g in Z]


def embryos(spacing=0.11, seed=0):
    return {"SYN_H": ("HESTA", make_embryo("SYN_H", "HESTA", np.linspace(0.0, 0.75, 6), spacing=spacing, seed=seed)),
            "SYN_M": ("MOSTA", make_embryo("SYN_M", "MOSTA", np.linspace(0.3, 0.75, 4), scale=0.8, shift=0.05,
                                           spacing=spacing, seed=seed + 1))}


def write_bench(root, cfg, seed=0, spacing=0.11):
    """Write the synthetic series, section caches and features for `cfg` and `seed` under root. Returns the series
    file. Idempotent per (section hash, feature hash)."""
    root = Path(root)
    emb = embryos(spacing, seed)
    sh, fh = cfg.section_hash(), cfg.feature_hash()
    f = cfg.raw["features"]
    by_section = f["zscore_by"] == "section"
    for name, (ds, secs) in emb.items():
        for s, d in secs.items():
            out = prepare.section_file(root, s, seed, sh)
            if not out.exists():
                prepare.atomic_savez(out, xy=d["xy"], label=d["label"], comp=d["comp"],
                                     n_bins=d["comp"].sum(1).astype(np.int64),
                                     purity=(d["comp"].max(1) / d["comp"].sum(1)).astype(np.float32),
                                     L_pair=d["L"], L_raw=d["L"], n_full=int(d["comp"].sum()), identity=False,
                                     organs=np.asarray(metrics.ORGANS), dataset=ds, section=s)
        # the within-embryo embedding that registers the stack (z-scored per section, as stack3d.joint_embedding)
        out = prepare.feature_file(root, f"stack_{name}", seed, fh)
        if not out.exists():
            E = _embed([[d["G"] for d in secs.values()]], f["pca_comp"], True)[0]
            prepare.atomic_savez(out, **{f"{s}/X_feat": e.astype(np.float32) for s, e in zip(secs, E)})
    pairs = {"SX": ("SYN_H", "SYN_M"), "W:SYN_H": ("SYN_H", "SYN_H"), "W:SYN_M": ("SYN_M", "SYN_M")}
    genes_ = np.asarray([f"g{i}" for i in range(29)])
    for pid, (a, b) in pairs.items():
        out = prepare.feature_file(root, pid, seed, fh)
        if out.exists():
            continue
        sa, sb = emb[a][1], emb[b][1]
        groups = [[d["G"] for d in sa.values()]] + ([] if a == b else [[d["G"] for d in sb.values()]])
        E = _embed(groups, f["pca_comp"], by_section)
        EA, EB = E[0], E[-1]
        arrays = dict(eval_a=np.asarray([f"y{i}" for i in range(N_EVAL)]), eval_b=np.asarray(
            [f"y{i}" for i in range(N_EVAL)]), emb_genes=genes_)
        for side, secs, Es in (("a", sa, EA), ("b", sb, EB)):
            for (s, d), e in zip(secs.items(), Es):
                arrays[f"{side}/{s}/emb"] = e.astype(np.float32)
                arrays[f"{side}/{s}/Y"] = d["Y"].astype(np.float16)
        prepare.atomic_savez(out, **arrays)
    series = root / "synthetic.yaml"
    if not series.exists():
        raw = dict(embryos={n: dict(dataset=ds, stage="syn", sections=list(secs)) for n, (ds, secs) in emb.items()},
                   pairs=dict(SX=dict(kind="cross", a="SYN_H", b="SYN_M", stage="syn", overlap="partial",
                                      tiers=["smoke", "dev"], note="synthetic pair")))
        series.parent.mkdir(parents=True, exist_ok=True)
        series.write_text(yaml.safe_dump(raw, sort_keys=False))
    return series


def write_stacks(root, cfg, seed=0, spacing=0.11, only=None):
    """Cache both stacks for `cfg` (which names the synthetic series): stage 1 as in `pipelines/stacked`
    (registration by moscot FGW and OT-ICP, file order) but with the true gaps instead of the variogram's.

    The variogram underestimates the synthetic gaps: genes are z-scored per section before the stack embedding,
    which removes the depth offsets (the effect §1c describes), and the cross-section neighbours sit closer in-plane
    than the variogram's first bin. The true gaps keep the tests about the field and stage 2. `only`: embryo names
    (default both)."""
    from mgw import stack3d as s3
    from mgw.bench import series as bseries, stack as bstack, store
    from mgw.bench.pipelines import stacked
    from mgw.bench.solvers import centred
    root = Path(root)
    ser = bseries.load(cfg.series_path())
    pair = ser.pairs["SX"]
    data = prepare.load_pair(pair, cfg, seed, root)
    emb = embryos(spacing, seed)
    for e, side in ((pair.a, "a"), (pair.b, "b")):
        f = stacked.stack_path(root, cfg, seed, e)
        if f.exists() or (only and e.name not in only):
            continue
        secs, names = data[side], list(e.sections)
        feats = prepare.load_stack_features(e, cfg, seed, root)
        scale = float(np.median([s3.rms_radius(secs[s].xy) for s in names]))
        gaps = np.diff([emb[e.name][1][s]["z"] for s in names]) * 100 / scale
        params = bstack.StackParams.from_mapping({**cfg.raw["stack"], "order": "file", "z_gaps": gaps.tolist()})
        ST, diag = bstack.build_stack(names, [centred(secs[s].xy, scale) for s in names], [feats[s] for s in names],
                                      [secs[s].label for s in names], params, dataset=e.dataset,
                                      fgw_alpha=cfg.raw["fgw"]["alpha"], seed=seed)
        store.savez(f, **bstack.to_arrays(ST, diag))
