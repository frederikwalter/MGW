"""The three tables of the report, all in kappa_bal at full resolution.

1. 2D tools (strategy real2d): per pair, the selected rows; per seed the mean over the two directions of the mean
   over each direction's queries, then mean and SD over seeds; the summary is the median over the pairs.
2. Within-embryo tools (strategy stack): per embryo, the mean over its neighbour rows per seed, then mean and SD over
   seeds; the summary is the mean over the embryos.
3. Between-embryo tools (strategy volume): as 1, plus delta = kappa_bal minus the best 2D tool's on the same pair.
"Best" is the tool with the highest summary.
"""
import numpy as np
import pandas as pd

from mgw.bench import store

GROUPS = {"2d": "real2d", "within": "stack", "3d": "volume"}


def pipelines(cfg, group):
    return [p for p, s in cfg.raw["pipelines"].items() if s["strategy"] == GROUPS[group]]


def load(cfg, root, seeds=None):
    """Scores of the config's pipelines (their current hashes) under root."""
    df = store.load_scores(root, [cfg.pipeline_dir(p) for p in cfg.raw["pipelines"]])
    if len(df) and seeds is not None:
        df = df[df.seed.isin(seeds)]
    return df


def _mean_sd(per_seed, by):
    """{key: (mean, sd)} over seeds; sd is NaN with one seed."""
    g = per_seed.groupby(by).kappa_bal
    return {k: (m, s) for k, m, s in zip(g.mean().index, g.mean(), g.std(ddof=1))}


def pair_values(df, pipeline):
    """{pair: (mean, sd)}: per seed the mean over directions of the mean over queries, then over seeds."""
    d = df[(df.pipeline == pipeline) & (df.role == "selected")]
    if d.empty:
        return {}
    per_dir = d.groupby(["pair", "seed", "direction"]).kappa_bal.mean().reset_index()
    return _mean_sd(per_dir.groupby(["pair", "seed"]).kappa_bal.mean().reset_index(), "pair")


def embryo_values(df, pipeline):
    """{embryo: (mean, sd)}: per seed the mean over the embryo's neighbour rows, then over seeds."""
    d = df[(df.pipeline == pipeline) & (df.role == "neighbour")]
    if d.empty:
        return {}
    return _mean_sd(d.groupby(["direction_label", "seed"]).kappa_bal.mean().reset_index(), "direction_label")


def _table(keys, key_name, pipes, values, summary, summary_name):
    """DataFrame with one row per key, a mean and an SD column per pipeline, and a summary row. Returns (table,
    {pipeline: summary value})."""
    rows = []
    for k in keys:
        r = {key_name: k}
        for p in pipes:
            r[p], r[f"{p}_sd"] = values[p].get(k, (np.nan, np.nan))
        rows.append(r)
    t = pd.DataFrame(rows, columns=[key_name] + [c for p in pipes for c in (p, f"{p}_sd")])
    summ = {p: float(summary(t[p].dropna())) if t[p].notna().any() else np.nan for p in pipes}
    if len(t):
        t = pd.concat([t, pd.DataFrame([{key_name: summary_name, **summ}])], ignore_index=True)
    return t, summ


def best(summ):
    ok = {p: v for p, v in summ.items() if np.isfinite(v)}
    return max(ok, key=ok.get) if ok else None


def tables(cfg, ser, root, tier, seeds=None):
    """(df, {"2d" | "within" | "3d": (table, summary, best)})."""
    df = load(cfg, root, seeds)
    pairs = [p for p in ser.tier(tier) if len(df) and p in set(df.pair)]
    out = {}
    # 1. 2D tools
    p2 = pipelines(cfg, "2d")
    t2, s2 = _table(pairs, "pair", p2, {p: pair_values(df, p) for p in p2}, np.median, "median")
    b2 = best(s2)
    out["2d"] = (t2, s2, b2)
    # 2. within-embryo tools
    pw = pipelines(cfg, "within")
    embryos = [e.name for pid in pairs for e in ser.pairs[pid].embryos]
    tw, sw = _table(embryos, "embryo", pw, {p: embryo_values(df, p) for p in pw}, np.mean, "mean")
    out["within"] = (tw, sw, best(sw))
    # 3. between-embryo tools, against the best 2D tool
    p3 = pipelines(cfg, "3d")
    v3 = {p: pair_values(df, p) for p in p3}
    t3, s3 = _table(pairs, "pair", p3, v3, np.median, "median")
    if b2 and len(t3):
        ref = pair_values(df, b2)
        t3.insert(1, f"best 2D ({b2})", [ref.get(k, (np.nan,))[0] for k in pairs] + [s2[b2]])
        for p in p3:
            d = [v3[p].get(k, (np.nan,))[0] - ref.get(k, (np.nan,))[0] for k in pairs]
            t3[f"Δ {p}"] = d + [float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan]
    out["3d"] = (t3, s3, best(s3))
    return df, out
