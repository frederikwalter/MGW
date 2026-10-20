"""Aggregation of scores.csv into the Go table (docs/st_3d_alignment_strategies.md §2a, §6 step 1).

Pair level: the mean over the two directions of the mean over each direction's queries (symmetric by construction).
Columns of the Go table (one row per cross pair):
- kappa_bal of A (the reference strategy, selected by the FGW objective), the label oracle and the regret;
- the nulls (random section, laterality partner, rigid only, feature-only OT, geometry-only GW, SCOT, organ prior);
- the ceiling (between embryos where available, else the mean of both W) and the ceiling share;
- OMS and Cohen's kappa alongside; delta kappa against the reference and win/loss against the noise margin;
- the guardrails: regret, fold-overs, and for X6 the end-of-stack check.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mgw.bench import store

# (column, pipeline, role)
COLUMNS = [
    ("A", "A_fgw", "selected"),
    ("oracle", "A_fgw", "oracle"),
    ("rand_section", "A_fgw", "null:random_section"),
    ("L_partner", "A_fgw", "L_partner"),
    ("rigid", "A_fgw", "null:rigid"),
    ("feat_OT", "A_linear", "selected"),
    ("space_GW", "A_spatialgw", "selected"),
    ("SCOT", "A_scot", "selected"),
    ("prior", "A_fgw", "null:prior"),
]


def load(cfg, root, pipelines=None, seeds=None):
    """Scores of the config's pipelines (their current hashes) under root."""
    dirs = [cfg.pipeline_dir(p) for p in (pipelines or cfg.raw["pipelines"])]
    df = store.load_scores(root, dirs)
    if len(df) and seeds is not None:
        df = df[df.seed.isin(seeds)]
    return df


def pair_level(df, value="kappa_bal", by=("pipeline", "pair", "seed", "role")):
    """Symmetric pair-level mean: mean over directions of the mean over queries."""
    if df.empty:
        return pd.DataFrame(columns=list(by) + [value])
    g = df.groupby(list(by) + ["direction"], dropna=False)[value].mean().reset_index()
    return g.groupby(list(by), dropna=False)[value].mean().reset_index()


def per_pair(df, pipeline, role, value="kappa_bal"):
    """{pair: mean over seeds of the pair-level value} and {pair: SD over seeds}."""
    d = df[(df.pipeline == pipeline) & (df.role == role)]
    pl = pair_level(d, value)
    return pl.groupby("pair")[value].mean().to_dict(), pl.groupby("pair")[value].std(ddof=1).to_dict()


def noise_margin(cfg, ser, root, tier=None, write=True):
    """factor x the pooled SD over seeds of the pair-level kappa_bal of the noise pipeline on the Dev tier."""
    nz = cfg.raw["noise"]
    tier = tier or nz["tier"]
    pairs = [p for p in ser.tier(tier) if ser.pairs[p].kind == "cross"]
    df = load(cfg, root, [nz["pipeline"]])
    out = dict(pipeline=nz["pipeline"], hash=cfg.pipeline_hash(nz["pipeline"]), tier=tier, factor=nz["factor"],
               pairs={}, margin=None)
    if df.empty:
        return out
    pl = pair_level(df[(df.role == "selected") & df.pair.isin(pairs)])
    var = []
    for pid, g in pl.groupby("pair"):
        v = g.kappa_bal.to_numpy()
        out["pairs"][pid] = dict(n_seeds=int(len(v)), mean=float(np.mean(v)),
                                 sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, values=[float(x) for x in v])
        if len(v) > 1:
            var.append(np.var(v, ddof=1))
    if var:
        out["pooled_sd"] = float(np.sqrt(np.mean(var)))
        out["margin"] = float(nz["factor"] * out["pooled_sd"])
    if write:
        f = noise_file(cfg, root)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, indent=1))
    return out


def noise_file(cfg, root):
    nz = cfg.raw["noise"]
    return Path(root) / "reports" / f"noise_{nz['pipeline']}-{cfg.pipeline_hash(nz['pipeline'])}_{nz['tier']}.json"


def read_noise(cfg, root):
    f = noise_file(cfg, root)
    return json.loads(f.read_text()) if f.exists() else None


# ---------- guardrails
def end_of_stack(df_sel, df_cand, end_fraction):
    """Share of queries whose selected target lies at the end of the target stack (by L) on the query's side."""
    ok = []
    for (direction, q), g in df_cand.groupby(["direction", "query"]):
        Ls = g.drop_duplicates("target").set_index("target").L_target.astype(float)
        sel = df_sel[(df_sel.direction == direction) & (df_sel["query"] == q)]
        if sel.empty or Ls.isna().all():
            continue
        Lq = float(sel.L_query.iloc[0])
        n_end = max(1, int(np.ceil(end_fraction * len(Ls))))
        end = Ls.sort_values(ascending=Lq < Ls.median()).index[:n_end]
        ok.append(sel.target.iloc[0] in set(end))
    return float(np.mean(ok)) if ok else np.nan


def guardrails(df, pair, cfg):
    """Pass/fail of the guardrails for one pair (A_fgw rows of all seeds)."""
    g = cfg.raw["guardrails"]
    a = df[(df.pipeline == "A_fgw") & (df.pair == pair.id)]
    out = {}
    if a.empty:
        return out
    sel, orc = pair_level(a[a.role == "selected"]), pair_level(a[a.role == "oracle"])
    regret = float(orc.kappa_bal.mean() - sel.kappa_bal.mean())
    fo = float(pair_level(a[a.role == "selected"], "foldover").foldover.mean())
    out.update(regret=regret, regret_ok=bool(regret <= g["regret_max"]), foldover=fo,
               foldover_ok=bool(fo <= g["foldover_max"]))
    if pair.guardrail == "end_of_stack":
        shares = [end_of_stack(a[(a.role == "selected") & (a.seed == s)], a[(a.role == "candidate") & (a.seed == s)],
                               g["end_fraction"]) for s in sorted(a.seed.unique())]
        share = float(np.nanmean(shares))
        out.update(end_share=share, end_ok=bool(share >= g["end_pass_share"]))
    return out


def selection_quality(df, pipeline="A_fgw"):
    """Per pair: Spearman between objective and kappa_bal over each query's candidates (mean), and the regret."""
    from scipy.stats import spearmanr
    rows = []
    c = df[(df.pipeline == pipeline) & (df.role == "candidate")]
    for pid, g in c.groupby("pair"):
        rhos = []
        for _, h in g.groupby(["seed", "direction", "query"]):
            if len(h) > 2 and h.kappa_bal.notna().sum() > 2:
                rhos.append(spearmanr(h.objective, h.kappa_bal, nan_policy="omit")[0])
        rows.append(dict(pair=pid, spearman_obj_kappa=float(np.nanmean(rhos)) if rhos else np.nan,
                         n_queries=int(g.groupby(["seed", "direction", "query"]).ngroups)))
    return pd.DataFrame(rows)


# ---------- the Go table
def ceiling_of(pair, ser, vals):
    """(kappa_bal, source) of a cross pair's ceiling: between embryos if run, else the mean of both W."""
    if pair.ceiling and pair.ceiling in vals:
        return vals[pair.ceiling], pair.ceiling
    ws = [vals[w] for w in (f"W:{pair.a.name}", f"W:{pair.b.name}") if w in vals]
    return (float(np.mean(ws)), "W") if ws else (np.nan, "")


def go_table(cfg, ser, root, tier, reference=None, seeds=None, margin=None):
    """The Go table (DataFrame), the ceiling table, the ranking table and the inputs used."""
    df = load(cfg, root, seeds=seeds)
    pairs = ser.tier(tier)
    cross = [p for p in pairs if ser.pairs[p].kind == "cross"]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), df
    vals, sds = {}, {}
    for col, pipe, role in COLUMNS:
        vals[col], sds[col] = per_pair(df, pipe, role)
    oms, _ = per_pair(df, "A_fgw", "selected", "oms")
    kap, _ = per_pair(df, "A_fgw", "selected", "kappa")
    K, _ = per_pair(df, "A_fgw", "selected", "K")
    ceil_vals, _ = per_pair(df, "A_fgw", "selected")      # ceilings are A on same-species pairs
    ceil_orc, _ = per_pair(df, "A_fgw", "oracle")
    # the reference: a pipeline (role selected) under its own hash, which may be an earlier result folder
    ref = reference or dict(pipeline="A_fgw", hash=cfg.pipeline_hash("A_fgw"))
    ref_df = store.load_scores(root, [f"{ref['pipeline']}-{ref['hash']}"])
    if len(ref_df) and seeds is not None:
        ref_df = ref_df[ref_df.seed.isin(seeds)]
    ref_vals, _ = per_pair(ref_df, ref["pipeline"], "selected") if len(ref_df) else ({}, {})
    rows = []
    for pid in cross:
        pair = ser.pairs[pid]
        c, src = ceiling_of(pair, ser, ceil_vals)
        r = dict(pair=pid, stage=pair.stage, overlap=pair.overlap, K=K.get(pid, np.nan))
        for col, *_ in COLUMNS:
            r[col] = vals[col].get(pid, np.nan)
        r["regret"] = r["oracle"] - r["A"]
        r.update(ceiling=c, ceiling_from=src, ceiling_share=r["A"] / c if c and np.isfinite(c) and c > 0 else np.nan,
                 W_human=ceil_vals.get(f"W:{pair.a.name}", np.nan), W_mouse=ceil_vals.get(f"W:{pair.b.name}", np.nan),
                 OMS=oms.get(pid, np.nan), kappa=kap.get(pid, np.nan), A_sd=sds["A"].get(pid, np.nan),
                 delta_ref=r["A"] - ref_vals.get(pid, np.nan))
        if margin is not None and np.isfinite(r["delta_ref"]):
            r["win"] = bool(r["delta_ref"] > margin)
        gr = guardrails(df, pair, cfg)
        r.update({f"g_{k}": v for k, v in gr.items()})
        rows.append(r)
    go = pd.DataFrame(rows)
    ranked = [p for p in cross if not ser.pairs[p].guardrail]
    if len(go):
        num = [c for c in go.columns if c not in ("pair", "stage", "overlap", "ceiling_from", "win")
               and not c.endswith("_ok") and pd.api.types.is_numeric_dtype(go[c])]
        med = go[go.pair.isin(ranked)][num].median()
        go = pd.concat([go, pd.DataFrame([dict(pair=f"median ({', '.join(ranked)})", **med.to_dict())])],
                       ignore_index=True)
    # ceilings
    crow = []
    for pid in pairs:
        pair = ser.pairs[pid]
        if pair.kind == "cross":
            continue
        crow.append(dict(pair=pid, kind=pair.kind, embryos=" / ".join(e.name for e in pair.embryos),
                         note=pair.note, kappa_bal=ceil_vals.get(pid, np.nan), oracle=ceil_orc.get(pid, np.nan)))
    ceil = pd.DataFrame(crow)
    # ranking of every pipeline (selected rows) against the reference
    rank = []
    for pipe in cfg.raw["pipelines"]:
        role = "selected"
        v, _ = per_pair(df, pipe, role)
        d = [v[p] - ref_vals[p] for p in ranked if p in v and p in ref_vals]
        rank.append(dict(pipeline=pipe, hash=cfg.pipeline_hash(pipe), role=role, n_pairs=len(d),
                         median_kappa_bal=float(np.median([v[p] for p in ranked if p in v])) if d else np.nan,
                         median_delta=float(np.median(d)) if d else np.nan,
                         wins=int(sum(x > margin for x in d)) if margin is not None and d else None))
    return go, ceil, pd.DataFrame(rank), df


def sweep_table(cfg_loader, key, values, ser, root, tier, pipeline="A_fgw"):
    """kappa_bal (selected, oracle) and solve time per value of one config key (each value its own hash)."""
    rows = []
    for v in values:
        cfg = cfg_loader([f"{key}={v}"])
        df = load(cfg, root, [pipeline])
        if df.empty:
            rows.append(dict(value=v, hash=cfg.pipeline_hash(pipeline)))
            continue
        for pid in ser.tier(tier):
            d = df[df.pair == pid]
            if d.empty:
                continue
            rows.append(dict(value=v, hash=cfg.pipeline_hash(pipeline), pair=pid,
                             A=pair_level(d[d.role == "selected"]).kappa_bal.mean(),
                             oracle=pair_level(d[d.role == "oracle"]).kappa_bal.mean(),
                             seconds_per_solve=float(d[d.role == "candidate"].seconds.mean()),
                             n_points=float(d[d.role == "candidate"].n_query.mean())))
    return pd.DataFrame(rows)
