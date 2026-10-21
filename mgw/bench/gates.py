"""Within-species gates for the field (docs/st_3d_alignment_strategies.md §3, §6 step 2): G1 (held-out section) and
G2 (odd/even split). Selection uses label-free scores only; label agreement is reported, never used to choose.

Each embryo's field regresses the features of one pair (`gates.embryos`: embryo -> the cross pair of stage 2), so
the gates test the features stage 2 will use. A unit is (setting, embryo, seed); its rows are cached as
results/gates/g1-<gate hash>/<embryo>/seed<s>/rows.csv (the hash covers the field and the scoring settings, so a
re-run skips done units), and `summarize_*` turns all units of a run into the summary and the pass table.

G1, for every interior section k:
- the stack without k: the other sections keep their registered positions and z in the full stack's frame, and
  k - 1 and k + 1 are joined by their direct coupling (as in `stack.loo_check`, cached once per embryo); re-registering
  without k would move the frame, and k's spots would then need a registration against k's own data;
- four predictions at k's own spot positions (z of k from the full stack): the field fitted without k; McCann
  (displacement interpolation between k - 1 and k + 1 at k's gap fraction, `VirtualSection.interp`); the nearer
  neighbour copied (its nearest spot in-plane); the nearer neighbour matched through the coupling (the McCann
  positions at that t, but every value copied from the nearer endpoint of its segment, no mixing). Each spot of k
  takes the nearest interpolated point in-plane;
- held-out genes: the field predicts features, not genes, so every predictor's features go through one
  feature -> held-out-gene kNN regressor (distance-weighted, k_regressor neighbours) fitted on the *other* sections'
  real spots; `r_heldout` (median over genes of the per-gene Pearson r over k's spots) is the selection score. The
  baselines that transport genes directly also report `r_direct`. The same regressor on k's true features
  (`true_features`) is the ceiling of the regressor path;
- also: `feat_err` (MSE of the features over their variance in k) and label agreement (a kNN classifier on the
  predicted features, and the transported labels for the baselines; reported only);
- pass (per embryo): the field beats the nearer-neighbour copy on r_heldout by more than the seed noise
  (factor x the SD over seeds of the field's r_heldout; 0 with one seed) and is no worse than McCann by more than it.
  A grid over `gates.g1.grid` (config keys -> values) x `gates.g1.zscore_by` picks the setting with the best mean
  r_heldout of the field over the embryos.

G2, for embryos with >= min_sections sections: the odd and the even sections (stack order) form two halves; each gets
a field and a volume (and its point cloud, the ablation), and stage 2 (unbalanced FGW, the volume pipeline's
settings in `gates.g2`) aligns the halves. The true map is the identity in the full stack's frame. Two variants:
`common` (the halves keep the full stack's z) and `reest` (each half re-estimates its gaps with `estimate_gaps`;
its z is carried back to the full frame through its own sections). Scores, per direction (odd -> even, even -> odd),
on the real spots of the source half whose true z lies within the target half's depth range, from the barycentric
map of A_q P, in units of the full stack's mean section spacing: median 3D error, median z error, and plane locking
(the share of mapped z within +-10 % of the target half's mean spacing of one of its section planes; `lock_chance`
is that share for z uniform over the target's range, about 20 %; `lock_true` the share of the true z). The field
also reports the errors of its samples. Pass: median 3D error < 0.5, locking <= 1.5 x chance, and the field's 3D error
below the point cloud's.
"""
import itertools
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from mgw import crossspecies as cs, stack3d as s3
from mgw.bench import config as bconfig, field, metrics, prepare, solvers, stack as bstack, store, volume
from mgw.bench.pipelines import stacked

PREDICTORS = ("field", "mccann", "nearest", "matched", "true_features")


# ---------- settings
def settings(cfg, gate="g1", grid=True):
    """[(label, overrides)] of a gate: the grid x zscore_by for G1 (one setting without the grid), the config as is
    for G2."""
    if gate != "g1" or not grid:
        return [("config", [])]
    g = cfg.raw["gates"]["g1"]
    keys = {"features.zscore_by": list(g["zscore_by"]), **{k: list(v) for k, v in g["grid"].items()}}
    out = []
    for vals in itertools.product(*keys.values()):
        ov = [f"{k}={v}" for k, v in zip(keys, vals)]
        out.append((",".join(ov), ov))
    return out


def unit_dir(root, cfg, gate, embryo, seed):
    return Path(root) / "results" / "gates" / f"{gate}-{cfg.gate_hash(gate)}" / embryo / f"seed{seed}"


def embryo_context(cfg, ser, root, seed, embryo, log=print):
    """(pair id, the embryo's side {name: Section}, its stack with the pair's features, skip pairs)."""
    pid = cfg.raw["gates"]["embryos"].get(embryo)
    if pid is None:
        pid = next((p for p in ser.pairs.values() if p.kind == "cross" and embryo in (p.a.name, p.b.name)), None)
        pid = pid.id if pid else None
    if pid is None:
        raise KeyError(f"{embryo}: no pair in gates.embryos and no cross pair with it")
    pair = ser.pairs[pid]
    f = prepare.feature_file(root, pid, seed, cfg.feature_hash())
    if not f.exists():
        raise FileNotFoundError(f"features of {pid} (seed {seed}, {cfg.feature_hash()}) missing: run `prepare` with "
                                f"these settings, e.g. --set features.zscore_by={cfg.raw['features']['zscore_by']}")
    side = "a" if pair.a.name == embryo else "b"
    secs = prepare.load_pair(pair, cfg, seed, root)[side]
    e = ser.embryos[embryo]
    ST = stacked.load_stack(cfg, root, seed, e, secs, log=log)
    skips = volume.skip_pairs(cfg, root, seed, embryo, ST, log=log) if len(ST["names"]) >= 3 else {}
    return pid, secs, ST, skips


# ---------- G1
def _r(Yp, Y):
    return float(np.nanmedian(cs.colwise_r(np.asarray(Yp, dtype=np.float64), np.asarray(Y, dtype=np.float64))))


def g1_predictions(ST, skips, k, fld):
    """{predictor: (features, transported genes or None, transported labels or None)} at the spots of section k."""
    off, z = ST["offsets"], np.asarray(ST["z"], dtype=np.float64)
    sk = slice(off[k], off[k + 1])
    xk = ST["xyz"][sk]
    F, Y, lab = ST["emb"]["X_ortho"], ST["Y"], ST["labels"]
    a, b = k - 1, k + 1
    out = {"field": (fld.features(xk), None, None)} if fld is not None else {}
    # the two neighbours at z 0 / 1 and their direct coupling (as loo_check); the plane at k's gap fraction
    two = volume.substack(ST, [a, b], skips, z=[0.0, 1.0])
    t = (z[k] - z[a]) / (z[b] - z[a]) if z[b] > z[a] else 0.5
    vs = s3.VirtualSection(two["xyz"], two["pairs"], s3.plane(t), ST["frame"], max_obs=1 << 30)
    nn = cKDTree(vs.xyz[:, :2]).query(xk[:, :2])[1]
    F2, Y2, l2 = two["emb"]["X_ortho"], two["Y"], two["labels"]
    near = np.where(vs.t < 0.5, vs.i, vs.j)
    lab_near = vs.labels(l2)[nn]
    out["mccann"] = (vs.interp(F2)[nn], vs.interp(Y2)[nn], lab_near)
    out["matched"] = ((vs._A @ F2[near])[nn], (vs._A @ Y2[near])[nn], lab_near)
    c = a if z[k] - z[a] <= z[b] - z[k] else b
    sc = slice(off[c], off[c + 1])
    j = cKDTree(ST["xyz"][sc, :2]).query(xk[:, :2])[1]
    out["nearest"] = (F[sc][j], Y[sc][j], lab[sc][j])
    out["true_features"] = (F[sk], None, None)
    return out


def g1_unit(cfg, ser, root, embryo, seed, log=print):
    """G1 rows of one embryo and seed: one per (interior section, predictor)."""
    pid, secs, ST, skips = embryo_context(cfg, ser, root, seed, embryo, log)
    n, off = len(ST["names"]), ST["offsets"]
    rows = []
    for k in range(1, n - 1):
        keep = [m for m in range(n) if m != k]
        sub = volume.substack(ST, keep, skips)
        fld = field.get_field(cfg, root, seed, pid, embryo, sub, sub["emb"]["X_ortho"], variant=f"loo{k}", log=log)
        kr = cfg.raw["gates"]["g1"]["k_regressor"]
        reg = KNeighborsRegressor(kr, weights="distance").fit(sub["emb"]["X_ortho"], sub["Y"])
        clf = KNeighborsClassifier(kr).fit(sub["emb"]["X_ortho"], sub["labels"])
        sk = slice(off[k], off[k + 1])
        Fk, Yk, lk = ST["emb"]["X_ortho"][sk], ST["Y"][sk], ST["labels"][sk]
        var = float(((Fk - Fk.mean(0)) ** 2).sum(1).mean())
        for name, (Fp, Yd, ld) in g1_predictions(ST, skips, k, fld).items():
            rows.append(dict(embryo=embryo, pair=pid, seed=seed, section=ST["names"][k], k=k, n_spots=len(Fk),
                             predictor=name, r_heldout=_r(reg.predict(Fp), Yk),
                             feat_err=float(((Fp - Fk) ** 2).sum(1).mean() / max(var, 1e-12)),
                             r_direct=_r(Yd, Yk) if Yd is not None else np.nan,
                             label_agree=float((clf.predict(Fp) == lk).mean()),
                             label_agree_direct=float((ld == lk).mean()) if ld is not None else np.nan,
                             gap_fraction=float((ST["z"][k] - ST["z"][k - 1]) / max(ST["z"][k + 1] - ST["z"][k - 1],
                                                                                      1e-12))))
        log(f"  G1 {embryo} {ST['names'][k]}: " + ", ".join(
            f"{r['predictor']} {r['r_heldout']:.3f}" for r in rows[-len(PREDICTORS):]))
    return pd.DataFrame(rows)


def summarize_g1(df, factor=2.0):
    """(per embryo and setting: mean r_heldout per predictor, the deltas, the margin and pass; per setting: the
    selection score and the number of embryos passed, best first)."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    per_seed = df.groupby(["setting", "embryo", "seed", "predictor"]).r_heldout.mean().unstack("predictor")
    out = []
    for (setting, embryo), g in per_seed.groupby(level=["setting", "embryo"]):
        sd = float(g["field"].std(ddof=1)) if len(g) > 1 else np.nan
        margin = factor * sd if np.isfinite(sd) else 0.0
        d_nn, d_mc = float((g["field"] - g["nearest"]).mean()), float((g["field"] - g["mccann"]).mean())
        lab = df[(df.setting == setting) & (df.embryo == embryo)].groupby("predictor").label_agree.mean()
        out.append(dict(setting=setting, embryo=embryo, n_seeds=len(g),
                        **{f"r_{p}": float(g[p].mean()) for p in PREDICTORS if p in g},
                        field_sd=sd, margin=margin, d_nearest=d_nn, d_mccann=d_mc,
                        label_agree_field=float(lab.get("field", np.nan)),
                        label_agree_nearest=float(lab.get("nearest", np.nan)),
                        passed=bool(d_nn > margin and d_mc >= -margin)))
    emb = pd.DataFrame(out)
    sel = emb.groupby("setting").agg(score=("r_field", "mean"), d_nearest=("d_nearest", "mean"),
                                     d_mccann=("d_mccann", "mean"), n_embryos=("embryo", "count"),
                                     n_passed=("passed", "sum")).reset_index()
    sel["all_passed"] = sel.n_passed == sel.n_embryos
    return emb, sel.sort_values("score", ascending=False).reset_index(drop=True)


# ---------- G2
def lock_share(zm, planes, tol):
    planes = np.asarray(planes)
    return float((np.abs(np.asarray(zm)[:, None] - planes[None]).min(1) <= tol).mean()) if len(zm) else np.nan


def lock_chance(planes, tol):
    """The share of [first plane, last plane] within tol of a plane (z uniform)."""
    planes = np.sort(np.asarray(planes, dtype=np.float64))
    lo, hi = planes[0], planes[-1]
    if hi <= lo:
        return np.nan
    iv = sorted((max(p - tol, lo), min(p + tol, hi)) for p in planes)
    cover, cur = 0.0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            cover += 0 if cur is None else cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return float((cover + cur[1] - cur[0]) / (hi - lo))


def _halves(ST, skips, variant):
    """{odd|even: substack} with zmap = (half z, full z) of its sections."""
    n, z = len(ST["names"]), np.asarray(ST["z"], dtype=np.float64)
    out = {}
    for h, keep in (("odd", list(range(0, n, 2))), ("even", list(range(1, n, 2)))):
        zh = z[keep]
        if variant == "reest":
            off, X = ST["offsets"], ST["emb"]["X_feat"]
            gp = bstack.estimate_gaps([ST["names"][k] for k in keep], [ST["reg_xy"][k] for k in keep],
                                      [X[off[k]:off[k + 1]] for k in keep])
            zh = z[keep[0]] + np.asarray(gp["z"])
        sub = volume.substack(ST, keep, skips, z=zh)
        sub["zmap"] = (zh, z[keep])
        out[h] = sub
    return out


def _to_full(x, sub):
    """Half-frame coordinates -> the full stack's frame (z through the half's own sections, piecewise linear)."""
    zh, zf = sub["zmap"]
    y = np.array(x, dtype=np.float64, copy=True)
    y[:, 2] = np.interp(y[:, 2], zh, zf) if zh[-1] > zh[0] else zf[0]
    return y


def g2_scores(ST, src, dst, ms, md, P, k_query, unit):
    """Identity-map scores of one direction (rows of P: the source half's measure)."""
    zf_dst = dst["zmap"][1]
    tol = 0.1 * float(np.mean(np.diff(zf_dst))) if len(zf_dst) > 1 else np.nan
    Xd = _to_full(md["X"], dst)
    bary, true, mass = [], [], []
    for s in range(len(src["names"])):
        Pq = np.asarray(volume.query_map(ms, src, s, k_query) @ P)
        bary.append(metrics.barycentric(Pq, Xd))
        mass.append(Pq.sum(1))
        true.append(ST["xyz"][src["rows"][src["offsets"][s]:src["offsets"][s + 1]]])
    bary, true, mass = np.vstack(bary), np.vstack(true), np.concatenate(mass)
    inside = (true[:, 2] >= zf_dst[0] - 1e-9) & (true[:, 2] <= zf_dst[-1] + 1e-9) & (mass > 0)
    e3 = np.linalg.norm(bary - true, axis=1)[inside] / unit
    ez = np.abs(bary[:, 2] - true[:, 2])[inside] / unit
    out = dict(n_scored=int(inside.sum()), med3d=float(np.median(e3)) if len(e3) else np.nan,
               medz=float(np.median(ez)) if len(ez) else np.nan,
               lock=lock_share(bary[inside, 2], zf_dst, tol), lock_true=lock_share(true[inside, 2], zf_dst, tol),
               lock_chance=lock_chance(zf_dst, tol))
    if ms["kind"] == "field":                  # the samples' own map
        Xs = _to_full(ms["X"], src)
        bs = metrics.barycentric(P, Xd)
        ok = (Xs[:, 2] >= zf_dst[0]) & (Xs[:, 2] <= zf_dst[-1]) & (P.sum(1) > 0)
        out.update(med3d_samples=float(np.median(np.linalg.norm(bs - Xs, axis=1)[ok]) / unit) if ok.any() else np.nan,
                   lock_samples=lock_share(bs[ok, 2], zf_dst, tol))
    return out


def g2_unit(cfg, ser, root, embryo, seed, log=print):
    """G2 rows of one embryo and seed: one per (variant, representation, direction)."""
    pid, secs, ST, skips = embryo_context(cfg, ser, root, seed, embryo, log)
    g = cfg.raw["gates"]["g2"]
    n, z = len(ST["names"]), np.asarray(ST["z"], dtype=np.float64)
    if n < max(g["min_sections"], 4):
        log(f"  G2 {embryo}: {n} sections, skipped (min {g['min_sections']})")
        return pd.DataFrame()
    unit = float((z[-1] - z[0]) / (n - 1))
    p = solvers.FGWParams.from_mapping(cfg.raw["fgw"])
    f = cfg.raw["features"]
    rows = []
    for variant in g["variants"]:
        halves = _halves(ST, skips, variant)
        scale = solvers.cost_scale(halves["odd"]["emb"]["X_ortho"], halves["even"]["emb"]["X_ortho"],
                                   f["m_scale_quantile"], f["m_scale_max_entries"], seed=seed)
        for rep in ("field", "points"):
            base = dict(embryo=embryo, pair=pid, seed=seed, variant=variant, representation=rep, n_sections=n,
                        unit=unit, z_odd=json.dumps(np.round(halves["odd"]["zmap"][0], 4).tolist()),
                        z_even=json.dumps(np.round(halves["even"]["zmap"][0], 4).tolist()))
            try:
                m = {h: volume.measure(cfg, root, seed, pid, embryo, sub, rep, g["n_samples"],
                                       variant=f"{h}_{variant}", log=log) for h, sub in halves.items()}
                P, obj, info = volume.stage2("fgw", m["odd"]["X"], m["even"]["X"], m["odd"]["F"], m["even"]["F"],
                                             scale, p, a=m["odd"]["mass"], b=m["even"]["mass"], tau=g["tau"],
                                             epsilon=g["epsilon"], unbalanced_correction=g["unbalanced_correction"])
            except (ValueError, RuntimeError, FloatingPointError) as e:
                log(f"  G2 {embryo} {variant} {rep}: failed ({type(e).__name__}: {e})")
                rows.append(dict(base, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=3)))
                continue
            rej_o, rej_e, mass = volume.rejected(P, m["odd"]["mass"], m["even"]["mass"])
            for direction, (s, d, Pd) in (("odd->even", ("odd", "even", P)), ("even->odd", ("even", "odd", P.T))):
                sc = g2_scores(ST, halves[s], halves[d], m[s], m[d], Pd, g["k_query"], unit)
                rows.append(dict(base, direction=direction, objective=obj, mass_total=mass,
                                 rejected_source=rej_o if s == "odd" else rej_e,
                                 rejected_target=rej_e if s == "odd" else rej_o, n_measure=len(m[s]["X"]),
                                 ugw_fallback=bool(info.get("ugw_fallback", False)),
                                 seconds=info.get("seconds"), **sc))
            r = rows[-2:]
            log(f"  G2 {embryo} {variant} {rep}: median 3D error {np.mean([x['med3d'] for x in r]):.3f}, "
                f"locking {np.mean([x['lock'] for x in r]):.3f} (chance {np.mean([x['lock_chance'] for x in r]):.3f})")
    return pd.DataFrame(rows)


def summarize_g2(df):
    """Per (embryo, variant): field and point cloud side by side (means over directions and seeds) and the pass."""
    if df.empty or "med3d" not in df:
        return pd.DataFrame()
    d = df.dropna(subset=["med3d"])
    m = d.groupby(["embryo", "variant", "representation"])[["med3d", "medz", "lock", "lock_chance", "lock_true"]]
    m = m.mean().unstack("representation")
    m.columns = [f"{a}_{b}" for a, b in m.columns]
    m = m.reset_index()
    for c in ("med3d_field", "lock_field", "lock_chance_field", "med3d_points"):
        if c not in m:
            m[c] = np.nan
    m["passed"] = ((m.med3d_field < 0.5) & (m.lock_field <= 1.5 * m.lock_chance_field)
                   & (m.med3d_field < m.med3d_points.fillna(np.inf)))
    m["points_passed"] = ((m.med3d_points < 0.5) & (m.get("lock_points", np.nan) <= 1.5 * m.get(
        "lock_chance_points", np.nan)))
    return m


# ---------- runs
def run(cfg, ser, root, gate, embryos=None, seeds=None, grid=True, force=False, log=print):
    """Run a gate over embryos x seeds (x the G1 grid); write every unit's rows and the summaries under
    results/gates/<config name>_<gate>/. Returns (rows, per-embryo summary, selection or None)."""
    embryos = embryos or list(cfg.raw["gates"]["embryos"])
    seeds = seeds if seeds is not None else list(cfg.raw["seeds"]["dev"])
    frames = []
    for label, ov in settings(cfg, gate, grid):
        c = bconfig.load(cfg.path, list(cfg.overrides) + ov) if ov else cfg
        for e in embryos:
            for s in seeds:
                d = unit_dir(root, c, gate, e, s)
                f = d / "rows.csv"
                if f.exists() and not force:
                    log(f"[skip] {gate} {label} {e} seed {s}: done ({d})")
                    df = pd.read_csv(f)
                else:
                    log(f"[run] {gate} {label} {e} seed {s} -> {d}")
                    try:
                        df = (g1_unit if gate == "g1" else g2_unit)(c, ser, root, e, s, log=log)
                    except FileNotFoundError as err:
                        log(f"[missing] {gate} {label} {e} seed {s}: {err}")
                        continue
                    store.write_table(f, df)
                    (d / "key.json").write_text(json.dumps(json.loads(bconfig.canonical(c.gate_key(gate))), indent=1))
                if len(df):
                    frames.append(df.assign(setting=label, gate_hash=c.gate_hash(gate)))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = Path(root) / "results" / "gates" / f"{cfg.raw['name']}_{gate}"
    store.write_table(out / "rows.csv", rows)
    if gate == "g1":
        emb, sel = summarize_g1(rows, cfg.raw["gates"]["factor"])
        store.write_table(out / "summary.csv", emb)
        store.write_table(out / "selection.csv", sel)
        if len(sel):
            (out / "best.json").write_text(json.dumps(dict(sel.iloc[0]), indent=1, default=str))
        return rows, emb, sel
    summ = summarize_g2(rows)
    store.write_table(out / "summary.csv", summ)
    return rows, summ, None
