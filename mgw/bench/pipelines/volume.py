"""The volume strategy (docs/st_3d_alignment_strategies.md §1, §6 step 3): stage 1 per embryo, one stage-2 solve per
pair, scored on the real query spots exactly like A.

1. Stage 1 (per side): the registered stack (`stacked.get_stack`, cached), then either the field and its sampled
   volume (representation `field`, §1c–§1d; cached per pair, embryo, seed and field hash) or the stack points with
   slab masses and real features (representation `points`, the ablation of §1e).
2. Stage 2: one solve between the two measures with the pipeline's aligner (`fgw`, or the nulls `linear`: feature
   only, `spatial_gw`: geometry only, `scot`), unbalanced with the same tau on both sides, so P (a x b) serves the
   direction b_stack (queries a) and P^T a_stack. The cross cost is Task.cross_cost (cdist / the pair's scale of the
   real features, as in A).
3. Back to the spots (§1f): for every real query section, A_q P against the target samples (3D coordinates,
   compositions B_t T, held-out genes B_t Y) through `Task.score`, one row per (direction, query section) with role
   `selected` and target `volume` (or `points`), so the rows compare one-to-one with A's selected rows. K comes from
   the whole target embryo (Task.target_presence). Only the top-k entries are stored (`store.dense: []`).

Extra columns (§2a guardrail, §2c, §2d): rejected_query (share of the query section's own sample mass that P leaves
unmatched, signed: < 0 means mass was added), rejected_side_query / rejected_side_target (whole measures),
mass_total; L_reached (the mass-weighted L of the target spots reached, also in L_target), L_reached_rel (its
position in the target's L range, 0 = the lowest L) and z_reached_rel (the same along the target's depth);
normal_dev_deg (the angle of the section's image plane to the mean of the query embryo's sections) and order_rho
(Spearman between the query sections' own z and their images' order along the mean normal); spacing_query,
spacing_target, n_query_measure, n_target_measure, tau, ugw_fallback.
"""
import numpy as np

from mgw.bench import metrics, volume
from mgw.bench.pipelines import stacked


def _side(task, direction):
    """(embryo, stack, measure) of the target embryo of a direction."""
    p = task.spec["params"]
    emb = task.target_embryo(direction)
    ST = stacked.get_stack(task, direction)
    if len(ST["names"]) < 2:
        raise ValueError(f"{emb.name}: a volume needs >= 2 sections")
    with task.time("field" if p["representation"] == "field" else "points"):
        m = volume.measure(task.cfg, task.root, task.seed, task.pair.id, emb.name, ST, p["representation"],
                           p["n_samples"], log=task.log)
    task.log(f"  {emb.name}: {len(m['X'])} {m['kind']} points, spacing {m['spacing']:.4f}")
    return emb, ST, m


def run(task):
    p, aligner = task.spec["params"], task.spec["aligner"]
    side = {"a": _side(task, "a_stack"), "b": _side(task, "b_stack")}
    (_, STa, ma), (_, STb, mb) = side["a"], side["b"]
    eps_al = task.cfg.raw["aligners"].get(aligner, {}).get("epsilon")
    with task.time("solve"):
        P, obj, info = volume.stage2(aligner, ma["X"], mb["X"], ma["F"], mb["F"], task.scale, task.fgw, a=ma["mass"],
                                     b=mb["mass"], tau=p["tau"], epsilon=p["epsilon"], aligner_eps=eps_al,
                                     unbalanced_correction=p["unbalanced_correction"])
    rej_a, rej_b, mass = volume.rejected(P, ma["mass"], mb["mass"])
    task.log(f"  stage 2 ({aligner}): objective {obj:.4f}, mass {mass:.3f}, rejected a {rej_a:.3f} b {rej_b:.3f}, "
             f"{info.get('seconds', 0):.1f} s")
    tname = "volume" if p["representation"] == "field" else "points"
    for direction, qk, tk, Pd in (("b_stack", "a", "b", P), ("a_stack", "b", "a", P.T)):
        (_, STq, mq), (_, STt, mt) = side[qk], side[tk]
        qs, ts = task.sides(direction)
        Bt = volume.target_pool(mt, STt, p["k_target"])
        T_full, T_red = np.asarray(Bt @ STt["comp"]), np.asarray(Bt @ metrics.onehot(STt["labels"]))
        Yt, Lt = np.asarray(Bt @ STt["Y"]), np.asarray(Bt @ STt["L"])
        z = np.asarray(STt["z"])
        zrel = (mt["X"][:, 2] - z[0]) / max(z[-1] - z[0], 1e-12)
        Ls = np.array([ts[s].L for s in STt["names"]], dtype=np.float64)
        Lmin, Lmax = (np.nanmin(Ls), np.nanmax(Ls)) if np.isfinite(Ls).any() else (np.nan, np.nan)
        mass_q = np.full(len(mq["X"]), 1 / len(mq["X"])) if mq["mass"] is None else mq["mass"]
        rows, images = [], []
        for s, name in enumerate(STq["names"]):
            Aq = volume.query_map(mq, STq, s, p["k_query"])
            Pq = np.asarray((Aq @ Pd))
            col = Pq.sum(0)
            Lr = float(col @ Lt / col.sum()) if col.sum() > 0 else np.nan
            ratio = float((Aq @ Pd.sum(1)).sum() / max((Aq @ mass_q).sum(), 1e-300))
            extra = dict(representation=p["representation"], n_query_measure=len(mq["X"]),
                         n_target_measure=len(mt["X"]), spacing_query=mq["spacing"], spacing_target=mt["spacing"],
                         tau=p["tau"], rejected_query=1 - ratio, rejected_side_query=rej_a if qk == "a" else rej_b,
                         rejected_side_target=rej_b if qk == "a" else rej_a, mass_total=mass, L_reached=Lr,
                         L_reached_rel=(Lr - Lmin) / (Lmax - Lmin) if Lmax > Lmin else np.nan,
                         z_reached_rel=float(col @ zrel / col.sum()) if col.sum() > 0 else np.nan,
                         ugw_fallback=bool(info.get("ugw_fallback", False)))
            with task.time("score"):
                row = task.score(direction, qs[name], tname, mt["X"], T_full, T_red, Yt, Pq, role="selected",
                                 objective=obj, info=info, L_target=Lr, extra=extra)
            rows.append(row)
            images.append((metrics.barycentric(Pq, mt["X"]), Pq.sum(1)))
        ang, rho = volume.parallel_sections(images, np.asarray(STq["z"]))
        for row, a in zip(rows, ang):
            row.update(normal_dev_deg=float(a), order_rho=rho)
        task.targets[f"{direction}/{tname}/xy"] = mt["X"][:, :2].astype(np.float32)
        task.targets[f"{direction}/{tname}/xyz"] = mt["X"].astype(np.float32)
        task.targets[f"{direction}/{tname}/label"] = metrics.majority(T_full)
