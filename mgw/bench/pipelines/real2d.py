"""Strategy A: every query section against every real target section; selection by the aligner's objective.

One solve per pair of sections serves both directions (P and its transpose; see common.py on the cost scale). The
mirror image is not tried: FGW on Euclidean distances is invariant to reflections, so a mirrored query gives the same
coupling (tests/test_solvers.py); the orientation is reported as `orientation`.

Roles per (direction, query):
- candidate: every target section;
- selected: the lowest objective; oracle: the highest kappa_bal (labels: the headroom); L_partner: the target with the
  closest laterality score; null:random_section: the mean over the candidates (a uniformly random real section);
- with the FGW aligner also null:rigid (OT-ICP from the selected coupling: rigid placement, no deformation) and
  null:prior (the organ-frequency prior of the target embryo, P = a b^T).
"""
import numpy as np

from mgw.bench import metrics, solvers, store


def _solve(task, aligner, sx, sy, M, D):
    """Coupling of sections sx (rows) and sy, its objective and solver info. D caches max-normalized distances."""
    p = task.fgw
    if aligner == "fgw":
        P, info = solvers.fgw(None, None, M, p, Ca=D["xy", sx.name], Cb=D["xy", sy.name])
        obj = solvers.fgw_objective(None, None, M, P, p.alpha, Ca=D["xy", sx.name], Cb=D["xy", sy.name])
    elif aligner == "linear":
        P, info = solvers.linear_ot(M, task.cfg.raw["aligners"]["linear"]["epsilon"], p.inner_maxit, p.inner_tol)
        obj = float((M * P).sum() / P.sum())
    elif aligner == "spatial_gw":
        P, info = solvers.gw(D["xy", sx.name], D["xy", sy.name], p, task.cfg.raw["aligners"]["spatial_gw"]["epsilon"])
        obj = solvers.gw_energy(D["xy", sx.name], D["xy", sy.name], P / P.sum())
    elif aligner == "scot":
        P, info = solvers.gw(D["f", sx.name], D["f", sy.name], p, task.cfg.raw["aligners"]["scot"]["epsilon"])
        obj = solvers.gw_energy(D["f", sx.name], D["f", sy.name], P / P.sum())
    else:
        raise ValueError(aligner)
    return P, obj, info


def run(task):
    pair, aligner = task.pair, task.spec["aligner"]
    A, B = task.data["a"], task.data["b"]
    within = pair.kind == "within"
    if within:
        names = list(A)
        grid = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    else:
        grid = [(x, y) for x in A for y in B]
    D = {}
    for side in (A, B):
        for s in side.values():
            if ("xy", s.name) not in D:
                D["xy", s.name] = solvers.dist_norm(task.xy(s))
                if aligner == "scot":
                    D["f", s.name] = solvers.dist_norm(s.emb)
    keep_P = aligner == "fgw" or any(task.keep_dense(r) for r in ("selected", "oracle", "L_partner"))
    P16, cand_rows = {}, {}
    for n, (x, y) in enumerate(grid):
        sx, sy = A[x], (A if within else B)[y]
        with task.time("solve"):
            P, obj, info = _solve(task, aligner, sx, sy, task.cross_cost(sx.emb, sy.emb), D)
        task.log(f"  [{n + 1}/{len(grid)}] {x} x {y}: objective {obj:.4f}, {info.get('n_outer')} it "
                 f"(outer stopped {info.get('outer_stopped')}, inner converged {info.get('inner_converged', 1):.0%}), "
                 f"{info.get('seconds', 0):.1f} s")
        if keep_P:
            P16[x, y] = (P * P.size).astype(np.float16)
        # (query, target, coupling query x target) for each direction this solve serves
        uses = ([("self", sx, sy, P), ("self", sy, sx, P.T)] if within
                else [("b_stack", sx, sy, P), ("a_stack", sy, sx, P.T)])
        with task.time("score"):
            for direction, q, t, Pqt in uses:
                row = task.score(direction, q, t.name, task.xy(t), t.comp, t.onehot, t.Y, Pqt, objective=obj,
                                 info=info, L_target=t.L)
                cand_rows.setdefault((direction, q.name), []).append(row)

    def coupling(direction, q, t):
        if within:
            Pq = P16[q, t] if (q, t) in P16 else P16[t, q].T
        else:
            Pq = P16[q, t] if direction == "b_stack" else P16[t, q].T
        return Pq.astype(np.float64) / Pq.size

    for (direction, qname), rows in cand_rows.items():
        qs, ts = task.sides(direction)
        q = qs[qname]
        obj = np.array([r["objective"] for r in rows])
        kb = np.array([r["kappa_bal"] for r in rows], dtype=np.float64)
        sel = rows[int(np.nanargmin(obj))]
        orc = rows[int(np.nanargmax(np.where(np.isfinite(kb), kb, -np.inf)))] if np.isfinite(kb).any() else sel
        roles = {"selected": sel, "oracle": orc}
        Ls = np.array([r["L_target"] for r in rows], dtype=np.float64)
        if np.isfinite(q.L) and np.isfinite(Ls).any():
            roles["L_partner"] = rows[int(np.nanargmin(np.abs(Ls - q.L)))]
        for role, r in roles.items():
            task.add_role(r, role)
            if task.keep_dense(role) and keep_P:
                store.save_dense(task.out, direction, qname, role, coupling(direction, qname, r["target"]))
        task.mean_row(rows, "null:random_section")
        if aligner == "fgw" and task.spec["strategy"] == "real2d":
            # rigid placement only, started from the selected coupling
            t = ts[sel["target"]]
            rg = task.cfg.raw["aligners"]["rigid"]
            with task.time("rigid"):
                Pr, info = solvers.rigid_icp(task.xy(q), task.xy(t), q.emb, t.emb, coupling(direction, qname, t.name),
                                             lam=rg["icp_lambda"], n_iter=rg["icp_iter"], epsilon=rg["icp_eps"])
            task.score(direction, q, t.name, task.xy(t), t.comp, t.onehot, t.Y, Pr, role="null:rigid",
                       objective=solvers.fgw_objective(task.xy(q), task.xy(t), task.cross_cost(q.emb, t.emb), Pr,
                                                       task.fgw.alpha), info=info, L_target=t.L)
            # organ-frequency prior of the whole target embryo (without the query's own section in a within pair)
            T_full = np.vstack([s.comp for s in ts.values() if s.name != qname])
            T_red = np.vstack([s.onehot for s in ts.values() if s.name != qname])
            s, _, _ = metrics.prior_scores(q.comp, T_full, q.onehot, T_red, min_bins=task.mp["present_min_bins"],
                                           min_tiles=task.mp["present_min_tiles"],
                                           coverage_share=task.mp["coverage_share"],
                                           present_t=task.target_presence(direction, qname))
            s.pop("recall")
            task.rows.append(dict(sel, role="null:prior", target="prior", objective=np.nan, **{
                k: v for k, v in s.items()}, foldover=np.nan, cond_median=np.nan, sharpness=np.nan,
                coexpr_r=np.nan, coexpr_r0=np.nan, recall="{}"))
