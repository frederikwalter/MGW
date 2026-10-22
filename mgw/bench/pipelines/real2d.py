"""2D <-> 2D: every query section against every real target section; the target is the one with the lowest objective.

One balanced solve per pair of sections serves both directions (P and its transpose; see common.py on the cost
scale). The aligner sets the intra-section cost of FGW:
- fgw: Euclidean distances, max-normalized;
- mgw: geodesics under the pull-back metric of phi, trained once per section (solvers.mgw_cost), max-normalized.
The objective is the FGW objective with those costs. The mirror image is not tried: FGW on intra-section distances is
invariant to reflections.

Rows per (direction, query): one `candidate` per target section, and the lowest objective again as `selected`.
"""
import numpy as np

from mgw.bench import metrics, solvers, store


def run(task):
    aligner = task.spec["aligner"]
    A, B = task.data["a"], task.data["b"]
    D = {}
    for side in (A, B):
        for s in side.values():
            with task.time("cost"):
                if aligner == "mgw":
                    D[s.name] = solvers.mgw_cost(task.xy(s), s.emb, task.cfg.raw["mgw"], seed=task.seed,
                                                 log=task.log)
                else:
                    D[s.name] = solvers.dist_norm(task.xy(s))
    grid = [(x, y) for x in A for y in B]
    cand, preds = {}, {}
    for n, (x, y) in enumerate(grid):
        sx, sy = A[x], B[y]
        M = task.cross_cost(sx.emb, sy.emb)
        with task.time("solve"):
            P, info = solvers.fgw(None, None, M, task.fgw, Ca=D[x], Cb=D[y])
        obj = solvers.fgw_objective(None, None, M, P, task.fgw.alpha, Ca=D[x], Cb=D[y])
        task.log(f"  [{n + 1}/{len(grid)}] {x} x {y}: objective {obj:.4f}, {info.get('n_outer')} it, "
                 f"{info.get('seconds', 0):.1f} s")
        # (direction, query, target, coupling query x target): b_stack has a's sections as queries
        for direction, q, t, Pqt in (("b_stack", sx, sy, P), ("a_stack", sy, sx, P.T)):
            row = task.score(direction, q, t.name, t.comp, Pqt, objective=obj, info=info)
            cand.setdefault((direction, q.name), []).append(row)
            preds[direction, q.name, t.name] = metrics.hard_prediction(Pqt, t.comp).astype(np.uint8)
    for (direction, qname), rows in cand.items():
        sel = rows[int(np.nanargmin([r["objective"] for r in rows]))]
        task.select(sel)
        task.transfer[store.key(direction, qname)] = preds[direction, qname, sel["target"]]
