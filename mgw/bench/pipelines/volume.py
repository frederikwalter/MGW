"""3D <-> 3D: stage 1 per embryo, one stage-2 solve per pair, scored on the real query spots exactly like 2D.

1. Stage 1 (per side): the registered stack with the within-embryo coupling `stack.coupling` (`stacked.get_stack`,
   cached), then either the field and its sampled volume (representation `field`; cached per pair, embryo, seed and
   field hash) or the stack points with slab masses and real features (representation `points`).
2. Stage 2: one unbalanced FGW solve between the two measures, the intra-volume cost by the aligner (fgw: Euclidean,
   rfgw: kNN shortest paths, mgw: pull-back geodesics of the field), the same tau on both sides, so P (a x b) serves
   the direction b_stack (queries a) and P^T a_stack. The cross cost is cdist / the pair's scale of the real features.
3. Back to the spots: for every real query section, A_q P against the target samples (their compositions B_t T),
   one `selected` row per (direction, query section), K from the whole target embryo, so the rows compare one-to-one
   with the 2D pipelines' selected rows.
"""
import numpy as np

from mgw.bench import volume
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
    with task.time("cost"):
        Ca, Cb = volume.intra_cost(aligner, ma, task.cfg), volume.intra_cost(aligner, mb, task.cfg)
    with task.time("solve"):
        P, obj, info = volume.stage2(Ca, Cb, ma["F"], mb["F"], task.scale, task.fgw, a=ma["mass"], b=mb["mass"],
                                     tau=p["tau"], epsilon=p["epsilon"],
                                     unbalanced_correction=p["unbalanced_correction"])
    rej_a, rej_b, mass = volume.rejected(P, ma["mass"], mb["mass"])
    task.log(f"  stage 2 ({aligner}): objective {obj:.4f}, mass {mass:.3f}, rejected a {rej_a:.3f} b {rej_b:.3f}"
             f"{' (ugw fallback)' if info.get('ugw_fallback') else ''}, {info.get('seconds', 0):.1f} s")
    tname = "volume" if p["representation"] == "field" else "points"
    for direction, qk, tk, Pd in (("b_stack", "a", "b", P), ("a_stack", "b", "a", P.T)):
        (_, STq, mq), (_, STt, mt) = side[qk], side[tk]
        qs, _ = task.sides(direction)
        T = np.asarray(volume.target_pool(mt, STt, p["k_target"]) @ STt["comp"])
        for s, name in enumerate(STq["names"]):
            Pq = np.asarray(volume.query_map(mq, STq, s, p["k_query"]) @ Pd)
            with task.time("score"):
                task.score(direction, qs[name], tname, T, Pq, role="selected", objective=obj, info=info)
