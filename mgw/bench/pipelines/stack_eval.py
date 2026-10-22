"""Within embryo: how good is a within-embryo coupling tool (params.coupling, stack.register_sections)?

Both embryos of the pair are stacked with the tool (cached per embryo, seed and stack hash). Every kept coupling of
two neighbouring sections k, k + 1 (the coupling the stack's displacement pairs and the field's depth loss use) is
scored in both directions, k -> k + 1 with P and k + 1 -> k with P^T. Both sections come from one embryo, so K is the
organs present on both sections (no embryo-wide presence).

Rows: role `neighbour`, one per (embryo, query section, neighbour); direction_label is the embryo. The embryo's score
is the mean over its rows.
"""
from mgw.bench.pipelines import stacked


def run(task):
    coupling = task.spec["params"]["coupling"]
    for direction in task.pair.directions:
        emb = task.target_embryo(direction)
        _, ts = task.sides(direction)
        ST = stacked.get_stack(task, direction, coupling=coupling)
        names = ST["names"]
        if len(names) < 2:
            task.log(f"  {emb.name}: one section, no neighbours")
            continue
        for k, P in enumerate(ST["couplings"]):
            a, b = ts[names[k]], ts[names[k + 1]]
            for q, t, Pqt in ((a, b, P), (b, a, P.T)):
                row = task.score(direction, q, t.name, t.comp, Pqt, role="neighbour", present_t=None,
                                 direction_label=emb.name)
                task.log(f"  {emb.name} ({coupling}): {q.name} -> {t.name} kappa_bal {row['kappa_bal']:.3f}")
