"""What every pipeline shares: the task (config, pair, seed, data), the cross-cost scale and one scoring routine.

Coordinates: every section is centred and scaled to an RMS radius of 1 for the solvers (FGW normalizes its
distances by their maximum, so the scale does not change a coupling).

The cross cost M of a pair is cdist(features) / s with one s per pair: the 0.99 quantile of the distances between
all points of side a and all points of side b. One scale for both directions makes a balanced solve of (x, y) the
transpose of (y, x), so each pair of sections is solved once; within a direction it is fixed across the candidates
of a query, as the objective comparison needs.
"""
import time

import numpy as np
from scipy.spatial.distance import cdist

from mgw.bench import metrics, prepare, store
from mgw.bench.solvers import FGWParams, centred, cost_scale


class Task:
    def __init__(self, cfg, pipeline, pair, seed, root, log=print):
        self.cfg, self.pipeline, self.pair, self.seed, self.root, self.log = cfg, pipeline, pair, seed, root, log
        self.spec = cfg.raw["pipelines"][pipeline]
        self.hash = cfg.pipeline_hash(pipeline)
        self.fgw = FGWParams.from_mapping(cfg.raw["fgw"])
        self.mp = cfg.raw["metrics"]
        self.out = store.result_dir(root, cfg, pipeline, pair.id, seed)
        self.data = prepare.load_pair(pair, cfg, seed, root)
        self.rows, self.transfer = [], {}
        self._xy = {}
        self.timings = {}
        A = np.vstack([s.emb for s in self.data["a"].values()])
        B = np.vstack([s.emb for s in self.data["b"].values()])
        f = cfg.raw["features"]
        self.scale = cost_scale(A, B, f["m_scale_quantile"], f["m_scale_max_entries"], seed=seed)

    # ---------- sides
    def sides(self, direction):
        """(query sections, target sections) of a direction, as {name: Section}."""
        return (self.data["b"], self.data["a"]) if direction == "a_stack" else (self.data["a"], self.data["b"])

    def target_embryo(self, direction):
        return self.pair.sides(direction)[1]

    def xy(self, sec):
        """Section coordinates, centred, RMS radius 1."""
        if sec.name not in self._xy:
            self._xy[sec.name] = centred(sec.xy)
        return self._xy[sec.name]

    def cross_cost(self, Fq, Ft):
        return cdist(Fq, Ft) / self.scale

    def target_presence(self, direction):
        """The organ mask of the target side: the organs of the whole target embryo, so a 2D target section and a 3D
        target volume are scored on the same K. With metrics.presence = "target", None: each target alone."""
        if self.mp["presence"] == "target":
            return None
        _, ts = self.sides(direction)
        return metrics.present(np.vstack([s.comp for s in ts.values()]), self.mp["present_min_bins"])

    # ---------- scoring
    def score(self, direction, q, target, T, P, *, role="candidate", objective=np.nan, info=None,
              present_t="embryo", direction_label=None, keep=True):
        """kappa_bal of one coupling P of query section q (rows: its spots) against a target with compositions T.
        present_t: "embryo" (the target embryo's organs, see target_presence), None (the target's own) or a mask.
        Records the row (and, for role "selected", the transferred organ of every query spot). Returns the row."""
        P = np.asarray(P, dtype=np.float64)
        if isinstance(present_t, str):
            present_t = self.target_presence(direction)
        kb, K = metrics.score(P, q.comp, T, min_bins=self.mp["present_min_bins"], present_t=present_t)
        info = info or {}
        row = dict(pipeline=self.pipeline, hash=self.hash, pair=self.pair.id, seed=self.seed, direction=direction,
                   direction_label=direction_label or self.pair.direction_label(direction), query=q.name,
                   target=target, role=role, objective=objective, K=K, kappa_bal=kb, n_query=q.n,
                   n_target=len(P[0]), seconds=info.get("seconds"))
        if keep:
            self.rows.append(row)
            if role == "selected":
                self.transfer[store.key(direction, q.name)] = metrics.hard_prediction(P, T).astype(np.uint8)
        return row

    def select(self, row):
        """A copy of a candidate row as the selected one."""
        r = dict(row, role="selected")
        self.rows.append(r)
        return r

    def time(self, name):
        task = self

        class _T:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                task.timings[name] = task.timings.get(name, 0.0) + time.perf_counter() - self.t0
        return _T()

    def save(self):
        if self.transfer:
            store.savez(self.out / "transfer.npz", **self.transfer)
        return store.write_scores(self.out, self.rows)
