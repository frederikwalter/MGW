"""What every pipeline shares: the task (config, pair, seed, data), the cross-cost scale and one scoring routine.

Coordinates: every section is centred and scaled to an RMS radius of 1 for the solvers (FGW normalizes its
distances by their maximum, so the scale does not change a coupling; it keeps fold-overs and plots comparable).

The cross cost M of a pair is cdist(features) / s with one s per pair: the 0.99 quantile of the distances between
all points of side a and all points of side b. One scale for both directions makes a balanced solve of (x, y) the
transpose of (y, x), so each pair of sections is solved once; within a direction it is fixed across the candidates
of a query, as the objective comparison needs. (The 3D notebook scales per query instead.)
"""
import json
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
        self.rows, self.cand, self.targets = [], {}, {}
        self._tri, self._xy = {}, {}
        self.timings = {}
        A = np.vstack([s.emb for s in self.data["a"].values()])
        B = np.vstack([s.emb for s in self.data["b"].values()])
        f = cfg.raw["features"]
        self.scale = cost_scale(A, B, f["m_scale_quantile"], f["m_scale_max_entries"], seed=seed)

    # ---------- sides
    def sides(self, direction):
        """(query sections, target sections) of a direction, as {name: Section}."""
        if direction == "self":
            return self.data["a"], self.data["a"]
        return (self.data["b"], self.data["a"]) if direction == "a_stack" else (self.data["a"], self.data["b"])

    def target_embryo(self, direction):
        return self.pair.sides(direction)[1]

    def xy(self, sec):
        """Section coordinates, centred, RMS radius 1."""
        if sec.name not in self._xy:
            self._xy[sec.name] = centred(sec.xy)
        return self._xy[sec.name]

    def nbr(self, q):
        """The k nearest neighbours of every query point (for the local Jacobians of the fold-over check)."""
        if q.name not in self._tri:
            self._tri[q.name] = metrics.neighbourhoods(self.xy(q), self.mp["foldover_k"])
        return self._tri[q.name]

    def cross_cost(self, Fq, Ft):
        return cdist(Fq, Ft) / self.scale

    def target_presence(self, direction, query):
        """(full, reduced) organ masks of the target side: the organs of the whole target embryo (the query's own
        section left out in a within pair), so a 2D target section and a 3D target volume are scored on the same K.
        With metrics.presence = "target", None: the organs of each target alone (the first harness)."""
        if self.mp["presence"] == "target":
            return None
        _, ts = self.sides(direction)
        secs = [s for s in ts.values() if s.name != query]
        mp = self.mp
        return (metrics.present(np.vstack([s.comp for s in secs]), mp["present_min_bins"]),
                metrics.present(np.vstack([s.onehot for s in secs]), mp["present_min_tiles"]))

    # ---------- scoring
    def score(self, direction, q, target, xt, T_full, T_red, Yt, P, *, role="candidate", objective=np.nan,
              info=None, L_target=np.nan, extra=None, keep=True):
        """Score one coupling of query q against a target (coordinates xt, 2D or 3D; compositions; held-out genes)
        and record its row and compact arrays. Returns the row."""
        P = np.asarray(P, dtype=np.float64)
        mp = self.mp
        s, C_full, C_red = metrics.label_scores(P, q.comp, T_full, q.onehot, T_red, min_bins=mp["present_min_bins"],
                                                min_tiles=mp["present_min_tiles"], coverage_share=mp["coverage_share"],
                                                present_t=self.target_presence(direction, q.name))
        ms, folded = metrics.map_scores(self.xy(q), centred(xt), P, mp["foldover_k"], nbr=self.nbr(q))
        ce = metrics.coexpression(P, q.Y, Yt, seed=self.seed)
        pred, conf = metrics.hard_prediction(P, T_red)
        info = info or {}
        recall = s.pop("recall")
        row = dict(pipeline=self.pipeline, hash=self.hash, pair=self.pair.id, pair_kind=self.pair.kind,
                   seed=self.seed, direction=direction, direction_label=self.pair.direction_label(direction),
                   query=q.name, target=target, role=role, objective=objective, **s, **ms, **ce,
                   n_query=q.n, n_target=len(xt), L_query=q.L, L_target=L_target,
                   converged=info.get("converged"), outer_stopped=info.get("outer_stopped"),
                   inner_converged=info.get("inner_converged"), n_outer=info.get("n_outer"), seconds=info.get("seconds"),
                   recall=json.dumps({k: round(v, 4) for k, v in recall.items()}), **(extra or {}))
        if keep:
            self.rows.append(row)
            # a coupling scored under its own role (e.g. null:rigid on the selected target) must not overwrite the
            # arrays of the candidate with the same target
            tkey = target if role in ("candidate", "selected") else f"{target}@{role}"
            row["array_key"] = tkey
            k = lambda f: store.key(direction, q.name, tkey, f)
            self.cand[k("C_full")] = C_full.astype(np.float32)
            self.cand[k("C_red")] = C_red.astype(np.float32)
            self.cand[k("pred")] = pred.astype(np.uint8)
            self.cand[k("conf")] = conf.astype(np.float16)
            self.cand[k("folded")] = np.packbits(folded)
            if self.spec["store"]["topk"]:
                ir, vr, ic, vc = metrics.topk(P, mp["topk"])
                n_q, n_t = P.shape
                self.cand[k("top_row_idx")], self.cand[k("top_row_val")] = ir, (vr * n_q).astype(np.float16)
                self.cand[k("top_col_idx")], self.cand[k("top_col_val")] = ic, (vc * n_t).astype(np.float16)
        return row

    def keep_dense(self, role):
        st = self.spec["store"]
        return role in st["dense"] and self.pair.kind in st["dense_kinds"]

    def add_role(self, row, role):
        """A copy of a candidate row under another role (selected, oracle, L_partner)."""
        r = dict(row, role=role)
        self.rows.append(r)
        return r

    def mean_row(self, rows, role, target="mean"):
        """The expectation over candidates (e.g. a uniformly random section): the mean of every numeric score."""
        out = dict(rows[0], role=role, target=target)
        for k, v in rows[0].items():
            if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and k not in (
                    "seed", "n_query"):
                vals = np.array([r[k] for r in rows], dtype=np.float64)
                out[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        out["recall"] = "{}"
        self.rows.append(out)
        return out

    def time(self, name):
        task = self

        class _T:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                task.timings[name] = task.timings.get(name, 0.0) + time.perf_counter() - self.t0
        return _T()

    def save(self):
        if self.cand:
            store.savez(self.out / "candidates.npz", **self.cand)
        if self.targets:
            store.savez(self.out / "targets.npz", **self.targets)
        return store.write_scores(self.out, self.rows)
