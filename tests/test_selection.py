import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mgw import crossspecies as cs
from mgw.bench import metrics, selection as sel

REPO = Path(__file__).resolve().parents[1]
# one human stage serves two mouse stages (X3, X4), one mouse stage two human ones (X4, X5), both move at once (X5, X6)
LADDER = [("X1", "h1", "m1"), ("X2", "h2", "m2"), ("X3", "h3", "m2"), ("X4", "h3", "m3"), ("X5", "h4", "m4")]


def _script():
    spec = importlib.util.spec_from_file_location("select_sections", REPO / "bench" / "scripts" / "select_sections.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _instance(seed):
    rng = np.random.default_rng(seed)
    sections = {s: [f"{s}_{i}" for i in range(rng.integers(1, 4))] for s in ("h1", "h2", "h3", "h4", "m1", "m2", "m3", "m4")}
    secs = [x for v in sections.values() for x in v]
    return sections, {(a, b): float(rng.random()) for a in secs for b in secs}


def _brute(ladder, sections, cost, lam):
    h, m = sel.stage_orders(ladder)
    best = np.inf
    for ch in itertools.product(*(sections[s] for s in h)):
        for cm in itertools.product(*(sections[s] for s in m)):
            best = min(best, sel.chain_cost(ladder, dict(zip(h, ch)), dict(zip(m, cm)), cost, lam))
    return best


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", [0.0, 1.0, 2.0])
def test_chain_matches_brute_force(seed, lam):
    sections, cost = _instance(seed)
    total, ch, cm = sel.solve_chain(LADDER, sections, cost, lam)
    assert total == pytest.approx(_brute(LADDER, sections, cost, lam))
    assert total == pytest.approx(sel.chain_cost(LADDER, ch, cm, cost, lam))
    assert set(ch) == {"h1", "h2", "h3", "h4"} and set(cm) == {"m1", "m2", "m3", "m4"}


def test_chain_edges_count_every_cost_term():
    sections, cost = _instance(0)
    total, ch, cm = sel.solve_chain(LADDER, sections, cost, 1.0)
    assert sum(cost[a, b] for _, a, b in sel.chain_edges(LADDER, ch, cm)) == pytest.approx(total)
    assert len(sel.chain_edges(LADDER, ch, cm)) == 3 + 3 + len(LADDER)


def test_margin_is_the_rise_without_the_section():
    sections, cost = _instance(1)
    total, ch, cm = sel.solve_chain(LADDER, sections, cost, 1.0)
    t = sel.margins(LADDER, sections, cost, 1.0, ch, cm, total)
    assert (t.margin.dropna() >= -1e-12).all()
    for r in t.itertuples():
        if len(sections[r.stage]) < 2:
            assert np.isnan(r.margin) and r.alternative == ""
        else:
            assert r.alternative != r.section


def test_missing_edge_raises():
    sections = {s: [f"{s}_0"] for s in ("h1", "h2", "h3", "h4", "m1", "m2", "m3", "m4")}
    secs = [v[0] for v in sections.values()]
    cost = {(a, b): 0.5 for a in secs for b in secs}
    del cost["h2_0", "h3_0"]
    with pytest.raises(KeyError):
        sel.solve_chain(LADDER, sections, cost, 1.0)


def _hard(labels):
    return metrics.onehot(labels)


def test_w_cost_with_hard_labels_is_tv():
    rng = np.random.default_rng(0)
    A = _hard(rng.choice(["Brain", "Heart", "Liver"], 200, p=[.5, .3, .2]))
    B = _hard(rng.choice(["Brain", "Heart", "Liver"], 300, p=[.2, .3, .5]))
    row, _ = sel.edge_scores(A, B, 0.2, 0.5)
    assert row["w_cost"] == pytest.approx(row["tv"], abs=5e-3)
    assert row["dL"] == pytest.approx(0.3)
    assert row["converged"]


def test_label_cost_values():
    t = _hard(["Heart", "Heart", "Brain", "Other"])
    assert np.allclose(sel.label_cost(t, t), [[0, 0, 1, .5], [0, 0, 1, .5], [1, 1, 0, .5], [.5, .5, .5, .5]])


def test_marginals_follow_the_bins():
    rng = np.random.default_rng(1)
    A = _hard(rng.choice(["Brain", "Heart"], 80)) * rng.integers(1, 20, (80, 1))
    B = _hard(rng.choice(["Brain", "Heart"], 60)) * rng.integers(1, 20, (60, 1))
    row, P = sel.edge_scores(A, B, 0.0, 0.0)
    assert row["marg_err"] < 1e-3
    assert np.allclose(P.sum(1), A.sum(1) / A.sum(), atol=1e-3)
    assert np.allclose(P.sum(0), B.sum(1) / B.sum(), atol=1e-3)


def test_transported_r_is_coexpression_in_chunks():
    rng = np.random.default_rng(1)
    Yq, Yt = rng.random((120, 300), dtype=np.float32), rng.random((90, 300), dtype=np.float32)
    P = rng.random((120, 90))
    r, r0 = sel.transported_r(P, Yq, Yt, seed=0, chunk=97)
    ra, ra0 = cs.coexpression(P, Yq, Yt, seed=0)
    assert np.allclose(r, ra, atol=1e-5, equal_nan=True) and np.allclose(r0, ra0, atol=1e-5, equal_nan=True)


def _scores_table():
    rows = [("HESTA:A_E1S1_HESTA", "HESTA", "A", "1", 1.0, .2, .5, 100),
            ("HESTA:A_E1S2_HESTA", "HESTA", "A", "1", 2.0, .01, .5, 100),
            ("HESTA:CS23_E2S2_HESTA", "HESTA", "A", "1", 3.0, .5, .5, 100),
            ("HESTA:A_3D_HESTA", "HESTA", "A", "?", np.nan, .5, .5, 100),
            ("MOSTA:B_E1S1.MOSTA", "MOSTA", "B", "1", 1.0, .2, .1, 100),
            ("MOSTA:C_E1S1.MOSTA", "MOSTA", "C", "1", 1.0, .2, .1, 100)]
    t = pd.DataFrame(rows, columns=["key", "dataset", "stage", "embryo", "section", "frac_classified", "L_laterality",
                                    "n_spots"])
    return t.set_index("key")


def test_exclusion_rules():
    t = sel.candidates(_scores_table(), [("X1", "A", "B")]).set_index("section").excluded
    assert t["A_E1S1"] == "" and t["B_E1S1"] == ""
    assert "classified" in t["A_E1S2"]
    assert t["CS23_E2S2"] == "no counts"
    assert t["A_3D"] == "combined 3D file"
    assert t["C_E1S1"] == "stage not in the ladder"


def test_panel_flag():
    ov = pd.DataFrame(dict(slice=["B_E1S1", "x", "y"], stage=["B"] * 3, n_genes_panel=[10, 100, 100]))
    t = sel.candidates(_scores_table(), [("X1", "A", "B")], ov).set_index("section")
    assert t.flag["B_E1S1"] == "small gene panel" and t.flag["A_E1S1"] == ""
    assert t.excluded["B_E1S1"] == ""


def _control_edges():
    rows = []
    for a, b in itertools.combinations(range(1, 6), 2):
        rows.append(dict(kind="control", a=f"A_E1S{a}", b=f"A_E1S{b}", gap=b - a, w_cost=(b - a) / 10, tv=(b - a) / 10,
                         dL=.1, gene_r=1 - (b - a) / 10))
    return pd.DataFrame(rows)


def test_control_table_sees_a_perfect_score():
    t = sel.control_table(_control_edges()).iloc[0]
    assert t.embryo == "A_E1" and t.n_pairs == 10
    assert t.rho_w_cost == pytest.approx(1.0) and t.hit_w_cost == 1.0
    assert t.rho_gene_r == pytest.approx(1.0)      # a similarity, negated


def test_control_hit_uses_nearest_available():
    e = pd.DataFrame(dict(kind="control", a=["A_E1S1", "A_E1S1", "A_E1S3"], b=["A_E1S3", "A_E1S7", "A_E1S7"], gap=[2., 6., 4.],
                          w_cost=[.1, .2, .1], tv=[.1, .2, .1], dL=[.1, .1, .1]))
    assert sel.control_table(e).iloc[0].hit_w_cost == 1.0


def test_decide():
    good = pd.DataFrame(dict(rho_gene_r=[.6, .5, .7]))
    bad = pd.DataFrame(dict(rho_gene_r=[.1, .2, 0.]))
    wins = pd.DataFrame(dict(w=[0, .5, 0, .5], seed=[0, 0, 1, 1], gene_r=[.10, .20, .11, .21]))
    ties = pd.DataFrame(dict(w=[0, .5, 0, .5], seed=[0, 0, 1, 1], gene_r=[.10, .11, .15, .16]))
    assert sel.decide(good, wins)["w"] == 0.5
    assert sel.decide(good, ties)["w"] == 0.0      # the gain is below the seed difference
    assert sel.decide(bad, wins)["w"] == 0.0       # gene r fails its control
    assert sel.decide(pd.DataFrame(), wins)["w"] == 0.0
    assert sel.decide(good, wins[wins.seed == 0])["w"] == 0.0     # no noise level


def test_weak_steps():
    t = pd.DataFrame(dict(kind=["temporal"] * 4, dL=[.1, .3, .1, .1], gene_r=[.1, .2, .3, .4]))
    w = sel.weak_steps(t)
    assert w.large_dL.tolist() == [False, True, False, False]
    assert w.low_gene_r.tolist() == [True, False, False, False]


def test_costs_resumes_without_repeating(tmp_path, monkeypatch):
    mod = _script()
    calls = []
    n = 4
    cand = pd.DataFrame(dict(section=[f"H_E1S{i}" for i in range(n)] + ["M_E1S0"], dataset=["HESTA"] * n + ["MOSTA"],
                             stage=["H"] * n + ["M"], embryo="1", number=[*range(n), 0], L=.5, frac_classified=.5,
                             n_bins=100, excluded="", flag=""))
    ser = SimpleNamespace(pairs={"X1": SimpleNamespace(a=SimpleNamespace(stage="H"), b=SimpleNamespace(stage="M"))},
                          tier=lambda tier: ["X1"])
    out = tmp_path / "n5"
    out.mkdir()
    cand.to_csv(out / "candidates.csv", index=False)
    cfg = SimpleNamespace(section_hash=lambda: "h")
    monkeypatch.setattr(mod, "context", lambda args: (cfg, ser, tmp_path, out))
    monkeypatch.setattr(mod, "solve_params", lambda c: {})
    monkeypatch.setattr(mod, "load_sections", lambda root, names, seed, h: {s: SimpleNamespace(comp=s) for s in names})
    monkeypatch.setattr(mod, "SAVE_EVERY", 2)

    stop = [True]

    def fake(ca, cb, la, lb):
        calls.append((ca, cb))
        if stop[0] and len(calls) == 5:
            raise RuntimeError("killed")
        return dict(w_cost=.1, tv=.1, dL=0., K=2, kappa_ab=0., kappa_ba=0., bins_a=1., bins_b=1., converged=True, n_iters=1, marg_err=0.), None

    monkeypatch.setattr(mod.sel, "edge_scores", fake)
    args = SimpleNamespace(seed=0, force=False)
    with pytest.raises(RuntimeError):
        mod.cmd_costs(args)
    first = list(calls)
    stop[0] = False
    assert len(pd.read_csv(out / "edges_seed0.csv")) == 4     # saved up to the last flush
    calls.clear()
    mod.cmd_costs(args)
    d = pd.read_csv(out / "edges_seed0.csv")
    assert len(d) == len(d.drop_duplicates(["kind", "a", "b"])) == 4 + 6      # 4 cross, 6 control
    assert not set(first[:4]) & set(calls)                    # saved edges aren't solved again
    calls.clear()
    mod.cmd_costs(args)
    assert not calls
