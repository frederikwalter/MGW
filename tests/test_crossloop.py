import numpy as np
import pytest
from scipy.spatial.distance import cdist

from mgw.bench import config, crossloop as cl, metrics, prepare


def _kernel(rng, n1, n2, conc=1.0):
    K = rng.gamma(conc, size=(n1, n2))
    return K / K.sum(1, keepdims=True)


def _brute(K1, K2, y):
    D = cdist(y, y)
    e = lambda A, B: np.einsum("ij,ik,jk->i", A, B, D)
    return 2 * e(K1, K2) - e(K1, K1) - e(K2, K2)


# ---------- the energy distance
def test_energy_matches_brute_force_and_vanishes_for_equal_paths():
    rng = np.random.default_rng(0)
    y, b = rng.normal(size=(8, 2)), rng.random(8)
    F = metrics.row_normalize(rng.random((8, len(metrics.ORGANS))))
    K1, K2 = _kernel(rng, 6, 8), _kernel(rng, 6, 8)
    res = cl.energy_commutativity(K1, K2, y, b, F)
    assert np.allclose(res["comm"] ** 2, np.maximum(_brute(K1, K2, y), 0), atol=1e-4)
    assert (res["comm"] > 0).all()
    same = cl.energy_commutativity(K1, K1, y, b, F)
    assert np.allclose(same["comm"], 0, atol=2e-3)
    assert np.allclose(same["tv_organ"], 0, atol=1e-6) and np.allclose(same["overlap"], 1, atol=1e-5)


def test_sharpness_is_the_distance_from_the_uninformative_kernel():
    rng = np.random.default_rng(1)
    y, b = rng.normal(size=(10, 2)), rng.random(10)
    F = metrics.row_normalize(rng.random((10, len(metrics.ORGANS))))
    K1 = _kernel(rng, 7, 10)
    U = np.tile(b / b.sum(), (7, 1))
    res = cl.energy_commutativity(K1, U, y, b, F)
    assert np.allclose(res["comm"], res["sharp1"], atol=1e-3)
    assert np.allclose(res["sharp2"], 0, atol=2e-3)


# ---------- rungs and loops
def test_loops_of_the_ladder():
    cfg = config.load("bench.yaml")
    rungs = cl.rungs(cfg)
    assert [r.id for r in rungs] == [f"X{i}" for i in range(1, 11)]
    loops = cl.loops(rungs)
    sq = {l.id for l in loops if l.kind == "square"}
    tri = {l.id for l in loops if l.kind == "triangle"}
    assert sq == {"X1-X2", "X2-X3", "X3-X5", "X5-X7", "X6-X8", "X8-X9"}
    assert tri == {"X3-X4", "X4-X5", "X5-X6", "X6-X7", "X7-X8", "X9-X10"}
    used = {p for l in loops for p in (l.th, l.tm) if p}
    assert used == {f"H{i}" for i in range(1, 7)} | {f"M{i}" for i in range(1, 7)}
    l = next(x for x in loops if x.id == "X3-X4")           # human stays at CS17
    assert l.th == "" and l.tm == "M3" and l.intermediates("h") == ("", l.r1.m)
    assert l.intermediates("m") == (l.r2.m, "")


def test_path_kernels_identity_and_directions():
    rng = np.random.default_rng(2)
    r1 = cl.Rung("X1", "h1", "m1", "S1", "E1", 0, 0)
    r2 = cl.Rung("X2", "h2", "m2", "S2", "E2", 1, 1)
    r3 = cl.Rung("X3", "h2", "m3", "S2", "E3", 1, 2)
    sq = cl.loops([r1, r2])[0]
    tri = next(l for l in cl.loops([r1, r2, r3]) if l.id == "X2-X3")
    n = dict(h1=5, m1=6, h2=7, m2=8, m3=9)
    Q = {"X1": rng.random((5, 6)), "X2": rng.random((7, 8)), "X3": rng.random((7, 9))}
    KQ = {k: (cl.kernel(v, np.ones(v.shape[1]))[0], cl.kernel(v.T, np.ones(v.shape[0]))[0]) for k, v in Q.items()}
    KT = {"H1": _kernel(rng, 5, 7), "M1": _kernel(rng, 6, 8), "M2": _kernel(rng, 8, 9)}
    f1, f2 = cl.path_kernels(sq, "h", KQ, KT)
    assert [f.shape for f in f1] == [(5, 7), (7, 8)] and [f.shape for f in f2] == [(5, 6), (6, 8)]
    f1, f2 = cl.path_kernels(sq, "m", KQ, KT)
    assert [f.shape for f in f1] == [(6, 8), (8, 7)] and [f.shape for f in f2] == [(6, 5), (5, 7)]
    assert np.allclose(f1[1], KQ["X2"][1]) and np.allclose(f2[0], KQ["X1"][1])
    f1, f2 = cl.path_kernels(tri, "h", KQ, KT)            # human stays: path 1 is the second rung alone
    assert len(f1) == 1 and np.allclose(f1[0], KQ["X3"][0]) and [f.shape for f in f2] == [(7, 8), (8, 9)]
    P = np.asarray(cl.product(f2))
    assert np.allclose(P, f2[0] @ f2[1], atol=1e-5) and np.allclose(P.sum(1), 1, atol=1e-5)
    assert n["h2"] == P.shape[0]


def test_kernel_rows_and_floor():
    P = np.array([[1.0, 3.0], [1e-9, 0.0], [2.0, 2.0]])
    K, dead = cl.kernel(P, np.array([1.0, 3.0]))
    assert dead.tolist() == [False, True, False]
    assert np.allclose(K.sum(1), 1) and np.allclose(K[1], [0.25, 0.75]) and np.allclose(K[0], [0.25, 0.75])


def test_topk_rows_round_trip():
    rng = np.random.default_rng(3)
    K = _kernel(rng, 50, 400, conc=0.05)
    idx, q, mx, kept = cl.topk_rows(K, 48)
    approx = np.zeros_like(K)
    np.put_along_axis(approx, idx.astype(int), q / 255 * mx[:, None], axis=1)
    assert (approx.argmax(1) == K.argmax(1)).all()
    assert np.allclose(np.take_along_axis(K, idx.astype(int), 1).sum(1), kept, atol=1e-5)
    assert np.abs(approx.sum(1) - kept).max() < 0.01           # uint8 quantization
    assert kept.min() > 0.85


# ---------- a synthetic loop: exact maps commute, a scrambled one does not
def _section(name, xy, rng):
    comp = np.zeros((len(xy), len(metrics.ORGANS)))
    comp[np.arange(len(xy)), rng.integers(0, 3, len(xy))] = 1 + rng.integers(0, 4, len(xy))
    return cl.SectionInfo(xy, comp, comp.sum(1) / comp.sum())


def test_exact_square_commutes():
    rng = np.random.default_rng(4)
    n = 60
    x = rng.normal(size=(n, 2))
    rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    secs = {s: _section(s, x @ rot(k), rng) for k, s in enumerate(("h1", "h2", "m1", "m2"))}
    r1, r2 = cl.Rung("X1", "h1", "m1", "", "", 0, 0), cl.Rung("X2", "h2", "m2", "", "", 1, 1)
    loop = cl.loops([r1, r2])[0]
    I = np.eye(n, dtype=np.float32)
    KQ = {"X1": (I, I), "X2": (I, I)}
    row, tiles = cl.evaluate_loop(loop, "h", KQ, {"H1": I, "M1": I}, secs, per_tile=True)
    assert row["comm"] < 2e-3 and row["sharp"] > 0.5
    perm = I[rng.permutation(n)]
    bad, _ = cl.evaluate_loop(loop, "h", KQ, {"H1": I, "M1": perm}, secs)
    assert bad["comm"] > 0.5
    v = cl.view_arrays(loop, "h", KQ, {"H1": I, "M1": I}, secs, k=8, tiles=tiles)
    assert v["p1a_idx"].shape == (n, 8) and "p2b_idx" in v and list(v["sections"]) == ["h1", "h2", "m1", "m2"]
    R, t = v["h2|R"], v["h2|t"]
    assert np.allclose(secs["h2"].xy @ R + t, secs["h1"].xy, atol=1e-6)     # frame back to the start


def test_viewer_html_renders():
    from mgw.bench import loopview
    rng = np.random.default_rng(5)
    n = 30
    secs = {s: _section(s, rng.normal(size=(n, 2)), rng) for s in ("h1", "h2", "m1", "m2")}
    r1, r2 = cl.Rung("X1", "h1", "m1", "", "", 0, 0), cl.Rung("X2", "h2", "m2", "", "", 1, 1)
    loop = cl.loops([r1, r2])[0]
    KQ = {"X1": (_kernel(rng, n, n), _kernel(rng, n, n)), "X2": (_kernel(rng, n, n), _kernel(rng, n, n))}
    KT = {"H1": _kernel(rng, n, n), "M1": _kernel(rng, n, n)}
    v = cl.view_arrays(loop, "m", KQ, KT, secs, k=8)
    page = loopview.loop_view(dict(v, run="fgw_a0.5", temporal="destot"))
    assert "lv-data" in page and "</script>" in page and "__ID__" not in page


# ---------- the solvers on a toy rung
def _toy_problem(n1=40, n2=45, seed=6):
    rng = np.random.default_rng(seed)
    cfg = config.load("bench.yaml")

    def sec(name, n):
        comp = np.zeros((n, len(metrics.ORGANS)))
        comp[np.arange(n), rng.integers(0, 4, n)] = 1 + rng.integers(0, 5, n)
        return prepare.Section(name, "x", rng.normal(size=(n, 2)), metrics.majority(comp), comp, 0.0, n)
    d = cl.RungData(cl.Rung("X1", "h", "m", "", "", 0, 0), sec("h", n1), sec("m", n2), rng.normal(size=(n1, 5)),
                    rng.normal(size=(n2, 5)), rng.random((n1, 20)).astype(np.float32),
                    rng.random((n2, 20)).astype(np.float32), 100)
    return cl.RungProblem(d, cfg)


@pytest.mark.parametrize("name", ["fgw_a0", "fgw_a0.5", "fgw_a1", "ufgw_a0_r0.3", "ufgw_a0.5_r0.3", "label_wot",
                                  "label_fgw_a0.5"])
def test_solve_dispatch(name):
    prob = _toy_problem()
    m, p = cl.parse_run(name)
    assert cl.run_name(m, p) == name
    Q, info = prob.solve(m, p)
    assert Q.shape == (40, 45) and np.isfinite(Q).all() and Q.sum() > 0
    if m in ("fgw", "label_wot", "label_fgw"):
        assert np.allclose(Q.sum(0), prob.b, atol=2e-3)
    row = cl.evaluate_rung(Q, prob, m, p)
    assert -1 <= row["kappa"] <= 1 and np.isfinite(row["gene_r"])


def test_grid_and_names():
    cfg = config.load("bench.yaml")
    g = cl.grid(cfg)
    names = [cl.run_name(m, p) for m, p in g]
    assert len(names) == len(set(names)) == 7 + 28 + 1 + 3 + 2
    assert all(cl.parse_run(n) == (m, p) for n, (m, p) in zip(names, g))
