"""The field (§1c), its sampling (§1d) and the measure helpers of the volume pipeline (§1e, §1f), on a stack built from
the synthetic embryo's true positions (no registration, so these run in seconds)."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from mgw.bench import field, gates, stack as bstack, synthetic, volume


def truth_stack(name="SYN_H"):
    """A stack dict from the true positions (body units) with Gaussian neighbour couplings."""
    _, secs = synthetic.embryos()[name]
    names = list(secs)
    xyz = [d["u"] * synthetic.SEMI * (0.8 if name == "SYN_M" else 1) for d in secs.values()]
    F = synthetic._embed([[d["G"] for d in secs.values()]], 8, False)[0]
    couplings = []
    for k in range(len(names) - 1):
        K = np.exp(-cdist(xyz[k][:, :2], xyz[k + 1][:, :2], "sqeuclidean") / 0.01)
        couplings.append(K / K.sum())
    ST = bstack.assemble(names, [x[:, :2] for x in xyz], [x[0, 2] for x in xyz], [d["label"] for d in secs.values()],
                         F, couplings, feat_key="X_ortho")
    ST.update(Y=np.vstack([d["Y"] for d in secs.values()]), comp=np.vstack([d["comp"] for d in secs.values()]),
              L=np.concatenate([np.full(len(d["u"]), d["L"]) for d in secs.values()]),
              reg_xy=[x[:, :2] for x in xyz])
    return ST


P_FAST = field.FieldParams(steps=250, batch=256, width=64, depth=2, n_freq=32, sigma_xy=2.0, sigma_z=0.5,
                           device="cpu")


@pytest.fixture(scope="module")
def fitted():
    ST = truth_stack()
    return ST, field.fit(ST, ST["emb"]["X_ortho"], P_FAST, seed=0)


def test_field_learns_features_and_occupancy(fitted):
    ST, fld = fitted
    h = fld.meta["history"]
    assert h[-1]["feat"] < 0.5 * h[0]["feat"] and h[-1]["occ"] < h[0]["occ"]
    F = ST["emb"]["X_ortho"]
    err = ((fld.features(ST["xyz"]) - F) ** 2).sum(1).mean() / ((F - F.mean(0)) ** 2).sum(1).mean()
    assert err < 0.3
    assert (fld.occupancy(ST["xyz"]) > 0.5).mean() > 0.9
    # far outside the body, on a section plane: no tissue
    out = np.c_[np.full(50, 3.0), np.linspace(-1, 1, 50), np.full(50, ST["z"][2])]
    assert fld.occupancy(out).mean() < 0.2


def test_field_save_load_roundtrip(fitted, tmp_path):
    ST, fld = fitted
    fld.save(tmp_path / "f.pt")
    g = field.Field.load(tmp_path / "f.pt", "cpu")
    assert np.allclose(g.features(ST["xyz"][:20]), fld.features(ST["xyz"][:20]), atol=1e-5)


def test_sampling_fills_the_clipped_volume_evenly(fitted):
    ST, fld = fitted
    vol = field.sample(fld, 400, seed=1)
    z = np.asarray(ST["z"])
    assert vol["xyz"].shape == (400, 3) and vol["F"].shape == (400, 8)
    assert (vol["occ"] > 0.5).all() and vol["xyz"][:, 2].min() >= z[0] and vol["xyz"][:, 2].max() <= z[-1]
    # isotropic density: samples are not locked to the section planes (share near the planes ~ chance)
    tol = 0.1 * np.mean(np.diff(z))
    assert gates.lock_share(vol["xyz"][:, 2], z, tol) < 1.5 * gates.lock_chance(z, tol)
    assert vol["spacing"] > 0 and 0 < vol["accept"] <= 1


def test_substack_skips_a_section_with_the_skip_pairs():
    ST = truth_stack()
    skips = {(1, 3): (np.array([0, 1]), np.array([2, 3]), np.array([0.5, 0.5]))}
    sub = volume.substack(ST, [0, 1, 3, 4, 5], skips)
    off = sub["offsets"]
    assert sub["names"] == [ST["names"][k] for k in (0, 1, 3, 4, 5)] and len(sub["pairs"]) == 4
    i, j, _ = sub["pairs"][1]
    assert (i == off[1] + np.array([0, 1])).all() and (j == off[2] + np.array([2, 3])).all()
    # the full stack's own pairs, re-indexed: same points
    i0, j0, w0 = ST["pairs"][3]
    i3, j3, w3 = sub["pairs"][2]
    assert np.allclose(sub["xyz"][i3], ST["xyz"][i0]) and np.allclose(sub["xyz"][j3], ST["xyz"][j0])
    z = volume.substack(ST, [0, 2, 4], {(0, 2): skips[1, 3], (2, 4): skips[1, 3]}, z=[0, 1, 5])
    assert np.allclose(np.unique(z["xyz"][:, 2]), [0, 1, 5])


def test_slab_masses_and_point_cloud():
    m = volume.slab_masses([0.0, 1.0, 3.0])
    assert np.allclose(m, np.array([0.5, 1.5, 1.0]) / 3)
    ST = truth_stack()
    pc = volume.points(ST, 300, seed=0)
    assert len(pc["xyz"]) <= 300 and pc["mass"].sum() == pytest.approx(1)
    per = np.bincount(pc["section"], pc["mass"])
    assert np.allclose(per, volume.slab_masses(ST["z"]))
    # the identity carry when nothing is dropped
    full = volume.points(ST, 10 ** 6)
    A = volume.query_map(dict(full, kind="points", X=full["xyz"]), ST, 2, 8)
    off = ST["offsets"]
    assert (A.indices == np.arange(off[2], off[3])).all()


def test_spread_and_pool_rows_sum_to_one():
    rng = np.random.default_rng(0)
    xs, xq = rng.random((200, 3)), rng.random((50, 3))
    A = volume.spread(xq, xs, 8, 0.1)
    assert np.allclose(np.asarray(A.sum(1)).ravel(), 1) and (A.getnnz(1) == 8).all()
    B = volume.pool(xs, xq, 3)
    assert np.allclose(np.asarray(B.sum(1)).ravel(), 1)
    # far from every sample: the nearest one takes it all
    Af = volume.spread(np.array([[50.0, 50, 50]]), xs, 4, 0.01)
    assert Af.sum() == pytest.approx(1)


def test_g2_halves_and_the_way_back_to_the_full_frame():
    ST = truth_stack()
    ST["emb"]["X_feat"] = ST["emb"]["X_ortho"]
    off = ST["offsets"]
    skips = {(k, k + 2): (np.arange(5), np.arange(5), np.full(5, 0.2)) for k in range(len(ST["names"]) - 2)}
    for variant in ("common", "reest"):
        h = gates._halves(ST, skips, variant)
        assert h["odd"]["names"] == ST["names"][0::2] and h["even"]["names"] == ST["names"][1::2]
        for sub in h.values():
            zh, zf = sub["zmap"]
            assert np.all(np.diff(zh) >= 0) and zh[0] == zf[0]
            # every section plane of a half maps back onto its own plane in the full frame
            back = gates._to_full(sub["xyz"], sub)
            assert np.allclose(back, ST["xyz"][sub["rows"]]) or np.any(np.diff(zh) == 0)
    assert np.allclose(gates._halves(ST, skips, "common")["odd"]["zmap"][0], np.asarray(ST["z"])[0::2])


def test_parallel_sections_and_lock_chance():
    rng = np.random.default_rng(1)
    imgs = [(np.c_[rng.random((60, 2)), np.full(60, z) + rng.normal(0, 1e-3, 60)], np.ones(60)) for z in (0, 1, 2, 3)]
    ang, rho = volume.parallel_sections(imgs, [0, 1, 2, 3])
    assert np.all(ang < 1) and abs(rho) == pytest.approx(1)
    assert gates.lock_chance([0, 1, 2, 3], 0.1) == pytest.approx(0.2)
    assert gates.lock_share(np.array([0.05, 0.5, 1.95]), [0, 1, 2], 0.1) == pytest.approx(2 / 3)
