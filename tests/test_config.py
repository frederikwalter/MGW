import pytest

from mgw.bench import config, series

FGW_2D, MGW_2D = "sec2sec_s-euc_f-euc_bal", "sec2sec_s-pb_f-euc_bal"
STACK_ICP, STACK_PROCRUSTES, STACK_MGW = ("sec2nbr_s-euc_f-euc_icp", "sec2nbr_s-euc_f-euc_procrustes",
                                          "sec2nbr_s-pb_f-euc_icp")
FGW_VOL, MGW_VOL, FGW_PTS = "vol2vol_s-euc_f-euc_ub", "vol2vol_s-pb_f-euc_ub", "pts2pts_s-euc_f-euc_ub"


def test_hashes_are_stable_and_overrides_change_them():
    a, b = config.load("bench.yaml"), config.load("bench.yaml")
    assert a.pipeline_hash(FGW_2D) == b.pipeline_hash(FGW_2D)
    c = config.load("bench.yaml", ["reduce.max_obs=1000"])
    assert c.raw["reduce"]["max_obs"] == 1000
    assert c.pipeline_hash(FGW_2D) != a.pipeline_hash(FGW_2D) and c.section_hash() != a.section_hash()
    # an FGW setting changes the pipelines but not the section cache
    e = config.load("bench.yaml", ["fgw.epsilon=1e-3"])
    assert e.pipeline_hash(FGW_2D) != a.pipeline_hash(FGW_2D) and e.section_hash() == a.section_hash()
    # MGW settings: the MGW pipelines only
    m = config.load("bench.yaml", ["mgw.niter=100"])
    assert m.pipeline_hash(MGW_2D) != a.pipeline_hash(MGW_2D) and m.pipeline_hash(MGW_VOL) != a.pipeline_hash(MGW_VOL)
    assert m.pipeline_hash(STACK_MGW) != a.pipeline_hash(STACK_MGW)
    for p in (FGW_2D, FGW_VOL, STACK_ICP, STACK_PROCRUSTES):
        assert m.pipeline_hash(p) == a.pipeline_hash(p)


def test_the_first_caches_keep_their_names():
    # section, feature and stack caches of the first runs (stack.coupling moscot_icp is left out of the key)
    c = config.load("bench.yaml")
    assert (c.feature_hash(), c.section_hash(), c.stack_hash()) == ("574c162c", "78dc1a8a", "32574805")
    assert c.stack_hash("moscot_icp") == c.stack_hash()


def test_stack_coupling_names_the_stacks():
    a = config.load("bench.yaml")
    hashes = {k: a.stack_hash(k) for k in config.COUPLINGS}
    assert len(set(hashes.values())) == 3
    b = config.load("bench.yaml", ["stack.coupling=mgw_icp"])
    assert b.stack_hash() == hashes["mgw_icp"] and b.field_hash() != a.field_hash()
    assert b.pipeline_hash(FGW_VOL) != a.pipeline_hash(FGW_VOL)
    # a stack tool builds its own stack whatever stack.coupling says
    assert b.pipeline_hash(STACK_ICP) == a.pipeline_hash(STACK_ICP)
    assert b.pipeline_hash(FGW_2D) == a.pipeline_hash(FGW_2D)
    with pytest.raises(ValueError):
        config.load("bench.yaml", ["stack.coupling=paste"])


def test_unknown_keys_and_bad_pipelines_are_errors():
    with pytest.raises(KeyError):
        config.load("bench.yaml", ["reduce.max_ob=1000"])
    with pytest.raises(KeyError):
        config.from_dict({"pipelines": {"X": {"strategy": "real2d", "params": {"bogus": 1}}}})
    with pytest.raises(ValueError):
        config.from_dict({"pipelines": {"X": {"strategy": "real2d", "aligner": "rfgw"}}})
    with pytest.raises(ValueError):
        config.from_dict({"pipelines": {"X": {"strategy": "volume", "aligner": "mgw",
                                              "params": {"representation": "points"}}}})


def test_series_tiers_and_directions():
    s = series.load(config.load("bench.yaml").series_path())
    assert s.tier("dev") == ["X1", "X5"] and s.tier("screen") == ["X1", "X2", "X3", "X5", "X9"]
    assert s.tier("full") == [f"X{i}" for i in range(1, 11)]                # numbered by stage
    x5 = s.pairs["X5"]
    assert (x5.a.name, x5.b.name) == ("CS18_E1", "E12.5_E1")
    assert x5.direction_label("a_stack") == "human_stack"
    q, t = x5.sides("a_stack")
    assert q.dataset == "MOSTA" and t.dataset == "HESTA"     # human stack: mouse queries


def test_field_hash_keeps_the_cached_fields():
    a = config.load("bench.yaml")
    assert a.field_hash() == "5318e0cf"                      # the fields of the X5 grid winner
    assert config.load("bench.yaml", ["field.tf32=true", "field.device=cpu"]).field_hash() == a.field_hash()
    r = config.load("bench.yaml", ["field.residual=true"])
    assert r.field_hash() != a.field_hash() and r.pipeline_hash(FGW_VOL) != a.pipeline_hash(FGW_VOL)
    assert r.pipeline_hash(FGW_PTS) == a.pipeline_hash(FGW_PTS)


def test_residual_field_net_shapes():
    import torch
    from mgw.bench import field
    for kw in (dict(), dict(residual=True), dict(residual=True, layernorm=True)):
        p = field.FieldParams(width=32, depth=3, n_freq=8, **kw)
        net = field.FieldNet(5, p, seed=0)
        f, o = net(torch.zeros(7, 3))
        assert f.shape == (7, 5) and o.shape == (7,) and net.feat(torch.zeros(7, 3)).shape == (7, 5)
