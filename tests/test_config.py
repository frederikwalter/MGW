import pytest

from mgw.bench import config, series


def test_hashes_are_stable_and_overrides_change_them():
    a, b = config.load("bench.yaml"), config.load("bench.yaml")
    assert a.pipeline_hash("A_fgw") == b.pipeline_hash("A_fgw")
    c = config.load("bench.yaml", ["reduce.max_obs=1000"])
    assert c.raw["reduce"]["max_obs"] == 1000
    assert c.pipeline_hash("A_fgw") != a.pipeline_hash("A_fgw") and c.section_hash() != a.section_hash()
    # an FGW setting changes the pipelines but not the section cache
    e = config.load("bench.yaml", ["fgw.epsilon=1e-3"])
    assert e.pipeline_hash("A_fgw") != a.pipeline_hash("A_fgw") and e.section_hash() == a.section_hash()
    # MGW settings: the MGW pipelines only
    m = config.load("bench.yaml", ["mgw.niter=100"])
    assert m.pipeline_hash("A_mgw") != a.pipeline_hash("A_mgw") and m.pipeline_hash("V_mgw") != a.pipeline_hash("V_mgw")
    assert m.pipeline_hash("S_mgw_icp") != a.pipeline_hash("S_mgw_icp")
    for p in ("A_fgw", "V_fgw", "S_moscot_icp", "S_moscot_procrustes"):
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
    assert b.pipeline_hash("V_fgw") != a.pipeline_hash("V_fgw")
    # a stack tool builds its own stack whatever stack.coupling says
    assert b.pipeline_hash("S_moscot_icp") == a.pipeline_hash("S_moscot_icp")
    assert b.pipeline_hash("A_fgw") == a.pipeline_hash("A_fgw")
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
    assert s.tier("dev") == ["X1", "X4"] and s.tier("screen") == ["X1", "X2", "X3", "X4", "X5"]
    x4 = s.pairs["X4"]
    assert x4.direction_label("a_stack") == "human_stack"
    q, t = x4.sides("a_stack")
    assert q.dataset == "MOSTA" and t.dataset == "HESTA"     # human stack: mouse queries
