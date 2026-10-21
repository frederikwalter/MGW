import pytest

from mgw.bench import config, series


def test_hashes_are_stable_and_overrides_change_them():
    a, b = config.load("step1.yaml"), config.load("step1.yaml")
    assert a.pipeline_hash("A_fgw") == b.pipeline_hash("A_fgw")
    c = config.load("step1.yaml", ["reduce.max_obs=1000"])
    assert c.raw["reduce"]["max_obs"] == 1000
    assert c.pipeline_hash("A_fgw") != a.pipeline_hash("A_fgw") and c.section_hash() != a.section_hash()
    # report-time settings never enter a hash
    d = config.load("step1.yaml", ["guardrails.regret_max=0.2", "noise.factor=3"])
    assert d.pipeline_hash("A_fgw") == a.pipeline_hash("A_fgw")
    # an FGW setting changes the pipelines but not the section cache
    e = config.load("step1.yaml", ["fgw.epsilon=1e-3"])
    assert e.pipeline_hash("A_fgw") != a.pipeline_hash("A_fgw") and e.section_hash() == a.section_hash()


def test_zscore_option_keeps_the_first_caches():
    a = config.load("step1.yaml")
    assert a.raw["features"]["zscore_by"] == "section" and "zscore_by" not in a.feature_key()["features"]
    e = config.load("step1.yaml", ["features.zscore_by=embryo"])
    assert e.feature_hash() != a.feature_hash() and e.section_hash() == a.section_hash()
    with pytest.raises(ValueError):
        config.load("step1.yaml", ["features.zscore_by=gene"])
    # the organ presence is a metric setting: it changes the pipelines, not the caches
    t = config.load("step1.yaml", ["metrics.presence=target"])
    assert t.pipeline_hash("A_fgw") != a.pipeline_hash("A_fgw") and t.feature_hash() == a.feature_hash()


def test_unknown_keys_are_errors():
    with pytest.raises(KeyError):
        config.load("step1.yaml", ["reduce.max_ob=1000"])
    with pytest.raises(KeyError):
        config.from_dict({"pipelines": {"X": {"strategy": "real2d", "params": {"bogus": 1}}}})


def test_series_tiers_and_directions():
    cfg = config.load("step1.yaml")
    s = series.load(cfg.series_path())
    assert s.tier("dev")[:2] == ["X1", "X4"]
    assert {"X1", "X2", "X3", "X4", "X5", "X6", "H1", "H2", "M2"} <= set(s.tier("screen"))
    x4 = s.pairs["X4"]
    assert x4.direction_label("a_stack") == "human_stack"
    q, t = x4.sides("a_stack")
    assert q.dataset == "MOSTA" and t.dataset == "HESTA"     # human stack: mouse queries
    assert s.pairs["W:CS18_E1"].directions == ("self",)
    assert s.pairs["X4"].ceiling_ids == ["M2", "W:CS18_E1", "W:E12.5_E1"]


def test_step1_hashes_are_pinned():
    # the names of the existing caches and results: new config blocks must never change them
    c = config.load("step1.yaml")
    assert (c.feature_hash(), c.section_hash(), c.stack_hash(), c.pipeline_hash("A_fgw")) == (
        "574c162c", "78dc1a8a", "32574805", "6810d286")


def test_step3_and_gates_share_step1_caches_and_A():
    a, s3, g = config.load("step1.yaml"), config.load("step3.yaml"), config.load("gates.yaml")
    for c in (s3, g):
        assert (c.section_hash(), c.feature_hash(), c.stack_hash()) == (a.section_hash(), a.feature_hash(),
                                                                         a.stack_hash())
    for p in a.raw["pipelines"]:
        assert s3.pipeline_hash(p) == a.pipeline_hash(p)
    assert s3.field_hash() == g.field_hash()
    assert {"V_fgw", "V_linear", "V_gw", "PC_fgw"} <= set(s3.raw["pipelines"])
    assert s3.raw["pipelines"]["PC_fgw"]["params"]["representation"] == "points"


def test_field_settings_enter_only_the_field_pipelines_and_gates():
    s3 = config.load("step3.yaml")
    f = config.load("step3.yaml", ["field.sigma_z=0.5"])
    assert f.pipeline_hash("V_fgw") != s3.pipeline_hash("V_fgw") and f.field_hash() != s3.field_hash()
    assert f.gate_hash("g1") != s3.gate_hash("g1") and f.gate_hash("g2") != s3.gate_hash("g2")
    for p in ("A_fgw", "PC_fgw"):
        assert f.pipeline_hash(p) == s3.pipeline_hash(p)
    assert f.stack_hash() == s3.stack_hash()
    # the device is not a setting of the result
    d = config.load("step3.yaml", ["field.device=cpu"])
    assert d.field_hash() == s3.field_hash() and d.pipeline_hash("V_fgw") == s3.pipeline_hash("V_fgw")
    # stage-2 settings: the pipeline, not the field cache
    t = config.load("step3.yaml", ["pipelines.V_fgw.params.tau=0.9"])
    assert t.pipeline_hash("V_fgw") != s3.pipeline_hash("V_fgw") and t.field_hash() == s3.field_hash()
    # a G2 setting is not a G1 one
    g = config.load("gates.yaml", ["gates.g2.tau=0.9"])
    assert g.gate_hash("g1") == config.load("gates.yaml").gate_hash("g1")
    with pytest.raises(ValueError):
        config.from_dict({"pipelines": {"V": {"strategy": "volume", "params": {"representation": "mesh"}}}})


def test_gate_grid_settings():
    from mgw.bench import gates
    g = config.load("gates.yaml")
    s = gates.settings(g, "g1")
    assert len(s) == len(g.raw["gates"]["g1"]["zscore_by"]) * len(g.raw["gates"]["g1"]["grid"]["field.sigma_z"])
    assert all(ov[0].startswith("features.zscore_by=") for _, ov in s)
    assert gates.settings(g, "g1", grid=False) == [("config", [])] == gates.settings(g, "g2")
    assert set(g.raw["gates"]["embryos"]) == {"CS18_E1", "CS12-13_E2", "E12.5_E1", "E9.5_E2"}
