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
