"""End-to-end checks on the local smoke sections (skipped where the h5ad files are missing):
- the prepared cache gives the same full-resolution C as the bins of the raw section;
- the compact C stored for a candidate equals the C recomputed from its dense coupling."""
import numpy as np
import pytest

from mgw import crossspecies as cs, orthologs
from mgw.bench import config, metrics, prepare, series, store
from mgw.bench.pipelines import run_task

SEC = ("MOSTA", "E9.5_E2S1")


def _have(*secs):
    hesta, mosta = prepare.data_dirs()
    return all(cs.section_path(ds, s, hesta, mosta).exists() for ds, s in secs)


@pytest.fixture(scope="module")
def cfg():
    return config.load("step1.yaml", ["reduce.max_obs=150", "pipelines.A_fgw.store.dense_kinds=[cross, between]"])


@pytest.mark.skipif(not _have(SEC), reason="local section missing")
def test_cache_matches_raw_section(cfg, tmp_path):
    ortho = orthologs.load_one2one()
    prepare.prepare_section(*SEC, cfg, 0, tmp_path, ortho, prepare.laterality_table())
    sec = prepare.load_section(tmp_path, SEC[1], 0, cfg.section_hash())
    r = cfg.raw["reduce"]
    hesta, mosta = prepare.data_dirs()
    red, full = cs.load_section(*SEC, cs.section_path(*SEC, hesta, mosta), max_obs=r["max_obs"],
                                drop_labels=r["drop_labels"], random_state=42)
    assert np.allclose(sec.xy, red.obsm["spatial"])
    # C from the cache equals C summed over the raw bins, each carrying its tile's row of a coupling
    rng = np.random.default_rng(0)
    P = rng.random((sec.n, sec.n)) ** 8
    T = metrics.row_normalize(sec.comp)
    C_cache = metrics.confusion(P, sec.comp, sec.comp)
    rowmass = metrics.row_normalize(P) @ T
    codes = metrics.organ_codes(full.obs["label_h"].to_numpy(str))
    C_raw = np.zeros_like(C_cache)
    np.add.at(C_raw, codes, rowmass[full.obs["tile"].to_numpy()])
    assert np.allclose(C_cache, C_raw)


@pytest.mark.skipif(not _have(("HESTA", "CS12-13_E1S3"), ("HESTA", "CS12-13_E2S6")), reason="local sections missing")
def test_compact_C_equals_dense(cfg, tmp_path):
    ser = series.load(cfg.series_path())
    pair = ser.pairs["S_H"]
    ortho = orthologs.load_one2one()
    for e in pair.embryos:
        for s in e.sections:
            prepare.prepare_section(e.dataset, s, cfg, 0, tmp_path, ortho, prepare.laterality_table())
    prepare.pair_features(pair, cfg, 0, tmp_path, ortho)
    out = run_task(cfg, "A_fgw", pair, 0, tmp_path, log=lambda *a: None)
    assert store.is_done(out, cfg, "A_fgw")
    data = prepare.load_pair(pair, cfg, 0, tmp_path)
    cand = np.load(out / "candidates.npz")
    files = sorted((out / "couplings").glob("*__selected.npz"))
    assert files
    df = __import__("pandas").read_csv(out / "scores.csv")
    for f in files:
        direction, query, _ = f.stem.split("__")
        row = df[(df.direction == direction) & (df["query"] == query) & (df.role == "selected")].iloc[0]
        qs, ts = (data["b"], data["a"]) if direction == "a_stack" else (data["a"], data["b"])
        q, t = qs[query], ts[row["target"]]
        C = metrics.confusion(store.load_dense(f), q.comp, t.comp)
        C_stored = cand[store.key(direction, query, row["target"], "C_full")]
        assert np.allclose(C, C_stored, rtol=2e-2, atol=1e-2 * C.max())
        # and the stored kappa_bal is the one of that C
        kb = metrics.kappa_scores(C, metrics.present(q.comp, 5), metrics.present(t.comp, 5))["kappa_bal"]
        assert kb == pytest.approx(row["kappa_bal"], abs=5e-3)
