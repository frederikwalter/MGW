"""On a real section (skipped where the h5ad file is missing): the prepared cache gives the same full-resolution C as
the bins of the raw section."""
import numpy as np
import pytest

from mgw import crossspecies as cs, orthologs
from mgw.bench import config, metrics, prepare

SEC = ("MOSTA", "E9.5_E2S1")


def _have(*secs):
    hesta, mosta = prepare.data_dirs()
    return all(cs.section_path(ds, s, hesta, mosta).exists() for ds, s in secs)


@pytest.mark.skipif(not _have(SEC), reason="section missing")
def test_cache_matches_raw_section(tmp_path):
    cfg = config.load("bench.yaml", ["reduce.max_obs=150"])
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
