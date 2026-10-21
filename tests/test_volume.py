"""End-to-end smoke tests of step 2 and 3 on the synthetic embryo pair (mgw.bench.synthetic), through the real code
path: harness caches -> stacks (registered by moscot + OT-ICP; true gaps) -> fields -> volumes -> stage 2 -> Task.score
-> scores.csv; G1 and G2 through `gates.run` and the CLI; the report with the volume columns. Small sizes (a few
hundred points, short training), so the numbers are no evidence about the method, only about the plumbing."""
import numpy as np
import pandas as pd
import pytest

from mgw.bench import config, gates, series, store, synthetic
from mgw.bench.pipelines import run_task

FAST = ["features.pca_comp=8", "stack.icp_iter=4", "stack.global_sweeps=1", "field.steps=120", "field.batch=256",
        "field.width=64", "field.n_freq=32", "field.device=cpu", "gates.embryos={SYN_H: SX, SYN_M: SX}",
        "gates.g2.n_samples=200", "gates.g1.grid={}"]
FAST_V = [f"pipelines.{p}.params.n_samples=200" for p in ("V_fgw", "V_linear", "V_gw", "PC_fgw")]
SPACING = 0.15


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    for z, only in (("section", None), ("embryo", ["SYN_M"])):        # the G1 grid runs SYN_M with both
        c = config.load("step3.yaml", FAST + FAST_V + [f"features.zscore_by={z}"])
        sp = synthetic.write_bench(root, c, spacing=SPACING)
        synthetic.write_stacks(root, config.load("step3.yaml", FAST + FAST_V + [f"features.zscore_by={z}",
                                                                              f"series={sp}"]), spacing=SPACING,
                               only=only)
    over = FAST + [f"series={sp}"]
    cfg = config.load("step3.yaml", over + FAST_V)
    return root, over, cfg, series.load(cfg.series_path())


def test_volume_pipelines_end_to_end(bench):
    root, _, cfg, ser = bench
    pair = ser.pairs["SX"]
    out = {}
    for p in ("V_fgw", "PC_fgw", "V_linear"):
        d = run_task(cfg, p, pair, 0, root, log=lambda *a: None)
        assert store.is_done(d, cfg, p)
        out[p] = pd.read_csv(d / "scores.csv")
    n_q = {"b_stack": len(pair.a.sections), "a_stack": len(pair.b.sections)}
    for p, df in out.items():
        # one row per (direction, query section), role selected: the rows line up with A's
        assert (df.role == "selected").all() and df.groupby("direction").size().to_dict() == n_q
        assert df.kappa_bal.notna().all() and df.K.min() >= 2 and df.n_target.isin([200]).all() | (p == "PC_fgw")
        for c in ("rejected_query", "rejected_side_query", "rejected_side_target", "L_reached", "L_reached_rel",
                  "z_reached_rel", "normal_dev_deg", "order_rho", "foldover", "coexpr_r"):
            assert c in df and df[c].notna().all(), c
        assert ((df.L_reached_rel > -1e-9) & (df.L_reached_rel < 1 + 1e-9)).all()
    # the field and the point cloud carry real signal on the synthetic pair (chance is 0)
    assert out["V_fgw"].kappa_bal.mean() > 0.05 and out["PC_fgw"].kappa_bal.mean() > 0.05
    d = store.result_dir(root, cfg, "V_fgw", "SX", 0)
    cand, tz = np.load(d / "candidates.npz"), np.load(d / "targets.npz")
    assert not (d / "couplings").exists()                         # top-k only, no dense couplings
    assert any(k.endswith("top_row_idx") for k in cand.files)
    assert tz["b_stack/volume/xyz"].shape == (200, 3)
    # the cached field and volume are reused (a second run of another pipeline trains nothing)
    assert list((root / "cache" / "fields" / "SX").rglob("*_full.pt")) and list((root / "cache" / "volumes").rglob(
        "*.npz"))


def test_report_has_volume_columns(bench):
    from mgw.bench.report import table
    root, _, cfg, ser = bench
    run_task(cfg, "V_fgw", ser.pairs["SX"], 0, root, log=lambda *a: None)
    df = table.load(cfg, root)
    go, _, rank, _ = table.go_table(cfg, ser, root, "smoke")
    assert {"V", "PC", "dV", "dPC", "V_feat_OT", "V_GW"} <= set(go.columns)
    assert np.isfinite(go.V.iloc[0])
    vol = table.volume_table(cfg, ser, df, "smoke")
    assert set(vol.pipeline) >= {"V_fgw"} and {"rejected_query", "L_reached_rel", "order_rho_abs"} <= set(vol.columns)
    # step 1's table has no volume columns
    go1, *_ = table.go_table(config.load("step1.yaml", [o for o in FAST if not o.startswith("gates")]), ser, root,
                             "smoke")
    assert "V" not in go1.columns


def test_g1_grid_on_synthetic(bench):
    root, over, _, ser = bench
    cfg = config.load("gates.yaml", over + ["gates.g1.zscore_by=[section, embryo]"])
    rows, emb, sel = gates.run(cfg, ser, root, "g1", embryos=["SYN_M"], seeds=[0], log=lambda *a: None)
    assert set(rows.predictor) == set(gates.PREDICTORS) and rows.r_heldout.notna().all()
    assert set(rows.section) == {"SYN_MS2", "SYN_MS3"}                   # the interior sections
    assert rows[rows.predictor == "field"].r_direct.isna().all()         # the field transports no genes
    assert rows[rows.predictor == "mccann"].r_direct.notna().all()
    assert len(sel) == 2 and sel.score.is_monotonic_decreasing and {"passed", "margin"} <= set(emb.columns)
    assert (root / "results" / "gates" / "gates_g1" / "best.json").exists()
    # done units are skipped on a re-run
    again, _, _ = gates.run(cfg, ser, root, "g1", embryos=["SYN_M"], seeds=[0], log=lambda *a: None)
    pd.testing.assert_frame_equal(rows.reset_index(drop=True), again.reset_index(drop=True))


def test_g2_cli_on_synthetic(bench, capsys):
    from mgw.bench.__main__ import main
    root, over, _, ser = bench
    args = ["--root", str(root)] + [a for o in over for a in ("--set", o)]
    main(args + ["--set", "gates.g2.variants=[common]", "gates", "g2", "--embryos", "SYN_M", "--seeds", "0"])
    printed = capsys.readouterr().out
    assert "G2 per embryo" in printed
    summ = pd.read_csv(root / "results" / "gates" / "gates_g2" / "summary.csv")
    assert set(summ.variant) == {"common"}
    assert {"med3d_field", "med3d_points", "lock_field", "lock_points", "lock_chance_field", "passed"} <= set(summ)
    rows = pd.read_csv(root / "results" / "gates" / "gates_g2" / "rows.csv")
    ok = rows.dropna(subset=["med3d"])
    assert set(ok.direction) == {"odd->even", "even->odd"} and (ok.lock_chance.between(0.15, 0.25)).all()
    # with the common z, the raw point cloud locks onto the other half's planes
    pc = ok[(ok.variant == "common") & (ok.representation == "points")]
    assert (pc.lock > 2 * pc.lock_chance).all()
