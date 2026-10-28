"""One-off: move the results under a benchmark root to the run names of 2026-10-28 (sides_space_features_marginals,
docs/cross_species_alignment_results.md "Run names").

    python bench/scripts/rename_pipelines.py [--root DIR]            # dry run: lists every change
    python bench/scripts/rename_pipelines.py [--root DIR] --apply

Under the root:
- results/<old>-<hash>/ -> results/<new>-<hash>/, with the pipeline field of every manifest.json and the pipeline
  column of every scores.csv (the hashes do not change: names do not enter them);
- tasks/*.jsonl: the pipeline field of every task;
- diag/whole_embryo/*.csv (demos/whole_embryo_tests.ipynb): the run column; feature-variant labels (feat:*) stay.
Safe to run twice: names that are already new are left alone.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mgw.bench import prepare  # noqa: E402

PIPELINES = {"A_fgw": "sec2sec_s-euc_f-euc_bal", "A_mgw": "sec2sec_s-pb_f-euc_bal",
             "S_moscot_icp": "sec2nbr_s-euc_f-euc_icp", "S_moscot_procrustes": "sec2nbr_s-euc_f-euc_procrustes",
             "S_mgw_icp": "sec2nbr_s-pb_f-euc_icp", "V_fgw": "vol2vol_s-euc_f-euc_ub",
             "V_rfgw": "vol2vol_s-knn_f-euc_ub", "V_mgw": "vol2vol_s-pb_f-euc_ub",
             "PC_fgw": "pts2pts_s-euc_f-euc_ub"}


def run_name(old):
    """The new name of a run label of whole_embryo_tests.ipynb (unknown labels unchanged)."""
    if old in PIPELINES:
        return PIPELINES[old]
    fixed = {"PC_w": "pts2pts_s-none_f-euc_ub", "PC_w@20k": "pts2pts20k_s-none_f-euc_ub",
             "E_w": "sec2emb_s-none_f-euc_bal", "E_w_nn": "sec2emb_s-none_f-euc_nn",
             "cos_w_nn": "sec2emb_s-none_f-cos_nn", "E_label_nn": "sec2emb_s-none_f-lab_nn"}
    if old in fixed:
        return fixed[old]
    for pat, new in ((r"PC_w_semi@(.+)", "pts2pts_s-none_f-euc_semi{}"),
                     (r"E_w_semi@(.+)", "sec2emb_s-none_f-euc_semi{}"),
                     (r"cos_w_semi@(.+)", "sec2emb_s-none_f-cos_semi{}")):
        m = re.fullmatch(pat, old)
        if m:
            return new.format(m.group(1))
    return old


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="benchmark root (default $MGW_BENCH_DIR or <data>/bench)")
    ap.add_argument("--apply", action="store_true", help="make the changes (default: only list them)")
    args = ap.parse_args()
    root = prepare.bench_root(args.root)
    act = "renaming" if args.apply else "would rename"
    print(f"root {root} ({'apply' if args.apply else 'dry run'})")

    base = root / "results"
    for pdir in sorted(base.iterdir()) if base.exists() else []:
        name, _, h = pdir.name.rpartition("-")
        if name not in PIPELINES:
            continue
        new = base / f"{PIPELINES[name]}-{h}"
        if new.exists():
            raise FileExistsError(f"{new} exists already; merge by hand")
        n_scores = n_man = 0
        for man in pdir.glob("*/seed*/manifest.json"):
            n_man += 1
            if args.apply:
                m = json.loads(man.read_text())
                m["pipeline"] = PIPELINES.get(m.get("pipeline"), m.get("pipeline"))
                man.write_text(json.dumps(m, indent=1, default=str))
        for sc in pdir.glob("*/seed*/scores.csv"):
            n_scores += 1
            if args.apply:
                df = pd.read_csv(sc)
                df["pipeline"] = df["pipeline"].map(lambda p: PIPELINES.get(p, p))
                df.to_csv(sc, index=False)
        print(f"{act} {pdir.name} -> {new.name} ({n_man} manifests, {n_scores} scores.csv)")
        if args.apply:
            pdir.rename(new)

    for f in sorted((root / "tasks").glob("*.jsonl")):
        lines = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        n = sum(t.get("pipeline") in PIPELINES for t in lines)
        if n:
            print(f"{act} {n} tasks in {f.name}")
            if args.apply:
                for t in lines:
                    t["pipeline"] = PIPELINES.get(t["pipeline"], t["pipeline"])
                f.write_text("".join(json.dumps(t) + "\n" for t in lines))

    for f in sorted((root / "diag" / "whole_embryo").glob("*.csv")):
        df = pd.read_csv(f)
        if "run" not in df:
            continue
        new = df["run"].map(run_name)
        changed = sorted(set(df["run"][new != df["run"]]))
        if changed:
            print(f"{act} {len(changed)} run labels in {f.name}: {', '.join(changed)}")
            if args.apply:
                df.assign(run=new).to_csv(f, index=False)


if __name__ == "__main__":
    main()
