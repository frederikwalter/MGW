"""One-off: move the caches and results under a benchmark root to the pair ids of 2026-10-29 (bench/configs/series.yaml:
the cross pairs numbered by stage, X1-X10). Old X4 -> X5, X5 -> X9, X6 -> X8; X1-X3 keep their ids.

    python bench/scripts/rename_pairs.py [--root DIR]            # dry run: lists every change
    python bench/scripts/rename_pairs.py [--root DIR] --apply

The caches and results are keyed by the pair id alone (no hash covers the sections), so without this the new X4
(CS17_E1 <-> E12.5_E1) would load the old X4's (CS18_E1) features and count its results as done. Under the root:
- cache/{features,fields,volumes}/<pair>/ and results/<pipeline>-<hash>/<pair>/;
- the pair field of every manifest.json and the pair column of every scores.csv under results/;
- tasks/*.jsonl: the pair field of every task;
- diag/ and reports/: files and folders named <pair> or <pair>_..., and the pair column of every csv.
The ids are swapped in one pass (X5 -> X9 before X4 -> X5). Nothing is moved if a new id exists already. A run with
--apply writes RENAMED_PAIRS.json to the root, and later runs stop there: unlike rename_pipelines.py, the old and the
new ids overlap, so a second pass would move X5 (old X4) on to X9.
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

PAIRS = {"X5": "X9", "X6": "X8", "X4": "X5"}   # in move order: every target is free when its turn comes
MARKER = "RENAMED_PAIRS.json"
NAME = re.compile(r"^(X[456])(?=$|[_.\-])")     # a file or folder name that starts with an old id


def new_name(name):
    m = NAME.match(name)
    return PAIRS[m.group(1)] + name[m.end():] if m else name


def pair_dirs(root):
    """(old dir, new dir) of every pair folder of the caches and results, in move order."""
    parents = [root / "cache" / k for k in ("features", "fields", "volumes")]
    parents += sorted(p for p in (root / "results").glob("*") if p.is_dir())
    return [(p / old, p / new) for old, new in PAIRS.items() for p in parents if (p / old).is_dir()]


def named_paths(root):
    """(old path, new path) of the files and folders under diag/ and reports/ named after an old id, deepest first."""
    out = [(p, p.with_name(new_name(p.name))) for sub in ("diag", "reports") for p in (root / sub).rglob("X*")
           if NAME.match(p.name)]
    order = {old: i for i, old in enumerate(PAIRS)}
    return sorted(out, key=lambda t: (-len(t[0].parts), order[NAME.match(t[0].name).group(1)]))


def check_free(moves):
    """Every target must be free, or the source of an earlier move."""
    freed = set()
    for old, new in moves:
        if new.exists() and new not in freed:
            raise FileExistsError(f"{new} exists already; merge by hand")
        freed.add(old)


def rewrite_csv(f, apply):
    df = pd.read_csv(f, dtype={"pair": str}) if "pair" in pd.read_csv(f, nrows=0).columns else None
    if df is None or not df["pair"].isin(list(PAIRS)).any():
        return 0
    n = int(df["pair"].isin(list(PAIRS)).sum())
    if apply:
        df.assign(pair=df["pair"].map(lambda p: PAIRS.get(p, p))).to_csv(f, index=False)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="benchmark root (default $MGW_BENCH_DIR or <data>/bench)")
    ap.add_argument("--apply", action="store_true", help="make the changes (default: only list them)")
    args = ap.parse_args()
    root = prepare.bench_root(args.root)
    act = "renaming" if args.apply else "would rename"
    print(f"root {root} ({'apply' if args.apply else 'dry run'}); {', '.join(f'{o} -> {n}' for o, n in PAIRS.items())}")
    if (root / MARKER).exists():
        print(f"{MARKER} found: the pairs were renamed already ({(root / MARKER).read_text().strip()}); nothing to do")
        return

    dirs, named = pair_dirs(root), named_paths(root)
    check_free(dirs)
    check_free(named)

    # contents first, while the paths are still the old ones
    for f in sorted((root / "results").glob("*/*/seed*/manifest.json")):
        m = json.loads(f.read_text())
        if m.get("pair") in PAIRS:
            print(f"{act} pair {m['pair']} -> {PAIRS[m['pair']]} in {f.relative_to(root)}")
            if args.apply:
                m["pair"] = PAIRS[m["pair"]]
                f.write_text(json.dumps(m, indent=1, default=str))
    csvs = sorted((root / "results").glob("*/*/seed*/scores.csv"))
    csvs += sorted(f for sub in ("diag", "reports") for f in (root / sub).rglob("*.csv"))
    for f in csvs:
        n = rewrite_csv(f, args.apply)
        if n:
            print(f"{act} the pair of {n} rows in {f.relative_to(root)}")
    for f in sorted((root / "tasks").glob("*.jsonl")):
        lines = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        n = sum(t.get("pair") in PAIRS for t in lines)
        if n:
            print(f"{act} the pair of {n} tasks in {f.relative_to(root)}")
            if args.apply:
                for t in lines:
                    t["pair"] = PAIRS.get(t.get("pair"), t.get("pair"))
                f.write_text("".join(json.dumps(t) + "\n" for t in lines))

    for old, new in dirs + named:
        print(f"{act} {old.relative_to(root)} -> {new.relative_to(root)}")
        if args.apply:
            old.rename(new)

    if args.apply:
        (root / MARKER).write_text(json.dumps(PAIRS) + "\n")
        print(f"wrote {MARKER}")


if __name__ == "__main__":
    main()
