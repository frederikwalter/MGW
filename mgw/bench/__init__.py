"""Benchmark harness for cross-species (HESTA / MOSTA) alignment: two questions, one metric (kappa_bal), one figure.
See bench/README.md. The CLI is `python -m mgw.bench`; see `mgw/bench/__main__.py`.

Modules:
- config:     YAML config (bench/configs/bench.yaml) -> frozen parameter groups, overrides, hashes
- series:     embryos, cross-species pairs, tiers (bench/configs/series.yaml)
- prepare:    sections reduced to points (seeded), per-tile organ counts, ortholog counts; features per pair
- solvers:    FGW and its intra-domain costs (Euclidean, kNN shortest paths, MGW pull-back geodesics)
- stack:      a 3D stack from serial sections (order, registration with a within-embryo coupling tool, gaps)
- field:      the field of an embryo (features and occupancy over the stack) and its sampled volume
- volume:     the 3D measures, stage 2 between two of them, back to the real spots
- metrics:    the soft organ x organ matrix C and kappa_bal
- pipelines:  real2d (2D <-> 2D), stack (within-embryo couplings), volume (3D <-> 3D)
- store:      result layout and manifests
- report:     the three kappa_bal tables and the label-transfer figures
- placement:  plane search through a stack (used by demos/human_mouse_align_3d.ipynb only; not a benchmark pipeline)
"""
