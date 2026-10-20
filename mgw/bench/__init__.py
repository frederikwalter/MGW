"""Benchmark harness for cross-species (HESTA / MOSTA) alignment: metrics, series, pipelines and reports.

Step 1 of docs/st_3d_alignment_strategies.md §6. The CLI is `python -m mgw.bench`; see `mgw/bench/__main__.py`.
Modules:
- config:     YAML configs -> frozen parameter groups, overrides, hashes
- series:     embryos, pairs, tiers (bench/configs/series.yaml)
- prepare:    sections reduced to points (seeded), per-tile organ counts, ortholog counts; features per pair
- solvers:    FGW and the null aligners, without notebook globals
- stack:      a 3D stack from serial sections (order, registration, gaps)
- placement:  plane search through a stack (used by demos/human_mouse_align_3d.ipynb only; not a benchmark pipeline)
- metrics:    the soft organ x organ matrix C, OMS, kappa_bal, Cohen's kappa, fold-overs (2D and 3D), sharpness
- pipelines:  A (2D <-> 2D on real sections, the reference); the cached stacks for stage 1 of the volume approach
- store:      result layout and manifests
- report:     the Go table, noise margin, guardrails, viewer and PNG sheets
"""
