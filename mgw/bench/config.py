"""Benchmark configuration: one YAML file, validated against DEFAULTS, frozen, hashed.

A config holds the parameter groups shared by every pipeline (reduce, features, fgw, stack, metrics) and the
pipelines themselves (strategy + aligner + params). `--set a.b=c` overrides any value before validation; the value
is parsed as YAML (`--set reduce.max_obs=1000`, `--set fgw.epsilon=1e-3`).

Hashes name the caches and the result folders, so results of different settings never mix:
- section cache:  reduce
- feature cache:  reduce + features (features.zscore_by only when not "section", so the first caches stay valid)
- stack cache:    reduce + features + stack + fgw.alpha (the order inference uses the FGW objective)
- a pipeline:     reduce + features + fgw + metrics + its aligner's params + its spec (+ stack if it uses one)
Guardrail and noise settings are report-time only and never enter a hash.
"""
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "bench" / "configs"

STRATEGIES = {
    # strategy: (needs a stack, default params)
    "real2d": (False, {}),
}
ALIGNERS = ("fgw", "linear", "spatial_gw", "scot")
KINDS = ("cross", "between", "within")
ROLES_DENSE = ("selected", "oracle", "L_partner")

DEFAULTS = {
    "name": "unnamed",
    "series": "series.yaml",
    "reference": "reference.yaml",
    "reduce": {
        "max_obs": 5000,
        "subsample": "voronoi",
        "agg": "sum",
        "lloyd_iter": 3,
        "drop_labels": ["Cavity", "Extraembryonic Tissue"],
    },
    "features": {
        "pca_comp": 30,
        "n_hvg": 2000,
        "n_eval_genes": 500,
        "m_scale_quantile": 0.99,
        "m_scale_max_entries": 20_000_000,
        "zscore_by": "section",
    },
    "fgw": {
        "alpha": 0.5,
        "epsilon": 5e-3,
        "inner_maxit": 3000,
        "outer_maxit": 200,
        "inner_tol": 1e-7,
        "outer_tol": 1e-7,
    },
    "stack": {
        "order": "auto",
        "order_subsample": 400,
        "pair_alpha": 0.8,
        "pair_eps": 1e-2,
        "icp_lambda": 0.5,
        "icp_iter": 15,
        "icp_eps": 5e-3,
        "section_flip": "auto",
        "global_sweeps": 2,
        "z_gaps": None,
    },
    "aligners": {
        "linear": {"epsilon": 5e-3},
        "spatial_gw": {"epsilon": 5e-3},
        "scot": {"epsilon": 5e-3},
        "rigid": {"icp_lambda": 0.5, "icp_iter": 15, "icp_eps": 5e-3},
    },
    "pipelines": {},
    "metrics": {
        "present_min_bins": 5,
        "present_min_tiles": 1,
        "coverage_share": 0.1,
        "foldover_k": 16,
        "topk": 64,
        "presence": "embryo",
    },
    "guardrails": {
        "regret_max": 0.05,
        "foldover_max": 0.05,
        "end_fraction": 0.34,
        "end_pass_share": 0.75,
    },
    "noise": {"pipeline": "A_fgw", "tier": "dev", "factor": 2.0},
    "seeds": {"smoke": [0], "dev": [0, 1, 2, 3, 4], "screen": [0], "full": [0]},
}

PIPELINE_DEFAULTS = {
    "strategy": None,
    "aligner": "fgw",
    "kinds": ["cross"],
    "params": {},
    # what is written besides the scores: top-k rows/columns per candidate, dense P for these roles
    "store": {"topk": True, "dense": ["selected", "oracle"], "dense_kinds": ["cross"]},
}


def _merge(base, over, path=""):
    """Deep merge of `over` into a copy of `base`; unknown keys are errors (except below free maps)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        p = f"{path}.{k}" if path else k
        if k not in base and path not in ("pipelines", "aligners", "seeds"):
            raise KeyError(f"unknown config key {p!r}")
        if isinstance(v, dict) and isinstance(base.get(k), dict) and path != "pipelines":
            out[k] = _merge(base[k], v, p) if base[k] else copy.deepcopy(v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(raw, overrides):
    """`a.b.c=value` strings applied to a nested dict (values parsed as YAML)."""
    raw = copy.deepcopy(raw)
    for o in overrides or ():
        key, _, val = o.partition("=")
        if not _:
            raise ValueError(f"override {o!r} is not key=value")
        parts = key.strip().split(".")
        d = raw
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return raw


def _validate_pipeline(name, spec):
    spec = _merge(PIPELINE_DEFAULTS, spec, f"pipelines.{name}")
    if spec["strategy"] not in STRATEGIES:
        raise ValueError(f"pipeline {name}: strategy must be one of {sorted(STRATEGIES)}")
    if spec["aligner"] not in ALIGNERS:
        raise ValueError(f"pipeline {name}: aligner must be one of {ALIGNERS}")
    bad = set(spec["kinds"]) - set(KINDS)
    if bad:
        raise ValueError(f"pipeline {name}: unknown kinds {sorted(bad)}")
    needs_stack, pdef = STRATEGIES[spec["strategy"]]
    unknown = set(spec["params"]) - set(pdef)
    if unknown:
        raise KeyError(f"pipeline {name}: unknown params {sorted(unknown)} for strategy {spec['strategy']}")
    spec["params"] = {**pdef, **spec["params"]}
    spec["needs_stack"] = needs_stack
    return spec


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(obj, n=8):
    return hashlib.sha1(canonical(obj).encode()).hexdigest()[:n]


def _freeze(d):
    if isinstance(d, dict):
        return MappingProxyType({k: _freeze(v) for k, v in d.items()})
    if isinstance(d, list):
        return tuple(_freeze(v) for v in d)
    return d


@dataclass(frozen=True)
class Config:
    raw: dict
    path: Path | None = None
    overrides: tuple = ()
    _frozen: MappingProxyType = field(default=None, repr=False, compare=False)

    def __getattr__(self, k):
        if k.startswith("_") or k in ("raw", "path", "overrides"):
            raise AttributeError(k)
        try:
            return self._frozen[k]
        except KeyError:
            raise AttributeError(k) from None

    # ---------- hashes
    def section_key(self):
        return dict(reduce=self.raw["reduce"])

    def feature_key(self):
        f = dict(self.raw["features"])
        if f["zscore_by"] == "section":     # the default of the first caches, which had no such key
            del f["zscore_by"]
        return dict(reduce=self.raw["reduce"], features=f)

    def stack_key(self):
        return dict(**self.feature_key(), stack=self.raw["stack"], fgw_alpha=self.raw["fgw"]["alpha"])

    def section_hash(self):
        return short_hash(self.section_key())

    def feature_hash(self):
        return short_hash(self.feature_key())

    def stack_hash(self):
        return short_hash(self.stack_key())

    def pipeline_key(self, name):
        spec = self.raw["pipelines"][name]
        key = dict(**self.feature_key(), fgw=self.raw["fgw"], metrics=self.raw["metrics"],
                   spec={k: v for k, v in spec.items() if k != "kinds"})
        if spec["aligner"] != "fgw":
            key["aligner"] = self.raw["aligners"].get(spec["aligner"], {})
        if name == "A_fgw" or spec["strategy"] == "real2d":
            key["rigid"] = self.raw["aligners"]["rigid"]
        if spec["needs_stack"]:
            key["stack"] = self.raw["stack"]
        return key

    def pipeline_hash(self, name):
        return short_hash(self.pipeline_key(name))

    def pipeline_dir(self, name):
        return f"{name}-{self.pipeline_hash(name)}"

    # ---------- files
    def series_path(self):
        p = Path(self.raw["series"])
        return p if p.is_absolute() else (self.path.parent if self.path else CONFIG_DIR) / p

    def reference_path(self):
        p = Path(self.raw["reference"])
        return p if p.is_absolute() else (self.path.parent if self.path else CONFIG_DIR) / p


def from_dict(raw, path=None, overrides=()):
    raw = apply_overrides(raw, overrides)
    cfg = _merge(DEFAULTS, raw)
    for k, v, ok in (("features.zscore_by", cfg["features"]["zscore_by"], ("section", "embryo")),
                     ("metrics.presence", cfg["metrics"]["presence"], ("embryo", "target"))):
        if v not in ok:
            raise ValueError(f"{k} must be one of {ok}")
    cfg["pipelines"] = {n: _validate_pipeline(n, s) for n, s in cfg["pipelines"].items()}
    for n in cfg["pipelines"]:
        if "/" in n or "-" in n:
            raise ValueError(f"pipeline name {n!r}: no '/' or '-' (the folder is <name>-<hash>)")
    c = Config(raw=cfg, path=Path(path).resolve() if path else None, overrides=tuple(overrides or ()))
    object.__setattr__(c, "_frozen", _freeze(cfg))
    return c


def load(path=None, overrides=()):
    """Load a YAML config (default: bench/configs/step1.yaml) with `key=value` overrides."""
    path = Path(path) if path else CONFIG_DIR / "step1.yaml"
    if not path.exists() and (CONFIG_DIR / path).exists():
        path = CONFIG_DIR / path
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return from_dict(raw, path, overrides)
