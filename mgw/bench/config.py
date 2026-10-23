"""Benchmark configuration: one YAML file, validated against DEFAULTS, frozen, hashed.

A config holds the parameter groups shared by every pipeline (reduce, features, fgw, stack, field, mgw, rfgw, metrics)
and the pipelines themselves (strategy + aligner + params). `--set a.b=c` overrides any value before validation; the
value is parsed as YAML (`--set reduce.max_obs=1000`, `--set fgw.epsilon=1e-3`).

Hashes name the caches and the result folders, so results of different settings never mix:
- section cache:  reduce
- feature cache:  reduce + features (features.zscore_by only when not "section", so the first caches stay valid)
- stack cache:    reduce + features + stack + fgw.alpha (the order inference uses the FGW objective); stack.coupling
                  only when it is not "moscot_icp" (the first caches keep their names), plus the mgw block for "mgw_icp"
- field cache:    the stack's key + field (minus `device` and `tf32`; residual / layernorm only when set): the
                  trained weights and the sampled volumes
- a pipeline:     reduce + features + fgw + metrics + its spec (+ the stack it uses, + field for the volume strategy on
                  a field, + the mgw / rfgw block for those aligners)
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

# the within-embryo coupling of neighbouring sections (stage 1); see stack.register_sections
COUPLINGS = ("moscot_icp", "moscot_procrustes", "mgw_icp")

STRATEGIES = {
    # strategy: (needs a stack, default params)
    # 2D <-> 2D: every query section against every real target section, the target chosen by the objective
    "real2d": (False, {}),
    # within embryo: build both stacks with `coupling` and score every kept neighbour coupling
    "stack": (True, {"coupling": "moscot_icp"}),
    # 3D <-> 3D: one stage-2 solve between the embryos as 3D measures, scored on the real query spots
    "volume": (True, {
        "representation": "field",       # field (sampled volume) | points (the registered stack points)
        "n_samples": 6000,               # points per embryo, equal for both; `points` subsamples to at most this
        "tau": 0.95,                     # unbalanced on both sides with the same tau (1 = balanced)
        "epsilon": None,                 # None: fgw.epsilon
        "unbalanced_correction": True,   # OTT's gw_unbalanced_correction; a NaN solve is retried without it
        "k_query": 8,                    # a query spot spreads over its k nearest samples (Gaussian, spacing)
        "k_target": 3,                   # a target sample pools the bins of its k nearest real target spots
    }),
}
# fgw: Euclidean intra costs; mgw: geodesics under the pull-back metric of a learned phi (2D, one per section) or of
# the field's features (3D); rfgw: shortest paths in a Euclidean kNN graph (3D)
ALIGNERS = {"real2d": ("fgw", "mgw"), "stack": ("fgw",), "volume": ("fgw", "mgw", "rfgw")}
KINDS = ("cross",)

DEFAULTS = {
    "name": "unnamed",
    "series": "series.yaml",
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
        "coupling": "moscot_icp",        # the within-embryo coupling the volume pipelines build their stacks with
    },
    "field": {                           # one field per embryo, pair features and seed; `device`, `tf32` not hashed
        "width": 128,
        "depth": 3,
        "n_freq": 64,                    # Fourier frequencies (sin and cos each), Gaussian with the bandwidths below
        "sigma_xy": 4.0,                 # in-plane bandwidth, cycles per stack radius
        "sigma_z": 1.0,                  # depth bandwidth: lower, the data are coarser along z
        "steps": 3000,
        "batch": 2048,
        "lr": 2.0e-3,
        "lambda_depth": 1.0,             # the depth loss along the displacement pairs
        "lambda_occ": 0.5,               # occupancy (binary cross-entropy)
        "lambda_smooth": 1.0e-3,         # ||J_f||_F^2
        "occ_out": 2.0,                  # negatives: farther than this many spot spacings from tissue / every segment
        "occ_jitter": 0.5,               # positives along the segments, jittered in-plane by this many spot spacings
        "residual": False,               # residual blocks after an input projection (deep nets)
        "layernorm": False,              # pre-LayerNorm in the residual blocks
        "device": "auto",                # auto | cpu | cuda
        "tf32": False,                   # TF32 matmuls on A100 / H100
    },
    "mgw": {                             # MGW's phi (2D) and its geodesics (2D and 3D); defaults of mgw.mgw_align
        "widths": [128, 256, 256, 128],
        "lr": 1.0e-3,
        "niter": 20000,
        "knn_k": 12,
        "geodesic_eps": 1.0e-2,
    },
    "rfgw": {"knn_k": 12},               # the kNN graph of Riemannian FGW
    "pipelines": {},
    "metrics": {
        "present_min_bins": 5,           # an organ is present on a side with >= 5 bins (full resolution)
        "presence": "embryo",            # K: the query's organs present in the whole target embryo (or "target")
    },
    "seeds": {"smoke": [0], "dev": [0, 1, 2], "screen": [0], "full": [0]},
}

PIPELINE_DEFAULTS = {
    "strategy": None,
    "aligner": "fgw",
    "kinds": ["cross"],
    "params": {},
}


def _merge(base, over, path=""):
    """Deep merge of `over` into a copy of `base`; unknown keys are errors (except below free maps)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        p = f"{path}.{k}" if path else k
        if k not in base and path not in ("pipelines", "seeds"):
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
    if spec["aligner"] not in ALIGNERS[spec["strategy"]]:
        raise ValueError(f"pipeline {name}: aligner must be one of {ALIGNERS[spec['strategy']]}")
    bad = set(spec["kinds"]) - set(KINDS)
    if bad:
        raise ValueError(f"pipeline {name}: unknown kinds {sorted(bad)}")
    needs_stack, pdef = STRATEGIES[spec["strategy"]]
    unknown = set(spec["params"]) - set(pdef)
    if unknown:
        raise KeyError(f"pipeline {name}: unknown params {sorted(unknown)} for strategy {spec['strategy']}")
    spec["params"] = {**pdef, **spec["params"]}
    if spec["strategy"] == "volume":
        if spec["params"]["representation"] not in ("field", "points"):
            raise ValueError(f"pipeline {name}: representation must be field or points")
        if spec["aligner"] != "fgw" and spec["params"]["representation"] != "field":
            raise ValueError(f"pipeline {name}: aligner {spec['aligner']} needs representation field")
    if spec["strategy"] == "stack" and spec["params"]["coupling"] not in COUPLINGS:
        raise ValueError(f"pipeline {name}: coupling must be one of {COUPLINGS}")
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

    def stack_key(self, coupling=None):
        """The stack's settings, with `coupling` in place of stack.coupling if given."""
        st = dict(self.raw["stack"], coupling=coupling or self.raw["stack"]["coupling"])
        extra = {"mgw": self.raw["mgw"]} if st["coupling"] == "mgw_icp" else {}
        if st["coupling"] == "moscot_icp":   # the coupling of the first caches, which had no such key
            del st["coupling"]
        return dict(**self.feature_key(), stack=st, fgw_alpha=self.raw["fgw"]["alpha"], **extra)

    def section_hash(self):
        return short_hash(self.section_key())

    def feature_hash(self):
        return short_hash(self.feature_key())

    def stack_hash(self, coupling=None):
        return short_hash(self.stack_key(coupling))

    def field_key(self):
        """The stack's key and the field settings, without the hardware ones (device, tf32); residual and layernorm
        only when set, so the first field caches keep their names."""
        f = {k: v for k, v in self.raw["field"].items() if k not in ("device", "tf32")}
        for k in ("residual", "layernorm"):
            if not f[k]:
                del f[k]
        return dict(**self.stack_key(), field=f)

    def field_hash(self):
        return short_hash(self.field_key())

    def pipeline_key(self, name):
        spec = self.raw["pipelines"][name]
        key = dict(**self.feature_key(), fgw=self.raw["fgw"], metrics=self.raw["metrics"],
                   spec={k: v for k, v in spec.items() if k != "kinds"})
        if spec["aligner"] in ("mgw", "rfgw"):
            key[spec["aligner"]] = self.raw[spec["aligner"]]
        if spec["strategy"] == "stack":
            key["stack"] = self.stack_key(spec["params"]["coupling"])
        elif spec["needs_stack"]:
            key["stack"] = self.stack_key()
        if spec["strategy"] == "volume" and spec["params"]["representation"] == "field":
            key["field"] = self.field_key()["field"]
        return key

    def pipeline_hash(self, name):
        return short_hash(self.pipeline_key(name))

    def pipeline_dir(self, name):
        return f"{name}-{self.pipeline_hash(name)}"

    # ---------- files
    def series_path(self):
        p = Path(self.raw["series"])
        return p if p.is_absolute() else (self.path.parent if self.path else CONFIG_DIR) / p


def from_dict(raw, path=None, overrides=()):
    raw = apply_overrides(raw, overrides)
    cfg = _merge(DEFAULTS, raw)
    for k, v, ok in (("features.zscore_by", cfg["features"]["zscore_by"], ("section", "embryo")),
                     ("metrics.presence", cfg["metrics"]["presence"], ("embryo", "target")),
                     ("field.device", cfg["field"]["device"], ("auto", "cpu", "cuda")),
                     ("stack.coupling", cfg["stack"]["coupling"], COUPLINGS)):
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
    """Load a YAML config (default: bench/configs/bench.yaml) with `key=value` overrides."""
    path = Path(path) if path else CONFIG_DIR / "bench.yaml"
    if not path.exists() and (CONFIG_DIR / path).exists():
        path = CONFIG_DIR / path
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return from_dict(raw, path, overrides)
