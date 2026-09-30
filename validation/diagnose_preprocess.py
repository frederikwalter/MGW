"""Reproduce the loading/preprocess part of demos/mouse_align.ipynb step by step.

Used to find out why the notebook kernel dies (no Python traceback) on a machine.
Run from the repo root, once with and once without the notebook's cell-0 CPU pinning:

    python validation/diagnose_preprocess.py --pin-env; echo "exit=$?"
    python validation/diagnose_preprocess.py;           echo "exit=$?"

Exit codes: 0 ok, 1 Python error, 137 SIGKILL (usually OOM), 139 SIGSEGV,
132 SIGILL, 134 SIGABRT.  faulthandler prints the Python stack for the
last three.
"""
import os
import sys

# Cell 0 of mouse_align.ipynb; must be set before numpy/torch are imported.
PIN_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "MKL_CBWR": "COMPATIBLE",
    "ATEN_CPU_CAPABILITY": "avx2",
    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
    "NPY_DISABLE_CPU_FEATURES": "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR",
    "OPENBLAS_CORETYPE": "Haswell",
}
if "--pin-env" in sys.argv:
    os.environ.update(PIN_ENV)

import faulthandler
import platform
import resource
import time

faulthandler.enable()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_t0 = time.time()


def step(msg):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 2**30 if sys.platform == "darwin" else peak / 2**20   # bytes on macOS, KiB on Linux
    print(f"[diag {time.time() - _t0:7.1f}s  peak {peak_gb:6.2f} GB] {msg}", flush=True)


step(f"python {platform.python_version()} on {platform.machine()} / {platform.processor() or platform.platform()}")
step("pin-env: " + ("ON (notebook cell 0)" if "--pin-env" in sys.argv else "off"))
for k in PIN_ENV:
    if k in os.environ:
        print(f"    {k}={os.environ[k]}", flush=True)

step("importing numpy / scipy / threadpoolctl")
import numpy as np
import scipy
step(f"numpy {np.__version__}, scipy {scipy.__version__}")
try:
    from threadpoolctl import threadpool_info
    for info in threadpool_info():
        print(f"    {info.get('internal_api')} {info.get('version')} threads={info.get('num_threads')} "
              f"arch={info.get('architecture')} lib={info.get('filepath')}", flush=True)
except ImportError:
    print("    threadpoolctl not installed", flush=True)

step("importing torch")
import torch
step(f"torch {torch.__version__}, cuda {torch.version.cuda}, cuda available={torch.cuda.is_available()}, "
     f"cpu capability={torch.backends.cpu.get_cpu_capability()}")

step("importing jax / ott")
import jax
import ott
step(f"jax {jax.__version__}, devices={jax.devices()}, ott {ott.__version__}")

step("importing numba / h5py / anndata / scanpy")
import numba
import h5py
import anndata as ad
import scanpy as sc
step(f"numba {numba.__version__}, h5py {h5py.__version__} (HDF5 {h5py.version.hdf5_version}), "
     f"anndata {ad.__version__}, scanpy {sc.__version__}")

from mgw import mgw
from mgw.config import get_data_dir

# ---- notebook cell 2: load ---------------------------------------------------
DATA_DIR = get_data_dir()
files = [DATA_DIR / "mouse_embryo/E9.5_E1S1.MOSTA.h5ad", DATA_DIR / "mouse_embryo/E10.5_E1S1.MOSTA.h5ad"]
timepoints = ["E9.5", "E10.5"]
PCA_comp = 30

adatas = []
for i, fh in enumerate(files):
    step(f"reading {fh}")
    adata = sc.read_h5ad(fh)
    adata.X = adata.layers["count"]
    adata.obs["timepoint"] = [timepoints[i]] * adata.shape[0]
    adatas.append(adata)
    step(f"  -> {adata.shape}")

step("intersecting genes")
common_genes = sorted(set.intersection(*[set(a.var_names) for a in adatas]))
adatas = [a[:, common_genes] for a in adatas]
step(f"  -> {len(common_genes)} common genes")

# ---- notebook cell 3: joint normalisation + PCA + mgw_preprocess ------------
step("ad.concat")
joint = ad.concat(adatas, join="inner")
step("sc.pp.normalize_total")
sc.pp.normalize_total(joint)
step("sc.pp.log1p")
sc.pp.log1p(joint)
step("sc.pp.pca")
sc.pp.pca(joint, n_comps=PCA_comp)

A = joint[joint.obs["timepoint"] == timepoints[0]].copy()
B = joint[joint.obs["timepoint"] == timepoints[1]].copy()
step(f"split: A={A.shape}, B={B.shape}")

step("mgw.mgw_preprocess (CCA feeler: GW solve on the CPU with JAX)")
pre = mgw.mgw_preprocess(
    A, B,
    PCA_comp=30, CCA_comp=3, use_cca_feeler=True,
    use_pca_X=True, use_pca_Z=True, log1p_X=True, log1p_Z=True,
    verbose=True, feature_only=False, spatial_only=True, rep_norm="zscore",
)
step(f"done: X_rep={pre['X_rep'].shape}, Z_rep={pre['Z_rep'].shape}, cca_corr={np.round(pre['cca_corr'], 3)}")
