"""Reproduce the CCA-feeler GW solve from hesta_align.ipynb outside Jupyter.

Usage (from the repo root):
    python -X faulthandler demos/debug_feeler_gw.py [n]; echo "exit code $?"
    JAX_PLATFORMS=cpu python -X faulthandler demos/debug_feeler_gw.py [n]; echo "exit code $?"

Exit code 137 = killed (SIGKILL), 134 = abort, 139 = segfault, 132 = illegal instruction.
"""
import os, sys, time, resource
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch  # imported before JAX, as in the notebook
import jax
from scipy.spatial.distance import cdist
from mgw.gw import solve_gw_ott

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

def log(msg):
    # ru_maxrss is in KB on Linux, bytes on macOS
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    print(f"[{time.strftime('%X')}] {msg}  (peak RSS {peak_gb:.1f} GB)", flush=True)

log(f"torch {torch.__version__} (CUDA {torch.version.cuda}), jax {jax.__version__}, devices {jax.devices()}")

# Same spatial-only costs as the feeler: pairwise distances, scaled by their 99% quantile
rng = np.random.default_rng(0)
C1, C2 = (cdist(x, x) for x in (rng.random((n, 2)), rng.random((n, 2))))
C1, C2 = C1 / np.quantile(C1, 0.99), C2 / np.quantile(C2, 0.99)

log(f"solving GW, n={n}")
P = solve_gw_ott(C1, C2, verbose=True, inner_maxit=2000, outer_maxit=2000,
                 inner_tol=1e-6, outer_tol=1e-6, epsilon=1e-3)
log(f"done, coupling mass {P.sum():.6f}")
