import os
# JAX (OTT GW solver) otherwise grabs 75% of GPU memory on first use and never returns it,
# starving the PyTorch φ/ψ training that runs after the CCA feeler's GW solve.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from .models import PhiModel, PhiModelFFN, train_phi, train_phi_pro
from .geometry import pullback_metric_field, knn_graph, geodesic_distances, pairwise_squared_geodesic
from .gw import gw_distance, solve_gw
from .data import NumpyPairLoader
from .plotting import scatter_colored, plot_alignment_lines