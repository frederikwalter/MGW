from dataclasses import dataclass

@dataclass
class TrainConfig:
    lr: float = 1e-3
    niter: int = 2000
    print_every: int = 100

@dataclass
class GeoConfig:
    k: int = 10              # kNN for graph on E
    epsilon: float = 1e-9    # jitter for SPD
    use_symmetric_edge: bool = True

@dataclass
class GWConfig:
    # No parameters needed here for vanilla POT's GW, left for extension
    pass

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR_FILE = _REPO_ROOT / ".mgw_data_dir"   # per-machine, gitignored
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"


def get_data_dir(reset=False):
    """Return the data directory for this machine.

    Resolution order: $MGW_DATA_DIR, the path saved in <repo>/.mgw_data_dir,
    otherwise ask once (Enter accepts <repo>/data) and save the answer.
    Pass reset=True to ask again.
    """
    if "MGW_DATA_DIR" in os.environ:
        return Path(os.environ["MGW_DATA_DIR"]).expanduser()
    if not reset and _DATA_DIR_FILE.exists():
        return Path(_DATA_DIR_FILE.read_text().strip()).expanduser()
    try:
        answer = input(f"Path to MGW data directory [{_DEFAULT_DATA_DIR}]: ").strip()
    except Exception:  # non-interactive run (e.g. nbconvert, scripts without stdin)
        return _DEFAULT_DATA_DIR
    data_dir = Path(answer).expanduser().resolve() if answer else _DEFAULT_DATA_DIR
    if not data_dir.is_dir():
        print(f"Warning: {data_dir} does not exist.")
    _DATA_DIR_FILE.write_text(str(data_dir))
    print(f"Saved to {_DATA_DIR_FILE} (delete it or call get_data_dir(reset=True) to change).")
    return data_dir
