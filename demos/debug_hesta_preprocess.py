"""Run the first code cells of hesta_align.ipynb (setup → mgw_preprocess) outside Jupyter.

Usage (from demos/, like the notebook):
    python -X faulthandler debug_hesta_preprocess.py; echo "exit code $?"
    python -X faulthandler debug_hesta_preprocess.py --skip-env; echo "exit code $?"

--skip-env leaves out the environment variables set in the notebook's first cell.
A background thread prints process and system memory every 5 s, so the last lines
before a kill show whether memory was running out.
Exit code 137 = killed (SIGKILL), 134 = abort, 139 = segfault, 132 = illegal instruction.
"""
import json, os, sys, threading, time

N_CELLS = 4  # env setup, slice choice, loading, preprocessing

def meminfo():
    status = dict(l.split(":", 1) for l in open("/proc/self/status"))
    system = dict(l.split(":", 1) for l in open("/proc/meminfo"))
    gb = lambda s: int(s.split()[0]) / 1e6  # values are in kB
    return (f"RSS {gb(status['VmRSS']):.1f} GB, peak {gb(status['VmHWM']):.1f} GB, "
            f"system available {gb(system['MemAvailable']):.1f} GB")

def monitor():
    while True:
        print(f"    [mem {time.strftime('%X')}] {meminfo()}", file=sys.stderr, flush=True)
        time.sleep(5)

threading.Thread(target=monitor, daemon=True).start()

nb = json.load(open("hesta_align.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"][:N_CELLS]
if "--skip-env" in sys.argv:
    cells[0] = 'import os, sys\nsys.path.append("..")'

ns = {}
for i, src in enumerate(cells):
    print(f"\n===== code cell {i} [{time.strftime('%X')}] =====", flush=True)
    exec(compile(src, f"<code cell {i}>", "exec"), ns)
    if "jax" in sys.modules:
        print(f"jax devices: {sys.modules['jax'].devices()}", flush=True)

print(f"\nfinished [{time.strftime('%X')}] {meminfo()}", flush=True)
