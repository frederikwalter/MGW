#!/bin/bash
# The XL fields of the field-size test (bench/scripts/field_scale.py) on LRZ: one array task per embryo of X5.
#
# Fill in the lines marked TODO (partition, time limit, environment), then:
#   sbatch bench/scripts/lrz_field_scale.sbatch
#
# Before submitting, copy the caches the fields need from the NAS to $MGW_BENCH_DIR on LRZ (X5, seed 0):
#   cache/sections/{CS18_E1S1..S6,E12.5_E1S1..S5}/seed0_78dc1a8a.npz
#   cache/features/X5/seed0_574c162c.npz
#   cache/features/stack_CS18_E1/seed0_574c162c.npz, cache/features/stack_E12.5_E1/seed0_574c162c.npz
#   cache/stacks/CS18_E1/seed0_32574805.npz, cache/stacks/E12.5_E1/seed0_32574805.npz
# e.g. from the NAS: rsync -avR cache/sections/CS18_E1S* cache/sections/E12.5_E1S* cache/features/X5 \
#   cache/features/stack_CS18_E1 cache/features/stack_E12.5_E1 cache/stacks/CS18_E1/seed0_32574805.npz \
#   cache/stacks/E12.5_E1/seed0_32574805.npz <lrz>:<MGW_BENCH_DIR>/   (run in /nas/lnt/ga79jin/MGW/data/bench)
# Afterwards bring the two fields back to the same paths on the NAS and score there:
#   cache/fields/X5/{CS18_E1,E12.5_E1}/seed0_<XL field hash>_full.pt   (the hash is printed in the log)
#   python bench/scripts/field_scale.py score --sizes current,L,XL
#
#SBATCH --job-name=mgw-field-xl
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=TODO                 # e.g. an H100 or A100 partition of the LRZ AI systems
#SBATCH --output=mgw-field-xl_%A_%a.log

set -euo pipefail
export MGW_BENCH_DIR=TODO                # the bench root on LRZ (holds cache/)
cd TODO                                  # the MGW checkout on LRZ (branch sim/bench-slim-20261022)
# TODO: activate the environment, e.g. `source .venv/bin/activate` or `export UV_PROJECT_ENVIRONMENT=...`
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "[start $(date) host=$(hostname) task=${SLURM_ARRAY_TASK_ID} commit=$(git rev-parse --short HEAD)]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python bench/scripts/field_scale.py train --size XL --array-index "${SLURM_ARRAY_TASK_ID}" --set field.tf32=true
echo "[finished $(date)]"
