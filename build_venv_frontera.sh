#!/usr/bin/env bash
# Build an isolated virtualenv on TACC Frontera.
#
# Validated stack (this session, on a local CUDA-12.4 server): torch
# 2.6.0+cu124. Frontera's module set is TACC-specific and was not
# re-validated on Frontera itself this session; the module names below
# are kept as documented by TACC staff -- verify `module avail cuda`
# resolves to a CUDA 12.4-compatible runtime before relying on this.
set -euo pipefail

module reset

# Load modules (TACC/Lmod-specific -- Frontera only)
module load python3/3.9
module load cuda/12

# Always operate relative to this script's own location, and always use
# this venv's own interpreter explicitly from here on -- never a bare
# `python`/`pip` that could silently resolve to something else on PATH.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# create env
# ---------
python3 -m virtualenv venv

VENV_PY="$SCRIPT_DIR/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv creation failed -- $VENV_PY not found or not executable." >&2
  echo "Refusing to install anything against an unverified interpreter." >&2
  exit 1
fi

"$VENV_PY" -m pip install --upgrade pip

# Pinned to the validated stack: torch 2.6.0+cu124.
"$VENV_PY" -m pip install torch==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

"$VENV_PY" -m pip install triton==3.2.0

"$VENV_PY" -m pip install torch_geometric==2.6.1

# PyG extras pinned to the versions validated against torch 2.6.0+cu124.
"$VENV_PY" -m pip install \
  torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

"$VENV_PY" -m pip install -r "$SCRIPT_DIR/requirements.txt"

# test env
# --------

# Check if the first command line argument is "--run-tests=true"
if [ "${1:-}" = "--run-tests=true" ]; then
  echo 'Running tests...'

  echo 'which python -> venv'
  "$VENV_PY" -c "import sys; print(sys.executable)"

  echo 'test_pytorch.py -> random tensor'
  "$VENV_PY" test/test_pytorch.py

  echo 'test_pytorch_cuda_gpu.py -> True if GPU'
  "$VENV_PY" test/test_pytorch_cuda_gpu.py

  echo 'test_torch_geometric.py -> no return if import successful'
  "$VENV_PY" test/test_torch_geometric.py

else
  echo "Skipping tests. To run tests, use the argument --run-tests=true"
fi

# Clean up
# --------
#rm -r venv
