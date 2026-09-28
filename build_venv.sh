#!/usr/bin/env bash
# Build an isolated virtualenv for local (non-TACC) servers.
#
# Validated stack (this session): CUDA 12.4, torch 2.6.0+cu124.
# NOTE: this script intentionally does NOT assume `module`/Lmod exists —
# that is TACC-specific (see build_venv_frontera.sh). Do not add module
# lines here.
set -euo pipefail

# Always operate relative to this script's own location, and always use
# this venv's own interpreter explicitly from here on -- never a bare
# `python`/`pip` that could silently resolve to something else on PATH
# (this is exactly the failure mode that corrupted venv_cotopaxi).
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

# PyG extras pinned to the versions validated against torch 2.6.0+cu124.
"$VENV_PY" -m pip install \
  torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

"$VENV_PY" -m pip install torch_geometric==2.6.1

"$VENV_PY" -m pip install -r "$SCRIPT_DIR/requirements.txt"

# test env
# --------
echo 'which python -> venv'
"$VENV_PY" -c "import sys; print(sys.executable)"

echo 'test_pytorch.py -> random tensor'
"$VENV_PY" test/test_pytorch.py

echo 'test_pytorch_cuda_gpu.py -> True if GPU'
"$VENV_PY" test/test_pytorch_cuda_gpu.py

echo 'test_torch_geometric.py -> no return if import successful'
"$VENV_PY" test/test_torch_geometric.py

# Clean up
# --------
#rm -r venv
