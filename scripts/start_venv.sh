#!/bin/bash

module reset

# This script moved from repo root to scripts/ (PROJECT_RULES.md rule 9);
# module.sh is alongside it here, but venv/ stays at the repo root (built
# there by build_venv.sh / build_venv_frontera.sh), so resolve both
# relative to this script's own location, not the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# start env
# ---------
source "$SCRIPT_DIR/module.sh"

source "$REPO_ROOT/venv/bin/activate"

# test env
# --------
echo 'which python -> venv'
which python

# echo 'test_pytorch.py -> random tensor'
# python tests/test_pytorch.py

# echo 'test_pytorch_cuda_gpu.py -> True if GPU'
# python tests/test_pytorch_cuda_gpu.py

# echo 'test_torch_geometric.py -> no retun if import sucessful'
# python tests/test_torch_geometric.py
