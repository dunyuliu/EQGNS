"""Run a meshnet train.py in torch deterministic mode.

    python3 test/paper_parity/det_rollout.py {current|published} --mode=rollout ...

`published` runs meshnet/train.py.published (the code behind the paper).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
script = REPO / "meshnet" / {"current": "train.py", "published": "train.py.published"}[sys.argv[1]]
sys.argv = [str(script)] + sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")
