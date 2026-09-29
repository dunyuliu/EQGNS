"""Seeding for training runs: one seed, independent sub-streams, checkpointed RNG state.

Ported from gns_earthquake_cycle/src/meshnet/seeding.py (2026-09-24) into meshnet/train.py
here as an opt-in flag: `--seed` (default None here, unlike the sibling's default 0) leaves
all training RNGs exactly as unseeded as before this port; passing an int activates the
sub-stream scheme described below.

Design (2026-09-24):
* ONE integer seed per run (`--seed`). When set, it is recorded in train_state-*.pt
  (see meshnet/train.py) so a resumed run can confirm/continue the same streams.
* The seed is split with numpy's SeedSequence into INDEPENDENT sub-streams, one per source of
  randomness: model init, training-sample order, validation-sample order, training noise.
  Changing how one source draws (e.g. a different noise scheme) therefore leaves the others
  bit-identical -- paired comparisons at the same seed differ only in the knob.
* Every generator's state is saved in train_state-*.pt and restored on resume, so a resumed
  run continues the same streams instead of restarting them.
* Replicates are different seeds. A seed is a nuisance variable: a result that holds for one
  seed and not another is not a result. Bulk statistics (noise std, sampling frequencies) do
  not depend on the seed; only the particular draws do (tests/test_seeding.py checks both).
* `--deterministic` additionally asks torch for deterministic kernels (warn-only). GPU
  message-passing reductions may still be nondeterministic, so same-seed GPU runs match at
  the start and drift slowly; CPU runs are exactly reproducible.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

STREAMS = ("init", "data_train", "data_valid", "noise")


def sub_seeds(seed: int) -> dict:
    """seed -> {stream: independent 63-bit int}. Deterministic in `seed` only."""
    children = np.random.SeedSequence(int(seed)).spawn(len(STREAMS))
    return {name: int(c.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
            for name, c in zip(STREAMS, children)}


def seed_global(seed: int) -> None:
    """Seed the process-global RNGs (python, numpy, torch CPU, all CUDA devices).
    Called with the INIT sub-seed right before the model is built, so weight init is fixed."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic(on: bool) -> None:
    if not on:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def torch_generator(seed: int, device="cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _np_state_to_py(st):
    name, keys, pos, has_gauss, cached = st
    return (str(name), [int(k) for k in keys], int(pos), int(has_gauss), float(cached))


def capture(generators: dict, np_rngs: dict) -> dict:
    """Snapshot every RNG for train_state. generators: {name: torch.Generator}; np_rngs: {name: np.random.Generator}."""
    state = {
        "python": random.getstate(),
        # plain python types only: torch>=2.6 loads train_state with weights_only=True, which
        # rejects numpy arrays (found 2026-09-24 on the first resume test)
        "numpy_global": _np_state_to_py(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "generators": {k: g.get_state() for k, g in generators.items() if g is not None},
        "np_rngs": {k: r.bit_generator.state for k, r in np_rngs.items() if r is not None},
    }
    return state


def restore(state: dict, generators: dict, np_rngs: dict) -> None:
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy_global"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for k, s in state.get("generators", {}).items():
        if generators.get(k) is not None:
            generators[k].set_state(s)
    for k, s in state.get("np_rngs", {}).items():
        if np_rngs.get(k) is not None:
            np_rngs[k].bit_generator.state = s
