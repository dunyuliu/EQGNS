"""Shared pytest configuration for the eq_rupture_gns test suite.

Adds the repo root to sys.path (so `import meshnet` / `import gns` work
regardless of how pytest is invoked) and registers the tier markers used to
group the meshnet test pyramid (see tests/README.md).
"""
import os
import sys

_TEST_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_TEST_DIR, '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# NOTE: do not import fixtures as `test.fixtures...` -- the CPython stdlib
# ships its own top-level `test` package (this is also why this directory is
# named `tests/`, not `test/`), and if anything on the import path has
# already triggered `import test` (pulling in /usr/lib/pythonX/test),
# `test.fixtures` would silently resolve against the stdlib package instead
# of this directory. Add tests/ itself to sys.path and import as `fixtures...`.
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

# The dataprep guard tests (test_dataprep_*.py) exec scripts/utils/prepare*.4gns.py,
# which creates a matplotlib figure per frame. Locally that resolves to
# TkAgg; GitHub Actions CI has no display server, so force the headless Agg
# backend for the whole test session before anything imports pyplot.
import matplotlib  # noqa: E402
matplotlib.use("Agg")

# The meshnet test fixtures are intentionally tiny (a dozen nodes). PyTorch's
# default of "use every core" adds pure thread-scheduling overhead on inputs
# this small and makes the suite slower and less reproducible run-to-run.
# Force single-threaded CPU execution for the whole test session.
import torch  # noqa: E402
torch.set_num_threads(1)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "unit: fast (<50ms) unit test of a single function/class")
    config.addinivalue_line(
        "markers", "integration: medium-speed test of module interactions")
    config.addinivalue_line(
        "markers", "e2e: full mini train->rollout pipeline vs a golden file")
    config.addinivalue_line(
        "markers", "physical: physical-behaviour / invariant test")
    config.addinivalue_line(
        "markers", "slow: not required for the fast local loop, but always run in CI")
    config.addinivalue_line(
        "markers", "paper_parity: reruns published checkpoints (opt-in, see tests/paper_parity/README.md)")
    config.addinivalue_line(
        "markers", "dataprep: guards scripts/utils/prepare*.4gns.py (EQdyna output -> npz) against regressions")
    config.addinivalue_line(
        "markers", "training_golden: tier 1 training-guard gate -- seeded deterministic training on "
                   "real D1 vs a committed loss-curve reference (needs data/gns-sample/, skips if absent)")
    config.addinivalue_line(
        "markers", "convergence_gate_nightly: tier 2 training-guard gate -- small-budget real training "
                   "+ rollout on real D1 vs a committed rollout-metrics reference (needs data/gns-sample/, "
                   "skips if absent)")
    config.addinivalue_line(
        "markers", "nightly: not required for the fast local loop or the standard CI invocation; run "
                   "explicitly via its own marker (e.g. `pytest tests/ -m convergence_gate_nightly`)")
    config.addinivalue_line(
        "markers", "training_golden_falsify: self-verifying mutation check for tier 1 -- lr_init "
                   "perturbed +10% must make the training_golden comparison FAIL (needs "
                   "data/gns-sample/, skips if absent); run explicitly, not part of "
                   "`-m training_golden`")
    config.addinivalue_line(
        "markers", "convergence_gate_nightly_falsify: self-verifying mutation check for tier 2 -- "
                   "lr_init perturbed +10% must make the convergence_gate_nightly comparison FAIL "
                   "(needs data/gns-sample/, skips if absent); run explicitly, not part of "
                   "`-m convergence_gate_nightly`")


def pytest_addoption(parser):
    parser.addoption("--paper-parity", action="store_true", default=False,
                     help="run the paper-parity gate (needs data/gns-sample/ and a GPU)")
