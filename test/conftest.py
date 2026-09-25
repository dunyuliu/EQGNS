"""Shared pytest configuration for the eq_rupture_gns test suite.

Adds the repo root to sys.path (so `import meshnet` / `import gns` work
regardless of how pytest is invoked) and registers the tier markers used to
group the meshnet test pyramid (see test/README.md).
"""
import os
import sys

_TEST_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_TEST_DIR, '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# NOTE: do not import fixtures as `test.fixtures...` -- the CPython stdlib
# ships its own top-level `test` package, and if anything on the import path
# has already triggered `import test` (pulling in /usr/lib/pythonX/test),
# `test.fixtures` silently resolves against the stdlib package instead of
# this directory. Add test/ itself to sys.path and import as `fixtures...`.
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

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
        "markers",
        "paper_parity: full paper-parity rollout gate (tiers 1+4) against "
        "gns-sample/ (249GB, real published checkpoints/rollouts). Not run "
        "by default even in CI -- requires gns-sample/ locally and an "
        "explicit `-m paper_parity` selection. See test/paper_parity/README.md.")


def pytest_addoption(parser):
    parser.addoption(
        "--paper-parity", action="store_true", default=False,
        help="Opt in to running the paper-parity gate tests (test/paper_parity/). "
             "Requires gns-sample/ to be symlinked in locally; see "
             "test/paper_parity/README.md. Ignored unless combined with "
             "`-m paper_parity` or running test/paper_parity/ directly.")
