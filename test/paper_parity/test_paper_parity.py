"""pytest wrapper for the paper-parity gate (opt-in: needs gns-sample/ and a GPU).

    pytest test/paper_parity --paper-parity -q
"""
import pytest

from paper_parity import gate

pytestmark = pytest.mark.paper_parity

# Known discrepancy: the post-publication rollout speed-up (1b3fb9e) caches
# node_property at step 0; D2/D3 data have time-varying node_property, so
# current code no longer matches train.py.published on those cases.
KNOWN_FAIL = {c: "rollout() caches node_property at step 0 (1b3fb9e)" for c in ("M2_D3", "M3_D2")}


@pytest.fixture(autouse=True)
def _require_opt_in(request):
    if not request.config.getoption("--paper-parity"):
        pytest.skip("paper-parity gate is opt-in: pass --paper-parity")
    if not gate.DATA.exists():
        pytest.skip(f"{gate.DATA} not found")


@pytest.mark.parametrize("case", list(gate.CASES))
def test_reproduces_published_rollouts(case):
    if case in KNOWN_FAIL:
        pytest.xfail(KNOWN_FAIL[case])
    current = gate.fresh_rollout(case, cuda=0)
    assert gate.compare(case, current, gate.load(gate.REFERENCE))


def test_planted_regression_is_caught():
    assert gate.cmd_falsify("M1_D1", cuda=0)
