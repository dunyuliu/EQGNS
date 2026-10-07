"""pytest wrapper for the paper-parity gate (opt-in: needs gns-sample/ and a GPU).

    pytest tests/paper_parity --paper-parity -q
"""
import pytest

from paper_parity import gate

pytestmark = pytest.mark.paper_parity



@pytest.fixture(autouse=True)
def _require_opt_in(request):
    if not request.config.getoption("--paper-parity"):
        pytest.skip("paper-parity gate is opt-in: pass --paper-parity")
    if not gate.DATA.exists():
        pytest.skip(f"{gate.DATA} not found")


@pytest.mark.parametrize("case", list(gate.CASES))
def test_reproduces_published_rollouts(case):
    current = gate.fresh_rollout(case, cuda=0)
    assert gate.compare(case, current, gate.load(gate.REFERENCE))


def test_planted_regression_is_caught():
    assert gate.cmd_falsify("M1_D1", cuda=0)
