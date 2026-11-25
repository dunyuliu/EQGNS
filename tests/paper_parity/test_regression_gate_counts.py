"""CPU unit tests for gate.cmd_regression()'s trajectory-count guard: a
rollout that wrote zero files, or a fast/ref count mismatch, must FAIL
(never pass vacuously, never count as a CAUGHT planted regression).
Monkeypatches measure_vs_published; no GPU, no data."""
import pytest

import measure_vs_published as mvp
from paper_parity import gate


def good_row(i):
    return {"row_type": "regression_fp16", "case": "M1_D1", "traj": i,
            "delta_rt_rmse_s": 0.0, "delta_mw": 0.0, "missed": 0, "false": 0, "note": ""}


@pytest.fixture
def patched(monkeypatch):
    def install(fast, ref):
        def fake_rollout(case, cuda, code="current", extra=(), model_dir=None):
            return ref if code == "published" else fast
        monkeypatch.setattr(mvp, "fresh_raw_rollout", fake_rollout)
        # real rows_for semantics: min(len) rows, mismatch only in a note
        monkeypatch.setattr(mvp, "rows_for",
                            lambda rt, case, a, b: [good_row(i) for i in range(min(len(a), len(b)))])
        monkeypatch.setattr(gate, "scaled_model", lambda case, tmp, scale: tmp)
    return install


def run(falsify=False):
    return gate.cmd_regression(["M1_D1"], "0", "fp16", falsify=falsify)


def test_zero_trajectories_fails(patched, capsys):
    patched(fast=[], ref=[])
    assert run() is False
    assert "zero trajectories" in capsys.readouterr().out


def test_count_mismatch_fails(patched, capsys):
    patched(fast=["a"], ref=["a", "b"])
    assert run() is False
    assert "count mismatch" in capsys.readouterr().out


@pytest.mark.parametrize("fast,ref", [([], ["a"]), (["a", "b"], ["a", "b", "c"])],
                         ids=["fast0_ref1", "fast2_ref3"])
def test_count_mismatch_is_error_not_caught_or_missed_in_falsify_mode(patched, capsys, fast, ref):
    patched(fast=fast, ref=ref)
    assert run(falsify=True) is False
    out = capsys.readouterr().out
    assert "ERROR" in out, out
    assert "CAUGHT" not in out, out
    assert "MISSED" not in out, out


def test_matched_counts_with_passing_rows_passes(patched):
    patched(fast=["a", "b"], ref=["a", "b"])
    assert run() is True
