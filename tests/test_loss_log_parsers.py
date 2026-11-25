"""parse_loss_log() in both training-fixture helper copies: malformed or
blank lines raise ValueError naming file and line; one trailing newline ok."""
import pytest

from fixtures.training_golden import common as golden
from fixtures.training_gate import common as gate

PARSERS = [golden.parse_loss_log, gate.parse_loss_log]
IDS = ["training_golden", "training_gate"]


@pytest.mark.parametrize("parse", PARSERS, ids=IDS)
def test_parses_well_formed_lines(tmp_path, parse):
    p = tmp_path / "loss_log.txt"
    p.write_text("1 0.5 0.25 9 9 9 9\n2 0.125 0.0625 9 9 9 9\n")
    assert parse(p) == [{"step": 1, "train_loss": 0.5, "valid_loss": 0.25},
                        {"step": 2, "train_loss": 0.125, "valid_loss": 0.0625}]


@pytest.mark.parametrize("parse", PARSERS, ids=IDS)
def test_single_trailing_blank_line_tolerated(tmp_path, parse):
    p = tmp_path / "loss_log.txt"
    p.write_text("1 0.5 0.25 9 9 9 9\n\n")
    assert [r["step"] for r in parse(p)] == [1]


@pytest.mark.parametrize("parse", PARSERS, ids=IDS)
def test_interior_blank_line_raises_with_location(tmp_path, parse):
    p = tmp_path / "loss_log.txt"
    p.write_text("1 0.5 0.25\n\n2 0.1 0.1\n")
    with pytest.raises(ValueError, match=r"loss_log\.txt:2:"):
        parse(p)


@pytest.mark.parametrize("parse", PARSERS, ids=IDS)
def test_two_trailing_blank_lines_raise(tmp_path, parse):
    p = tmp_path / "loss_log.txt"
    p.write_text("1 0.5 0.25\n\n\n")
    with pytest.raises(ValueError, match=r"loss_log\.txt:2:"):
        parse(p)


@pytest.mark.parametrize("parse", PARSERS, ids=IDS)
def test_short_line_raises_value_error(tmp_path, parse):
    p = tmp_path / "loss_log.txt"
    p.write_text("1 0.5 0.25\n2 0.5\n")
    with pytest.raises(ValueError, match=r"loss_log\.txt:2:"):
        parse(p)


@pytest.mark.parametrize("parse", PARSERS, ids=IDS)
def test_nonnumeric_line_raises_value_error(tmp_path, parse):
    p = tmp_path / "loss_log.txt"
    p.write_text("1 0.5 0.25\n3 abc 0.1\n")
    with pytest.raises(ValueError, match=r"loss_log\.txt:2:"):
        parse(p)
