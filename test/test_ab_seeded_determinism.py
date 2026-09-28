"""A/B seeded training-determinism test (refactor-gating tier).

This test protects the invariant demanded by PROJECT_RULES.md rule 1: a
refactor of meshnet/train.py's train()/validation() loop must not change
observable behaviour. It runs the SAME tiny deterministic training config
(same synthetic dataset, same fixed seed via
test/fixtures/meshnet/seeded_pipeline_cli.py) through TWO separate
subprocess invocations of the real `python3 -m meshnet.train` CLI entry
point, each importing `meshnet` from its own tree ("tree A" = this
worktree's checked-out `meshnet/` as-is, "tree B" = a byte-for-byte copy of
tree A's `meshnet/` package made at test time), and asserts the per-step
training loss sequences the two runs log to `loss_log.txt` are identical.

Why copy tree A onto itself as tree B rather than diff two git refs?
This test's job is to prove the HARNESS is deterministic (A vs a copy of A
is the null hypothesis: if this doesn't pass bit-for-bit, no cross-ref
comparison ever will). The real use case -- gating an actual refactor -- is
to point `SECOND_TREE_MESHNET_SRC` (see below) at a candidate worktree's
`meshnet/` directory instead of copying tree A, e.g.:

    SECOND_TREE_MESHNET_SRC=/home/utig5/dliu/eqgns-worktrees/kai-refactor/meshnet \
        pytest test/test_ab_seeded_determinism.py -q

then the run-B loss sequence would come from the refactored code and any
divergence beyond float32 round-off fails the test immediately, per-step,
without needing to run the full training pipeline to convergence.

See docs/refactor_gating.md for the full usage note.
"""
import glob
import os
import re
import shutil
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLI_RELATIVE = os.path.join("test", "fixtures", "meshnet", "seeded_pipeline_cli.py")

DATASET_SEED = 555
TIMESTEPS = 24
NTRAINING_STEPS = 6
# Override the tiny fixture's default loss_report_step (1_000_000, i.e. "only
# ever log step 0") so train() writes loss_log.txt once per step -- this is a
# config override passed at test time, not an edit to synth.py itself.
LOSS_REPORT_STEP = 1

_TENSOR_FLOAT_RE = re.compile(r"tensor\(([^,)]+)")


def _parse_loss_log(path):
    """Parse meshnet/train.py's `loss_log.txt` lines:
    "{step} {loss_tensor} {valid_loss_tensor} {rms} {rms} {max} {max}\n"
    into a list of (step, train_loss, valid_loss) float triples.

    train.py formats tensors with an f-string (`f"{loss}"`), which for a
    0-d tensor calls Python's scalar float formatting (full precision), not
    `repr()` (which truncates to 4 significant figures) -- verified by hand
    against the actual file contents before relying on it here.
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            step = int(parts[0])
            train_loss = float(parts[1])
            valid_loss = float(parts[2])
            rows.append((step, train_loss, valid_loss))
    return rows


def _run_training(cli_path, tree_root, data_dir, model_dir, overrides_model_dir):
    sys.path.insert(0, os.path.join(REPO_ROOT, "test"))
    from fixtures.meshnet.synth import write_config

    write_config(str(overrides_model_dir), overrides={"loss_report_step": LOSS_REPORT_STEP})

    env = dict(os.environ, PYTHONPATH=tree_root)
    cmd = [
        sys.executable, cli_path, "--mode=train",
        f"--data_path={data_dir}/", f"--model_path={model_dir}/",
        "--batch_size=2", f"--ntraining_steps={NTRAINING_STEPS}",
        "--nsave_steps=1000",
    ]
    result = subprocess.run(cmd, cwd=tree_root, env=env,
                             capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"training subprocess failed (tree_root={tree_root}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return _parse_loss_log(os.path.join(str(model_dir), "loss_log.txt"))


def _make_second_tree(tmp_path):
    """Build tree B: a byte-for-byte copy of this worktree's `meshnet/`
    package plus the fixture's `seeded_pipeline_cli.py` (preserved at the
    same relative depth -- 3 levels under repo root -- since the CLI wrapper
    locates its own repo root from `__file__`), so the subprocess for run B
    imports `meshnet` from an entirely separate directory tree than run A.

    Override point for real refactor-gating: set
    `SECOND_TREE_MESHNET_SRC` to a candidate worktree's `meshnet/` dir
    instead of copying tree A onto itself.
    """
    second_tree = tmp_path / "second_tree"
    meshnet_src = os.environ.get(
        "SECOND_TREE_MESHNET_SRC", os.path.join(REPO_ROOT, "meshnet"))

    dest_meshnet = second_tree / "meshnet"
    shutil.copytree(meshnet_src, dest_meshnet,
                     ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # meshnet/learned_simulator.py imports `gns.graph_network` (the shared
    # GNN building blocks live in the sibling `gns/` package, not `meshnet/`
    # itself) -- copy it too so tree B's import tree is self-contained.
    gns_src = os.environ.get(
        "SECOND_TREE_GNS_SRC", os.path.join(REPO_ROOT, "gns"))
    shutil.copytree(gns_src, second_tree / "gns",
                     ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    dest_cli_dir = second_tree / "test" / "fixtures" / "meshnet"
    dest_cli_dir.mkdir(parents=True)
    shutil.copy(os.path.join(REPO_ROOT, CLI_RELATIVE), dest_cli_dir / "seeded_pipeline_cli.py")

    return str(second_tree)


def test_ab_training_loss_sequence_is_bitwise_reproducible_across_trees(tmp_path):
    """Same seeded config, run once against tree A (this checkout) and once
    against tree B (a fresh copy of tree A's meshnet/) in a separate
    subprocess/import tree -- the resulting per-step training loss
    sequences must match exactly.

    This is the null-hypothesis case (A vs copy-of-A). A real refactor gate
    points SECOND_TREE_MESHNET_SRC at the refactor worktree instead.
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "test"))
    from fixtures.meshnet.synth import build_dataset

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    build_dataset(str(data_dir), seed=DATASET_SEED, timesteps=TIMESTEPS)

    # --- Run A: tree A (this worktree's checked-out meshnet/) ---
    model_dir_a = tmp_path / "model_a"
    cli_a = os.path.join(REPO_ROOT, CLI_RELATIVE)
    loss_a = _run_training(cli_a, REPO_ROOT, data_dir, model_dir_a, model_dir_a)

    # --- Run B: tree B (separate copy / candidate worktree) ---
    second_tree_root = _make_second_tree(tmp_path)
    model_dir_b = tmp_path / "model_b"
    cli_b = os.path.join(second_tree_root, CLI_RELATIVE)
    loss_b = _run_training(cli_b, second_tree_root, data_dir, model_dir_b, model_dir_b)

    assert len(loss_a) == NTRAINING_STEPS + 1, (
        f"expected {NTRAINING_STEPS + 1} logged steps (0..{NTRAINING_STEPS}), "
        f"got {len(loss_a)} from run A's loss_log.txt -- loss_report_step "
        f"override did not take effect")
    assert len(loss_b) == len(loss_a), (
        "run A and run B logged a different number of training steps -- "
        "the two trees are not running the same training loop")

    steps_a = [row[0] for row in loss_a]
    steps_b = [row[0] for row in loss_b]
    assert steps_a == steps_b, "step numbering diverged between run A and run B"

    train_losses_a = [row[1] for row in loss_a]
    train_losses_b = [row[1] for row in loss_b]
    valid_losses_a = [row[2] for row in loss_a]
    valid_losses_b = [row[2] for row in loss_b]

    # Bit-for-bit (float32 round-trip through text) is achievable here:
    # both processes run single-threaded CPU (torch.set_num_threads(1)) with
    # torch.use_deterministic_algorithms(True) set by seeded_pipeline_cli.py,
    # on the same tiny fixed-seed dataset and identical model config, so
    # there is no floating-point-order source of non-determinism left. We
    # assert exact equality; any observed spread here would indicate a real
    # non-determinism source (e.g. an unseeded op, thread-count sensitivity)
    # and must be reported, not tolerance-papered-over.
    assert train_losses_a == train_losses_b, (
        "training loss sequence diverged between tree A and tree B for an "
        "identical seeded config -- this should never happen for two "
        "identical trees, and signals a real behaviour change if tree B "
        "was pointed at a refactor worktree via SECOND_TREE_MESHNET_SRC.\n"
        f"A: {train_losses_a}\nB: {train_losses_b}")
    assert valid_losses_a == valid_losses_b, (
        "validation loss sequence diverged between tree A and tree B for "
        "an identical seeded config.\n"
        f"A: {valid_losses_a}\nB: {valid_losses_b}")
