# Refactor gating: A/B seeded training-determinism test

`test/test_ab_seeded_determinism.py` protects PROJECT_RULES.md's rule
that a refactor of `meshnet/train.py`'s `train()`/`validation()` must not
change observable behaviour. It is a fast, per-step, bit-for-bit
regression check that sits alongside (not in place of) the full test
pyramid in `test/README.md`.

## What it proves

Given the SAME seeded tiny config (fixed dataset seed, fixed model seed,
`torch.use_deterministic_algorithms(True)`, single CPU thread -- all via
the existing `test/fixtures/meshnet/seeded_pipeline_cli.py` wrapper), it
runs `python3 -m meshnet.train --mode=train` **twice**, each time
importing the `meshnet` (and `gns`) packages from a **separate directory
tree**, and asserts the two runs' per-step training-loss and
validation-loss sequences (parsed from `train()`'s own
`loss_log.txt`, which it already writes -- no new logging was added to
`meshnet/train.py`) are identical.

- Run A always uses this checkout's `meshnet/` as-is.
- Run B, by default, uses a **fresh copy** of tree A's `meshnet/` +
  `gns/` packages -- i.e. the test's baseline case is "A vs a copy of
  A". This proves the harness itself (subprocess isolation, seeding,
  loss-log parsing) is deterministic, which is the precondition for
  trusting any A-vs-B comparison at all.

## Using it to gate a real refactor

Point run B at a candidate refactor worktree's `meshnet/` (and, if it
also touched the shared GNN building blocks, `gns/`) directory instead
of copying tree A:

```bash
source venv/bin/activate
SECOND_TREE_MESHNET_SRC=/home/utig5/dliu/eqgns-worktrees/kai-refactor/meshnet \
SECOND_TREE_GNS_SRC=/home/utig5/dliu/eqgns-worktrees/kai-refactor/gns \
    pytest test/test_ab_seeded_determinism.py -q
```

If the refactor changed `train()`/`validation()`'s observable
behaviour -- a different loss at any logged step, in either the train
or the validation sequence -- this test fails immediately, naming the
first diverging step, without needing to run (or diff against) the
full golden-file end-to-end pipeline.

## What this test is NOT

- **Not a substitute for the full suite.** It only compares scalar
  loss values at a handful of steps on a 12-node synthetic mesh. It
  says nothing about rollout stability, checkpoint I/O, config
  loading, or the physical-behaviour invariants in
  `test_meshnet_physical.py` -- those tiers must still pass.
- **Not a numerical-tolerance check.** It asserts exact equality
  (`==`, not `np.testing.assert_allclose`) because both runs are
  single-threaded CPU with deterministic algorithms enabled and an
  identical, tiny, fixed-seed input -- there is no legitimate source of
  float noise left. If tree B's refactor is a deliberate,
  numerically-different-but-equivalent change (e.g. a reduction order
  change that shifts float32 round-off), this test is expected to fail
  and that failure should be triaged by a human, not loosened in this
  test -- use the golden-file e2e tier's documented tolerance
  (`TOLERANCE = 1e-3` in `test_meshnet_e2e_golden.py`) for that kind of
  intentional-but-bounded change instead.
- **Not a way to detect performance regressions.** It says nothing
  about speed; see `CLAUDE.md`'s rollout-optimization notes for that
  axis.

## Cost

Two full `python3 -m meshnet.train` subprocess launches, each paying
~8s of `torch`/`torch_geometric` import overhead (measured on this
machine) plus <1s of actual training on the 6-step tiny config. It is
marked `@pytest.mark.slow` (excluded from the fast local loop, included
in the CI default per `test/README.md`'s CI-parity convention) for the
same reason `test_meshnet_e2e_golden.py` is.
