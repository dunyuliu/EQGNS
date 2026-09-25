# Convergence gate (tier 3) — design

Board row: `convergence-tier3-nightly` in `PATHWAY_FORWARD.md` (P3, "ready to
schedule, not scheduled — GPUs saturated"). This directory holds the design
and a working script; **no cron/systemd timer is installed** by this commit —
scheduling is a separate, deliberate step (see "Installing the schedule"
below) once a slot is free.

## What this proves, and why it's a separate tier from 1/2/4

Tiers 1 (`test/paper_parity/`) and 2 (`test/test_ab_seeded_determinism.py`)
both replay a FROZEN, already-trained checkpoint or a fixed short loss
sequence. Neither one trains a model from a fresh random initialization for
very long. A regression that breaks the ability of `train()` to actually
*learn* from scratch (a bad gradient path, a normalizer that saturates, a
loss that silently NaNs after enough steps, an optimizer state bug that only
shows up after hundreds of steps) can slip past both of those gates, because
they never run that many real fresh-init steps.

This tier does: fresh random init, real `MeshSimulator.predict_acceleration`
-> `acceleration_loss` -> backward -> `optimizer.step()` wiring (identical to
`test/test_meshnet_integration_train_rollout.py`'s tier-2 integration test,
just run for many more steps: 2000 by default vs. that test's 8), and checks
the loss actually goes down and stays finite.

## What it is NOT

- Not a parity check — it says nothing about whether today's numbers match
  yesterday's on real data. That's tier 1.
- Not a substitute for the real GPU training runs in `work.cnn/` or
  `gns-sample/*/models.*` — it uses the tiny synthetic dataset
  (`test/fixtures/meshnet/synth.py`), CPU-only, and is intentionally cheap
  (~20s wall for 2000 steps on this box with 4 threads) so it can run
  nightly without touching a GPU or the 249GB `gns-sample/` data.
- Not a performance/perf benchmark — no timing assertions in the gate logic
  itself (the runtime is only logged for the operator's information).

## Convention statement (PROJECT_RULES.md rule 7)

Not applicable: this script never computes a rupture time or slip rate. It
only measures acceleration-loss convergence over training steps, so none of
the `DT` / `SLIPRATE_THRESHOLD` / `+1.2s` conventions from
`utils/plot.rupture.dynamics.py` apply.

## Calibration (2026-09-24)

Run: `python3 test/convergence_gate/convergence_gate.py --nsteps 2000 --seed
20260101 --timesteps 64` on this box (CPU, `torch.set_num_threads(4)`).
Result: `head_mean_loss=6.339`, `tail_mean_loss=6.051`, wall time ≈19s user
CPU-bound but ≈19s wall too (thread-capped; an earlier uncapped run at
`nsteps=100` took 105s wall at 3861% CPU — see "Thread discipline" below).
`threshold.json`'s `final_mean_loss_max=7.5` is set with ~25% margin above the
observed tail loss (see `_calibration_note` inside `threshold.json`).
Re-calibrate (rerun and update `threshold.json`) if `TINY_CONFIG`/`synth.py`
change shape, or if a legitimate architecture change to `MeshSimulator`
changes the achievable loss floor on this tiny dataset.

## Thread discipline (shared box)

This box runs GPU training jobs (`work.cnn/`, `dynamo_gns/`, `gns_earthquake_cycle/`
experiments — see PATHWAY_FORWARD.md context) whose CPU-side dataloader
workers also want cores. An uncapped `torch.set_num_threads` call spawned
~38 intra-op threads for trivial tiny-tensor ops and was 20x SLOWER wall
time than capping to 4 threads (105s vs 5s for 100 steps) — pure contention,
not real work. The script defaults to
`torch.set_num_threads(int(os.environ.get("CONVERGENCE_GATE_NUM_THREADS", "4")))`;
override via the env var if you know the box is otherwise idle, but 4 is the
tested-good default and should not be raised without re-measuring.

## Installing the schedule (not done yet)

When a slot is free (GPUs unsaturated, or moving this to a CPU-only cron
box), install e.g.:
```
# crontab -e (as the user this repo runs under)
0 3 * * * cd /home/utig5/dliu/eq_rupture_gns && source venv/bin/activate && \
  python3 test/convergence_gate/convergence_gate.py --nsteps 2000 >> test/convergence_gate/nightly.log 2>&1
```
`test/convergence_gate/history/*.json` accumulates one record per run
(gitignored, see `.gitignore` entry added alongside this design) so a trend
can be plotted later; nothing here auto-alerts yet — the exit code is the
gate signal a wrapping CI/cron job should act on (e.g. page/email on
non-zero exit, or 2+ consecutive failures per PROJECT_RULES.md escalation
conventions).

## Usage

```
python3 test/convergence_gate/convergence_gate.py \
    [--nsteps 2000] [--seed 20260101] [--timesteps 64] \
    [--threshold-file test/convergence_gate/threshold.json]
```
Exit 0 = converged; exit 1 = did not converge (loss didn't drop, or went
non-finite, or stayed above `final_mean_loss_max`). Prints and saves a JSON
record with `head_mean_loss`/`tail_mean_loss`/`converged`.
