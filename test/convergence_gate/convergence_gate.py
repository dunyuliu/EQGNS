#!/usr/bin/env python3
"""Tier-3 nightly seeded mini-training convergence gate (DESIGN + SCRIPT ONLY
per PATHWAY_FORWARD.md row `convergence-tier3-nightly` — no cron/systemd timer
is installed by this script; see DESIGN.md for how to schedule it).

Purpose
-------
Tiers 1/2/4 (test/paper_parity/, test/test_ab_seeded_determinism.py) prove
*parity* — that today's code reproduces yesterday's numbers on real or
seeded-synthetic data. None of them prove that a real training loop still
*converges* at all: a change could keep every parity gate green (because
parity gates replay a FROZEN checkpoint) while quietly breaking the ability
to train a NEW model from scratch (e.g. a broken gradient path, a NaN'ing
normalizer, an exploding loss). This tier trains a tiny model from a fresh
random init, for real, every night, and checks the loss actually goes down.

This intentionally reuses the exact real training wiring already exercised
by test/test_meshnet_integration_train_rollout.py (data_loader ->
transformer -> MeshSimulator.predict_acceleration -> acceleration_loss ->
backward -> optimizer.step), NOT a re-implementation, so a regression in
that wiring shows up here too.

Convention statement (PROJECT_RULES.md rule 7): this script does not compute
rupture time or slip rate at all -- it only checks acceleration-loss
convergence -- so none of the DT / SLIPRATE_THRESHOLD / +1.2s conventions
from utils/plot.rupture.dynamics.py apply here.

Usage
-----
    python3 test/convergence_gate/convergence_gate.py \
        [--nsteps 2000] [--seed 20260101] [--threshold-file test/convergence_gate/threshold.json]

Exit code 0 if the run converges under threshold; 1 otherwise. Writes a
timestamped JSON record to test/convergence_gate/history/ so trend can be
tracked across nightly runs once scheduled (see DESIGN.md).

CPU-only, tiny synthetic data (test/fixtures/meshnet/synth.py) -- this never
touches gns-sample/ or a GPU, so it is safe to run any time, including
alongside the GPU training jobs this repo may have in flight elsewhere.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch
import torch_geometric.transforms as T

# This is a CPU-only tiny-data probe sharing a box with GPU training jobs
# whose CPU-side dataloader workers also want cores; cap intra-op threads
# so we don't grab the whole machine (see PROJECT_RULES.md / conductor
# babysitting rules: cap BLAS/OpenMP threads per process on a shared node).
torch.set_num_threads(int(os.environ.get("CONVERGENCE_GATE_NUM_THREADS", "4")))

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TEST_DIR = os.path.join(_REPO_ROOT, "test")
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from meshnet import data_loader          # noqa: E402
from meshnet import train as train_mod   # noqa: E402
from meshnet.learned_simulator import MeshSimulator  # noqa: E402
from meshnet.noise import get_velocity_noise          # noqa: E402
from meshnet.utils import NodeType                    # noqa: E402
from fixtures.meshnet.synth import build_dataset, TINY_CONFIG  # noqa: E402

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "history")
DEFAULT_THRESHOLD_FILE = os.path.join(os.path.dirname(__file__), "threshold.json")

transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


def run_convergence_probe(nsteps: int, seed: int, dataset_dir: str):
    """Real gradient-descent loop, identical wiring to the tier-2 integration
    test, just run for many more steps. Returns the per-step loss list."""
    torch.manual_seed(seed)
    cfg = TINY_CONFIG
    simulator = MeshSimulator(
        simulation_dimensions=cfg["simulator_simulation_dimensions"],
        nnode_in=cfg["simulator_nnode_in"],
        nedge_in=cfg["simulator_nedge_in"],
        latent_dim=cfg["simulator_latent_dim"],
        nmessage_passing_steps=cfg["simulator_nmessage_passing_steps"],
        nmlp_layers=cfg["simulator_nmlp_layers"],
        mlp_hidden_dim=cfg["simulator_mlp_hidden_dim"],
        nnode_types=cfg["simulator_nnode_types"],
        node_type_embedding_size=cfg["simulator_node_type_embedding_size"],
        device="cpu")
    simulator.train()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-3)

    ds = data_loader.get_data_loader_by_samples(
        path=os.path.join(dataset_dir, "train.npz"), input_length_sequence=1, dt=0.1,
        batch_size=4, shuffle=False)

    losses = []
    step = 0
    while step < nsteps:
        for graph in ds:
            if step >= nsteps:
                break
            graph = transformer(graph)
            node_types = graph.x[:, 0]
            node_property = graph.x[:, 1]
            current_velocities = graph.x[:, 2:4]
            velocity_noise = get_velocity_noise(graph, noise_std=0.0, device="cpu")

            pred_acc, target_acc = simulator.predict_acceleration(
                current_velocities=current_velocities, node_type=node_types,
                node_property=node_property, edge_index=graph.edge_index,
                edge_features=graph.edge_attr, target_velocities=graph.y,
                velocity_noise=velocity_noise)

            non_kinematic_mask = torch.logical_or(node_types == NodeType.NORMAL,
                                                   node_types == NodeType.HIGH_STRESS)
            loss = train_mod.acceleration_loss(pred_acc, target_acc, non_kinematic_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
    return losses


def evaluate(losses, threshold):
    """Convergence criterion: mean loss over the final 10% of steps must be
    below `threshold`'s 'final_mean_loss_max', AND finite throughout, AND the
    mean loss over the final 10% must be below the mean loss over the first
    10% (i.e. it actually went down, not just started low)."""
    n = len(losses)
    head = losses[:max(1, n // 10)]
    tail = losses[-max(1, n // 10):]
    head_mean = sum(head) / len(head)
    tail_mean = sum(tail) / len(tail)
    all_finite = all(v == v and abs(v) < 1e6 for v in losses)  # v==v excludes NaN
    converged = (all_finite
                 and tail_mean < threshold["final_mean_loss_max"]
                 and tail_mean < head_mean)
    return {
        "converged": converged,
        "all_finite": all_finite,
        "head_mean_loss": head_mean,
        "tail_mean_loss": tail_mean,
        "final_mean_loss_max_threshold": threshold["final_mean_loss_max"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsteps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--threshold-file", default=DEFAULT_THRESHOLD_FILE)
    parser.add_argument("--timesteps", type=int, default=64,
                         help="length of each synthetic trajectory fed to build_dataset")
    args = parser.parse_args()

    with open(args.threshold_file) as f:
        threshold = json.load(f)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        build_dataset(tmp, seed=args.seed, timesteps=args.timesteps)
        losses = run_convergence_probe(args.nsteps, args.seed, tmp)

    result = evaluate(losses, threshold)
    result["nsteps"] = args.nsteps
    result["seed"] = args.seed
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(HISTORY_DIR, exist_ok=True)
    out_path = os.path.join(HISTORY_DIR, f"run_{result['timestamp_utc'].replace(':', '')}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
    sys.exit(0 if result["converged"] else 1)


if __name__ == "__main__":
    main()
