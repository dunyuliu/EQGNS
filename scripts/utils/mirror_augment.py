#!/usr/bin/env python3
"""Offline x-mirror augmentation of a meshnet trajectory npz.

For every `trajectory<i>` in the input (dict with pos, velocity, node_type,
node_property, pressure, cells) append `trajectory<N+i>` whose along-strike
coordinate is reflected, `pos[..., 0] *= -1`; everything else is copied
unchanged. This is the transform defined in
docs/dev/M1_ARRESTING_MIRROR_EXPANSION_DESIGN.md section 2.2: the fault is a
vertical plane with x-symmetric boundaries and friction, so the mirrored
trajectory is a valid sample with vx'(x) = vx(-x); vy is identically zero in
every M1/M2/M3 dataset so its sign flip is moot; node_type is x-symmetric
(asserted for all 19 D1 scenarios, 0 mismatches); cells index the same nodes
and need no change; per-node node_property travels with its node. Measured
truth asymmetry of a mirrored label: 2.5e-3 to 4.5e-3 rms/peak slip rate for
propagating ruptures, 1.3e-2 for the arresting H0/H16 pair (same doc).

Leakage: mirroring a train split can place the mirror image of a validation
or test scenario into training (M1 D1: test traj 1, 2 and all 3 valid
scenarios). The caller owns the split; this script only writes the data.

Usage:
  python3 scripts/utils/mirror_augment.py in.npz out.npz
"""
import argparse
import numpy as np

def mirror_trajectory(traj):
    out = {k: np.array(v, copy=True) for k, v in traj.items()}
    out["pos"][..., 0] *= -1
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    data = np.load(a.src, allow_pickle=True)
    keys = sorted(data.files, key=lambda k: int(k.replace("trajectory", "")))
    n = len(keys)
    out = {k: data[k].item() for k in keys}
    for i, k in enumerate(keys):
        out[f"trajectory{n + i}"] = mirror_trajectory(out[k])
    np.savez(a.dst, **out)
    print(f"{a.src}: {n} trajectories -> {a.dst}: {2 * n} (trajectory{n}..{2 * n - 1} are x-mirrors of 0..{n - 1})")

if __name__ == "__main__":
    main()
