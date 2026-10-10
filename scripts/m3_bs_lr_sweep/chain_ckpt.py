#!/usr/bin/env python3
"""Checkpoint / loss-log helpers for the chained-segment arm driver
(scripts/m3_bs_lr_sweep/arm_segment_gh200.sbatch; design doc section 3.4).

meshnet/train.py has no wall-clock stop and no NaN check, and its resume path
(train.py, `if FLAGS.model_file is not None:` block) raises on a checkpoint
that does not load. A SLURM kill or `timeout` SIGINT that lands inside
torch.save leaves a truncated newest model-N.pt / train_state-N.pt pair, which
`--model_file latest` would pick and crash on at the next segment. This script
is what the driver calls between segments; train.py itself is not modified.

Sub-commands (all print one machine-readable line on stdout, diagnostics on stderr):

  arm NAME                     -> "batch seed target_steps lr_init" for the design's
                                  section 3.2 arm table (A-arms only; B-arms need a
                                  fresh owner decision and are not defined here)
  latest-valid --model-dir D   -> step of the newest model/train_state pair that
                                  torch.load()s and carries the resume keys, or NONE.
                                  Pairs that fail are MOVED to D/corrupt/ (train.py's
                                  `latest` glob is non-recursive) and reported.
  check-loss --model-dir D --since-step S
                               -> parses D/loss_log.txt ("step train_loss valid_loss
                                  ..." per train.py); exit 1 if any train or valid
                                  loss at step > S is non-finite (NaN/inf); prints
                                  "last_step=N n_rows=M". Resumed segments re-log
                                  steps between the resume point and the kill, so
                                  duplicate step rows are expected and tolerated.
  prune --model-dir D --keep-every K --keep-newest M
                               -> delete model/train_state pairs whose step is not a
                                  multiple of K, except the M newest pairs. Never
                                  touches step 0, config.json or the logs.
"""
import argparse
import glob
import math
import os
import re
import shutil
import sys

# Design doc section 3.2, reduced 4-arm scope (owner-approved). Sample budget per arm
# is 21.6M = steps x batch. lr_init is the published 3e-5 on every A-arm and is
# asserted against config.json by the driver, never written by it.
ARMS = {
    #  name : (batch, seed, target_steps, lr_init)
    "A0a": (8, 1, 2_700_000, 3e-5),
    "A0b": (8, 2, 2_700_000, 3e-5),
    "A1": (4, 1, 5_400_000, 3e-5),
    "A2": (12, 1, 1_800_000, 3e-5),
    # DRY: driver self-test only (short target, overridable save interval); never scored.
    "DRY": (8, 1, 3_000, 3e-5),
}

_STEP_RE = re.compile(r"^model-(\d+)\.pt$")


def _pairs(model_dir):
    """{step: (model_path, train_state_path)} for every model-N.pt that has its train_state."""
    out = {}
    for p in glob.glob(os.path.join(model_dir, "model-*.pt")):
        m = _STEP_RE.match(os.path.basename(p))
        if not m:
            continue
        step = int(m.group(1))
        ts = os.path.join(model_dir, f"train_state-{step}.pt")
        if os.path.exists(ts):
            out[step] = (p, ts)
        else:
            print(f"[chain_ckpt] model-{step}.pt has no train_state-{step}.pt; ignored", file=sys.stderr)
    return out


def _loads(model_path, ts_path):
    """True iff both files load the way train.py will load them (torch.load default,
    weights_only=True on torch>=2.6) and carry the keys the resume path reads."""
    import torch
    try:
        m = torch.load(model_path, map_location="cpu")
        if not isinstance(m, dict) or "model" not in m:
            raise ValueError("model file lacks 'model' state dict")
        ts = torch.load(ts_path, map_location="cpu")
        if "optimizer_state" not in ts or "step" not in ts.get("global_train_state", {}):
            raise ValueError("train_state lacks optimizer_state / global_train_state.step")
        return True, None
    except Exception as e:  # noqa: BLE001 -- any failure here is exactly what we are screening for
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"


def cmd_arm(a):
    if a.name not in ARMS:
        sys.exit(f"unknown arm {a.name!r}; defined: {', '.join(ARMS)}")
    b, s, n, lr = ARMS[a.name]
    print(f"{b} {s} {n} {lr:g}")


def cmd_latest_valid(a):
    pairs = _pairs(a.model_dir)
    corrupt_dir = os.path.join(a.model_dir, "corrupt")
    for step in sorted(pairs, reverse=True):
        ok, why = _loads(*pairs[step])
        if ok:
            print(step)
            return
        os.makedirs(corrupt_dir, exist_ok=True)
        for p in pairs[step]:
            shutil.move(p, os.path.join(corrupt_dir, os.path.basename(p)))
        print(f"[chain_ckpt] step {step} does not load ({why}); moved pair to {corrupt_dir}/, "
              f"falling back one save", file=sys.stderr)
    print("NONE")


def cmd_check_loss(a):
    path = os.path.join(a.model_dir, "loss_log.txt")
    if not os.path.exists(path):
        print("last_step=NONE n_rows=0")
        return 0
    bad, last, n = [], None, 0
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            step = int(parts[0]); n += 1
            last = step if last is None else max(last, step)
            if step <= a.since_step:
                continue
            for col, name in ((1, "train"), (2, "valid")):
                v = float(parts[col])
                if not math.isfinite(v):
                    bad.append((step, name, parts[col]))
    print(f"last_step={last if last is not None else 'NONE'} n_rows={n}")
    if bad:
        for step, name, raw in bad[:5]:
            print(f"[chain_ckpt] non-finite {name} loss at step {step}: {raw}", file=sys.stderr)
        print(f"[chain_ckpt] {len(bad)} non-finite loss rows after step {a.since_step} -- diverged", file=sys.stderr)
        return 1
    return 0


def cmd_prune(a):
    pairs = _pairs(a.model_dir)
    steps = sorted(pairs)
    newest = set(steps[-a.keep_newest:]) if a.keep_newest > 0 else set()
    removed = 0
    for step in steps:
        if step == 0 or step in newest or step % a.keep_every == 0:
            continue
        for p in pairs[step]:
            os.remove(p)
        removed += 1
    print(f"pruned={removed} kept={len(steps) - removed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("arm"); p.add_argument("name"); p.set_defaults(fn=cmd_arm)
    p = sub.add_parser("latest-valid"); p.add_argument("--model-dir", required=True); p.set_defaults(fn=cmd_latest_valid)
    p = sub.add_parser("check-loss"); p.add_argument("--model-dir", required=True)
    p.add_argument("--since-step", type=int, default=-1); p.set_defaults(fn=cmd_check_loss)
    p = sub.add_parser("prune"); p.add_argument("--model-dir", required=True)
    p.add_argument("--keep-every", type=int, required=True); p.add_argument("--keep-newest", type=int, default=3)
    p.set_defaults(fn=cmd_prune)
    a = ap.parse_args()
    rc = a.fn(a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
