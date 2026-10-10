#!/usr/bin/env python3
"""Training throughput (samples/s) from a wall-clock-timestamped meshnet/train.py log.

Input: the stdout of `python -u meshnet/train.py --mode=train ...` with every line
prefixed by a unix timestamp (seconds, float) -- produced by pilot_gh200.sbatch's
`while read line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done` filter.
train.py prints `Training step: <step>/<n>. Train loss: ...` every
config.json:loss_report_step steps; those (timestamp, step) pairs are the data.

Method: ordinary least-squares fit of step vs wall time over all report lines AFTER
the first one (the step-0 report includes dataset load, CUDA context and cuDNN warm-up,
none of which is steady-state training). samples/s = slope [step/s] x batch_size.
With >= 3 points the fit residual is reported so a non-steady run is visible.

Optional --dmon: an `nvidia-smi dmon -s um -d <sec> -o T` log; its sm-utilisation
column is averaged over the same steady-state window (rows whose wall time falls
between the first and last report). `-o T` prints HH:MM:SS without a date, so the
date is taken from the training log's first timestamp (UTC on the compute node, same
clock) -- a run that crosses midnight is handled by allowing one day rollover.

Output: one JSON file (--out) and a one-line human summary on stdout.
"""
import argparse
import datetime as dt
import json
import re
import sys

import numpy as np

STEP_RE = re.compile(r"^(\d+(?:\.\d+)?)\s+Training step:\s+(\d+)/\d+\.")


def parse_train_log(path):
    pts, first_ts = [], None
    with open(path) as f:
        for line in f:
            if first_ts is None:
                m0 = re.match(r"^(\d+(?:\.\d+)?)\s", line)
                if m0:
                    first_ts = float(m0.group(1))
            m = STEP_RE.match(line)
            if m:
                pts.append((float(m.group(1)), int(m.group(2))))
    return first_ts, pts


def parse_dmon(path, day0_ts):
    """-> list of (unix_ts, sm_util_pct). Header lines start with '#'."""
    rows, sm_col, time_col = [], None, None
    day0 = dt.datetime.utcfromtimestamp(day0_ts).replace(hour=0, minute=0, second=0, microsecond=0)
    prev_secs = None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "#":
                if "sm" in parts and sm_col is None:
                    sm_col = parts.index("sm") - 1
                    time_col = parts.index("Time") - 1 if "Time" in parts else 0
                continue
            if sm_col is None:
                continue
            try:
                hh, mm, ss = (int(x) for x in parts[time_col].split(":"))
                secs = hh * 3600 + mm * 60 + ss
                if prev_secs is not None and secs < prev_secs - 3600:
                    day0 += dt.timedelta(days=1)
                prev_secs = secs
                ts = (day0 + dt.timedelta(seconds=secs) - dt.datetime(1970, 1, 1)).total_seconds()
                rows.append((ts, float(parts[sm_col])))
            except (ValueError, IndexError):
                continue
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--dmon", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    first_ts, pts = parse_train_log(a.log)
    res = {"label": a.label, "batch_size": a.batch, "log": a.log, "n_reports": len(pts)}
    if len(pts) < 3:
        res["status"] = "insufficient_reports"
        res["note"] = f"need >= 3 'Training step' reports to fit a slope, got {len(pts)}"
    else:
        res["load_and_warmup_s"] = pts[0][0] - first_ts if first_ts is not None else None
        t = np.array([p[0] for p in pts[1:]])
        s = np.array([p[1] for p in pts[1:]], dtype=float)
        slope, icpt = np.polyfit(t - t[0], s, 1)
        resid = s - (slope * (t - t[0]) + icpt)
        res.update({
            "status": "ok",
            "steady_window_s": float(t[-1] - t[0]),
            "steady_steps": int(s[-1] - s[0]),
            "steps_per_s": float(slope),
            "ms_per_step": float(1e3 / slope),
            "ms_per_sample": float(1e3 / (slope * a.batch)),
            "samples_per_s": float(slope * a.batch),
            "fit_residual_rms_steps": float(np.sqrt(np.mean(resid ** 2))),
            "first_report_step": int(pts[0][1]),
            "last_report_step": int(pts[-1][1]),
        })
    if a.dmon and res.get("status") == "ok":
        rows = parse_dmon(a.dmon, pts[0][0])
        win = [u for ts, u in rows if pts[1][0] <= ts <= pts[-1][0]]
        res["gpu_sm_util_mean_pct"] = float(np.mean(win)) if win else None
        res["gpu_sm_util_n"] = len(win)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    if res.get("status") == "ok":
        print(f"[{a.label or 'b%d' % a.batch}] b{a.batch}: {res['samples_per_s']:.1f} samples/s, "
              f"{res['ms_per_step']:.1f} ms/step, {res['ms_per_sample']:.2f} ms/sample over "
              f"{res['steady_steps']} steps / {res['steady_window_s']:.0f} s "
              f"(load+warmup {res['load_and_warmup_s']:.0f} s"
              + (f", GPU sm {res['gpu_sm_util_mean_pct']:.0f}%" if res.get("gpu_sm_util_mean_pct") is not None else "")
              + ")")
    else:
        print(f"[{a.label or 'b%d' % a.batch}] {res['status']}: {res.get('note', '')}", file=sys.stderr)


if __name__ == "__main__":
    main()
