# Rollout speed: what was measured, what to do

Two measurements: a 3D mesh (below) and the 2D paper models (`## 2D paper models`), which led
to the opt-in `--rollout_fast` path (`meshnet/fast_rollout.py`).

Measured 2026-10-08 with this repo's `meshnet` rollout
(`predict_velocity` in a loop over a static graph, batched disjoint graph as in
`rollout_batched`): 3D mesh, ~30k nodes / ~130k edges per trajectory, 10 message-passing layers,
latent 128, 1726 rollout steps, one idle A100 40 GB, torch 2.9.1. Batch of 5 trajectories.

| variant | ms / step / trajectory | vs eager |
|---|---|---|
| eager fp32 (current `rollout_batched`) | 38.2 | 1x |
| `torch.compile` on `EncodeProcessDecode.forward`, fp32 | 27.0 | 1.4x |
| same, TF32 matmuls | 13.4 | 2.9x |
| same, bf16 `torch.autocast` | 9.0 | 4.2x |
| bf16 + `mode="reduce-overhead"` (CUDA graphs) | 8.9 | no extra gain |

Batch of 1 vs 5: 40.8 vs 38.2 ms per trajectory-step. At this graph size one trajectory already
saturates the GPU, so batching buys little; it matters for small graphs (e.g. the 2D M1-M3 cases).

## Lessons

1. **The win is precision, not graph capture.** Compile alone is 1.4x; bf16 under compile is the
   4x. CUDA graphs add nothing once kernels are large (they help only when launch overhead
   dominates, i.e. small graphs).
2. **Pointwise parity is not a usable gate for any of these.** The rollout is autoregressive and
   amplifies rounding: even fp32 compile (kernel fusion only) differs from eager by ~4% of the
   field maximum after 200 steps; TF32/bf16 by ~30%. Gate on outcome metrics over full rollouts
   (rupture-time error, arrest/propagation classification), per model, never on bitwise match.
   Consistent with the cycle-GNS note in the `rollout-compile-optin` row (run-to-run diffs ~0.056).
3. **Never mix precisions inside one comparison.** Score every checkpoint being compared with
   the same variant; confirm a final winner in fp32.
4. **A speed check needs one or two trajectories and a few minutes.** Time ~200 steps after a
   5-step warm-up (compile costs ~11 s once per graph shape), with `torch.cuda.synchronize()`
   around the timed loop. Full-split rollouts are for the accuracy gate, not for timing.
5. **The static graph is already built once per rollout.** The edge encoder still re-runs on the
   unchanged edge features every step; caching its output is a small further saving (1 of 11
   edge MLP passes).
6. **No production edit needed to use it.** A launcher that wraps
   `gns.graph_network.EncodeProcessDecode.forward` in `torch.compile` + `torch.autocast(bf16)` and
   then `runpy.run_module("meshnet.train")` leaves `meshnet/` and checkpoints untouched (rule 1),
   so it stays opt-in and out of the paper-parity gate.

## 2D paper models: `--rollout_fast`

Measured 2026-10-08 on the M1-M3 checkpoints (`tests/paper_parity/gate.py` CASES): ~4.7k nodes /
~28k edges per trajectory, 826 steps, one idle A100 (GPU 1), torch 2.9.1.

`--rollout_fast {fp32,tf32,fp16,bf16}` (default `off`) runs `meshnet/fast_rollout.py`: same math
as `predict_velocity` on the static graph, with normalizers, one-hot and the edge encoder folded
into constants, the edge MLP's first layer factorized per node, edges laid out as (node, in-degree
slot) so aggregation is a fixed-order masked sum (no scatter, deterministic), `torch.compile`, and
the whole step captured as one CUDA graph.

ms / step / trajectory (min of 3 x 200 steps; the factorized path with `index_add_`
aggregation; the slot layout replaced it after these timings and is not yet timed):

| variant | batch 1 | batch 6 (M1_D1) | batch 15 (M2_D3) |
|---|---|---|---|
| eager fp32 (default path) | 10.1 | 8.06 | 7.77 |
| eager TF32 | 6.4 | 4.32 | 4.23 |
| fast tf32 | 1.88 | 1.87 | 1.86 |
| fast fp16 | 1.83 | | |
| fast bf16 | 1.81 | 1.46 | 1.37 |

So ~5.4x (tf32) to ~5.5x (fp16, batch 1) over eager; most of it is the CUDA graph plus compile,
which on these small graphs removes launch overhead that dominates (unlike the 3D case above).
Once graphed, batching adds little. `max-autotune` compile gains a further ~15% at a ~30 s
warm-up per graph shape; not exposed.

Accuracy, full test sets (outcome metrics, `gate.metrics`), against the deterministic
`reference.json` and two nondeterministic eager runs (the noise floor): mean rt_rmse per case

| case | reference | eager runs | fast fp32 | fast tf32 | fast fp16 | fast bf16 |
|---|---|---|---|---|---|---|
| M1_D1 | 0.352 | 0.352, 0.352 | 0.352 | 0.347 | 0.343 | 0.339 |
| M2_D3 | 0.341 | 0.331, 0.337 | 0.326 | 0.304 | 0.318 | **0.398** |
| M3_D3 | 0.272 | 0.272, 0.272 | 0.272 | 0.274 | 0.274 | **0.287** |

No trajectory collapsed in any variant. bf16 is the one that leaves the noise floor (+17% on
M2_D3, +5% on M3_D3); fp16 is as fast and stays inside it. **Use `--rollout_fast fp16`**; tf32
when fp16 is unavailable.

Gate: `gate.py fast [--precision P]` runs the full M1_D1 / M2_D3 / M3_D3 test sets with the flag,
deterministically, and requires per-case mean rt_rmse, missed+false and mean mse_vx within
`FAST_TOL` of the reference, with nothing collapsed; `--falsify` plants the weights x1.005
regression, which must fail. The default path stays gated by `gate.py run` (flag off, unchanged).

GATE_RESULTS_PLACEHOLDER
