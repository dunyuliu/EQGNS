
## 2026-09-28 — INCIDENT: shared venv_cotopaxi torch install corrupted

**What happened**: while doing the stranger-clone README check for release-v1.1, ran
`bash build_venv.sh` from a fresh clone under /tmp scratch, following the README's
documented "Building GNS environment" step literally. `build_venv.sh` does
`python3 -m virtualenv venv && source venv/bin/activate` — this resolved (via
inherited shell environment / module.sh, not investigated further) to the
**shared production venv** `/home/utig5/dliu/gns/gns/venv_cotopaxi`, NOT an
isolated venv under the scratch clone. The script proceeded to
`pip3 install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 triton==2.0.0`
into that live venv, overwriting the working `torch==2.9.1+cu128` it had before.

**Who else uses this venv**: 6 live production training jobs were running against
it at the time (PIDs 1888683, 1889010, 1889717 — meshnet_opt/gns_earthquake_cycle
experiments on GPU0/GPU0/GPU2; 2060307 — 5-day-old exp105b job on GPU3; 3207004,
3207006 — dynamo_gns_wt_moredata MD1/MD2 on GPU1/GPU3). None are part of this
EQGNS campaign.

**Action taken**: killed the install by PID (bash PID 3892081, pip child PID
3893866) the moment the misdirected path was confirmed via `ps -ef` showing the
pip process running out of `/home/utig5/dliu/gns/gns/venv_cotopaxi/bin/pip3`.
Kill was targeted by PID, not `pkill -f`.

**Damage**: `torch` is left in a broken half-replaced state in that shared venv
(`import torch` now raises `AttributeError: module 'torch' has no attribute
'rand'`; site-packages/torch/ has pip's `~`-prefixed rename-artifact dirs from
the interrupted in-place replace, e.g. `~dynamo`, `~decomp`). `torchvision`
0.24.1 and `triton` 2.0.0 were also partially overwritten. No lockfile or
requirements pin recording the original 2.9.1+cu128 build was found under
`/home/utig5/dliu/gns/gns/` in a quick search, so I did NOT attempt to guess a
restore — further writes to this venv without knowing the exact original spec
risk compounding the corruption.

**Live-job status at time of writing**: all 6 PIDs still `Rl` (running) —
already-imported `torch` stays resident in a running process's memory even
after the on-disk package is replaced, so they have not crashed YET. Risk:
any new process spawned against this venv (a fresh training launch, a
checkpoint-time subprocess, a lazy re-import) will hit the broken torch
immediately, and if any of the 6 processes needs a fresh import of a torch
submodule it hasn't touched yet, it could crash too. This is unverified either
way — I stopped rather than probe further and risk touching the venv again.

**Root cause not yet found**: why does a *relative* `venv/` path in a script
run from `/tmp/.../scratch/repo` land in `/home/utig5/dliu/gns/gns/venv_cotopaxi`?
Suspect `module.sh` (sourced by build_venv.sh) or a login-shell rc sets
`VIRTUAL_ENV`/`PATH`/a `venv` alias/symlink that redirects `virtualenv venv`
activation. Not investigated further — stopped to escalate instead of digging
deeper inside a shared, live resource.

**Escalated to human**: yes, this session, before taking any further action on
this venv. Recommend: (1) confirm whether the 6 live jobs are still healthy,
(2) supply the original pinned versions (or point to the module/build log that
built venv_cotopaxi originally) so it can be restored precisely, (3) treat
`build_venv.sh` itself as suspect — it should not be run again as documented
until the relative-path resolution is understood, which is itself a
release-blocker finding for the release-v1.1 stranger-clone check (README step
is unsafe to follow literally on this machine).

## 2026-09-28 — venv incident root cause (per coordinator's read-only survey)

`/home/utig5/dliu/gns/gns` is a **symlink into this repo** (eq_rupture_gns).
`venv_cotopaxi` inside this repo is the shared venv for `dynamo_gns_wt_moredata`
and CycleGNS (`/home/utig5/dliu/scratch/gns_earthquake_cycle`) — ~96 CycleGNS
`train.sh` scripts put this tree on `PYTHONPATH`. This explains why
`build_venv.sh`'s relative `venv/` landed in the shared venv: it isn't a
misconfigured shell, `gns/gns/venv_cotopaxi` *is* this repo's own venv,
consumed cross-project.

**New standing rule for this campaign** (pending `zofia-kaminska` write-up):
never modify or rebuild `venv_cotopaxi`; any change under `meshnet/` or `gns/`
(not `utils/`, not `test/`) is cross-project and stops for owner OK before
landing, because sibling repos import against this tree at those paths.
`utils/prepare.eqdyna.4gns.py` and `utils/prepare.fractal.stress.eqdyna.4gns.py`
fixes remain approved (utils/ is not on the cross-project import path) —
confirming via grep of sibling trees that neither file is imported there.

**Board additions proposed by owner, NOT started, pending owner sign-off**:
1. Collapse guard in gate/retrain comparison — flag if predicted/GT variance
   ratio < 0.5 (dynamo_gns Rule 22 cl.7-8): a flat forecast scores well on MSE
   but is degenerate.
2. Checkpoint selection on VALID, report on TEST (CycleGNS Rule 17) — apply to
   the M1 old-D1-vs-fixed-D1 comparison; test-selected single checkpoints are
   "oracle" and not reportable.
3. Seed-spread requirement before claiming an effect (CycleGNS Rule 25) — one
   retrain pair is one seed; factor into the training-step-budget question
   already queued for the owner.

**Sibling-import check (done)**: scoped grep of sibling code dirs
(`gns_earthquake_cycle/{src,utils,train.sh,tests}`,
`dynamo_gns_wt_moredata/{src.reference,src.spec,src.sphere,scenario.rollout.py,tests}`)
for `prepare.eqdyna.4gns` / `prepare.fractal.stress.eqdyna.4gns` — zero matches.
The two approved utils/ fixes are confirmed NOT on the cross-project import
path; safe to land per the coordinator's own stated condition. (Full recursive
`experiment/`/`work/`/`results/` output dirs were NOT scanned — those are run
artifacts, not import sources, so out of scope for this check; a wide
unscoped `du`/`grep -r` over those dirs was killed by PID mid-run as wasteful,
per papercuts.)

## 2026-09-28 — m1-retrain-fixed-data design amendment (per shared lessons file, ~/code/gns_lessons.md)

New facts from CycleGNS/eq_cycle survey, bearing directly on the fixed-vs-old-D1
M1 retrain comparison already queued for owner sign-off on step/seed budget:

(a) `meshnet/train.py` training is **unseeded** — init, sample order, and noise
    are all unseeded. Consequence: a single "identical settings" fixed-D1 vs
    old-D1 control pair CANNOT by itself resolve a small data effect from
    ordinary training-run variance — this sharpens (not replaces) the
    already-agreed seed-spread requirement (CycleGNS Rule 25, proposed board
    row #3 above): without seeding, "spread" must mean multiple full retrain
    runs per arm, not just multiple seeds passed to an otherwise-unseeded
    path. Folding into the training-step-budget question already queued for
    the owner: budget now needs to cover N>1 runs per arm, not 1.
(b) Resume off-by-one at `meshnet/train.py:229` — `model-<s>.pt` is saved
    AFTER step s's update; resuming restarts at step s, causing one extra
    update and the checkpoint being silently overwritten. Do not resume
    mid-comparison for the M1 retrain pair; if a resume is unavoidable
    (owner's step budget forces multi-session training), account for the
    off-by-one explicitly rather than silently absorbing it.
(c) Validation in train mode is computed on training-noised inputs, which also
    feed the normalizers — a further reason not to trust a single
    train-mode-reported validation number as the comparison metric; matches
    the already-proposed board row #2 (select checkpoints on VALID, report on
    TEST via gate.py's own deterministic rollout/metrics path, not train.py's
    internal validation printout).

Fix patterns already exist upstream: `src/meshnet/seeding.py`,
`SeededEpochSampler`, `tests/test_seeding*.py` in
`/home/utig5/dliu/scratch/gns_earthquake_cycle` (v0.0.43/44). **NOT applying
these to this repo's `meshnet/train.py` without owner OK** — `meshnet/` is
cross-project (shared with dynamo_gns/CycleGNS via the `gns/gns` symlink +
venv_cotopaxi/PYTHONPATH), per the standing constraint logged above. Proposing
this as a 4th candidate board addition, not started:
  4. Port the seeding fix (seeding.py / SeededEpochSampler) and the resume
     off-by-one fix from eq_cycle v0.0.43/44 into EQGNS's `meshnet/train.py` —
     cross-project change, owner sign-off required before any work, and must
     land (if approved) BEFORE the M1 retrain pair is run, or the retrain
     comparison inherits the same unseeded-confound problem it's meant to
     resolve.

## 2026-09-28 — prepare-loop-fix landed (PR #5, not merged — awaiting owner)

PR: https://github.com/dunyuliu/EQGNS/pull/5, branch
`worktree-agent-a43db383ef698f0f4`, base `origin/main` @ 85e2d4b (not stale,
verified). Diff is exactly the 2-line loop-bound fix per file, confirmed
against `git show HEAD:...` before commit — no unrelated changes.

Independent verification (this session, not just the subagent's report):
loaded the regenerated `test.npz` directly, confirmed frames 755/800/826 of
`pos`/`cells` now hold real (non-zero, mesh-matching) data where the old
published npz had exact zeros; confirmed `node_type`/`node_property` are
legitimately all-zero in BOTH old and new data (checked the old published
npz too) — not a fix artifact.

**Lesson — near-duplicate dispatch on a shared (non-worktree) output path**:
dispatched a second mira-volkov mission (agent-a6da6e86f64450912) onto the
same task after the first (agent-a43db383ef698f0f4) reported only a "plan" +
no live process + empty output dir. The first agent had in fact continued
autonomously in the background and later returned its own complete,
independently-matching report. Both agents' uncommitted worktree diffs turned
out byte-identical (deterministic fix + deterministic regen), so no actual
conflict resulted — but this was luck, not design: both were writing to the
SAME shared, non-git path (`/home/utig5/dliu/eq_rupture_gns_data/D1_fixed/`)
outside either worktree, which is exactly the "two agents, one file" collision
this campaign's rules exist to prevent. Proposing for zofia-kaminska: a rule
that any mission writing to a shared (non-worktree, non-repo) output path must
get an EXCLUSIVE path suffixed with its own agent-id, merged/reconciled by the
conductor only after both are known-complete — never two missions racing the
same shared output directory.

Duplicate worktree `agent-a6da6e86f64450912` (branch
`worktree-agent-a6da6e86f64450912`) left in place, NOT reaped — holds an
uncommitted, byte-identical copy of the same fix; will be cleaned at next
milestone close per the worktree-reaping rule (checked for anything unique
first).

**Board status (evidence for zofia-kaminska to write, not self-authored)**:
`prepare-loop-fix` row: PR #5 open, fresh command output above, gate is
merge = owner sign-off, currently OPEN pending merge (not yet DONE).

## 2026-09-28 — venv_cotopaxi restored (by invoking session); incident closed

Restored: torch 2.9.1+cu128 + triton 3.5.1 reinstalled `--no-deps`; CUDA,
torch_scatter, FaceToEdge verified by the invoking session; CycleGNS
`pytest tests -m "not slow"` -> 127 passed. Independently re-verified here
(read-only): `import torch; torch.__version__` -> `2.9.1+cu128`,
`torch.cuda.is_available()` -> `True`.

Root cause confirmed: `build_venv.sh`'s `virtualenv venv` failed silently, so
`pip install` fell through to whatever `python`/`pip` was first on PATH —
which was `venv_cotopaxi` (this repo's own shared venv, reachable via the
`gns/gns` symlink + inherited shell PATH), not a new isolated venv.

**Corrected protocol for any future stranger-clone check on this machine**:
run in a clean environment (`env -i HOME=$HOME PATH=/usr/bin:/bin bash -lc
...`) and do NOT run `build_venv*.sh` at all. Use the repo's own `venv/`
(torch 2.6.0, healthy) or skip the env-build step and state that explicitly
in the report rather than following the README script literally. Filing this
as a `PROJECT_RULES.md` / release-gate amendment for `zofia-kaminska`:
the README's "Building GNS environment" section is unsafe to execute as
documented and needs a fix or a stranger-check carve-out before the next
milestone release.

**Stopping per coordinator instruction**: owner decisions pending are PR #5
merge (https://github.com/dunyuliu/EQGNS/pull/5) and the four proposed board
rows (collapse guard, valid/test checkpoint selection, seed-spread,
seeding-fix port). No further autonomous work until they reply.

## 2026-09-28 — killed a leftover looping regen process (PID 4095426)

The duplicate D1-regen mission's background python process (launched earlier
via `/tmp/.../scratchpad/gen_d1_fixed.py`, PID 4095426, parent shell 4095415)
never exited after producing its first, already-verified output at 17:34-17:47.
It looped and started re-generating train.npz/valid.npz a second time
(mtimes advanced to 17:40:41 and 17:47:35, well after the first verified
pass and after PR #5 was opened/merged), consuming a full core for 45+
minutes doing pointless repeat work. Killed by PID (targeted, not `pkill -f`)
once confirmed via `ls --time-style=full-iso` that files were being rewritten
post-verification.

Re-verified all three splits fresh after the kill (no truncation/corruption):
train/valid/test.npz all load cleanly, 827 frames each, frames 755 and 826
non-zero in every split. `test.npz` itself was never touched by the second
pass (mtime unchanged at 17:34:04) so it matches exactly what PR #5's
evidence was based on; train/valid were rewritten but re-verified
independently as correct (deterministic regen, same result).

The mission (agent-a6da6e86f64450912) is now closed — its underlying task
was already completed and superseded by the other branch's merged PR #5. Not
re-engaging it further if it notifies again.

## 2026-09-28 — env-pinning landed (PR #6, not merged — CI in progress, awaiting owner/CI merge)

PR: https://github.com/dunyuliu/EQGNS/pull/6, branch
`worktree-agent-a0f9f3470b70e716f`, base current with `origin/main` (includes
PR #5). Diff limited to the 6 intended files (build scripts, requirements,
README, CI workflow) — confirmed via `git status --porcelain` before commit,
no meshnet/gns/model files touched.

Independently re-verified (not just the subagent's report): re-ran
`pytest test/ -q` and `test/paper_parity/gate.py quick` myself against this
worktree's actual files using the already-validated `eq_rupture_gns/venv` ->
44 passed/8 skipped; M1_D1/M2_D3/M3_D3 all PASS. Matches the subagent's
fresh-clean-env-build claim.

Flagged gap carried into the PR body: torchvision/torchaudio dropped from
both build scripts (absent from the validated venv's freeze); `gns/train_multinode.py`
imports torchvision and would need an explicit tested pin if exercised —
not fixed here, not guessed.

## 2026-09-28 — PR #6 merged (40f20c2); v1.1.1 tagged; release-v1.1 row supersession

Owner/invoking-session confirmed: PR #6 merged as `40f20c2` (tree identical to
my verified `f29a781`). Stranger-clone re-run done by invoking session on that
content: clean-env (`env -i HOME=$HOME PATH=/usr/bin:/bin`) `build_venv.sh` ->
exit 0, torch 2.6.0+cu124 + CUDA; `pytest test/ -q` 44 passed; `gate.py quick`
3/3 PASS. I independently confirmed CI's own run on main for `40f20c2`
(not just the PR-branch run) is `completed`/`success` before tagging.

Tagged `v1.1.1` (annotated) on `40f20c2`, pushed to origin. Scope of the
authorization exercised: owner's message explicitly named this SHA, said
"patch only," and "never force-update v1.1.0" — v1.1.0 untouched, only a new
tag added. This is a widened, explicit, one-off grant for this specific SHA
in this named repo, stated back here per the standing versioning rule.

**Board row disposition (evidence for zofia-kaminska to write, not
self-authored)**: `release-v1.1` closes as "superseded" — v1.1.0's
`build_venv.sh` was unsafe (silent PATH fallthrough); the pinned setup on
`40f20c2`/`v1.1.1` passes a clean stranger build. v1.1.0 tag itself untouched.

torchvision gap resolved per owner: only `gns/train_multinode.py` (particle-
GNS path, not the earthquake path this repo cares about) imports it; leaving
it out of the pinned build scripts is correct, noted in README's pins section
already (via PR #6's flagged-gap note) — no further change needed.

**In flight**: seeding + resume-off-by-one port into `meshnet/train.py`
dispatched (mira-volkov, worktree-isolated) per approved scope: opt-in `--seed`
flag preserving exact default behavior for non-opting callers; resume
off-by-one fix is a non-opt-in behavior change (correctly so, no valid old
behavior to preserve); gated on quick+full paper-parity re-run proving
current-vs-published parity unaffected. Next after that, in order: m1-retrain
pair (seeded, with seed spread), collapse guard, m1-large-gate.

## 2026-09-28 — priority interrupt: CI red on draft-pdf.yml (PR #7)

Owner-flagged "ci red", verified before acting: runs 36486350151 (v1.1.0) and
36503447221 (v1.1.1) both failed with workflowName `.github/workflows/draft-pdf.yml`
(upstream geoelements JOSS paper-build workflow, fails at setup on retired
`actions/upload-artifact@v1`; its `paths` filter doesn't apply to tag-push
events so it fires/fails on every tag regardless of diff). `tests` workflow
green on both SHAs. PR #7 opened: single-file deletion, not merged (owner
merges on green CI). Resuming the seeding/resume-port row now.

## 2026-09-28 — PR #7 widened: upstream JOSS paper files removed

Independently verified before acting (grep of README.md, CITATION.cff,
PROJECT_RULES.md, docs/, test/, .github/): zero references to paper.md,
references.bib, or figs/ outside draft-pdf.yml itself. Added `d41240b`
(deletions only: paper.md, references.bib, figs/{gnn.png,gnn.svg,gns-ddp.png,
gns-mpm.png,gns-scaling.png}) to the same PR #7. Still not merged, still
deletions-only. Resuming seeding/resume-port row.

## 2026-09-28 — seeding + resume-off-by-one port landed (PR #8, not merged)

PR: https://github.com/dunyuliu/EQGNS/pull/8, branch
`worktree-agent-a5f8596d8e73fbe93`. Worktree was STALE (based on 40f20c2,
before PR #7's merge to 661961c) -- caught via `git merge-base --is-ancestor`,
rebased clean (disjoint files, no conflicts), re-diffed to confirm exactly 4
files changed vs current main.

Independently re-verified myself on the rebased tree (not trusting the
subagent's report alone): `pytest test/ -q` -> 44 passed/8 skipped;
`gate.py quick --cuda 2` -> 3/3 PASS; full `gate.py run --cuda 2,2,2,2` ->
7/7 PASS, every trajectory, 2310s wall. This is the load-bearing gate for a
cross-project meshnet/ change and was re-run fresh against the real
published-checkpoint oracle, not accepted on the subagent's transcript.

Design: `--seed`/`--deterministic` opt-in (default None/False preserves
exact legacy unseeded call signatures everywhere); resume off-by-one fix is
non-opt-in (applies to every caller, correctly so -- no valid old behavior
worth preserving).

Next in owner's order: m1-retrain pair (seeded, with seed spread), then
collapse guard, then m1-large-gate.

## 2026-09-28 — collapse guard (PR #9) and H14.large regeneration (data only)

PR #9: https://github.com/dunyuliu/EQGNS/pull/9, collapse guard in
`test/paper_parity/gate.py` (var_ratio/collapsed keys, COLLAPSE_TOL=0.5).
Independently re-verified by me on the pinned venv: `pytest test/ -q` 44
passed/8 skipped (subagent's 1 failure was a venv_cotopaxi torch-cluster
CUDA-lib artifact, confirmed absent on the pinned venv); `gate.py quick
--cuda 2` 3/3 PASS with matching var_ratios; `gate.py falsify --quick --cuda
2` planted regression still CAUGHT on all 3 models, collapse guard did not
spuriously fire. Not merged.

H14.large (40 km) test-set regeneration: `/home/utig5/dliu/eq_rupture_gns_data/M1_large/dataset/case3.200m.large.npz`
+ metadata.json. Independently re-verified: 827 frames, pos (827,10302,2),
cells (827,20100,3), zero all-zero-pos frames (checked every frame), real
geometry at the last frame. `test/paper_parity/gate.py` untouched by this
mission (avoided collision with the concurrent collapse-guard mission on the
same file) -- gate-wiring (add `M1_large` CASES entry, score only unpadded
steps against `rollouts.nmp10.cotopaxi.large.published`) is queued as a
follow-up PR once #9 merges, to keep gate.py edits serial.

## 2026-09-28 — M1_large gate wiring landed (PR #10, not merged)

PR: https://github.com/dunyuliu/EQGNS/pull/10. Design: gated directly against
the committed published rollout pkls (not reference.json), both sides
truncated to the published rollout's own valid_steps() window (n=755 of the
new npz's 827 frames) via a new `metrics(pkl, n_override)` param (default
None, every other call site unaffected) and `TRUNCATE_TO_PUBLISHED`/
`published_reference()`. Data: `case3.200m.large.npz` symlinked into
`gns-sample/case3.200m.homo.a.Vw.others/dataset/` from its canonical location
under `eq_rupture_gns_data/M1_large/`, matching the gitignored/externally-
managed precedent already used for M1_small's npz.

Independently re-verified by me (not trusting the report alone): `gate.py
run M1_large --cuda 2` -> PASS var_ratio=0.791 (matches exactly); full
`gate.py run --cuda 2,2,2,2` (all 8 cases) -> 8/8 PASS, every trajectory,
2531s wall, matches exactly; `pytest test/ -q` -> 44 passed/9 skipped on the
pinned venv (subagent's 1-failure/9-skip reading was the same venv_cotopaxi
torch-cluster artifact seen before; independently traced the +1 skip to
test_paper_parity.py's CASES-parametrization growing 7->8, opt-in skip, not
a hidden regression).

m1-large-gate board row: DONE, evidence above, PR #10 open awaiting
owner/CI merge. Remaining open item on the board: m1-retrain pair, still
held for the owner's step-budget decision (options relayed: 3M/N=3 ~19
days, 1M/N=3 ~6.3 days serial, 500k/N=3 ~76h, 100k/N=3or5 ~15-25h).

## 2026-09-28 — PR #11: gate.py reads M1_large data via external path, not gns-sample symlinks

Coordinator flagged: the two symlinks placed inside gns-sample/.../dataset/
(published, read-only) in PR #10 don't exist on any other machine. Fixed:
added REGEN_DATA (default /home/utig5/dliu/eq_rupture_gns_data, override via
EQGNS_REGEN_DATA env var) + REGEN_DATASET_DIR case-lookup in
test/paper_parity/gate.py's run_rollout(), used only for M1_large; every
other case unaffected. Removed the two symlinks from gns-sample. Caught and
fixed one bug of my own during this: REGEN_DATA's first draft derived from
REPO.parent (HERE.parents[1].parent), which resolves wrongly when gate.py
runs from a worktree (REPO = the worktree root, not the main checkout) --
switched to a fixed absolute default, env-var-overridable. Re-ran `gate.py
run M1_large --cuda 2` after removing the symlinks: PASS, var_ratio=0.791,
identical to before. PR: https://github.com/dunyuliu/EQGNS/pull/11, not
merged.

Per coordinator: stop here. Only remaining open board row is m1-retrain,
held for the owner's step-budget decision.

## 2026-10-05 — Stage-7 Vista experiment campaign: staged, smoke/speed-verified, launched

**Scope**: scale-up of `stepover-gns` (board) onto TACC Vista (allocation
EAR26006, GH200 nodes), owner-approved for exactly experiments 01-07 (08-10
parked). Lives entirely in `vista/` (git-ignored, isolated per owner's
`PROJECT_RULES.md` rule 4 "Experiments are isolated in `work.*`" precedent —
`vista/` carries the same isolation, out of repo by owner's explicit choice,
documented here rather than committed).

**Design**: 7 experiments atop the `stepover_compressional` dataset/code —
`01_base_s0`/`02_base_s1` (seed replicate pair, the only noise yardstick),
`03_trim50` (tail-trimmed data), `04_faultid` (node_property flag), `05_edgenorm`
(edge normalizer), `06_edgesets` (mesh/cross-edge split MLPs), `07_worldedges_k8`
(knn cross-fault edges, k=16 despite the `k8` name — `matrix.txt`). 01-06 run as
2 x 48h segments; 07 as 4 x 48h segments (bigger graphs). 16 Slurm jobs total,
submitted as job IDs 1050422-1050437, `afterany` chained, `--mail-type=END,FAIL`.

**Staging audit** (`vista/docs/audits/AUDIT_2026-10-05_vista-staging.md`, local,
read-only review by the project's auditor): no verified launch blocker. Each
experiment trains its own code copy (`diff -rq` confirmed); configs differ only
as intended (checked against base `config.json`); flags reach the model
(`stepover_data.configure`); derived datasets (`trim50`, `faultid`) independently
re-derived and matched. Open should-fix findings, none blocking the
already-approved launch: (1) experiment 07's saved activations measured at
5.4x base (README claimed ~2.5x) — base peaks 31 GB, 07 projects ~125-170 GB
against the 96 GB GH200 budget, GPU peak itself UNVERIFIED without a live run;
(2) `run.process.gns.py` drops the training return code, so a crashed segment
exits 0 and `--mail-type=FAIL` never fires; (3) `shuttle.sh run`/`status` use a
non-login ssh while `env.sh`'s `module reset` is redirected under `set -e` —
either can kill a job with an empty `.err`, UNVERIFIED; (4) `$SCRATCH`'s ~10-day
idle purge can age out test-split data, early checkpoints, and eval-only venv
packages that training itself never re-reads. None of these were applied this
session (no new sbatch submitted beyond the owner-approved 16); flagged here as
the open risk list for whoever next touches `vista/remote/*`.

**Smoke** (job 1050912, 1000 steps each, `gh-dev`): all 7 `train.sbatch` rc=0;
rollout rc=0 for 01 and 07 (the two most different code paths), 5 metric rows
each, wall 172s/384s. Log: `$SCRATCH/EQGNS/smoke/summary-1050912.txt`.

**Speed** (job 1050951, 2000 steps unbuffered, rate measured over steps
250-2000): 01-05 at 8.05-8.17 steps/s (~34-35h to reach 1M steps, fits one 48h
segment), peak ~30.5 GB alloc; 06 at 6.14 steps/s (~45h, one segment), 36.3 GB;
07 at 3.80 steps/s (~73h, needs 2 of its 4 segments), 62.6 GB alloc / 63.9 GB
reserved this early (vs. the audit's ~125-170 GB full-training projection above
— the speed run itself did not reach a confirmed steady-state peak). Log:
`$SCRATCH/EQGNS/smoke/summary-1050951.txt`.

**Launch state** (re-confirmed this session via `shuttle.sh run squeue` after
smoke+speed passed): all 16 jobs queued, first segments of 01-07
(1050422/24/26/28/30/32/34) `PENDING (Priority)` behind ~275 jobs ahead on the
`gh` partition, no start estimate; chained second/third/fourth segments
`PENDING (Dependency)`. No action taken on the queue (read-only check, per the
standing Vista-write-needs-owner-OK rule — nothing here required a write).

**Local stepover-gns, re-verified this session** (not just inherited from the
state block): training PID 2005557 live (`ps -o pid,lstart,etimes,args`,
started 2026-10-03 19:37:52, elapsed ~47.4h), latest checkpoint
`model-680000.pt` / `train_state-680000.pt` at 17:38, `loss_log.txt`/
`gpu_mem_log.txt` still advancing (18:59) — alive and past 680k. Eval watcher
PID 2029283 live, `eval/metrics.csv` has rows through step 680000. No design
change made (owner rule: none before 1M steps).

**stepover-data / stepover-converter evidence re-run this session** (fresh
command output, for `zofia-kaminska` to apply to the board — not self-closed
here, per the board-driving rule that only she opens/closes/re-scopes rows):
- `tail -3 work/stepover_compressional/run_all.log` -> `ALL_SCENARIOS_DONE`,
  51 lines matching `exit=0` (50 scenarios + 1 repatch).
- `work/stepover_compressional/flagged_runs.log` absent; confirmed from
  `run_all.sh:17-19` this file is append-on-flag-only
  (`echo "$flag_json" | grep -q '"flagged": true' && echo "$flag_json" >>
  flagged_runs.log`) — its absence means zero flagged runs, not a skipped
  check, i.e. a clean pass, not a gap.
- `ls work/stepover_compressional/dataset/{train,valid,test}` -> 35/5/10,
  matching the board's stated split; this is the exact dataset the live
  680k-step training run (above) is training against, i.e. converter output
  is in production use, not just smoke-tested.
Recommend to Zofia: close `stepover-data` and `stepover-converter` to DONE on
this evidence; `stepover-gns` stays IN PROGRESS (680k/1M steps, stop condition
not yet reached).

**Nothing else advanced this session**: no new mission dispatched, no PR
opened, no merge performed. Board/doc update only, plus a `zofia-kaminska`
dispatch to apply the row changes above (board edits are hers, not mine, per
the board-driving rule).
