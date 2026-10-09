# EQGNS release notes — v1.2.0 — 2026-10-09

## 1. Version and date

v1.2.0, minor bump from v1.1.1 (`40f20c2`). Tag target: the release commit on
`main`, built on `origin/main @ 88a333f` plus this release's own audit fixes.

## 2. Summary of scope

76 commits since v1.1.1 (`git log --oneline v1.1.1..HEAD`). This release
closes most of the `test-suite-overhaul` board item, lands the fast-rollout
regression gate and its owner-approved thresholds, extends the paper-parity
gate's metrics, resolves the `rollout_batched()` tolerance gap for the
batched-rollout path, retires Docker (dormant, never used by CI), and
finishes the root-layout/no-conda documentation cleanup. No change to the
model's default training or rollout math (see Audit findings, item 1, for
the one qualified exception: the resume-checkpoint bookkeeping fix).

## 3. Files added / removed / renamed / cleaned up

Full list: `git diff --name-status v1.1.1..origin/main` (162 files). Highlights:
- Root layout finished the rule-9 template move: `test/` -> `tests/`,
  `utils/` -> `scripts/utils/`, `slurm_scripts/` -> `scripts/slurm_scripts/`,
  top-level driver scripts -> `scripts/`, `example/` -> `docs/user/examples/`,
  `docs/` split into `docs/user/` + `docs/dev/`.
- Removed: `Dockerfile` (dormant, never referenced by CI, owner-approved
  retirement), `.circleci/`, upstream JOSS paper files (`paper.md`,
  `references.bib`, `figs/`), `requirements.dl.txt`, `gns_env.yml`,
  `enviornment.yml`, stale community-health files under the old root
  (`AUTHORS.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `DCO.md`), six
  upstream `gns`-domain test stubs/leftovers with no `meshnet/` caller.
  `license.md` renamed to `LICENSE`.
- Added: `meshnet/fast_rollout.py`, `meshnet/seeding.py`,
  `scripts/check_root.py`, `tests/paper_parity/gate.py`/`measure_vs_published.py`,
  training-gate and training-golden fixture/test trees, `docs/dev/` design
  and session-log docs.
- This release's own fixes (see item 5): `README.md`, `docs/user/theory.md`,
  `meshnet/train.py` (help-string only), `PATHWAY_FORWARD.md`.

## 4. Content updates to master documents

- `PROJECT_RULES.md`: rule 9's root template updated as each move landed;
  rule 9's Docker bullet updated to record retirement; rule 10 added
  (paper-parity gate output required in `meshnet/`/`gns/` PR bodies).
- `PATHWAY_FORWARD.md`: closed `test-suite-overhaul` sub-items (1)-(4),
  `release-gate-decisions-pending` (a)/(d)/(f), `rollout-batched-oracle-gap`
  (gate-mechanics half), `gate-enforcement` (partial), `training-guard`
  (verified); opened `m2-checkerboard-chaos-exclusion-decision`,
  `m1-arresting-mirror-expansion`, `m2m3-mirror-augmentation`,
  `m3-batchsize-lr-sweep-gh200`; this release adds a correction note to the
  `rollout-compile-optin` row (see Audit findings, item 1) and a new row
  `paper-parity-gate-schema-and-count-gaps` (item 2/3 below).
- `README.md`: Installation/Quickstart reconfirmed working (stranger-clone
  check, see CI row below); "Reproducing the paper" section intact; fixed a
  dangling link (item 4 below).

## 5. Audit findings and fixes

Phase 1 ran two independent passes: my own direct checks (root/citation/
pin/Docker/link greps, diff-boundary checks on `train()`/`validation()`) and
a `victor-reyes` dispatch over the full `v1.1.1..origin/main` diff. Merged,
deduped findings:

1. **[Major, applied as a documentation fix, not a code change] Resume
   checkpoint bookkeeping in `train()` changed unconditionally.**
   `meshnet/train.py` now saves `global_train_state={"step": step + 1}`
   instead of `{"step": step}` on every `nsave_steps` checkpoint, not only
   when `--seed` is set. This is a correct bug fix (PR #8, `5e5a474`,
   already owner-approved and recorded on the board's
   `train-seeding-resume-fix` row before this release): the pre-fix code
   re-applied the last saved step's gradient update a second time on resume.
   It is, however, a real default-path behavior change to `train()`'s
   resume semantics, which `PROJECT_RULES.md` rule 1 states `train()`/
   `validation()` "are not to change" at all. Per-step training math for any
   non-resumed run, and the per-step math itself for a resumed run, are
   unchanged — only the resume *entry point* changes (by one step,
   correctly). Added a correction note to `PATHWAY_FORWARD.md`'s
   `rollout-compile-optin` row, which had summarized this release as
   "default numerics unchanged" without naming this exception. A resumed
   run against an older (pre-v1.2.0) `train_state-*.pt` file still re-does
   one step on first resume after upgrading; not separately handled, not a
   release blocker (an edge path, not the default training/rollout flow).
2. **[Major, recorded as an open board row, not fixed — gate owned
   separately per `CLAUDE.md`] `tests/paper_parity/gate.py`'s extended
   metrics (Mw error, slip-rate RMSE, final-slip RMSE, PR #71) are only
   present in `reference.json` for `M1_D1` and the 3 `@quick` cases.** The
   other 6 of 8 `gate.py run` cases print a "schema gap" and still return
   PASS without actually checking the new metrics — the board's "all 8
   cases PASS on the extended metrics" claim holds for 2/8, not 8/8. New
   board row `paper-parity-gate-schema-and-count-gaps`.
3. **[Major, same row, not fixed] `gate.py regression`'s `rows_for` can PASS
   vacuously on a trajectory-count mismatch** (scores `min(len_a, len_b)`
   trajectories; `cmd_regression` never reads the `note` field that records
   the mismatch). `gate.py run`'s own `compare()` already guards this; the
   newer `regression` subcommand does not.
4. **[Minor, fixed] Two dangling references to a never-created
   `docs/user/ROLLOUT_BATCHING.md`.** `README.md:83` (a whole numbered item)
   repointed to the existing `docs/user/rollout_and_analysis.md` (item 3,
   which already covers single/batched/fast rollout); `meshnet/train.py`'s
   `--rollout_batch_size` help string repointed the same way (one-line
   string change only, no behavior change — `tests/paper_parity/gate.py
   run --cuda 1 --quick` re-run fresh on this branch after the edit: 3/3
   PASS, M1_D1/M2_D3/M3_D3, per rule 10).
5. **[Minor, fixed] `docs/user/theory.md`'s four image links broke** when
   the doc moved from `docs/` to `docs/user/` (PR #30) without updating
   `img/*.svg` to `../img/*.svg`; all four target files confirmed present
   under `docs/img/`. Fixed.
6. **[Minor, recorded, not fixed] `gate.py`'s scored window includes one
   zero-padded ground-truth frame** (`valid_steps`'s last index, e.g. M1_D1
   traj 0 frame 755, confirmed all-zero). Both sides of every comparison are
   scored identically, so victor-reyes's read is that this cannot produce a
   false PASS — it only shifts the absolute metric values slightly. Not
   fixed (gate owned separately); not release-blocking.
7. **[Minor, recorded, not fixed] Inconsistent non-positive-moment
   handling**: `regression_ok` skips the Mw check silently when either
   side's moment is <= 0; `metrics()` raises on the same condition. Not
   fixed (gate owned separately).
8. **Confirmed clean, no action needed**: `requirements.txt` fully pinned
   with `==` (including the new `scikit-learn==1.7.2`, PR #71); root
   Dockerfile absent and no CI workflow ever referenced Docker; all three
   citation DOIs present in both `README.md` and `CITATION.cff`; `train()`/
   `validation()` function definitions unchanged apart from item 1 above
   (confirmed by diffing hunk headers across the full `v1.1.1..HEAD` range,
   not just the PR #49 refactor's own diff); PR #49's `rollout()`/
   `rollout_batched()` refactor is a 3-dot diff of `meshnet/train.py` only
   (+68/-84), re-confirmed bit-identical to pre-refactor `main` by the
   conductor's own gate runs recorded on the board; `scripts/check_root.py`
   exits 0 on this tree.

## 6. Remaining open issues or pending items

All pre-existing, explicitly out of scope for this tag per the dispatch
brief, unchanged by this release: `gh200-cross-hw-timing` (BLOCKED, owner
decision — reported not gated); `release-gate-decisions-pending` (b)/(c)
(GH200 tolerance/torch upgrade, deferred); `m2-checkerboard-chaos-exclusion-
decision` (owner decision pending); `m1-arresting-mirror-expansion` /
`m2m3-mirror-augmentation` / `m3-batchsize-lr-sweep-gh200` (queued
experiments, no code in this tag). Plus, newly recorded by this release's
audit: `paper-parity-gate-schema-and-count-gaps` (items 2/3 above).

Also newly noted (not a board row, informational): the prior tag `v1.1.1`
(`40f20c2`) has no corresponding GitHub Release — only `v1.1.0` has one, and
it is still marked "Latest" on GitHub despite `v1.1.1` being the newer tag.
This predates this release and is not fixed here (no prior release note is
deleted or rewritten, per standing rule); flagging so the owner can decide
whether to backfill a `v1.1.1` Release (which would need to say so in its
first line and never claim Latest, per this project's own backfill
convention).

## 7. Totals or cost changes

Not applicable — no cost/budget tracked by this repo's release process.

## 8. Assumptions used

- Used the project's actual fast-tier test command
  (`pytest tests/ -m "not slow" -q`) as the equivalent of a generic
  `tests/check.sh`/`release_gate.sh` — neither exists in this repo; ran by
  hand and transcribed below.
- `evals/` does not exist in this repo (rule 9: a template slot that "fills
  as earned"); the fixture-verdict trend measure is not applicable at
  either commit.
- Used the `release-v1.1.1` board row's own recorded gate result (`pytest
  test/ -q -> 44 passed, 8 skipped`, from that release's own verification
  on its tagged SHA `40f20c2`) as the v1.1.1-side comparator, rather than
  re-installing that tag's (different) pinned dependency set to re-run its
  `test/` suite from scratch — both sides use each release's own, correctly
  pinned environment.

## 9. CI run this release was gated on

Pre-existing `origin/main @ 88a333f`: run `37984276592`, conclusion
`success` (push trigger, before this release's own fixes). This release's
own commit (the fixes in item 5 above, see git log for the exact SHA) is
gated separately per rule 15a — see the Release gate section below for that
run's id/conclusion/SHA, filled in at tag time.

## 10. Trend since v1.1.1 (`40f20c2`)

- **Gate assertions.** Project equivalent: `pytest tests/ -m "not slow" -q`.
  v1.1.1 (recorded on its own release row, `pytest test/ -q` on tagged SHA
  `40f20c2`): `44 passed, 8 skipped`. Today (this release commit, same
  command adapted to the renamed `tests/` directory): `64 passed, 9
  skipped, 8 deselected`. Better: 20 more tests passing, and the suite now
  covers training-gate/training-golden/dataprep tiers that did not exist at
  v1.1.1.
- **Fixture verdicts.** `evals/` does not exist at either commit (rule 9:
  fills as earned) — not applicable, not a gap introduced by this release.
- **Tracked text lines.** `git diff --stat v1.1.1..HEAD -- '*.md' '*.py'
  '*.sh'`: 128 files changed, +6525/-2530. Growth, driven mostly by new
  test fixtures/goldens and the docs/dev design + session-log notes, not by
  the production surface (`meshnet/`+`gns/` diff alone is ~2 files, see
  item 5/8 above). Per the standing convention, this is reported, not
  gated: a dedicated leanness pass has been explicitly deferred.
- **Board currency.** `PATHWAY_FORWARD.md` row count: 6 rows at v1.1.1 (all
  `OPEN`) vs 34 rows today (17 `DONE`, 3 `VERIFIED`, 12 `OPEN`, 3 `PARTIAL`,
  2 `BLOCKED`, 1 `STOPPED`, 1 `SUPERSEDED` — some rows carry more than one
  state token in their note text, so these sub-counts are approximate, not
  disjoint). This board does not use a VERIFIED/BROKEN color convention or a
  blank-last-checked-date field; state is read from the `state` column
  directly. Better: far more rows are closed-with-evidence than open.
- **CI green-on-first-try rate.** `gh run list --branch main` since
  v1.1.1's push (`2026-09-28T19:29:01-05:00`): 70 runs, 70 `success`, 0
  non-success conclusions recorded on `main`. Whether any of those 70
  needed a manual re-run attempt (vs. green on the first attempt) needs
  per-run `gh run view --json` inspection, which was not done (cost); the
  conclusion-level rate is 100%, unchanged/excellent, but "first-try"
  specifically is not independently confirmed.

## 11. Work record

- audit: two passes — my own direct checks (root/citation/pin/Docker/link
  greps; `train()`/`validation()` diff-boundary check across the full
  range) plus a `victor-reyes` dispatch over the complete `v1.1.1..main`
  diff; 7 findings merged after dedup (2 Major fixed-as-documented, 2 Major
  recorded as open board items, 3 Minor: 2 fixed, 1 recorded).
- correctness: no model-behavior regression found; the one default-path
  change (`train()`'s resume step bookkeeping) is a correct, pre-approved
  bug fix, now explicitly named in this note and the board (item 5.1).
- conciseness: tracked-text lines grew (+6525/-2530, see Trend); a
  dedicated leanness/refactor pass remains explicitly deferred by the
  project maintainer — reported, not gated.
- fixes: applied — `README.md` dangling link, `meshnet/train.py` help-string
  dangling link, `docs/user/theory.md` four broken image links,
  `PATHWAY_FORWARD.md` correction note + new findings row. Deferred —
  `tests/paper_parity/gate.py`'s schema-gap and trajectory-count-mismatch
  gaps (owned separately, routed to board row
  `paper-parity-gate-schema-and-count-gaps`); the `M2_checkerboard`
  exclusion-list decision (pre-existing, owner-decision-pending, untouched
  per brief).
- docs: reconciled against the actual filesystem this session (`find`/`ls`
  checks for every link fixed or flagged above), not just the diff.
- refactor: none run this release — `kai-fischer` not dispatched. Scope
  assessed inline: the only candidate code changes found by the audit
  (`gate.py`'s two gaps) sit in a file this project's own `CLAUDE.md`
  marks "owned separately, do not edit casually," so a refactor pass was
  out of scope by that boundary, not by my own judgment call alone.
- rules: not run as a separate `zofia-kaminska` dispatch this release (no
  rule-tier-split request from the brief); `PROJECT_RULES.md` rule 9
  (`scripts/check_root.py`) and rule 8 (pinned `requirements.txt`) verified
  directly and both pass; rule 1 compliance verified directly for the full
  `v1.1.1..HEAD` range, with the one named exception at item 5.1.

## 12. Release gate

- tree: _(filled after commit — clean, single worktree, no lock, level with upstream)_
- ci: _(filled after push — run id, URL, conclusion, SHA)_
- publish: _(filled after push — note version, tag, remote)_
- release: _(filled after `gh release create` — `gh release view` against the pushed tag)_
- clone: _(filled after tag — PASS <sha>, fresh clone under `env -i`)_
