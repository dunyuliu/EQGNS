# PROJECT_RULES.md — EQGNS

Rule book for `dunyuliu/EQGNS` (fork of `geoelements/gns`, upstream remote),
published as Liu & Becker (2025), *Earthquake Rupture Dynamics From Graph
Neural Networks*, JGR Solid Earth, doi:10.1029/2025JB031981, archived at
Zenodo doi:10.5281/zenodo.17095311. The repo is now in post-publication
maintenance + new-experiment mode.

## Index
1. Published-paper state is frozen
2. Citation surfaces stay intact
3. Large/raw data directories never get committed
4. Experiments are isolated in `work.*`
5. Docs must match the drivers they document
6. Remotes and commit style
7. Rupture-analysis conventions are stated explicitly
8. `requirements.txt` vs `requirements.dl.txt` differ only in the numpy pin
9. A curated root — the template is the target, not the status quo

---

## 1. Published-paper state is frozen

`meshnet/train.py.published` is the exact snapshot of `train.py` used to
produce the paper's results. It is never edited. The current
`meshnet/train.py` may add inference-speed optimizations (rollout path only)
but must stay mathematically equivalent for training to the published
version — the `train()` and `validation()` functions are not to change
behavior. Tag `v1.0-jgr2025` marks the commit corresponding to the Zenodo
archive; the Zenodo archive (doi:10.5281/zenodo.17095311), not the current
`main`, is the authoritative record of "what produced the paper."

**Rationale**: a paper's results must remain reproducible from a fixed
snapshot indefinitely, independent of later maintenance on `main`.

**How to apply**: any PR touching `meshnet/train.py` states whether it
changes `train()`/`validation()` (forbidden without a version bump and
explicit note) or only `rollout()` (the sanctioned optimization surface, see
`/home/utig5/dliu/eq_rupture_gns/CLAUDE.md`). Never edit
`meshnet/train.py.published`.

## 2. Citation surfaces stay intact

`README.md`'s "How to cite" section lists, in order: the JGR paper, the
Zenodo software DOI, and the upstream Kumar & Vantassel (2023) JOSS paper.
`CITATION.cff` carries the JOSS paper as a `references:` entry alongside the
`preferred-citation` for the JGR paper. Any edit to `README.md` or
`CITATION.cff` must leave all three citations present and consistent with
each other.

**Rationale**: this is a fork of published, credited upstream work; dropping
a citation on an edit is silent plagiarism-by-omission.

**How to apply**: before merging a README or CITATION.cff change, grep both
files for `10.1029/2025JB031981`, `10.5281/zenodo.17095311`, and
`10.21105/joss.05025` — all three must still appear in each file where they
appeared before.

## 3. Large/raw data directories never get committed

`gns-sample/` (249GB), `work.test/`, `dataset_archive/`, `model/`, `venv*/`,
`misc/` are gitignored and stay that way. Raw datasets under these paths are
read-only inputs to experiments — nothing writes through them in place.

**Rationale**: multi-hundred-GB directories and generated model artifacts do
not belong in git history; treating raw data as read-only prevents an
experiment from silently corrupting the ground truth another experiment
depends on.

**How to apply**: `git check-ignore -v gns-sample work.test dataset_archive
model venv venv_cotopaxi misc` must report each as ignored; `git status
--porcelain` must never list a file under these paths as untracked-to-add.

## 4. Experiments are isolated in `work.*`

New/exploratory code lives in git-ignored `work.*` directories (e.g.
`work.cnn/`), excluded via `.gitignore` or `.git/info/exclude`. Experimental
code never lands in `gns/` or `meshnet/` until it is proven and consciously
merged as a reviewed change. `work.cnn/` carries its own `PROJECT_RULES.md`
scoped to that experiment — this rule book does not edit or govern its
contents.

**Rationale**: keeps `gns/` and `meshnet/` — the paper-adjacent, load-bearing
code — free of half-finished experimental branches.

**How to apply**: `git ls-files work.cnn work.test 2>/dev/null` should be
empty (both are ignored, not tracked); a change touching `gns/` or `meshnet/`
that originated in a `work.*` directory names the experiment it was promoted
from in the commit body.

## 5. Docs must match the drivers they document

`docs/data_preparation.md`, `docs/training.md`, and
`docs/rollout_and_analysis.md` document the actual entry points:
`train_cli.py`, `run.process.gns.py`, `scenario.rollout.py`, and
`meshnet/batch_rollout.py`. A doc's claimed flags, defaults, and file outputs
must match those scripts as they exist today.

**Rationale**: spec-drift between docs and drivers is exactly the failure
mode that costs the most time to a returning user or a fresh agent.

**How to apply**: when any of the four drivers above changes its CLI flags,
defaults, or output filenames, the corresponding doc changes in the same
commit. An audit greps each doc for `--flag` style references and diffs them
against `argparse`/`click` definitions in the named script.

## 6. Remotes and commit style

`origin` = `dunyuliu/EQGNS`, `upstream` = `geoelements/gns`. Commits use a
short imperative subject line (e.g. "Add rollout caching", not "Added" or
"Adding"). Author is Dunyu Liu (dl27583@eid.utexas.edu).

**Rationale**: keeps history skimmable and keeps fork/upstream sync
unambiguous when pulling from `upstream`.

**How to apply**: `git remote -v` shows exactly these two remotes at these
URLs; `git log --oneline -20` subjects read as imperative commands.

## 7. Rupture-analysis conventions are stated explicitly

`utils/plot.rupture.dynamics.py` fixes: `SLIPRATE_THRESHOLD=0.1` m/s for
rupture-time plots, `get_rupture_time(threshold=0.001)` for SCEC benchmark
comparison files, `dt=1/60` s, and a `+1.2` s time offset. These are not
universal constants — they are this script's calibrated choices. Any new
analysis script that computes a rupture time, slip rate, or aligns two time
series must state, in a comment or docstring, which of these conventions
(or which alternative, and why) it uses.

**Rationale**: two rupture-time definitions silently mixed across scripts
produce numbers that look comparable and are not (starter invariant 5: one
calibrated definition of "pass").

**How to apply**: grep a new analysis script for `threshold=`, `dt=`, and any
time-offset constant; if a rupture-time or slip-rate calculation appears
with none of these named, it is a Tier-2 finding routed to whoever owns that
script.

## 8. `requirements.txt` vs `requirements.dl.txt` differ only in the numpy pin

`requirements.txt` leaves numpy unpinned (local servers); `requirements.dl.txt`
pins `numpy==1.23.1`. This is the only intended difference between the two
files.

**Rationale**: two requirements files that silently diverge on more than the
one documented axis make it unclear which environment a bug report came
from.

**How to apply**: `diff requirements.txt requirements.dl.txt` — every line of
the diff must be the numpy pin (or trailing-newline noise); any other diff
line is a Tier-1 violation and gets documented here or reverted.

## 9. A curated root — the template is the target, not the status quo

This rule states where every root-level entry **belongs**, independent of
where it happens to sit today. The template:

- the four control docs (`PROJECT_RULES.md`, `PATHWAY_FORWARD.md`,
  `README.md`, `CLAUDE.md`)
- `license.md`, `CITATION.cff` (GitHub's citation widget only reads a
  root-level `CITATION.cff`; this file does not move), `.gitignore`,
  `Dockerfile`, the two requirements files (rule 8), and the two environment
  specs `enviornment.yml` / `gns_env.yml`
- `.circleci/`, `.github/` — community-health files (`AUTHORS.md`,
  `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `DCO.md`) live under `.github/`;
  GitHub reads them there equally
- `docs/`, split `docs/user/` (tutorials, how-to, reference, explanation —
  the published site builds from here) and `docs/dev/` (design notes, logs,
  archived release notes)
- `evals/` — fixtures and golden/reference data for the test gate. A slot
  that fills as earned (rule 1): absent until something is moved or added
  to it is not itself a violation
- `data/` — small reference data only; large data links to the shared
  dataset store, never committed. Same "fills as earned" status as `evals/`
- `gns/`, `meshnet/` — the two packages. They stay at root: an external
  project symlinks them, so root is their stable address
- `scripts/` — every root-level `.sh`/`.py` entry point (`train.sh`, `run.sh`,
  `resume.train.sh`, `asp.rollout.sh`, `module.sh`, `render.sh`,
  `render.cpu.sh`, `build_venv.sh`, `build_venv_frontera.sh`,
  `start_venv.sh`, `train_cli.py`, `run.process.gns.py`,
  `scenario.rollout.py`), plus `utils/` and `slurm_scripts/` as
  subdirectories (`scripts/utils/`, `scripts/slurm_scripts/`)
- `tests/` — never `test/`: the name shadows Python's stdlib `test` package

No other root-level entry is added without updating this list in the same
change. No tracked file, anywhere in the tree, exceeds 5MB.

**The repo does not yet match this template.** Every entry below is an open
violation, not an accepted exception — each is tagged with the move that
clears it and who owns the coupled edit. This rule does not authorize moving
the files; it names the target each move lands on:

- **in-repo move, same-PR, no coupled edit found**: `AUTHORS.md`,
  `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `DCO.md` -> `.github/`;
  `train.sh`, `run.sh`, `resume.train.sh`, `asp.rollout.sh`, `module.sh` ->
  `scripts/` — no doc or CI reference to these root paths was found at the
  time of writing.
- **in-repo move, touches CI in the same commit, owner iris-vermeulen**:
  `test/` -> `tests/`, with `.github/workflows/tests.yml`,
  `.circleci/config.yml` (both run `pytest test/ ...`), and any
  `python -m test.` / `from test import` reference updated in the same
  change; `utils/` -> `scripts/utils/` and `slurm_scripts/` ->
  `scripts/slurm_scripts/`, with every in-repo reference (docs, other
  scripts, test fixtures) updated in the same change — both directories are
  named from inside `test/` and from `scripts/run_m1_retrain_eval_queue.sh`.
- **in-repo move, touches a doc in the same commit, owner anya-petrov**:
  `render.sh`, `render.cpu.sh` -> `scripts/` (named by root path in
  `docs/rollout_and_analysis.md`); `example/` -> `docs/user/examples/`
  (it is a usage tutorial, not a test fixture, so it does not go to
  `evals/`) — no in-repo reference to `example/` was found besides this
  rule book, but the move still lands with any new doc link in the same
  commit.
- **breaks outside paths, waits for the owner, not a move this rule
  authorizes**: `build_venv.sh`, `build_venv_frontera.sh`, `start_venv.sh`,
  `train_cli.py`, `run.process.gns.py`, `scenario.rollout.py` -> `scripts/`
  — external job scripts invoke these by root path; moving them is a
  cross-repo breaking change until the owner coordinates it.
- **pending owner, not a path move — rewrites a tracked asset**:
  `docs/img/meshnet.gif` is currently 10.7MB, over the 5MB cap; converting
  it to Git LFS or external hosting rewrites history for that path and
  needs the owner's sign-off, not just a PR. This is the only size
  exception the gate carries.

The `docs/` split into `docs/user/` and `docs/dev/` is part of the template
but is not a root-level violation the gate below can see (`docs/` is already
an allowed root entry); it is a standing to-do for whoever next touches
`docs/` (anya-petrov), not blocking on this PR.

**Rationale**: a root with no agreed membership accretes one script at a
time until a build artifact is indistinguishable from an entry point;
stating the template as the target — rather than whitelisting whatever is
currently there — is what lets the gate fail honestly instead of laundering
the status quo into "compliant."

**How to apply**: `scripts/check_root.py` enforces the template above
directly, with exactly one exception mechanism left: a named, tagged entry
in `PENDING_LARGE_FILES` for an oversized tracked file awaiting owner
sign-off (today: `docs/img/meshnet.gif`). Every other current violation
listed above fails the gate until its move lands — the gate is expected to
be red until the specialists above land their moves, and that is the
correct state, not a bug in the check. A PR adding a new root-level file
either lands on the allow-list or updates this rule in the same commit — an
unlisted new entry fails the gate with no exception available. A PR
clearing a listed violation makes the matching coupled edit (CI config,
doc) in the same commit as the move, and removes that entry from this rule
in the same commit — leaving a cleared entry listed here is itself a
rule-9 violation the next reader will trust wrongly.
