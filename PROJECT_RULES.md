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
4. Experiments are isolated in `scratch/` and `runs/`
5. Docs must match the drivers they document
6. Remotes and commit style
7. Rupture-analysis conventions are stated explicitly
8. One `requirements.txt`, every entry pinned with `==`
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

Raw data lives under git-ignored `data/` (`data/gns-sample/`, 249GB, and
`data/dataset_archive/`); experiment outputs under `runs/<YYYYMMDD>_<slug>/`;
throwaway work under `scratch/`. All are gitignored and stay that way. Raw
datasets under `data/` are read-only inputs — nothing writes through them in
place. Two ignored venvs stay at the root because their compiled extensions
bake the path and other projects activate them by path: `venv/` and
`venv_cotopaxi/` (moving either = a rebuild, owner-only). A root symlink
`gns-sample -> data/gns-sample` stays until no outside script reads the old
path (board row untracked-root-reorg).

**Rationale**: multi-hundred-GB directories and generated model artifacts do
not belong in git history; treating raw data as read-only prevents an
experiment from silently corrupting the ground truth another experiment
depends on.

**How to apply**: `git check-ignore -v data runs scratch venv venv_cotopaxi`
must report each as ignored; `python3 scripts/check_root.py` lists any
untracked root entry outside the template (report-only).

## 4. Experiments are isolated in `scratch/` and `runs/`

New/exploratory code lives in git-ignored `scratch/` (or a dated
`runs/<YYYYMMDD>_<slug>/` when it produces results). Experimental
code never lands in `gns/` or `meshnet/` until it is proven and consciously
merged as a reviewed change. An experiment directory may carry its own
`PROJECT_RULES.md` scoped to it; this rule book does not govern its contents.

**Rationale**: keeps `gns/` and `meshnet/` — the paper-adjacent, load-bearing
code — free of half-finished experimental branches.

**How to apply**: `git ls-files scratch runs 2>/dev/null` should be
empty (both are ignored, not tracked); a change touching `gns/` or `meshnet/`
that originated in `scratch/` or `runs/` names the experiment it was promoted
from in the commit body.

## 5. Docs must match the drivers they document

`docs/user/data_preparation.md`, `docs/user/training.md`, and
`docs/user/rollout_and_analysis.md` document the actual entry points:
`scripts/train_cli.py`, `scripts/run.process.gns.py`, `scripts/scenario.rollout.py`, and
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

`scripts/utils/plot.rupture.dynamics.py` fixes: `SLIPRATE_THRESHOLD=0.1` m/s for
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

## 8. One `requirements.txt`, every entry pinned with `==`

There is a single `requirements.txt` for every environment (local servers
included). Every line is pinned with `==` — no unpinned minimum version, and
no transitive `pip freeze` dump: an `nvidia-*` CUDA wheel pinned verbatim
would break CPU CI.

**Rationale**: a second requirements file (or an unpinned line in the one
file) drifts silently from the stack a bug report actually ran on; one
pinned file is the only thing that can be diffed against what shipped.

**How to apply**: a check in `scripts/check_root.py` fails on any
`requirements.txt` line without `==`. (As of this writing that check does
not exist yet; it ships with the `no-conda-docs-hygiene` PR. Until it lands,
`grep -vE '==' requirements.txt` returning empty is the manual gate.)

## 9. A curated root — the template is the target, not the status quo

This rule states where every root-level entry **belongs**, independent of
where it happens to sit today. The template:

- the four control docs (`PROJECT_RULES.md`, `PATHWAY_FORWARD.md`,
  `README.md`, `CLAUDE.md`)
- `LICENSE`, `CITATION.cff` (GitHub's citation widget only reads a
  root-level `CITATION.cff`; this file does not move), `.gitignore`,
  `Dockerfile`, `requirements.txt` (rule 8), and the two environment specs
  `enviornment.yml` / `gns_env.yml`
- `.github/workflows/` — CI job definitions only. `.circleci/` is retired
  (one CI provider). The community-health files that used to live under
  `.github/` (`AUTHORS.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`,
  `DCO.md`, `ISSUE_TEMPLATE/`, `pull_request_template.md`) are retired, not
  relocated — an MIT-licensed repo needs neither CONTRIBUTING nor DCO
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

**The repo now matches this template.** `scripts/check_root.py` exits 0.
Every move below has landed, each in the commit that made its coupled edit:

- `AUTHORS.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `DCO.md` ->
  `.github/` (subsequently retired, not relocated, under
  `no-conda-docs-hygiene` — an MIT-licensed repo needs neither);
  `render.sh`, `render.cpu.sh` -> `scripts/` (with
  `docs/user/rollout_and_analysis.md` updated in the same commit);
  `example/` -> `docs/user/examples/` — landed, owner anya-petrov.
- `test/` -> `tests/`, with `.github/workflows/tests.yml` and
  `.circleci/config.yml` (both now run `pytest tests/ ...`) updated in the
  same commit; `utils/` -> `scripts/utils/` and `slurm_scripts/` ->
  `scripts/slurm_scripts/`, with every in-repo reference (docs, other
  scripts, test fixtures) updated in the same commit — landed, owner
  iris-vermeulen.
- `train.sh`, `run.sh`, `resume.train.sh`, `asp.rollout.sh`, `module.sh`,
  `build_venv.sh`, `build_venv_frontera.sh`, `start_venv.sh`,
  `train_cli.py`, `run.process.gns.py`, `scenario.rollout.py` ->
  `scripts/` — landed; the venv scripts split `SCRIPT_DIR` vs `REPO_ROOT` so
  the venv and `requirements.txt` still resolve at the repo root, and the
  Python drivers resolve `REPO_ROOT` from `__file__` and inject it into
  subprocess `PYTHONPATH`, so external job scripts invoking these by the old
  root path keep working.
- `docs/` split into `docs/user/` (tutorials, how-to, reference,
  explanation) and `docs/dev/` (design notes, logs, internal status) —
  landed, owner anya-petrov.

**Standing exception**: `docs/img/meshnet.gif` is 10.7MB, over the 5MB cap.
Owner-approved: it stays in git, tracked normally, no Git LFS and no history
rewrite. This is the only exception the gate carries (`PENDING_LARGE_FILES`
in `scripts/check_root.py`).

**Rationale**: a root with no agreed membership accretes one script at a
time until a build artifact is indistinguishable from an entry point;
stating the template as the target — rather than whitelisting whatever is
currently there — is what let the gate fail honestly instead of laundering
the status quo into "compliant," until every move above landed.

**How to apply**: `scripts/check_root.py` enforces the template above
directly, with exactly one exception mechanism left: a named, tagged entry
in `PENDING_LARGE_FILES` for an oversized tracked file the owner has
approved to keep (today: `docs/img/meshnet.gif`). A PR adding a new root-level file
either lands on the allow-list or updates this rule in the same commit — an
unlisted new entry fails the gate with no exception available.
