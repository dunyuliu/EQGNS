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
9. A curated root — whitelist, not a preference

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

## 9. A curated root — whitelist, not a preference

The repo root carries only: the four control docs (`PROJECT_RULES.md`,
`PATHWAY_FORWARD.md`, `README.md`, `CLAUDE.md`), `license.md`, `CITATION.cff`
(GitHub's citation widget only reads a root-level `CITATION.cff`; this file
does not move), `.gitignore`, `Dockerfile`, the two requirements files (rule
8), and the two environment specs `enviornment.yml` / `gns_env.yml` — plus
the ten tracked top-level directories: `.circleci/`, `.github/`, `docs/`,
`example/`, `gns/`, `meshnet/`, `scripts/`, `slurm_scripts/`, `test/`,
`utils/`. No other root-level entry is added without updating this list in
the same change. No tracked file, anywhere in the tree, exceeds 5MB.

Everything else currently at root is a named, flagged exception, not an open
invitation:
- `AUTHORS.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `DCO.md` — GitHub
  reads these equally from `.github/`; no reference to their root path was
  found elsewhere, so they can move there on their own.
- `train.sh`, `run.sh`, `resume.train.sh`, `asp.rollout.sh`, `module.sh` —
  no doc or CI reference to these root paths was found at the time of
  writing; they can move to `scripts/` on their own.
- `build_venv.sh`, `build_venv_frontera.sh`, `start_venv.sh`,
  `train_cli.py`, `run.process.gns.py`, `scenario.rollout.py`, `render.sh`,
  `render.cpu.sh` — root-level entry points invoked by path in `README.md`
  (and, for the three Python drivers, named in rule 5). A move to
  `scripts/` is legitimate but only together with the matching README.md
  (and rule 5, if the Python drivers move) edit in the same commit (rule
  11) — never the file move alone.
- `docs/img/meshnet.gif` is currently 10.7MB, over the 5MB cap. It is a
  violation to fix (Git LFS or external hosting), not a precedent for a
  second large asset.

**Rationale**: a root with no agreed membership accretes one script at a
time until a build artifact is indistinguishable from an entry point;
GitHub-recognized alternate locations exist for exactly the community-health
files above and cost nothing to use.

**How to apply**: `scripts/check_root.py` enforces this list and the 5MB
cap, and report-only lists other worktrees and already-merged branches for
manual tidy-up (never fails the gate on those). A PR adding a root-level
file either lands on this list or edits this rule to add it, in the same
commit. A PR moving one of the named exceptions above routes its doc-sync
half (README.md / rule 5) to whoever owns docs, not to this rule.
