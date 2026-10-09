# EQGNS — project pointer

EQGNS is a Graph Network-based Simulator (mesh-based, MeshNet) for 2D
earthquake dynamic rupture, the code behind Liu & Becker (2025), *JGR Solid
Earth* (doi:10.1029/2025JB031981). Start at [README.md](README.md).

## Where things live

- **Project rules** (frozen-paper-state, data/commit conventions, root
  layout): [PROJECT_RULES.md](PROJECT_RULES.md).
- **Status board** (open work, owners): [PATHWAY_FORWARD.md](PATHWAY_FORWARD.md).
- **User docs** (data prep, training, rollout, examples): `docs/user/`.
- **Dev notes** (rollout-speed work, retrain campaigns): `docs/dev/`.
- **Test gates**: `tests/README.md` for the suite; `tests/paper_parity/`
  for the paper-reproduction gate (owned separately, do not edit casually).
- **Environment**: `requirements.txt` (pinned, no conda) + `scripts/build_venv.sh`
  / `scripts/build_venv_frontera.sh`; see the README's Installation section.

## Working in this repo

Follow `PROJECT_RULES.md` for commit style, what never gets committed, and
where experiments live. Anything about the historical `rollout()` speed
optimizations belongs in `docs/dev/ROLLOUT_SPEED.md`, not here.
