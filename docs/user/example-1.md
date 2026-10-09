# Inverse problem example (upstream, granular flow — not part of the EQGNS rupture workflow)

`docs/user/examples/inverse_problem/` carries an upstream example — gradient-based
inversion for initial velocity in a granular-flow column collapse, using
GNS's differentiability and automatic differentiation — inherited from
[geoelements/gns](https://github.com/geoelements/gns). It is not part of
the EQGNS earthquake-rupture workflow (see the README's "EQGNS workflow"
section for that).

The example data, configuration format (`config.toml`), and the gradient
checkpoint / resume mechanics it demonstrates are documented upstream; the
code is unchanged here. Kept for reference since the same differentiable
rollout mechanics underlie `meshnet/train.py`.
