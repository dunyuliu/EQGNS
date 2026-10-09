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

## Example figures

These are upstream granular-flow figures (not EQGNS rupture output) kept
from the original example for reference.

![initial condition](../img/initial_vel.png)

> Problem setup: a multi-layered granular column (10 particle groups), each
> given a different initial x-velocity to be recovered by the inversion.

![ground truth MPM](../img/true_ani.gif)

> Ground-truth deposit evolution for the configuration above, computed with
> the material point method (MPM).

![loss history](../img/loss_hist.png)

> Inverse-optimization loss (MSE) vs. iteration, over the `nepoch=30` loop in
> `config.toml`.

![velocity history](../img/vel_hist.png)

> Recovered initial x-velocity per particle group across iterations (color =
> iteration), converging from the initial guess (grey) toward the true
> profile (black).

![GNS prediction](../img/pred_ani.gif)

> GNS rollout using the velocity estimated at iteration 29, showing good
> agreement with the ground-truth MPM simulation above.
