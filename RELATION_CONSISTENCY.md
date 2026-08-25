# Terminal time-aware VSRA for ZeroDiff

This branch keeps the original DFG data path and places VSRA at the generator
output. It restores the stable relation space from `exp/vsra` and coordinates
semantic and contrastive topology with each sample's diffusion timestep.

## Stable terminal relation space

Two learned projectors map visual and PaCo features into a shared relation
space. Before every generator update they are calibrated on the ordinary real
training batch with the original full-batch distance and angle RKD objectives:

```text
L_real = w_s RKD(E_v(x_real), S)
       + w_c RKD(E_v(x_real), E_c(C))
       + w_a RKD(E_c(C), S).
```

The projectors are then frozen. The main DFG branch samples one independent
timestep per sample, predicts `x_0_fake`, and applies terminal VSRA to that same
prediction:

```text
L_legacy = w_s RKD(E_v(x_0_fake), S)
         + w_c RKD(E_v(x_0_fake), E_c(C)).
```

There is no balanced relation episode, auxiliary fake branch, shared batch
timestep, class loop, or class-centering projection.

## Diffusion-timestep coordination

Let `u=t/(T-1)` and `d=2u-1`. The recommended schedule is

```text
w_class(t)    = 1 + rho d
w_instance(t) = 1 - rho d.
```

High-noise predictions therefore receive stronger semantic class-topology
supervision and weaker instance-level PaCo supervision; low-noise predictions
receive the reverse allocation. Each pair `(i,j)` is weighted by the geometric
mean `sqrt(w(t_i) w(t_j))`.

Pair space is split without per-class loops:

```text
M_class(i,j)    = 1[y_i != y_j]
M_instance(i,j) = 1[y_i == y_j and i != j].
```

The masks are disjoint. Cross-class pairs align with semantic distances and
within-class pairs align with PaCo distances. The final relation objective is

```text
L_relation = L_legacy
           + eta (w_s L_time_class + w_c L_time_instance).
```

`eta=0` exactly selects the restored VSRA anchor. The proposed time-aware method
uses `eta=0.1` by default, so the diffusion correction remains subordinate to
the stable VSRA objective.

## AWA2 runs

Restored VSRA anchor:

```bash
python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 1.0 \
  --rel_time_pair_weight 0.0
```

Time-aware terminal VSRA:

```bash
python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 1.0 \
  --rel_time_pair_weight 0.1 \
  --rel_time_mode class_up_instance_down \
  --rel_time_strength 0.5
```

Both runs use `gamma_dist=0`, projection dimension 512, distance ratio 1, and
angle ratio 2 in the AWA2 launcher, matching the original VSRA configuration.
