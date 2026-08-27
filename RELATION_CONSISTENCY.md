# Time-aware multi-granularity relation consistency for ZeroDiff

This branch keeps the original DFG data path, calibrates the stable relation
space from `exp/vsra`, and coordinates semantic and contrastive topology at
matched diffusion timesteps.

## Stable terminal relation space

Two learned projectors map visual and PaCo features into a shared relation
space. Before every generator update they are calibrated on the ordinary real
training batch with the original full-batch distance and angle RKD objectives:

```text
L_real = w_s RKD(E_v(x_real), S)
       + w_c RKD(E_v(x_real), E_c(C))
       + w_a RKD(E_c(C), S).
```

The projectors are then frozen. Static terminal VSRA remains the `eta=0`
control:

```text
L_legacy = w_s RKD(E_v(x_0_fake), S)
         + w_c RKD(E_v(x_0_fake), E_c(C)).
```

There is no auxiliary fake branch, class loop, or class-centering projection.

## Diffusion-timestep coordination

Let `u=t/(T-1)` and `d=2u-1`. The recommended schedule is

```text
w_class(t)    = 1 + rho d
w_instance(t) = 1 - rho d.
```

High-noise predictions therefore receive stronger semantic class-topology
supervision and weaker instance-level PaCo supervision; low-noise predictions
receive the reverse allocation. Classes are assigned to balanced timestep
groups before the generator update. All occurrences of one class share a
timestep, while Gaussian noise remains independent per sample. Relation pairs
are formed only when `t_i == t_j` and weighted by
`sqrt(w(t_i) w(t_j))`.

Pair space is split without per-class loops:

```text
M_class(i,j)    = 1[y_i != y_j and t_i == t_j]
M_instance(i,j) = 1[y_i == y_j and i != j and t_i == t_j].
```

The masks are disjoint. Cross-class pairs align with semantic distances and
within-class pairs align with PaCo distances. Each topology is normalized only
over the pairs selected by its own mask:

```text
D_hat_M = D / mean(D_ij for (i,j) in M and D_ij > 0).
```

Class-level scale is therefore independent of within-class dispersion, and
instance-level scale is independent of cross-class separation. The final
relation objective is

```text
L_time = w_s L_time_class + w_c L_time_instance
L_relation = (1 - eta) L_legacy + eta L_time.
```

`eta=0` exactly selects the restored VSRA anchor. `eta=1` removes the overlapping
static generator constraint and uses only the time-aware dual topology, while
retaining real-space projector calibration. Intermediate values keep a constant
relation-loss budget instead of adding duplicate regularization.

## Three-dataset runs

The AWA2, CUB, and SUN DFG launchers all default to the proposed time-aware
dual-topology setting (`gamma_rel=1`, `eta=1`, `rho=0.5`). Each launcher keeps
the dataset-specific ZeroDiff hyperparameters and accepts trailing command-line
arguments as overrides.

Prepare the dataset-specific clean DRG checkpoint once:

```bash
python scripts/run_awa2_zerodiff_DRG_train.py
python scripts/run_cub_zerodiff_DRG_train.py
python scripts/run_sun_zerodiff_DRG_train.py
```

Run the proposed method:

```bash
python scripts/run_awa2_zerodiff_DFG_train.py
python scripts/run_cub_zerodiff_DFG_train.py
python scripts/run_sun_zerodiff_DFG_train.py
```

For a controlled comparison on any launcher, append one of these overrides:

```bash
# ZeroDiff-equivalent control: no relation objective
--gamma_rel 0

# Restored static VSRA anchor
--gamma_rel 1 --rel_time_pair_weight 0

# Constant-budget midpoint ablation
--gamma_rel 1 --rel_time_pair_weight 0.5

# Proposed time-aware dual topology
--gamma_rel 1 --rel_time_pair_weight 1
```

Use the same DRG checkpoint and seed for the four DFG runs of a dataset. The
launcher appends overrides after its defaults, so the final occurrence of a
scalar option is used by `argparse`.

## AWA2 examples

Restored VSRA anchor:

```bash
python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 1.0 \
  --rel_time_pair_weight 0.0
```

Time-aware dual topology with calibrated relation space:

```bash
python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 1.0 \
  --rel_time_pair_weight 1.0 \
  --rel_time_mode class_up_instance_down \
  --rel_time_strength 0.5
```

All three launchers use projection dimension 512, distance ratio 1, and angle
ratio 2. The angle loss applies to calibration and the static VSRA control; the
proposed `eta=1` generator objective is the masked distance topology itself.
