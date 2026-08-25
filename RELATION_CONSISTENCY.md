# Time-aware VSRA for ZeroDiff

This branch implements the proposed relation method on top of the runnable
ZeroDiff DFG. The clean `diagnostics/` package remains the source of baseline
measurements and is not imported by training code.

## Motivation and scope

The clean baseline shows a mild but consistent granularity-dependent temporal
response: class-level semantic topology is stable or slightly improves over
diffusion timesteps, while instance-level contrastive topology becomes less
faithful at higher timesteps. Adjacent-timestep correlations remain close to
one, so severe relational drift and temporal discontinuity are not the claims of
this method.

The method first restores the calibrated relation space used by VSRA and then
aligns two mathematically separated components at the sampled timestep:

1. **Class-level VSRA:** projected visual class-mean relations are aligned with
   semantic-attribute relations.
2. **Instance-level VSRA:** projected, class-centered visual residual relations
   are aligned with projected PaCo residual relations.
3. **Timestep coordination:** class supervision stays fixed while instance
   supervision is fixed, strengthened, or weakened toward high timesteps.

Let `Z` be the episode class-indicator matrix and define

```text
P = Z (Z^T Z)^-1 Z^T,    H = I - P.
```

For a projected episode feature matrix `Q`, `PQ` is the expanded class-mean
component and `HQ` is the within-class residual component. Because `P` and `H`
are complementary orthogonal projections,

```text
Q = PQ + HQ,    <PQ, HQ>_F = 0.
```

Class relations are computed only from the unique means represented by `PQ`;
instance relations are computed only within each class in `HQ`. The two losses
therefore do not reuse the same sample-space component. Both use scale-normalized
RKD distances and optionally RKD angles.

## Training integration

When `gamma_rel > 0`, the generator update uses one balanced `N`-way, `K`-shot
episode and one shared timestep. Two learned projectors map visual and PaCo
features into a common relation space. Before each generator update, the
projectors are calibrated on real data with three constraints:

1. visual class means follow semantic topology;
2. visual residuals follow the projected PaCo residual topology;
3. the PaCo projector preserves semantic class topology and raw PaCo residual
   topology.

The projectors are then frozen and the relation loss is applied to the same
`x_0_fake` used by the VAE, adversarial, and semantic reconstruction objectives.
This prevents an auxiliary fake branch from changing the meaning of the VSRA
gradient. With `gamma_rel=0`, projectors and their optimizer are not created, so
the baseline random-number stream remains unchanged.

The three instance schedules are symmetric around weight one. With normalized
time `u=t/(T-1)` and strength `rho`:

```text
fixed:         w_i(t) = 1
instance_up:   w_i(t) = 1 + rho * (2u - 1)
instance_down: w_i(t) = 1 - rho * (2u - 1)
```

Their average strength is equal under uniform timestep sampling. Class weight is
fixed at one in all three modes.

## Run and ablate

The dataset DFG launchers pass additional arguments to the training entry point.
For example:

```bash
python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 1.0 \
  --rel_n_way 8 --rel_k_shot 8 \
  --rel_class_weight 1.0 \
  --rel_instance_weight 1.0 \
  --rel_proj_dim 512 \
  --rel_teacher_anchor_weight 1.0 \
  --rel_dist_ratio 1.0 --rel_angle_ratio 2.0 --rel_use_angle \
  --rel_time_mode fixed \
  --rel_time_strength 0.5
```

Use the same command with `instance_up` and `instance_down`. The recommended
experiment order is:

1. `gamma_rel=0` baseline equivalence.
2. Class-only VSRA.
3. Instance-only VSRA.
4. Joint VSRA with `fixed` weights.
5. Joint VSRA with `instance_up`.
6. Joint VSRA with `instance_down`.

Adjacent-timestep consistency and gradient reconciliation are intentionally not
part of the core implementation. They should only be reconsidered after the
direct topology-alignment experiments establish a need.
