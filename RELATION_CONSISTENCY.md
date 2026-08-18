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

The method therefore aligns each generated prediction with two relation
teachers at its sampled timestep:

1. **Class-level VSRA:** generated visual class-prototype distances are aligned
   with semantic-attribute distances.
2. **Instance-level VSRA:** generated within-class instance distances are
   aligned with real PaCo/contrastive-feature distances.
3. **Timestep coordination:** class supervision stays fixed while instance
   supervision is fixed, strengthened, or weakened toward high timesteps.

This retains the teacher-student relation-transfer idea of VSRA without learned
relation projectors, C-teacher embedders, adaptive gates, temporal losses, or
relation-specific optimizers. Relation matrices use Euclidean distances in the
original feature space and are normalized by their mean off-diagonal distance,
matching the diagnostics.

## Training integration

The original DFG discriminator and generator batches remain unchanged. When
`gamma_rel > 0`, each generator update additionally samples a balanced `N`-way,
`K`-shot relation episode and one shared timestep. The auxiliary prediction is
used only for the relation objective; its latent input is detached so VSRA
directly regularizes the DFG generator. The combined loss is backpropagated once
through the existing generator optimizer.

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
