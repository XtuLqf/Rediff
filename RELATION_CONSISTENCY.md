# Time-aware multi-granularity relation consistency

This branch adds the proposed method on top of the clean runnable ZeroDiff DFG
commit `d9da5ab`. It does not reuse VSRA, relation projectors, C-teacher modules,
or learned gates. The separate `diagnostics/` package remains the only place used
to establish the baseline problem.

## Method boundary

For a balanced `N`-way, `K`-shot seen-class episode at a single diffusion
timestep, training adds three directly measurable constraints:

1. **Class relation consistency:** align pairwise generated class-prototype
   distances with pairwise semantic-attribute distances.
2. **Instance relation consistency:** within each class, align pairwise generated
   instance distances with pairwise real contrastive-feature distances.
3. **Temporal relation consistency:** preserve both granularities between
   predictions at adjacent timesteps under the same episode, latent variables,
   and diffusion noise.

All diagnostic and training topologies use the same definition: Euclidean
pairwise distances in the original feature space, normalized by the mean
off-diagonal distance. No learnable relation projector or pre-distance feature
normalization is used.

The class and instance coefficients are deterministic functions of the sampled
timestep. There is no learned gating network. The default `uniform` mode is the
neutral implementation check; `class_high_noise` and `instance_high_noise` are
opposing hypotheses, and `custom` supports endpoints inferred from the clean
diagnostic curves. The two named directional schedules keep the class/instance
weight sum equal to 2 at every timestep, matching `uniform`; this prevents total
regularization strength from confounding the schedule comparison.

After the original generator loss is backpropagated, relation gradients are
computed only for the DFG generator. If their global dot product with the base
generator gradient is negative, `base_anchor` removes the opposing component
from the relation gradient. The base gradient is never projected or replaced.

## Recommended experimental order

1. Run `diagnostics/` on the clean checkpoint and select the schedule direction
   from multi-seed, multi-dataset evidence.
2. Verify equivalence with `--gamma_rel 0`.
3. Add static relation consistency with `--rel_time_mode uniform`.
4. Compare the two directional schedules or use diagnostic-derived `custom`
   endpoints.
5. Compare `--rel_gradient_mode sum` against `base_anchor`.
6. Ablate class, instance, and temporal terms using their three component weights.

The dataset DFG launchers pass any extra command-line flags through to the
training script. For example, run AWA2 with the method enabled (the current
default batch size is 64):

```bash
python scripts/run_awa2_zerodiff_DFG_train.py \
--gamma_rel 1.0 \
--rel_n_way 8 --rel_k_shot 8 \
--rel_class_weight 1.0 \
--rel_instance_weight 1.0 \
--rel_temporal_weight 1.0 \
--rel_time_mode uniform \
--rel_gradient_mode base_anchor
```

Use the CUB or SUN launcher identically. Relation runs receive a compact
configuration hash in their log and checkpoint names, so they cannot overwrite a
baseline run with the same ZeroDiff hyperparameters. The complete unhashed
configuration is written to the log and checkpoint metadata.

For a diagnostic-derived custom schedule:

```bash
--rel_time_mode custom \
--rel_class_t0 0.5 --rel_class_tT 1.5 \
--rel_instance_t0 1.5 --rel_instance_tT 0.5
```

Relation-enabled checkpoints contain a `method_metadata` marker. The clean
baseline checkpoint guard rejects this marker, preventing a method checkpoint
from being used to produce baseline problem figures.
