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

The clean ZeroDiff diagnostics show different class/instance topology drift
across diffusion time. AWA2 v1.04 then falsified the original linear seesaw:
the fixed dual topology reached VCS H=0.8139, while absolute high-noise class
upweighting reached only H=0.7902. The proposed method therefore keeps the
research hypothesis but replaces the unsupported direction rule.

For a generator step `t`, let the actual signal power in its noisy input
`x_{t+1}` be

```text
s_t = alpha_bar[t+1] = SNR(t+1) / (1 + SNR(t+1)).
```

With reliability floor `f` and sensitivity `rho`, the topology weights are

```text
w_class(t)    = f + (1 - f) s_t^(rho/2)
w_instance(t) = f + (1 - f) s_t^rho.
```

Both relations become less trusted as diffusion signal disappears, but the
fine-grained instance topology decays faster. Class topology is therefore
relatively more important at high noise without being absolutely amplified.
`rho=0` exactly recovers fixed topology weights, and `f` prevents either
relation from vanishing at the terminal noise state. Together with
`rel_topology_norm=global`, it reproduces the complete v1.04 fixed-weight
objective. The legacy
`class_up_instance_down` mode remains available only as an ablation.

Classes are assigned to balanced timestep groups before the generator update.
All occurrences of one class share a timestep, while Gaussian noise remains
independent per sample. Relation pairs are formed only when `t_i == t_j` and
weighted by `sqrt(w(t_i) w(t_j))`.

Pair space is split without per-class loops:

```text
M_class(i,j)    = 1[y_i != y_j and t_i == t_j]
M_instance(i,j) = 1[y_i == y_j and i != j and t_i == t_j].
```

The masks are disjoint. Cross-class pairs align with semantic distances and
within-class pairs align with PaCo distances. Each topology is normalized
independently inside every timestep block:

```text
D_hat_(M,t) = D / mean(D_ij for (i,j) in M_t and D_ij > 0).
```

Class-level scale is therefore independent of within-class dispersion, and
instance-level scale is independent of cross-class separation. Different
diffusion noise scales can no longer leak through a shared normalization
constant. This also matches the per-timestep diagnostic definition. The final
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

The AWA2, CUB, and SUN DFG launchers default to diffusion-reliability dual
topology (`gamma_rel=1`, `eta=1`, `rho=0.5`, `f=0.5`). Each launcher keeps the
dataset-specific ZeroDiff hyperparameters and accepts trailing command-line
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

# Proposed diffusion-reliability dual topology
--gamma_rel 1 --rel_time_pair_weight 1 --rel_time_mode diffusion_reliability

# Exact v1.04 fixed-weight winner
--gamma_rel 1 --rel_time_pair_weight 1 --rel_time_mode fixed \
  --rel_time_strength 0 --rel_topology_norm global

# New normalization only, without reliability gating
--gamma_rel 1 --rel_time_pair_weight 1 --rel_time_strength 0 \
  --rel_topology_norm timestep

# Diffusion reliability only, with v1.04 global normalization
--gamma_rel 1 --rel_time_pair_weight 1 --rel_time_mode diffusion_reliability \
  --rel_time_strength 0.5 --rel_topology_norm global

# Reproduce the rejected v1.04 linear seesaw
--gamma_rel 1 --rel_time_pair_weight 1 --rel_time_mode class_up_instance_down
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
  --rel_time_mode diffusion_reliability \
  --rel_time_strength 0.5 \
  --rel_reliability_floor 0.5 \
  --rel_topology_norm timestep
```

All three launchers use projection dimension 512, distance ratio 1, and angle
ratio 2. The angle loss applies to calibration and the static VSRA control; the
proposed `eta=1` generator objective is the masked distance topology itself.

## Diffusion-method positioning

SNR-dependent weighting is established in diffusion training by
[P2 weighting](https://arxiv.org/abs/2204.00227) and
[Min-SNR](https://arxiv.org/abs/2303.09556). This method does not reuse their
denoising objectives. It uses ZeroDiff's own `alpha_bar[t+1]` to coordinate two
ZSL-specific relational tasks whose distinct temporal drift was measured on a
frozen clean baseline. The diffusion state therefore controls relation
comparability, topology normalization, and granularity reliability.
