# Time-aware multi-granularity relation consistency for ZeroDiff

On `codex/ds-reg`, this file describes the inherited time-aware v1.07.3
implementation. The proposed DS-ReG modules, diagnostic extensions, and revised
contribution statements are in [DS_REG_DESIGN.md](DS_REG_DESIGN.md); they are
research targets, not changes already implemented in the training code.

This branch keeps the original DFG data path, calibrates the stable relation
space from `exp/vsra`, and coordinates semantic and contrastive topology at
matched diffusion timesteps.

For environment setup and copyable experiment commands, see
[SERVER_RUNBOOK.md](SERVER_RUNBOOK.md). This document covers method details only.

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

The historical baseline diagnosis motivated studying class/instance topology
drift across diffusion time. The AWA2 v1.04 record reports fixed dual topology
VCS H=0.8139 and the original linear seesaw H=0.7902. This motivated replacing
the original direction rule. These historical numbers have not been revalidated
in this documentation update and do not replace current controlled runs.

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

Both raw weights decrease as diffusion signal disappears, with the instance
weight decaying faster. Their effect on the loss also depends on the weighted
mean below. `rho=0` exactly recovers fixed topology weights; a positive `f`
keeps each raw weight above zero. Together with `rel_topology_norm=global`,
this selects the current implementation's fixed-weight reference objective.
The legacy
`class_up_instance_down` mode remains available only as an ablation.

Classes are assigned to balanced timestep groups before the generator update.
All occurrences of one class share a timestep, while Gaussian noise remains
independent per sample. Relation pairs are formed only when `t_i == t_j` and
weighted by `sqrt(w(t_i) w(t_j))`.

The actual implementation uses a normalized weighted mean for each topology:

```text
a_ij = M(i,j) * sqrt(w(t_i) * w(t_j))
L_topology = sum(a_ij * SmoothL1(D_hat_visual, D_hat_teacher))
             / max(sum(a_ij), eps).
```

The weights redistribute contributions across timesteps within each topology;
they do not directly attenuate the entire loss of a high-noise batch. A common
positive scaling of all weights cancels when the denominator is above `eps`.
With just one valid timestep, its common weight also cancels. Thus the raw
class/instance weight ratio is not the ratio of the final topology losses.
`global` still normalizes class and instance topology separately, but pools
valid pairs across timesteps. `timestep` uses one scale per topology per timestep.
Empty masks contribute zero loss. `f=1` also gives fixed weights.

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
L_time = rel_dist_ratio * (w_s L_time_class + w_c L_time_instance)
L_relation = (1 - eta) L_legacy + eta L_time.
```

`eta=0` exactly selects the restored VSRA anchor. `eta=1` removes the overlapping
static generator constraint and uses only the time-aware dual topology, while
retaining real-space projector calibration. Intermediate values use a convex
combination whose coefficients sum to one; this does not guarantee equal loss
magnitudes or gradient budgets.

## Implementation defaults and ablation boundaries

All training, ablation, diagnosis, and recovery commands are maintained only in
[SERVER_RUNBOOK.md](SERVER_RUNBOOK.md). This document explains the method.

The three DFG launchers default to `gamma_rel=1`, `eta=1`,
`rel_time_mode=diffusion_reliability`, `rho=0.5`, `f=0.5`, and
`rel_topology_norm=timestep`. They use projection dimension 512, distance ratio
1, and angle ratio 2. Direct training-entry invocation still defaults to
`gamma_rel=0`.

- `gamma_rel=0` disables the relation module: it is the clean control on this
  branch, not a verified reproduction of the original paper's numerical result.
- `eta=0` uses independent sample timesteps; `eta>0` uses class-grouped timesteps.
  A static-versus-dynamic comparison therefore changes both the objective and
  timestep sampling.
- `rel_class_weight` and `rel_instance_weight` affect both real-space calibration
  and generator supervision. Setting either to zero is not an isolated ablation
  of only the generator topology.
- In calibration, `E_c(C)` is detached in the visual-to-contrastive loss.
  Setting `rel_teacher_anchor_weight=0` removes the current training source for
  the PaCo projector, leaving a fixed random contrastive projection.
- With `eta=1`, the generator relation objective uses only masked distances.
  `rel_angle_ratio=0` therefore removes the calibration angle term only.
- Ordinary batches do not guarantee repeated samples of every class. Inspect
  effective instance-pair counts, especially on datasets with many classes.
- The diagnostic runner accepts only clean baseline checkpoints; it measures
  baseline behavior and does not establish improvement in a trained method model.

## Diffusion-method positioning

SNR-dependent weighting is established in diffusion training by
[P2 weighting](https://arxiv.org/abs/2204.00227) and
[Min-SNR](https://arxiv.org/abs/2303.09556). This method does not reuse their
denoising objectives. It uses ZeroDiff's own `alpha_bar[t+1]` to coordinate two
ZSL-specific relational tasks whose distinct temporal drift was measured on a
frozen clean baseline. The diffusion state therefore controls relation
comparability, topology normalization, and granularity reliability.
