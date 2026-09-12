# DS-ReG / Time-aware relation consistency for ZeroDiff

On `codex/ds-reg`, v1.02 implements module two, State-Matched Dual-Granularity
Alignment (SDGA), alongside the inherited `legacy` objective. The remaining
research targets and contribution statements are in [DS_REG_DESIGN.md](DS_REG_DESIGN.md).
Classification improvements and new calibration/reweighting designs remain unverified.

Calibration and D updates retain their ordinary real minibatches. An optional
P×K minibatch supplies all terms in a single G update. The relation projectors
are calibrated before this minibatch is selected, then frozen during G's update.

For environment setup and copyable experiment commands, see
[SERVER_RUNBOOK.md](SERVER_RUNBOOK.md). This document covers method details only.

## Implemented SDGA objective (v1.02)

Select `--rel_objective sdga --rel_time_pair_weight 1 --rel_time_mode fixed --rel_time_strength 0`.
Incompatible settings fail early. Module three's reliability weighting is not
applied to this objective.

For each relation group `b`, compare different-class pairs against semantic
attributes and same-class, distinct-instance pairs against projected PaCo
features. Normalize student and teacher distances separately on the selected
pairs, compute the mean Smooth L1 loss within each block, and then use

```text
L_SDGA = rel_dist_ratio * (lambda_C * mean_valid_b L_C(b)
                         + lambda_I * mean_valid_b L_I(b))
L_G = L_ZeroDiff + gamma_rel * L_SDGA
```

The student normalization stays differentiable; teacher geometry is detached.
Empty blocks do not enter the corresponding granularity's average. All-empty
cases produce a differentiable zero. A block with one unordered edge (two
ordered pairs) is retained and flagged as degenerate, since its separately
normalized distance loss cannot carry useful shape information. This flag does
not detect every possible geometric degeneracy or collapsed representation.

`rel_topology_norm=timestep` normalizes inside each relation block; `global`
is an ablation pooling the selected pairs across blocks. In `matched` mode a
block is the actual generation timestep `t`, whose noisy input has signal
`alpha_bar[t+1]`. In `mixed` mode a block is a relation group, not a timestep.

The inherited fixed objective averages over pairs. SDGA averages over valid
states. With equal per-state pair counts they are mathematically equal; a
refactor alone is not evidence of a new performance gain.

### Sampling and controlled ablations

- `g_batch_mode=random` reuses the final D batch. `pk` caches eligible class
  indices from the actual training split, samples P different classes and K
  different examples per class, and reports excluded classes with fewer than K
  samples. Insufficient eligible classes cause an error; no repeated-row fallback.
- P×K must equal batch size, P must be divisible by `n_T`, and the PK configuration
  requires at least three classes per state and K≥2. With explicit `class_group`,
  P=16, K=4, T=4 gives four classes and 16 examples per state: 192 ordered class
  pairs and 48 ordered instance pairs per state (768/192 total).
- `g_timestep_policy=auto` keeps historical behavior; `independent` and
  `class_group` are explicit policies that also work when `gamma_rel=0`.
  PK alone does not enable class grouping; the main commands specify both.
- `rel_pair_grouping=matched` uses the actual G timestep. `mixed` requires SDGA,
  PK, explicit class grouping and at least two states. It redistributes whole
  classes across equally sized relation blocks, with multiple true generation
  states in every block. Neither noisy inputs nor G's timestep condition changes.
- Mixed groups preserve every class's within-class pairs. Cross-class comparisons
  now mix generation states; instance pairs still share one true state, while
  their block normalization pools different classes/states. This is a test of
  the overall grouping design, not separate proof of both granularities' causal
  contributions.
- `rel_generator_class_weight` / `rel_generator_instance_weight` affect G only
  and otherwise inherit `rel_class_weight` / `rel_instance_weight`. Existing
  coefficients still control calibration. Setting both G coefficients to zero
  with `gamma_rel=1` keeps calibration active for the primary control.

Private batch, timestep and grouping RNG streams prevent mixed grouping from
shifting subsequent sampling/noise draws. Full checkpoints save these streams
and validate relation plus training settings. Old checkpoints acquire only
legacy defaults and cannot silently resume as SDGA.

### Outputs and limits

New sampling/objective/coefficient configurations require `--run_dir` to avoid
overwriting experiments. The directory contains `config.json` (including seed,
DRG path and SHA-256), `train.log`, short best-model filenames and
`dfg_training_last.tar`. Existing nonempty directories require explicit resume.

`SDGA blocks:` records include per-block losses, pair counts, sample/class
coverage, validity and degeneracy frequencies, raw distance scales, actual
generation-state composition, and signal retention. Statistics are detached
and averaged over G updates in each epoch; an empty block contributes zero to
its logged per-block loss, with the validity frequency identifying missing
coverage. The optimized loss averages only valid blocks on each update.

RSC calibration math, GSR weighting math, the clean baseline diagnostic and
classification/model-selection protocol are unchanged. SDGA models must not be
sent to `diagnostics.run_baseline`; a common method-diagnostic entrypoint is
still future work. The following sections describe the inherited legacy path.

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
