# Clean ZeroDiff relational diagnostics

These tools diagnose the unmodified ZeroDiff DFG at source commit `d9da5ab`.
They deliberately do not import or use VSRA, relation projectors, C-teacher
embedders, adaptive gates, or relation optimizers.

## Isolation guarantees

- `checkpoint_guard.py` rejects checkpoints containing known relation/VSRA state.
- No diagnostic script calls `optimizer.step()` or writes a checkpoint.
- Model-dependent scripts write only under `out/diagnostics/baseline/` by default.
- Plotting scripts read `.npz`/`.csv` artifacts and never import a model.
- Class relations use unique class prototypes; instance relations use within-class
  pairs, so repeated class attributes cannot masquerade as instance supervision.

The original clean DFG checkpoints do not save `state_dict_E`. When it is absent,
the diagnostic runtime uses a seeded random latent and records
`latent_source=seeded_random`. This exactly matches the latent source used by
ZeroDiff feature synthesis, but the limitation must be disclosed when interpreting
the counterfactual training-gradient probe.

## 1. Export controlled timestep trajectories

Run from the repository root in the same PyTorch environment used for ZeroDiff:

```bash
python -m diagnostics.export_trajectory \
  --dataset AWA2 \
  --dataroot Dataset \
  --checkpoint out/AWA2/<clean-dfg-checkpoint>.tar \
  --ways 8 \
  --shots 8 \
  --episodes 20 \
  --device cuda
```

For every episode, the exporter holds samples, latent variables, attributes,
contrastive features, and Gaussian noise fixed while evaluating all four exact
timesteps. It writes `trajectory.npz`, `metrics.csv`, and `metadata.json`.

## 2. Plot multi-granularity relational drift

```bash
python -m diagnostics.plot_relation_drift \
  --trajectory out/diagnostics/baseline/AWA2/<hash>/trajectory_seed_20260814/trajectory.npz \
  --metrics out/diagnostics/baseline/AWA2/<hash>/trajectory_seed_20260814/metrics.csv \
  --output-dir out/diagnostics/baseline/AWA2/<hash>/figures
```

This produces class/instance relation heatmaps and dataset-level timestep curves.

## 3. Probe gradient interference without training

```bash
python -m diagnostics.gradient_probe \
  --dataset AWA2 \
  --dataroot Dataset \
  --checkpoint out/AWA2/<clean-dfg-checkpoint>.tar \
  --ways 8 \
  --shots 8 \
  --episodes 20 \
  --device cuda
```

The probe computes gradients of the existing generator objective and two
counterfactual relation losses with `torch.autograd.grad`. It performs zero
optimizer updates. Plot the result with:

```bash
python -m diagnostics.plot_gradient_conflicts \
  --metrics out/diagnostics/baseline/AWA2/<hash>/gradients_seed_20260814/gradient_metrics.csv \
  --output out/diagnostics/baseline/AWA2/<hash>/figures/gradient_conflicts.png
```

## Interpretation rules

Do not claim timestep-dependent applicability unless the class and instance curves
show reproducible differences across episodes, datasets, and seeds. Do not claim
gradient conflict unless negative cosine frequency is substantial and stable. A
single selected episode is only a visualization; paper claims must use the
multi-episode CSV statistics.

