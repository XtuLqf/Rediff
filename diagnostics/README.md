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

## Target Linux environment

The code is compatible with Python 3.10, PyTorch 2.9.1+cu130, torchvision
0.24.1+cu130, RTX 5090, and NVIDIA driver 580.126.09. Plotting additionally
requires Matplotlib, which is not included in the base experiment dependencies:

```bash
pip install matplotlib==3.7.5 pytest==7.4.4
```

Both plotting commands accept one or more metrics CSV files after `--metrics`.
The default is 10 balanced 8-way/8-shot episodes. All timesteps reuse the same
samples and noise within an episode, so timestep differences are directly
comparable. The diagnostic seed only controls episode sampling; it is not a new
model-training seed.

## 1. Export controlled timestep trajectories

Run from the repository root in the same PyTorch environment used for ZeroDiff:

```bash
python -m diagnostics.export_trajectory \
  --dataset AWA2 \
  --dataroot Dataset \
  --checkpoint out/AWA2/<clean-dfg-checkpoint>.tar \
  --ways 8 \
  --shots 8 \
  --episodes 10 \
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

This produces `relation_heatmaps.png` and a three-panel
`topology_dynamics.png`. The latter shows topology fidelity, alignment error,
and class/instance temporal stability as episode mean +/- standard deviation.
Existing trajectory archives can be upgraded without model inference using
`python -m diagnostics.recompute_trajectory_metrics`; see `SERVER_RUNBOOK.md`.

## 3. Probe gradient interference without training

```bash
python -m diagnostics.gradient_probe \
  --dataset AWA2 \
  --dataroot Dataset \
  --checkpoint out/AWA2/<clean-dfg-checkpoint>.tar \
  --ways 8 \
  --shots 8 \
  --episodes 10 \
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

The gradient figure contains raw cosine compatibility, conflict frequency, and
the opposing gradient magnitude that the counterfactual base-anchor projection
would remove. The probe still performs no optimizer update.

## Interpretation rules

Treat this as a motivation experiment: first inspect one checkpoint with 10
episodes. Only expand to more training seeds or datasets if the curves are
promising enough for the final paper experiment. A single heatmap episode is
illustrative; the curves summarize all exported episodes.
