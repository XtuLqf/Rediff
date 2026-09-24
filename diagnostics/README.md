# ZeroDiff baseline relation diagnosis

This package measures cross-class and within-class relation preservation
across diffusion states without updating the clean ZeroDiff baseline model.

At each diffusion timestep it asks how closely generated cross-class distances
align with semantic distances, how closely generated within-class distances
align with PaCo distances, and whether their pairwise rankings agree. Each
relation is normalized by its own positive mean. Samples, latent variables,
and Gaussian noise are fixed across timesteps inside an episode so that
timestep comparisons are paired. Figure shading shows the standard deviation
of episode measurements, not a confidence interval.

## Clean AWA2 baseline and diagnosis

Run these commands from the repository root. If there is no AWA2 DRG checkpoint,
train it first:

```bash
python scripts/run_awa2_zerodiff_DRG_train.py
```

The AWA2 DFG launcher normally enables a legacy relation loss. Explicitly
disable it and use a separate output directory for the clean baseline:

```bash
python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 0 \
  --run_dir out/AWA2/clean_baseline_seed_9182
```

Diagnose the saved VCS GZSL checkpoint:

```bash
python -m diagnostics.run_baseline \
  --dataset AWA2 --dataroot Dataset \
  --checkpoint out/AWA2/clean_baseline_seed_9182/dfg_gzsl_VCS.tar \
  --ways 8 --shots 8 --episodes 10 --seed 9182 --device cuda:0
```

Use a new `--run_dir` for a new training run. The diagnostic seed controls
balanced episode sampling and paired noise, not the DFG training seed.

## Diagnose another clean checkpoint

```bash
python -m diagnostics.run_baseline \
  --dataset AWA2 --dataroot Dataset \
  --checkpoint out/AWA2/<clean-dfg-checkpoint>.tar \
  --ways 8 --shots 8 --episodes 10 --seed 9182 --device cuda:0
```

One invocation writes metrics and these three figures:

```text
out/diagnostics/AWA2/
├── metrics_seed_9182.csv
├── diagnosis_a_relation_error.png
├── diagnosis_b_rank_agreement.png
└── diagnosis.png
```

The first figure shows normalized distance alignment error; the second shows
Spearman rank agreement; `diagnosis.png` combines the two panels. No gradient
panel or gradient metrics are produced. The x-axis reports both the generator
timestep and actual `alpha_bar[t+1]` signal retention.

Running another seed adds one CSV and rebuilds all three figures from matching
CSV files for the same dataset, checkpoint, and episode configuration. Running
the same seed again replaces that seed's CSV. Each CSV records checkpoint and
code provenance along with the run configuration.

## Baseline isolation and latent source

`checkpoint_guard.py` rejects checkpoints containing relation/VSRA state. The
diagnostic code never calls `optimizer.step()` and never writes a checkpoint.

New DFG checkpoints save `state_dict_E`, allowing the diagnostic latent to
match the encoder-conditioned training path. Older clean checkpoints remain
usable; when the encoder is absent, the runner prints a warning, uses one
seeded random latent shared by all timesteps, and records
`latent_source=seeded_random` in the CSV.
