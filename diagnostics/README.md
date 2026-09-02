# ZeroDiff baseline relation diagnosis

This package measures the problem targeted by time-aware multi-granularity
relation consistency without updating or extending the baseline model.

The diagnosis asks three questions at every diffusion timestep:

1. How well does generated visual space preserve cross-class semantic topology?
2. How well does it preserve within-class PaCo instance topology?
3. Do the two relation objectives conflict or dominate one another in generator
   gradient space?

Class relations use different-class sample pairs and semantic attributes.
Instance relations use same-class, different-instance pairs and PaCo features.
Each topology is normalized by its own positive mean, exactly as in the proposed
training loss. Samples, latent variables, and Gaussian noise are fixed across
timesteps inside an episode so that timestep comparisons are paired.

## Run

Run the complete small experiment directly from the repository root:

```bash
python -m diagnostics.run_baseline \
  --dataset AWA2 \
  --dataroot Dataset \
  --checkpoint out/AWA2/<clean-dfg-checkpoint>.tar \
  --ways 8 \
  --shots 8 \
  --episodes 10 \
  --seed 9182 \
  --device cuda:0
```

There is no smoke mode and no separate export/plot sequence. One invocation
runs inference, computes relation gradients, writes metrics, and refreshes the
figure.

The default output is deliberately flat:

```text
out/diagnostics/AWA2/
├── metrics_seed_9182.csv
└── diagnosis.png
```

Running another seed adds one CSV and rebuilds `diagnosis.png` from all
`metrics_seed_*.csv` files belonging to the same dataset, checkpoint, and
episode configuration. Running the same seed again replaces that seed's CSV.
The CSV contains checkpoint/code provenance and run configuration, so no
separate metadata file is written.

`diagnosis.png` contains topology alignment error, topology rank fidelity, and
cross-granularity gradient coordination. The x-axis reports both the ZeroDiff
generator timestep and the actual `alpha_bar[t+1]` signal retention.

## Baseline isolation and latent source

`checkpoint_guard.py` rejects checkpoints containing relation/VSRA state. The
diagnostic code never calls `optimizer.step()` and never writes a checkpoint.

New DFG checkpoints save `state_dict_E`, allowing the diagnostic latent to match
the encoder-conditioned training path. Older clean checkpoints remain usable;
when the encoder is absent, the runner prints a warning, uses one seeded random
latent shared by all timesteps, and records `latent_source=seeded_random` in the
CSV.

The diagnostic seed controls balanced episode sampling and paired random inputs;
it is not a model-training seed.
