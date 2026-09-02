# Linux server runbook

Run all commands from the repository root.

## Branch

Use the current method branch for both the clean control and the proposed
method. The diagnostic runner loads only clean ZeroDiff state and rejects a
checkpoint containing relation/VSRA parameters, so a separate diagnostic branch
is no longer required.

On the server:

```bash
git fetch Rediff
git switch codex/time-aware-relational-consistency
```

## Environment

```bash
conda create -n zerodiff python=3.10 -y
conda activate zerodiff
python -m pip install --upgrade pip
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install scikit-learn==1.3.0 scipy==1.10.0 numpy==1.24.3 pillow==9.4.0 matplotlib==3.7.5
```

## Train or select a clean baseline

```bash
export DATASET=AWA2
export CLEAN_DFG="out/AWA2/your_clean_zerodiff_DFG_checkpoint.tar"
```

The checkpoint must come from a run without relation consistency. If needed:

```bash
python scripts/run_awa2_zerodiff_DRG_train.py
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 0
find out/AWA2 -maxdepth 1 -type f -name 'zerodiff_DFG*.tar' -print
```

New checkpoints contain `state_dict_E`, which lets the diagnosis reproduce the
encoder-conditioned training path. Older clean checkpoints are accepted with a
visible seeded-random-latent warning. Relation/method checkpoints are rejected.

## Run the baseline diagnosis

There is one complete command and no smoke workflow:

```bash
python -m diagnostics.run_baseline \
  --dataset "$DATASET" \
  --dataroot Dataset \
  --checkpoint "$CLEAN_DFG" \
  --ways 8 \
  --shots 8 \
  --episodes 10 \
  --seed 9182 \
  --device cuda:0
```

The command performs the paired timestep passes, relation-gradient measurements,
CSV export, and plotting. It creates only:

```text
out/diagnostics/AWA2/
├── metrics_seed_9182.csv
└── diagnosis.png
```

Optional additional diagnostic sampling seeds add one CSV each and are
automatically included in the refreshed figure:

```bash
python -m diagnostics.run_baseline \
  --dataset "$DATASET" --dataroot Dataset --checkpoint "$CLEAN_DFG" \
  --ways 8 --shots 8 --episodes 10 --seed 19182 --device cuda:0
```

These are episode-sampling seeds, not independently trained model seeds. Files
from another checkpoint or episode configuration in the same directory are
ignored when rebuilding the current figure.

The previous `baseline/`, `smoke/`, `topology_v2/`, `trajectory/`, `gradients/`,
`paper_figures_old/`, and `metrics_topology_v2.csv` layouts are no longer read by
the code. After confirming the new CSV and figure, old server artifacts can be
archived or removed manually.

## Train the proposed method

```bash
python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 1.0 \
  --rel_class_weight 1.0 \
  --rel_instance_weight 1.0 \
  --rel_proj_dim 512 \
  --rel_teacher_anchor_weight 1.0 \
  --rel_dist_ratio 1.0 \
  --rel_angle_ratio 2.0 \
  --rel_use_angle \
  --rel_time_pair_weight 1.0 \
  --rel_time_mode diffusion_reliability \
  --rel_time_strength 0.5 \
  --rel_reliability_floor 0.5 \
  --rel_topology_norm timestep
```

Useful controlled comparisons:

```bash
# ZeroDiff-equivalent control
python scripts/run_awa2_zerodiff_DFG_train.py --gamma_rel 0

# Static VSRA anchor
python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 1 --rel_time_pair_weight 0

# Fixed dual topology
python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 1 --rel_time_pair_weight 1 \
  --rel_time_mode fixed --rel_time_strength 0 --rel_topology_norm global

# Timestep normalization without reliability gating
python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 1 --rel_time_pair_weight 1 \
  --rel_time_strength 0 --rel_topology_norm timestep
```

Use the same DRG checkpoint and training seed for controlled comparisons.

## Resume interrupted DFG training

DFG training atomically overwrites a recoverable checkpoint at the configured
interval. Its name ends in `_training_last.tar`.

Resume using the same launcher arguments plus:

```bash
python scripts/run_awa2_zerodiff_DFG_train.py \
  --resume_training 'out/AWA2/<matching-run>_training_last.tar'
```

Do not pass a `gzsl_*.tar` or `zsl_*.tar` model-selection checkpoint to
`--resume_training`; those files intentionally exclude optimizer state.
