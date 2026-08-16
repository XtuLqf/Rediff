# Linux server runbook

## Branch map

- `codex/relational-drift-diagnostics` is the clean-problem branch. It starts
  from runnable ZeroDiff commit `d9da5ab` and contains only read-only diagnostic
  exporters, gradient probes, and offline plotting. Use it to train/select a
  clean baseline checkpoint and produce the motivation figures.
- `codex/time-aware-relational-consistency` is the proposed-method branch. It
  contains the same diagnostics plus timestep-aware class, instance, and
  temporal relation consistency and base-anchored gradient reconciliation. Use
  it only after the baseline evidence is fixed.

These branches are currently local until they are pushed. From the development
machine, publish them once:

```bash
git push -u Rediff codex/relational-drift-diagnostics
git push -u Rediff codex/time-aware-relational-consistency
```

Then on the Linux server:

```bash
git fetch Rediff
git switch --track Rediff/codex/relational-drift-diagnostics
```

## Environment

```bash
conda create -n zerodiff python=3.10 -y
conda activate zerodiff
python -m pip install --upgrade pip
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install scikit-learn==1.3.0 scipy==1.10.0 numpy==1.24.3 pillow==9.4.0 matplotlib==3.7.5 pytest==7.4.4
```

Verify that the intended environment is active:

```bash
python - <<'PY'
import torch, numpy, scipy, sklearn, matplotlib
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("numpy/scipy/sklearn/matplotlib:", numpy.__version__, scipy.__version__, sklearn.__version__, matplotlib.__version__)
PY
```

Run every command below from the repository root. Set the clean DFG checkpoint
once; it must have been trained without the relation method:

```bash
git switch codex/relational-drift-diagnostics
export DATASET=AWA2
export CLEAN_DFG="out/AWA2/your_clean_zerodiff_DFG_checkpoint.tar"
```

If no clean checkpoint exists yet, train DRG first and then the unmodified DFG
on this diagnostic branch:

```bash
python scripts/run_awa2_zerodiff_DRG_train.py
python scripts/run_awa2_zerodiff_DFG_train.py
find out/AWA2 -maxdepth 1 -type f -name 'zerodiff_DFG*.tar' -print
```

`CLEAN_DFG` must point to a DFG checkpoint from the final `find` output, not the
DRG checkpoint used as DFG conditioning input.

The checkpoint guard checks both required ZeroDiff module keys and known
relation/method markers. A method checkpoint fails immediately.

## Generate paper-aligned motivation diagnostics

Export trajectories and gradient measurements for three seeds:

```bash
# Optional smoke test: first use one seed and --episodes 2.
for SEED in 9182 19182 29182; do
  python -m diagnostics.export_trajectory \
    --dataset "$DATASET" --dataroot Dataset --checkpoint "$CLEAN_DFG" \
    --ways 8 --shots 8 --episodes 30 --seed "$SEED" --device cuda:0 \
    --output-dir "out/diagnostics/topology_v2/$DATASET/seed_$SEED/trajectory"

  python -m diagnostics.gradient_probe \
    --dataset "$DATASET" --dataroot Dataset --checkpoint "$CLEAN_DFG" \
    --ways 8 --shots 8 --episodes 30 --seed "$SEED" --device cuda:0 \
    --output-dir "out/diagnostics/topology_v2/$DATASET/seed_$SEED/gradients"
done
```

Render one representative heatmap and pool all three seeds for the curves:

```bash
FIG_DIR="out/diagnostics/topology_v2/$DATASET/paper_figures"

python -m diagnostics.plot_relation_drift \
  --trajectory "out/diagnostics/topology_v2/$DATASET/seed_9182/trajectory/trajectory.npz" \
  --metrics \
    "out/diagnostics/topology_v2/$DATASET/seed_9182/trajectory/metrics.csv" \
    "out/diagnostics/topology_v2/$DATASET/seed_19182/trajectory/metrics.csv" \
    "out/diagnostics/topology_v2/$DATASET/seed_29182/trajectory/metrics.csv" \
  --episode 0 --output-dir "$FIG_DIR"

python -m diagnostics.plot_gradient_conflicts \
  --metrics \
    "out/diagnostics/topology_v2/$DATASET/seed_9182/gradients/gradient_metrics.csv" \
    "out/diagnostics/topology_v2/$DATASET/seed_19182/gradients/gradient_metrics.csv" \
    "out/diagnostics/topology_v2/$DATASET/seed_29182/gradients/gradient_metrics.csv" \
  --output "$FIG_DIR/gradient_conflicts.png"
```

The old CSV files do not contain the granularity-specific temporal and
base-anchor columns, so they cannot be reused with the new plotting scripts.
The `topology_v2` directory deliberately keeps the new run separate.

This creates:

- `relation_heatmaps.png`: shared-scale class/instance topology matrices;
- `topology_error_heatmaps.png`: absolute deviation from each reference topology;
- `topology_dynamics.png` (also saved as `relation_curves.png`): fidelity, error,
  granularity-specific adjacent stability, and paired endpoint effects;
- `gradient_conflicts.png`: raw compatibility, conflict activation frequency,
  and the magnitude removed by counterfactual base-anchor projection;
- `topology_statistics.csv` and `gradient_statistics.csv`: numerical bootstrap
  intervals and paired permutation results used by the figures.

Repeat with `DATASET=CUB` and `DATASET=SUN`, changing `CLEAN_DFG` to the matching
clean checkpoint. Do not reuse an AWA2 checkpoint for another dataset.

## Train the proposed method

After the diagnostic curves determine the schedule hypothesis:

```bash
git switch --track Rediff/codex/time-aware-relational-consistency  # first use on server
# Later uses: git switch codex/time-aware-relational-consistency

python scripts/run_awa2_zerodiff_DFG_train.py \
  --gamma_rel 1.0 \
  --rel_n_way 8 --rel_k_shot 8 \
  --rel_class_weight 1.0 \
  --rel_instance_weight 1.0 \
  --rel_temporal_weight 1.0 \
  --rel_time_mode uniform \
  --rel_gradient_mode base_anchor
```

First run `uniform`; only choose `class_high_noise`, `instance_high_noise`, or a
`custom` schedule after inspecting the clean curves. Use `--gamma_rel 0` as the
same-code baseline-equivalence control and `--rel_gradient_mode sum` as the
no-conflict-reconciliation ablation.
