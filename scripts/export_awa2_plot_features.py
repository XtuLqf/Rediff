#!/usr/bin/env python3
"""Launch AWA2 feature export using the flat checkpoint names in out/AWA2."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "out" / "AWA2"
DFG_SETTINGS = {100: ("0.0", 5400), 30: ("5.0", 1800), 10: ("5.0", 600)}
DRG_COUNTS = {100: 1800, 30: 600, 10: 200}


def dfg_path(percent):
    distance, count = DFG_SETTINGS[percent]
    return CHECKPOINTS / (
        f"zerodiff_DFG_{percent}percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_"
        f"gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:{distance}_f:1.5_num:{count}gzsl_VCS.tar"
    )


def drg_path(percent):
    return CHECKPOINTS / (
        f"zerodiff_DRG_{percent}percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_"
        f"gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:{DRG_COUNTS[percent]}_gzsl.tar"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-percent", type=int, choices=[100, 30, 10], default=100)
    pairing = parser.add_mutually_exclusive_group()
    pairing.add_argument("--drg-percent", type=int, choices=[100, 30, 10],
                         help="Actual DRG used to train this DFG. Defaults to 100 only for the 100%% DFG.")
    pairing.add_argument("--drg-checkpoint", type=Path, help="Explicit actual training DRG, overriding the filename mapping.")
    parser.add_argument("--dfg-checkpoint", type=Path, help="Override the mapped best GZSL VCS checkpoint.")
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--method", default="Rediff")
    parser.add_argument("--sampling", choices=["one-step", "progressive"], default="one-step")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9182)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataroot", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", type=Path,
                        help="Reference protocol; for 30/10%%, defaults to the existing 100pct/protocol.json if present.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without checking or loading checkpoints.")
    args = parser.parse_args(argv)
    if args.samples_per_class <= 0 or args.batch_size <= 0 or not 0 <= args.seed < 2**63 - 1:
        parser.error("Positive sample/batch counts and a seed in [0, 2**63-1) are required.")
    if args.drg_checkpoint is None and args.drg_percent is None:
        if args.split_percent != 100:
            parser.error("For 30/10% DFG, explicitly provide --drg-percent or --drg-checkpoint "
                         "matching the actual training run; the current training launcher defaults to 100% DRG.")
        args.drg_percent = 100

    dfg = args.dfg_checkpoint or dfg_path(args.split_percent)
    drg = args.drg_checkpoint or drg_path(args.drg_percent)
    output = args.output_dir or ROOT / "out" / "plot_features" / "AWA2" / f"{args.split_percent}pct"
    protocol = args.protocol
    default_protocol = ROOT / "out" / "plot_features" / "AWA2" / "100pct" / "protocol.json"
    if protocol is None and args.split_percent != 100 and default_protocol.is_file():
        protocol = default_protocol

    command = [
        sys.executable, "-m", "feature_export.export", "--dataset", "AWA2",
        "--dataroot", str(args.dataroot.expanduser().resolve()),
        "--dfg-checkpoint", str(dfg.expanduser().resolve()),
        "--drg-checkpoint", str(drg.expanduser().resolve()),
        "--method", args.method, "--split-percent", str(args.split_percent),
        "--view", "both", "--samples-per-class", str(args.samples_per_class),
        "--sampling", args.sampling, "--seed", str(args.seed),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--output-dir", str(output.expanduser().resolve()),
    ]
    for flag, path in (("--training-config", args.training_config), ("--protocol", protocol)):
        if path is not None:
            command.extend([flag, str(path.expanduser().resolve())])
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return command
    for path in (dfg, drg):
        if not path.expanduser().is_file():
            parser.error(f"Checkpoint not found: {path}")
    if output.expanduser().exists():
        parser.error(f"Output already exists: {output}; use --output-dir to choose a new directory.")
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    return command


if __name__ == "__main__":
    main()
