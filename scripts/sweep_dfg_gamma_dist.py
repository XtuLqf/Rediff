#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep DFG gamma_dist for the first VSRA search round."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATAROOT = ROOT / "Dataset"


DATASET_CONFIGS = {
    "CUB": {
        "omp_num_threads": "3",
        "default_values": [0.0, 1.0, 2.0, 4.0],
        "netr_candidates": [
            "zerodiff_DRG_100percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:0.0_x0:1.0_xt:1.0_dist:1.0_num:300_gzsl.tar",
            "zerodiff_DRG_100percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:0.0_x0:1.0_xt:1.0_dist:1.0_num:300_zsl.tar",
        ],
        "args": [
            "--gzsl", "--encoded_noise", "--manualSeed", "3483", "--preprocessing", "--cuda", "--image_embedding", "res101",
            "--class_embedding", "sent", "--nepoch", "300", "--ngh", "4096", "--ndh", "4096", "--lambda1", "10", "--critic_iter", "5",
            "--nclass_all", "200", "--dataset", "CUB", "--eval_interval", "5",
            "--batch_size", "64", "--noiseSize", "1024", "--attSize", "1024", "--resSize", "2048",
            "--lr", "0.0001", "--classifier_lr", "0.001", "--gamma_recons", "0.01", "--dec_lr", "0.0001",
            "--gamma_ADV", "10", "--gamma_VAE", "1.0", "--embed_type", "VA",
            "--n_T", "4", "--dim_t", "1024", "--gamma_x0", "1.0", "--gamma_xt", "1.0",
            "--factor_dist", "1.5",
            "--gamma_rel", "1.0", "--rel_sem_weight", "1.0", "--rel_con_weight", "1.0", "--rel_proj_dim", "512",
            "--rel_dist_ratio", "1.0", "--rel_angle_ratio", "2.0", "--rel_angle_max_samples", "128", "--rel_use_angle",
            "--split_percent", "100", "--syn_num", "1440",
        ],
    },
    "AWA2": {
        "omp_num_threads": "4",
        "default_values": [0.0, 2.5, 5.0, 10.0],
        "netr_candidates": [
            "zerodiff_DRG_100percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_gzsl.tar",
            "zerodiff_DRG_100percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_zsl.tar",
            "diffzero_pretrain_100percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_gzsl.tar",
            "diffzero_pretrain_100percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_zsl.tar",
        ],
        "args": [
            "--gzsl", "--encoded_noise", "--manualSeed", "9182", "--preprocessing", "--cuda", "--image_embedding", "res101",
            "--class_embedding", "att", "--class_embedding_norm", "--nepoch", "300", "--ngh", "4096", "--ndh", "4096", "--lambda1", "10", "--critic_iter", "5",
            "--nclass_all", "50", "--dataset", "AWA2", "--eval_interval", "5",
            "--batch_size", "64", "--noiseSize", "85", "--attSize", "85", "--resSize", "2048",
            "--lr", "0.0005", "--classifier_lr", "0.001", "--gamma_recons", "1.0", "--freeze_dec", "--dec_lr", "0.0001",
            "--gamma_ADV", "10", "--gamma_VAE", "1.0", "--embed_type", "VA",
            "--n_T", "4", "--dim_t", "85", "--gamma_x0", "1.0", "--gamma_xt", "1.0",
            "--split_percent", "100", "--syn_num", "5400", "--factor_dist", "1.5",
            "--gamma_rel", "1.0", "--rel_sem_weight", "1.0", "--rel_con_weight", "1.0", "--rel_proj_dim", "512",
            "--rel_dist_ratio", "1.0", "--rel_angle_ratio", "2.0", "--rel_angle_max_samples", "128", "--rel_use_angle",
        ],
    },
    "SUN": {
        "omp_num_threads": "3",
        "default_values": [0.0, 0.5, 1.0, 2.0],
        "netr_candidates": [
            "zerodiff_DRG_100percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:400_gzsl.tar",
            "zerodiff_DRG_100percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:400_zsl.tar",
            "diffzero_pretrain_100percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:400_gzsl.tar",
            "diffzero_pretrain_100percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:400_zsl.tar",
        ],
        "args": [
            "--dataset", "SUN", "--image_embedding", "res101", "--class_embedding", "att", "--class_embedding_norm", "--eval_interval", "5",
            "--gzsl", "--manualSeed", "4115", "--encoded_noise", "--preprocessing", "--cuda",
            "--nepoch", "400", "--ngh", "4096", "--ndh", "4096", "--lr", "0.0001", "--classifier_lr", "0.0005", "--lambda1", "10", "--critic_iter", "5",
            "--nclass_all", "717", "--batch_size", "64", "--noiseSize", "102", "--attSize", "102", "--resSize", "2048",
            "--gamma_recons", "0.01", "--dec_lr", "0.0001",
            "--gamma_ADV", "1", "--gamma_VAE", "1.0", "--embed_type", "VA",
            "--n_T", "4", "--dim_t", "102", "--gamma_x0", "1.0", "--gamma_xt", "1.0", "--factor_dist", "1.5",
            "--gamma_rel", "1.0", "--rel_sem_weight", "1.0", "--rel_con_weight", "1.0", "--rel_proj_dim", "512",
            "--rel_dist_ratio", "1.0", "--rel_angle_ratio", "2.0", "--rel_angle_max_samples", "128", "--rel_use_angle",
            "--split_percent", "100", "--syn_num", "400",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep DFG gamma_dist for VSRA experiments.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS), help="Dataset to sweep.")
    parser.add_argument("--values", nargs="+", type=float, help="Override default gamma_dist values.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue sweeping after a failed run.")
    return parser.parse_args()


def resolve_netr_model(dataset, config, dry_run=False):
    out_dir = ROOT / "out" / dataset
    candidates = [out_dir / name for name in config["netr_candidates"]]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    available = sorted(out_dir.glob("*.tar")) if out_dir.exists() else []
    if available:
        return available[0]

    if dry_run and candidates:
        print(
            "[dry-run] No DRG checkpoint found; using the first configured candidate path for command preview:",
            candidates[0],
        )
        return candidates[0]

    raise FileNotFoundError(
        f"No DRG checkpoint found in {out_dir}. Please run the {dataset} DRG script first."
    )


def build_command(config, gamma_dist, netr_model):
    return [
        sys.executable,
        "zerodiff_DFG_train.py",
        "--dataroot",
        str(DATAROOT),
        *config["args"],
        "--gamma_dist",
        str(gamma_dist),
        "--netR_model_path",
        str(netr_model),
    ]


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    values = args.values if args.values is not None else config["default_values"]
    netr_model = resolve_netr_model(args.dataset, config, dry_run=args.dry_run)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = config["omp_num_threads"]

    for gamma_dist in values:
        command = build_command(config, gamma_dist, netr_model)
        print(f"\n[{args.dataset}] gamma_dist={gamma_dist}")
        print(shlex.join(command))

        if args.dry_run:
            continue

        try:
            subprocess.run(command, cwd=ROOT, check=True, env=env)
        except subprocess.CalledProcessError:
            if not args.continue_on_error:
                raise
            print(f"[{args.dataset}] gamma_dist={gamma_dist} failed; continuing.")


if __name__ == "__main__":
    main()
