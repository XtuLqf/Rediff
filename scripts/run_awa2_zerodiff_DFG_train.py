#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZihanYe
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATAROOT = ROOT / 'Dataset'
OUT_DIR = ROOT / 'out' / 'AWA2'
NETR_MODEL_CANDIDATES = [
	OUT_DIR / 'zerodiff_DRG_100percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_gzsl.tar',
	OUT_DIR / 'zerodiff_DRG_100percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_zsl.tar',
	OUT_DIR / 'diffzero_pretrain_100percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_gzsl.tar',
	OUT_DIR / 'diffzero_pretrain_100percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_zsl.tar',
]

override_parser = argparse.ArgumentParser(add_help=False)
override_parser.add_argument('--experiment', type=str.upper, choices=['S0', 'S1', 'S2'],
	                        help='S0: G relation off; S1: matched SDGA; S2: mixed SDGA')
overrides, training_args = override_parser.parse_known_args()
path_parser = argparse.ArgumentParser(add_help=False)
path_parser.add_argument('--netR_model_path')
path_parser.add_argument('--run_dir')
path_parser.add_argument('--resume_training')
path_parser.add_argument('--manualSeed', type=int, default=9182)
paths, _ = path_parser.parse_known_args(training_args)
NETR_MODEL = Path(paths.netR_model_path).expanduser().resolve() if paths.netR_model_path else next(
	(candidate for candidate in NETR_MODEL_CANDIDATES if candidate.exists()),
	None,
)
if NETR_MODEL is None:
	raise FileNotFoundError(
		f'No matching AWA2 100% DRG checkpoint found in {OUT_DIR}. '
		'Please run the AWA2 DRG script first.'
	)
if not NETR_MODEL.is_file():
	raise FileNotFoundError(f'DRG checkpoint does not exist: {NETR_MODEL}')

env = os.environ.copy()
env['OMP_NUM_THREADS'] = '4'
env.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

command = [
	sys.executable,
	'zerodiff_DFG_train.py',
	'--gzsl', '--encoded_noise', '--manualSeed', '9182', '--preprocessing', '--cuda', '--image_embedding', 'res101',
	'--class_embedding', 'att', '--class_embedding_norm', '--nepoch', '300', '--ngh', '4096', '--ndh', '4096', '--lambda1', '10', '--critic_iter', '5',
	'--nclass_all', '50', '--dataroot', str(DATAROOT), '--dataset', 'AWA2', '--eval_interval', '5',
	'--batch_size', '64', '--noiseSize', '85', '--attSize', '85', '--resSize', '2048',
	'--lr', '0.0005', '--classifier_lr', '0.001', '--gamma_recons', '1.0', '--freeze_dec', '--dec_lr', '0.0001',
	'--gamma_ADV', '10', '--gamma_VAE', '1.0', '--embed_type', 'VA',
	'--n_T', '4', '--dim_t', '85', '--gamma_x0', '1.0', '--gamma_xt', '1.0',
	'--split_percent', '100', '--syn_num', '5400', '--gamma_dist', '0.0', '--factor_dist', '1.5',
	'--gamma_rel', '1.0', '--rel_class_weight', '1.0', '--rel_instance_weight', '1.0',
	'--rel_proj_dim', '512', '--rel_teacher_anchor_weight', '1.0',
	'--rel_dist_ratio', '1.0', '--rel_angle_ratio', '2.0', '--rel_angle_max_samples', '128', '--rel_use_angle',
	'--rel_time_pair_weight', '1.0', '--rel_time_mode', 'diffusion_reliability', '--rel_time_strength', '0.5',
	'--rel_reliability_floor', '0.5',
	'--rel_topology_norm', 'timestep',
	'--netR_model_path', str(NETR_MODEL),
]
if overrides.experiment:
	name, grouping, class_weight, instance_weight = {
		'S0': ('s0_control', 'matched', '0', '0'),
		'S1': ('s1_matched', 'matched', '1', '1'),
		'S2': ('s2_mixed', 'mixed', '1', '1'),
	}[overrides.experiment]
	command.extend([
		'--gamma_rel', '1', '--rel_objective', 'sdga',
		'--rel_time_pair_weight', '1', '--rel_time_mode', 'fixed', '--rel_time_strength', '0',
		'--g_batch_mode', 'pk', '--g_pk_classes', '16', '--g_pk_samples', '4',
		'--g_timestep_policy', 'class_group', '--rel_pair_grouping', grouping,
		'--rel_generator_class_weight', class_weight, '--rel_generator_instance_weight', instance_weight,
	])
	if not paths.run_dir:
		if paths.resume_training:
			run_dir = Path(paths.resume_training).expanduser().resolve().parent
		else:
			base_dir = ROOT / 'out' / 'ds_reg' / 'AWA2' / f'{name}_seed{paths.manualSeed}'
			run_dir = base_dir
			attempt = 2
			while run_dir.exists():
				run_dir = base_dir.with_name(f'{base_dir.name}_run{attempt}')
				attempt += 1
		command.extend(['--run_dir', str(run_dir)])
	else:
		run_dir = paths.run_dir
	print(f'实验 {overrides.experiment} | 输出目录: {run_dir}', flush=True)
command.extend(training_args)
# Resolve explicit relative paths before the subprocess changes directory.
command.extend(['--netR_model_path', str(NETR_MODEL)])

try:
	subprocess.run(command, cwd=ROOT, check=True, env=env)
except subprocess.CalledProcessError as error:
	# The trainer already printed the cause; do not repeat its entire argument list.
	raise SystemExit(error.returncode) from None
