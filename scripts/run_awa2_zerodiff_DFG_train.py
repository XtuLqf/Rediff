#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZihanYe
"""
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

NETR_MODEL = next(
	(candidate for candidate in NETR_MODEL_CANDIDATES if candidate.exists()),
	None,
)
if NETR_MODEL is None:
	raise FileNotFoundError(
		f'No matching AWA2 100% DRG checkpoint found in {OUT_DIR}. '
		'Please run the AWA2 DRG script first.'
	)

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
command.extend(sys.argv[1:])

subprocess.run(command, cwd=ROOT, check=True, env=env)
