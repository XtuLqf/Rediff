#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATAROOT = ROOT / 'Dataset'
OUT_DIR = ROOT / 'out' / 'SUN'
NETR_MODEL_CANDIDATES = [
	OUT_DIR / 'zerodiff_DRG_100percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:400_gzsl.tar',
	OUT_DIR / 'zerodiff_DRG_100percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:400_zsl.tar',
	OUT_DIR / 'diffzero_pretrain_100percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:400_gzsl.tar',
	OUT_DIR / 'diffzero_pretrain_100percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:400_zsl.tar',
]

for candidate in NETR_MODEL_CANDIDATES:
	if candidate.exists():
		NETR_MODEL = candidate
		break
else:
	available = sorted(OUT_DIR.glob('*.tar')) if OUT_DIR.exists() else []
	if available:
		NETR_MODEL = available[0]
	else:
		raise FileNotFoundError(
			f'No DRG checkpoint found in {OUT_DIR}. Please run the SUN DRG script first.'
		)

env = os.environ.copy()
env['OMP_NUM_THREADS'] = '3'

command = [
	sys.executable,
	'zerodiff_DFG_train.py',
	'--dataroot', str(DATAROOT),
	'--dataset', 'SUN', '--image_embedding', 'res101', '--class_embedding', 'att', '--class_embedding_norm', '--eval_interval', '5',
	'--gzsl', '--manualSeed', '4115', '--encoded_noise', '--preprocessing', '--cuda',
	'--nepoch', '400', '--ngh', '4096', '--ndh', '4096', '--lr', '0.0001', '--classifier_lr', '0.0005', '--lambda1', '10', '--critic_iter', '5',
	'--nclass_all', '717', '--batch_size', '64', '--noiseSize', '102', '--attSize', '102', '--resSize', '2048',
	'--gamma_recons', '0.01', '--dec_lr', '0.0001',
	'--gamma_ADV', '1', '--gamma_VAE', '1.0', '--embed_type', 'VA',
	'--n_T', '4', '--dim_t', '102', '--gamma_x0', '1.0', '--gamma_xt', '1.0', '--gamma_dist', '1.0', '--factor_dist', '1.5',
	'--split_percent', '100', '--syn_num', '400',
	'--netR_model_path', str(NETR_MODEL),
]

subprocess.run(command, cwd=ROOT, check=True, env=env)

# split_percent 100:
# --split_percent 100 --syn_num 400 \
# --netR_model_path ./out/SUN/diffzero_pretrain_100percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:400_gzsl.tar

# split_percent 30:
# --split_percent 30 --syn_num 120 \
# --netR_model_path ./out/SUN/diffzero_pretrain_30percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:120_gzsl.tar

# split_percent 10:
# --split_percent 10 --syn_num 40 \
# --netR_model_path ./out/SUN/diffzero_pretrain_10percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:40_gzsl.tar