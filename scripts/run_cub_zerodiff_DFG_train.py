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
OUT_DIR = ROOT / 'out' / 'CUB'
NETR_MODEL_CANDIDATES = [
	OUT_DIR / 'zerodiff_DRG_100percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:0.0_x0:1.0_xt:1.0_dist:1.0_num:300_gzsl.tar',
	OUT_DIR / 'zerodiff_DRG_100percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:0.0_x0:1.0_xt:1.0_dist:1.0_num:300_zsl.tar',
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
			f'No DRG checkpoint found in {OUT_DIR}. Please run the CUB DRG script first.'
		)

env = os.environ.copy()
env['OMP_NUM_THREADS'] = '3'

command = [
	sys.executable,
	'zerodiff_DFG_train.py',
	'--gzsl', '--encoded_noise', '--manualSeed', '3483', '--preprocessing', '--cuda', '--image_embedding', 'res101',
	'--class_embedding', 'sent', '--nepoch', '300', '--ngh', '4096', '--ndh', '4096', '--lambda1', '10', '--critic_iter', '5',
	'--nclass_all', '200', '--dataroot', str(DATAROOT), '--dataset', 'CUB', '--eval_interval', '5',
	'--batch_size', '64', '--noiseSize', '1024', '--attSize', '1024', '--resSize', '2048',
	'--lr', '0.0001', '--classifier_lr', '0.001', '--gamma_recons', '0.01', '--dec_lr', '0.0001',
	'--gamma_ADV', '10', '--gamma_VAE', '1.0', '--embed_type', 'VA',
	'--n_T', '4', '--dim_t', '1024', '--gamma_x0', '1.0', '--gamma_xt', '1.0',
	'--gamma_dist', '2.0', '--factor_dist', '1.5',
	'--split_percent', '100', '--syn_num', '1440',
	'--netR_model_path', str(NETR_MODEL),
]

subprocess.run(command, cwd=ROOT, check=True, env=env)

# split_percent 100:
# --split_percent 100 --syn_num 1440
# --netR_model_path ./out/CUB/zerodiff_DRG_100percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:300_gzsl.tar \

# split_percent 30:
# --split_percent 30 --syn_num 440
# --netR_model_path ./out/CUB/zerodiff_DRG_30percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:100_gzsl.tar \

# split_percent 10:
# --split_percent 10 --syn_num 140
# --netR_model_path ./out/CUB/zerodiff_DRG_10percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:40_gzsl.tar \