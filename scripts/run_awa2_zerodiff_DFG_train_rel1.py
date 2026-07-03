#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the AWA2 DFG gamma_rel=1.0 contrastive/semantic teacher control."""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / 'scripts' / 'run_awa2_zerodiff_DFG_train.py'

command = [
    sys.executable,
    str(BASE_RUNNER),
    '--gamma_rel', '1.0',
    '--rel_sem_weight', '1.0',
    '--rel_con_weight', '1.0',
    '--eval_c_scales', '1.0,0.5,0.25,0.1,0.0',
    '--eval_c_scale_modalities', 'VC,VCS',
]

command.extend(sys.argv[1:])
subprocess.run(command, cwd=ROOT, check=True)
