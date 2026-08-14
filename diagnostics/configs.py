"""Dataset-specific architecture presets copied from clean launch scripts."""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from typing import Dict


PRESETS: Dict[str, Dict[str, object]] = {
    "AWA2": {
        "class_embedding": "att",
        "class_embedding_norm": True,
        "noiseSize": 85,
        "attSize": 85,
        "dim_t": 85,
        "ngh": 4096,
        "ndh": 4096,
        "lr": 5e-4,
        "dec_lr": 1e-4,
        "gamma_ADV": 10.0,
        "gamma_VAE": 1.0,
        "gamma_recons": 1.0,
        "gamma_x0": 1.0,
        "gamma_xt": 1.0,
        "factor_dist": 1.5,
    },
    "CUB": {
        "class_embedding": "sent",
        "class_embedding_norm": False,
        "noiseSize": 1024,
        "attSize": 1024,
        "dim_t": 1024,
        "ngh": 4096,
        "ndh": 4096,
        "lr": 1e-4,
        "dec_lr": 1e-4,
        "gamma_ADV": 10.0,
        "gamma_VAE": 1.0,
        "gamma_recons": 0.01,
        "gamma_x0": 1.0,
        "gamma_xt": 1.0,
        "factor_dist": 1.5,
    },
    "SUN": {
        "class_embedding": "att",
        "class_embedding_norm": True,
        "noiseSize": 102,
        "attSize": 102,
        "dim_t": 102,
        "ngh": 4096,
        "ndh": 4096,
        "lr": 1e-4,
        "dec_lr": 1e-4,
        "gamma_ADV": 1.0,
        "gamma_VAE": 1.0,
        "gamma_recons": 0.01,
        "gamma_x0": 1.0,
        "gamma_xt": 1.0,
        "factor_dist": 1.5,
    },
}


COMMON = {
    "image_embedding": "res101",
    "resSize": 2048,
    "n_T": 4,
    "ddpmbeta1": 0.1,
    "ddpmbeta2": 20.0,
    "embed_type": "VA",
    "encoder_layer_sizes": [8192, 4096],
    "decoder_layer_sizes": [4096, 8192],
    "conditional": True,
    "preprocessing": True,
    "standardization": False,
    "validation": False,
    "split_percent": 100,
    "freeze_dec": False,
    "batch_size": 64,
    "lambda1": 10.0,
}


def make_options(dataset: str, dataroot: str, split_percent: int = 100) -> Namespace:
    name = dataset.upper()
    if name not in PRESETS:
        raise KeyError(f"Unknown dataset {dataset!r}; choose from {sorted(PRESETS)}")
    values = deepcopy(COMMON)
    values.update(deepcopy(PRESETS[name]))
    values.update(dataset=name, dataroot=dataroot, split_percent=split_percent)
    values["encoder_layer_sizes"][0] = values["resSize"]
    values["decoder_layer_sizes"][-1] = values["resSize"]
    values["latent_size"] = values["attSize"]
    return Namespace(**values)

