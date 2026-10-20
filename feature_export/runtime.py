"""Frozen DRG -> DFG inference matching the trainer's sampling equations."""

from __future__ import annotations

from argparse import Namespace
import json
import math
from pathlib import Path

import torch

import zerodiff_tools


DEFAULTS = {
    "dataset": "AWA2", "resSize": 2048, "attSize": 85, "noiseSize": 85,
    "dim_t": 85, "n_T": 4, "ddpmbeta1": 0.1, "ddpmbeta2": 20.0,
    "decoder_layer_sizes": [4096, 2048], "class_embedding": "att",
    "class_embedding_norm": True, "preprocessing": True,
    "standardization": False, "validation": False,
}


def load_options(split_percent: int, config_path: Path | None = None):
    values = dict(DEFAULTS)
    training = {}
    if config_path is not None:
        document = json.loads(config_path.read_text(encoding="utf-8"))
        training = document.get("training", document)
        if not isinstance(training, dict):
            raise ValueError("Training config must be a JSON object or contain a 'training' object.")
        if "split_percent" in training and training["split_percent"] != split_percent:
            raise ValueError("--split-percent disagrees with the training config.")
        values.update({key: training[key] for key in DEFAULTS if key in training})
    if values["dataset"] != "AWA2" or values["class_embedding"] != "att" or values["validation"]:
        raise ValueError("Only the AWA2 att/test_unseen protocol is supported.")
    if values["resSize"] != 2048 or values["attSize"] != 85:
        raise ValueError("This export interface requires AWA2 V=2048 and attributes=85.")
    if (not isinstance(values["n_T"], int) or values["n_T"] < 1
            or not isinstance(values["noiseSize"], int) or values["noiseSize"] < 1
            or not isinstance(values["dim_t"], int) or values["dim_t"] < 2):
        raise ValueError("Invalid generator dimensions or diffusion step count.")
    if not all(math.isfinite(values[key]) and values[key] > 0 for key in ("ddpmbeta1", "ddpmbeta2")):
        raise ValueError("Diffusion betas must be finite and positive.")
    layers = list(values["decoder_layer_sizes"])
    if len(layers) != 2 or any(not isinstance(size, int) or size <= 0 for size in layers):
        raise ValueError("decoder_layer_sizes must contain two positive integers.")
    # config_zerodiff.py rewrites the final layer after parsing, as does this tool.
    values["decoder_layer_sizes"] = [layers[0], values["resSize"]]
    values["split_percent"] = split_percent
    return Namespace(**values), training


def load_weights(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 1.12 compatibility, matching the original project.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a checkpoint dictionary: {path}")
    return checkpoint


class FeatureRuntime:
    def __init__(self, options, dfg_checkpoint: Path, drg_checkpoint: Path, device: str):
        self.options = options
        self.device = torch.device(device)
        # Initialization should not affect the local generation streams.
        self.netG = zerodiff_tools.DFG_Generator(options).to(self.device)
        self.netR = zerodiff_tools.DRG_Generator(options).to(self.device)
        dfg = load_weights(dfg_checkpoint)
        if "state_dict_G" not in dfg:
            raise KeyError("DFG checkpoint is missing state_dict_G.")
        self.netG.load_state_dict(dfg["state_dict_G"], strict=True)
        self.method_metadata = dfg.get("method_metadata")
        self.has_relation_state = "state_dict_VSRA" in dfg
        del dfg
        drg = load_weights(drg_checkpoint)
        key = next((key for key in ("state_dict_G_con", "state_dict_R") if key in drg), None)
        if key is None:
            raise KeyError("DRG checkpoint needs state_dict_G_con or state_dict_R.")
        self.netR.load_state_dict(drg[key], strict=True)
        del drg
        self.netG.eval().requires_grad_(False)
        self.netR.eval().requires_grad_(False)
        self.posterior = zerodiff_tools.ddpmgan_posterior_coefficients(
            options.ddpmbeta1, options.ddpmbeta2, options.n_T, self.device, False,
        )

    @torch.no_grad()
    def sample(self, attributes: torch.Tensor, sampling: str, cpu_rng, posterior_rng):
        if sampling not in ("one-step", "progressive"):
            raise ValueError(f"Unknown sampling mode: {sampling}")
        attributes = attributes.to(self.device)
        count = attributes.shape[0]
        options = self.options

        def initial_noise(width):
            # The original sample_from_model draws these four tensors on CPU.
            return torch.randn(count, width, generator=cpu_rng).to(self.device)

        x_tp1 = initial_noise(options.resSize)
        z = initial_noise(options.noiseSize)
        z_con = initial_noise(options.noiseSize)
        times = torch.full((count,), options.n_T - 1, dtype=torch.long, device=self.device)
        r_tp1 = initial_noise(2048)
        contrastive = self.netR(z_con, attributes, r_tp1, times)
        if sampling == "one-step":
            return self.netG(z, attributes, contrastive, x_tp1, times).cpu()
        for timestep in reversed(range(options.n_T)):
            times.fill_(timestep)
            prediction = self.netG(z, attributes, contrastive, x_tp1, times)
            posterior = self.posterior
            mean = (zerodiff_tools.extract(posterior.posterior_mean_coef1, times, x_tp1.shape) * prediction
                    + zerodiff_tools.extract(posterior.posterior_mean_coef2, times, x_tp1.shape) * x_tp1)
            log_var = zerodiff_tools.extract(posterior.posterior_log_variance_clipped, times, x_tp1.shape)
            noise = torch.randn(x_tp1.shape, generator=posterior_rng, device=self.device, dtype=x_tp1.dtype)
            x_tp1 = mean + (times != 0).float()[:, None] * torch.exp(0.5 * log_var) * noise
        return prediction.cpu()


def generate_features(runtime, data, count: int, seed: int, batch_size: int, sampling: str):
    features, labels = [], []
    for class_id in data.class_ids.tolist():
        class_seed = (seed + 1_000_003 * class_id) % (2**63 - 1)
        cpu_rng = torch.Generator(device="cpu").manual_seed(class_seed)
        posterior_rng = (cpu_rng if runtime.device.type == "cpu" else
                         torch.Generator(device=runtime.device).manual_seed(class_seed))
        for start in range(0, count, batch_size):
            size = min(batch_size, count - start)
            attributes = data.attributes[class_id].repeat(size, 1)
            prediction = runtime.sample(attributes, sampling, cpu_rng, posterior_rng)
            if prediction.shape != (size, runtime.options.resSize) or not torch.isfinite(prediction).all():
                raise ValueError("Generator returned invalid visual features.")
            features.append(prediction)
            labels.extend([class_id] * size)
    import numpy as np

    return torch.cat(features).numpy().astype(np.float32, copy=False), np.asarray(labels, dtype=np.int64)
