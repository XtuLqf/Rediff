"""Minimal frozen ZeroDiff generator runtime used only for diagnosis."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

import zerodiff_tools
from diagnostics.checkpoint_guard import load_baseline_checkpoint


class BaselineRuntime:
    """Load only the clean generator and optional encoder needed by the probe."""

    def __init__(self, options, checkpoint_path: Union[str, Path], device: str = "cpu"):
        self.options = options
        self.device = torch.device(device)
        self.n_T = int(options.n_T)
        self.prior = zerodiff_tools.ddpmgan_prior_coefficients(
            options.ddpmbeta1,
            options.ddpmbeta2,
            self.n_T,
            self.device,
            False,
        )

        checkpoint = load_baseline_checkpoint(checkpoint_path, map_location="cpu")
        self.netG = zerodiff_tools.DFG_Generator(options).to(self.device)
        self.netG.load_state_dict(checkpoint["state_dict_G"], strict=True)
        self.netG.eval()

        self.netE = None
        if "state_dict_E" in checkpoint:
            self.netE = zerodiff_tools.Encoder(options).to(self.device)
            self.netE.load_state_dict(checkpoint["state_dict_E"], strict=True)
            self.netE.eval()
            for parameter in self.netE.parameters():
                parameter.requires_grad_(False)

    @property
    def latent_source(self) -> str:
        return "encoder" if self.netE is not None else "seeded_random"

    @property
    def relation_signal_retention(self) -> torch.Tensor:
        """Signal power in the x_{t+1} input used to predict x_0."""
        return self.prior.sqrt_alphas_bar[1 : self.n_T + 1].square()

    def make_latent(
        self,
        visual: torch.Tensor,
        attributes: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        if self.netE is not None:
            with torch.no_grad():
                latent, _, _ = self.netE(visual, attributes)
            return latent.detach()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(
            visual.shape[0],
            self.options.noiseSize,
            generator=generator,
            device=self.device,
        )

    def make_shared_noise(self, visual: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(
            visual.shape,
            generator=generator,
            device=self.device,
            dtype=visual.dtype,
        )

    def noisy_input(
        self,
        visual: torch.Tensor,
        timestep: int,
        shared_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Return a coupled q(x_{t+1}|x_0) sample using common random numbers."""
        index = timestep + 1
        signal = self.prior.sqrt_alphas_bar[index]
        noise = self.prior.sigmas_bar[index]
        return signal * visual + noise * shared_noise

    def predict_x0(
        self,
        visual: torch.Tensor,
        contrastive: torch.Tensor,
        attributes: torch.Tensor,
        timestep: int,
        latent: torch.Tensor,
        shared_noise: torch.Tensor,
    ) -> torch.Tensor:
        times = torch.full(
            (visual.shape[0],), timestep, dtype=torch.long, device=self.device
        )
        x_tp1 = self.noisy_input(visual, timestep, shared_noise)
        return self.netG(latent, attributes, contrastive, x_tp1, times)
