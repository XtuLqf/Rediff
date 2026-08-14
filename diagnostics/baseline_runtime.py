"""Minimal frozen ZeroDiff DFG runtime used only for measurement."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

import zerodiff_tools
from diagnostics.checkpoint_guard import load_baseline_checkpoint


def weighted_l1_attributes(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = (prediction - target).pow(2)
    weights = weights / weights.sum(1).sqrt().clamp_min(1e-12).unsqueeze(1)
    return (weights * (prediction - target).abs()).sum() / prediction.shape[0]


class BaselineRuntime:
    """Loads only modules that exist in the clean d9da5ab DFG checkpoint."""

    def __init__(self, options, checkpoint_path: str | Path, device: str = "cpu"):
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
        self.posterior = zerodiff_tools.ddpmgan_posterior_coefficients(
            options.ddpmbeta1,
            options.ddpmbeta2,
            self.n_T,
            self.device,
            False,
        )

        checkpoint = load_baseline_checkpoint(checkpoint_path, map_location=self.device)
        self.checkpoint = checkpoint
        self.netG = zerodiff_tools.DFG_Generator(options).to(self.device)
        self.netDec = zerodiff_tools.V2S_mapping(options, options.attSize).to(self.device)
        self.netD_x0 = zerodiff_tools.DFG_Discriminator_x0(options).to(self.device)
        self.netD_xt = zerodiff_tools.DFG_Discriminator_xt(options).to(self.device)
        self.netD_xc = zerodiff_tools.DFG_Discriminator_xc(options).to(self.device)
        self.netG.load_state_dict(checkpoint["state_dict_G"], strict=True)
        self.netDec.load_state_dict(checkpoint["state_dict_Dec"], strict=True)
        self.netD_x0.load_state_dict(checkpoint["state_dict_D_x0"], strict=True)
        self.netD_xt.load_state_dict(checkpoint["state_dict_D_xt"], strict=True)
        self.netD_xc.load_state_dict(checkpoint["state_dict_D_xc"], strict=True)

        self.netE = None
        if "state_dict_E" in checkpoint:
            self.netE = zerodiff_tools.Encoder(options).to(self.device)
            self.netE.load_state_dict(checkpoint["state_dict_E"], strict=True)
            self.netE.eval()

        for module in (self.netG, self.netDec, self.netD_x0, self.netD_xt, self.netD_xc):
            module.eval()
        for module in (self.netDec, self.netD_x0, self.netD_xt, self.netD_xc):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @property
    def latent_source(self) -> str:
        return "encoder" if self.netE is not None else "seeded_random"

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

    def predict_all_timesteps(
        self,
        visual: torch.Tensor,
        contrastive: torch.Tensor,
        attributes: torch.Tensor,
        latent: torch.Tensor,
        shared_noise: torch.Tensor,
        track_gradients: bool = False,
    ) -> List[torch.Tensor]:
        context = torch.enable_grad if track_gradients else torch.no_grad
        with context():
            return [
                self.predict_x0(
                    visual,
                    contrastive,
                    attributes,
                    timestep,
                    latent,
                    shared_noise,
                )
                for timestep in range(self.n_T)
            ]

    def posterior_mean(
        self, prediction: torch.Tensor, x_tp1: torch.Tensor, timestep: int
    ) -> torch.Tensor:
        times = torch.full(
            (prediction.shape[0],), timestep, dtype=torch.long, device=self.device
        )
        coefficient_x0 = zerodiff_tools.extract(
            self.posterior.posterior_mean_coef1, times, x_tp1.shape
        )
        coefficient_xt = zerodiff_tools.extract(
            self.posterior.posterior_mean_coef2, times, x_tp1.shape
        )
        return coefficient_x0 * prediction + coefficient_xt * x_tp1

    def base_generator_losses(
        self,
        visual: torch.Tensor,
        contrastive: torch.Tensor,
        attributes: torch.Tensor,
        prediction: torch.Tensor,
        timestep: int,
        shared_noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        times = torch.full(
            (visual.shape[0],), timestep, dtype=torch.long, device=self.device
        )
        x_tp1 = self.noisy_input(visual, timestep, shared_noise)
        x_t_fake = self.posterior_mean(prediction, x_tp1, timestep)

        reconstruction = F.binary_cross_entropy(
            prediction.clamp(1e-7, 1.0 - 1e-7), visual.detach(), reduction="sum"
        ) / visual.shape[0]
        adversarial_x0 = -self.netD_x0(prediction, attributes).mean()
        adversarial_xt = -self.netD_xt(
            x_t_fake, x_tp1, attributes, contrastive, times
        ).mean()
        adversarial_xc = -self.netD_xc(prediction, contrastive).mean()
        adversarial = (
            self.options.gamma_x0 * adversarial_x0
            + self.options.gamma_xt * adversarial_xt
            + adversarial_xc
        )
        semantic_reconstruction = weighted_l1_attributes(
            self.netDec(prediction), attributes
        )
        total = (
            self.options.gamma_VAE * reconstruction
            + self.options.gamma_ADV * adversarial
            + self.options.gamma_recons * semantic_reconstruction
        )
        return {
            "total": total,
            "reconstruction": reconstruction,
            "adversarial": adversarial,
            "semantic_reconstruction": semantic_reconstruction,
        }

