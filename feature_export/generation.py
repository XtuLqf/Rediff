"""Rediff's frozen V sampler, with the class-relations-v1 interface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from feature_export.reference import sha256_file
from feature_export.runtime import FeatureRuntime, generate_features, load_options, load_weights
from feature_export.specs import DATASET_SPECS


RNG_POLICY = {
    "class_seed": "(seed + 1000003 * original_zero_based_class_id) % (2**63 - 1)",
    "initial_rng": "torch.Generator(device='cpu')",
    "draw_order": ["visual_noise[B,2048]", "dfg_latent[B,noiseSize]",
                   "drg_latent[B,noiseSize]", "representation_noise[B,2048]"],
    "batch_order": "ascending class_ids; consecutive batches within each class",
    "extra_random_operations": [],
}


class VisualFeatureGenerator:
    """Reusable adapter; model initialization never consumes the class RNGs."""

    def __init__(self, *, dataset, device, dfg_checkpoint, drg_checkpoint, training_config=None, feature_space="V"):
        if feature_space not in ("V", "VCS"):
            raise ValueError("feature_space must be V or VCS.")
        self.feature_space = feature_space
        dfg_checkpoint, drg_checkpoint = Path(dfg_checkpoint), Path(drg_checkpoint)
        config = Path(training_config) if training_config is not None else None
        if config is None and (dfg_checkpoint.parent / "config.json").is_file():
            config = dfg_checkpoint.parent / "config.json"
        embedded = load_weights(dfg_checkpoint).get("training_config")
        self.options, self.training = load_options(100, config, dataset=dataset,
                                                  training_values=embedded if config is None else None)
        if config is not None and embedded is not None:
            embedded_options, embedded_training = load_options(100, dataset=dataset, training_values=embedded)
            if vars(embedded_options) != vars(self.options):
                raise ValueError("Training config disagrees with embedded DFG inference options.")
            embedded_hash = embedded_training.get("drg_sha256")
            if embedded_hash and self.training.get("drg_sha256", embedded_hash) != embedded_hash:
                raise ValueError("DRG checkpoint hash declarations disagree.")
            self.training = {**embedded_training, **self.training}
        spec = DATASET_SPECS[dataset]
        fixed = {"split_percent": 100, "preprocessing": True, "standardization": False,
                 "class_embedding_norm": spec["normalize"], "n_T": 4,
                 "noiseSize": spec["semantic_dim"], "dim_t": spec["semantic_dim"]}
        for key, expected in fixed.items():
            if getattr(self.options, key) != expected:
                raise ValueError(f"class-relations-v1 requires {key}={expected!r}.")
        self.config_path = config
        self.config_status = ("provided_or_autodetected" if config else
                              "embedded_checkpoint" if embedded is not None else "published_defaults_user_declared")
        self.dfg_sha256 = sha256_file(dfg_checkpoint)
        self.drg_sha256 = sha256_file(drg_checkpoint)
        declared_hash = self.training.get("drg_sha256")
        if declared_hash and declared_hash != self.drg_sha256:
            raise ValueError("DRG checkpoint hash disagrees with the DFG training config.")
        self.pairing_status = "sha256_verified" if declared_hash else "user_supplied_unverified"
        self._runtime = None
        self._runtime_args = (self.options, dfg_checkpoint, drg_checkpoint, device)

    @property
    def runtime(self):
        if self._runtime is None:
            self._runtime = FeatureRuntime(*self._runtime_args, with_decoder=self.feature_space == "VCS")
        return self._runtime

    def generate(self, *, class_ids, attributes, samples_per_class, seed, batch_size, sampling):
        if sampling != "one-step":
            raise ValueError("This fixed diagnostic requires one-step sampling.")
        if (type(samples_per_class) is not int or samples_per_class < 1
                or type(batch_size) is not int or batch_size < 1
                or type(seed) is not int or not 0 <= seed < 2**63 - 1):
            raise ValueError("Invalid generation counts, batch size or seed.")
        spec = DATASET_SPECS[self.options.dataset]
        class_ids = np.asarray(class_ids)
        if (class_ids.dtype != np.dtype("int64") or class_ids.shape != (spec["unseen"],)
                or np.any(np.diff(class_ids) <= 0) or class_ids.min() < 0 or class_ids.max() >= spec["classes"]):
            raise ValueError("Expected ascending int64 IDs for all unseen classes.")
        attributes = torch.as_tensor(attributes, dtype=torch.float32, device="cpu")
        if attributes.shape != (spec["classes"], spec["semantic_dim"]) or not torch.isfinite(attributes).all():
            raise ValueError("Invalid full semantic table indexed by original class ID.")
        if spec["normalize"] and not torch.allclose(
                attributes.norm(dim=1), torch.ones(spec["classes"]), rtol=1e-5, atol=1e-6):
            raise ValueError("The generation semantic table must already be L2 normalized.")
        features, labels = generate_features(self.runtime, SimpleNamespace(class_ids=class_ids, attributes=attributes),
                                             samples_per_class, seed, batch_size, sampling, feature_space=self.feature_space)
        if not np.array_equal(labels, np.repeat(class_ids, samples_per_class)):
            raise ValueError("Unexpected generated class order.")
        return features, labels


def generate_visual_features(*, dataset: str, class_ids: np.ndarray, attributes,
                             samples_per_class: int, seed: int, batch_size: int,
                             sampling: str, device: str, dfg_checkpoint: str,
                             drg_checkpoint: str, training_config: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 V and int64 labels, grouped in the supplied class order.

    attributes is the full generation semantic table, with the dataset's
    prescribed L2 normalization already applied, indexed by original class ID.
    """
    adapter = VisualFeatureGenerator(dataset=dataset, device=device, dfg_checkpoint=dfg_checkpoint,
                                     drg_checkpoint=drg_checkpoint, training_config=training_config)
    return adapter.generate(class_ids=class_ids, attributes=attributes, samples_per_class=samples_per_class,
                            seed=seed, batch_size=batch_size, sampling=sampling)
