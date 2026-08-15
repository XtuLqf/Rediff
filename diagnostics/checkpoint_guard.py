"""Fail-fast validation for uncontaminated ZeroDiff checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Union

import torch


FORBIDDEN_KEY_FRAGMENTS = (
    "relproj",
    "relationprojector",
    "cteach",
    "vsra",
    "rel_gate",
    "optimizerrel",
    "method_metadata",
)

REQUIRED_DFG_KEYS = (
    "state_dict_G",
    "state_dict_Dec",
    "state_dict_D_x0",
    "state_dict_D_xt",
    "state_dict_D_xc",
)


class ContaminatedCheckpointError(ValueError):
    """Raised when a checkpoint contains post-baseline relation modules."""


def checkpoint_sha256(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_keys(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            yield full_key
            yield from _flatten_keys(child, full_key)


def validate_baseline_checkpoint(
    checkpoint: Mapping[str, Any],
    required_keys: Iterable[str] = REQUIRED_DFG_KEYS,
) -> None:
    flattened = tuple(key.lower() for key in _flatten_keys(checkpoint))
    contaminated = sorted(
        key
        for key in flattened
        if any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS)
    )
    if contaminated:
        preview = ", ".join(contaminated[:8])
        raise ContaminatedCheckpointError(
            "Checkpoint contains post-baseline relation/VSRA state: " + preview
        )

    missing = sorted(set(required_keys) - set(checkpoint))
    if missing:
        raise KeyError(f"Baseline DFG checkpoint is missing keys: {missing}")


def load_baseline_checkpoint(
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    required_keys: Iterable[str] = REQUIRED_DFG_KEYS,
) -> Dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        # Compatibility fallback for the original PyTorch 1.12 environment.
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary checkpoint.")
    validate_baseline_checkpoint(checkpoint, required_keys=required_keys)
    return checkpoint

