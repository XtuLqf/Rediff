"""Read the same AWA2 visual space used by ZeroDiff's DATA_LOADER."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def matlab_name(value) -> str:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.ndarray):
        return "".join(str(item) for item in value.reshape(-1))
    return str(value)


@dataclass
class AWA2Data:
    real_features: np.ndarray
    real_labels: np.ndarray
    real_indices: np.ndarray
    class_ids: np.ndarray
    class_names: np.ndarray
    attributes: torch.Tensor
    alignment: dict
    files: dict


def load_awa2(dataroot: Path, options) -> AWA2Data:
    root = dataroot / "AWA2"
    paths = {name: root / name for name in ("res101.mat", "att_splits.mat", "ce_ce.mat")}
    if options.split_percent != 100:
        name = f"split_{options.split_percent}percent.mat"
        paths[name] = root / name
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = sio.loadmat(paths["res101.mat"], variable_names=["labels"])["labels"]
    labels = np.asarray(labels, dtype=np.int64).reshape(-1) - 1
    splits = sio.loadmat(paths["att_splits.mat"])
    train_indices = np.asarray(splits["trainval_loc"], dtype=np.int64).reshape(-1) - 1
    real_indices = np.asarray(splits["test_unseen_loc"], dtype=np.int64).reshape(-1) - 1
    for indices in (train_indices, real_indices):
        if not indices.size or indices.min() < 0 or indices.max() >= labels.size:
            raise ValueError("Invalid AWA2 split indices.")
        if np.unique(indices).size != indices.size:
            raise ValueError("Duplicate AWA2 split indices.")
    if np.intersect1d(train_indices, real_indices).size:
        raise ValueError("Training and unseen-test samples overlap.")
    if np.intersect1d(labels[train_indices], labels[real_indices]).size:
        raise ValueError("Seen and unseen class labels overlap.")
    features = np.asarray(sio.loadmat(paths["ce_ce.mat"])["features"])
    if features.shape != (labels.size, options.resSize):
        raise ValueError("ce_ce.mat must contain features shaped [samples, resSize].")
    train = features[train_indices]
    real = features[real_indices]
    if not np.isfinite(train).all() or not np.isfinite(real).all():
        raise ValueError("Non-finite AWA2 visual features.")

    preprocessing = {"enabled": bool(options.preprocessing), "standardization": bool(options.standardization),
                     "fit_split": "full_trainval_before_percentage_subsampling"}
    if options.preprocessing:
        scaler = StandardScaler() if options.standardization else MinMaxScaler()
        # DATA_LOADER passes float32 *Torch tensors* to sklearn. Passing NumPy
        # arrays directly can select a different compute dtype in sklearn.
        train = scaler.fit_transform(torch.from_numpy(train).float())
        real = scaler.transform(torch.from_numpy(real).float())
        # Match DATA_LOADER: float32 first, then multiply by reciprocal max.
        maximum = torch.from_numpy(train).float().max()
        if not torch.isfinite(maximum) or maximum <= 0:
            raise ValueError("Invalid training feature normalization maximum.")
        real = torch.from_numpy(real).float().mul_(1 / maximum).numpy()
        preprocessing["float32_training_max"] = maximum.item()
        preprocessing["scaler_state"] = {
            key: array_sha256(value) for key, value in vars(scaler).items()
            if isinstance(value, np.ndarray)
        }
    real = np.ascontiguousarray(real, dtype=np.float32)
    if not np.isfinite(real).all():
        raise ValueError("Non-finite preprocessed real features.")
    attributes = torch.from_numpy(np.asarray(splits["original_att"].T)).float()
    if options.class_embedding_norm:
        attributes = torch.nn.functional.normalize(attributes, p=2, dim=1)
    if attributes.shape != (50, options.attSize) or not torch.isfinite(attributes).all():
        raise ValueError("Expected finite AWA2 attributes shaped [50, attSize].")
    if labels.min() < 0 or labels.max() >= 50:
        raise ValueError("AWA2 labels must be MATLAB class IDs 1..50 before conversion.")
    real_labels = labels[real_indices]
    class_ids = np.unique(real_labels)
    if class_ids.size != 10:
        raise ValueError("This AWA2 protocol requires all 10 unseen classes.")
    names = splits["allclasses_names"].reshape(-1)
    if names.size != 50:
        raise ValueError("Expected 50 AWA2 class names.")
    class_names = np.asarray([matlab_name(names[index]) for index in class_ids], dtype=np.str_)
    files = {name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
             for name, path in paths.items()}
    # The percentage-specific split is provenance, not a shared-space invariant.
    alignment = {
        "dataset": "AWA2", "feature_space": "V/ce_ce", "feature_dim": options.resSize,
        "label_convention": "dataset_zero_based", "class_ids": class_ids.tolist(),
        "class_names": class_names.tolist(),
        "data_sha256": {name: files[name]["sha256"] for name in ("res101.mat", "att_splits.mat", "ce_ce.mat")},
        "preprocessing": preprocessing,
        "attributes_sha256": array_sha256(attributes.numpy()),
        "real_features_sha256": array_sha256(real),
        "real_labels_sha256": array_sha256(real_labels),
        "real_indices_sha256": array_sha256(real_indices),
    }
    return AWA2Data(real, real_labels, real_indices, class_ids, class_names, attributes, alignment, files)
