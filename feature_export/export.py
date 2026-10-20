"""Export AWA2 high-dimensional features for a separate plotting project.

Run from a ZeroDiff/Rediff repository root: python -m feature_export.export --help
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import scipy
import sklearn
import torch

from feature_export import SCHEMA_VERSION
from feature_export.data import array_sha256, file_sha256, load_awa2
from feature_export.runtime import FeatureRuntime, generate_features, load_options


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["AWA2"], default="AWA2")
    parser.add_argument("--dataroot", type=Path, default=Path("Dataset"))
    parser.add_argument("--dfg-checkpoint", type=Path, required=True)
    parser.add_argument("--drg-checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path,
                        help="Training JSON; auto-detects config.json beside the DFG checkpoint.")
    parser.add_argument("--method", required=True, help="Display name, e.g. ZeroDiff or Rediff.")
    parser.add_argument("--split-percent", type=int, choices=[10, 30, 100], required=True)
    parser.add_argument("--view", choices=["generated-only", "real-vs-generated", "both"], default="both")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--sampling", choices=["one-step", "progressive"], required=True)
    parser.add_argument("--seed", type=int, default=9182)
    parser.add_argument("--batch-size", type=int, default=100, help="Generation batch size per class.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--protocol", type=Path, help="Reuse another export's protocol.json for alignment checks.")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: out/plot_features/AWA2/<split-percent>pct; existing paths are never overwritten.")
    args = parser.parse_args(argv)
    if args.samples_per_class <= 0 or args.batch_size <= 0:
        parser.error("sample count and batch size must be positive")
    if not 0 <= args.seed < 2**63 - 1:
        parser.error("seed must be in [0, 2**63 - 1)")
    if not args.method.strip():
        parser.error("method must not be blank")
    if args.output_dir is None:
        args.output_dir = Path("out") / "plot_features" / args.dataset / f"{args.split_percent}pct"
    return args


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def make_protocol(data, args):
    return {
        "schema_version": SCHEMA_VERSION,
        "alignment": data.alignment,
        "generation": {
            "samples_per_class": args.samples_per_class, "sampling": args.sampling,
            "seed": args.seed, "batch_size": args.batch_size,
            "rng_policy": "per_class_seed=(seed+1000003*class_id)%(2**63-1);initial_cpu;posterior_device",
        },
    }


def validate_protocol(expected, actual, prefix="protocol"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise ValueError(f"Protocol fields differ at {prefix}.")
        for key in expected:
            validate_protocol(expected[key], actual[key], prefix + "." + key)
    elif expected != actual:
        raise ValueError(f"Protocol mismatch at {prefix}; use matching data and export settings.")


def make_packages(data, fake_features, fake_labels, count):
    expected_labels = np.repeat(data.class_ids, count)
    if not np.array_equal(fake_labels, expected_labels):
        raise ValueError("Generated labels must follow class_ids, with count samples per class.")
    if fake_features.shape != (expected_labels.size, data.real_features.shape[1]):
        raise ValueError("Generated feature shape does not match the data protocol.")
    if not np.isfinite(fake_features).all():
        raise ValueError("Non-finite generated features.")
    fake_ids = np.asarray([f"generated:{class_id}:{index}" for class_id in data.class_ids
                           for index in range(count)], dtype=np.str_)
    shared = {"class_ids": data.class_ids, "class_names": data.class_names,
              "schema_version": np.asarray(SCHEMA_VERSION)}
    generated = {
        **shared, "features": fake_features.astype(np.float32, copy=False), "labels": fake_labels,
        "source": np.ones(fake_labels.size, dtype=np.uint8), "sample_ids": fake_ids,
        "dataset_indices": np.full(fake_labels.size, -1, dtype=np.int64),
    }
    real_ids = np.asarray([f"real:{index}" for index in data.real_indices], dtype=np.str_)
    combined = {
        **shared,
        "features": np.concatenate([data.real_features, generated["features"]]),
        "labels": np.concatenate([data.real_labels, fake_labels]),
        "source": np.concatenate([np.zeros(data.real_labels.size, dtype=np.uint8), generated["source"]]),
        "sample_ids": np.concatenate([real_ids, fake_ids]),
        "dataset_indices": np.concatenate([data.real_indices, generated["dataset_indices"]]),
    }
    return {"generated_only.npz": generated, "real_vs_generated.npz": combined}


def git_provenance(root: Path):
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                                           text=True, encoding="utf-8", stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain")}


def write_export(output: Path, packages, protocol, metadata):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the directory only after every array and metadata file is written.
    with tempfile.TemporaryDirectory(prefix=".feature-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        artifacts = {}
        for name, arrays in packages.items():
            path = staging / name
            np.savez_compressed(path, **arrays)
            artifacts[name] = {
                "sha256": file_sha256(path), "rows": int(arrays["features"].shape[0]),
                "feature_dim": int(arrays["features"].shape[1]),
                "real_rows": int((arrays["source"] == 0).sum()),
                "generated_rows": int((arrays["source"] == 1).sum()),
                "array_sha256": {key: array_sha256(value) for key, value in arrays.items()},
            }
        manifest = {
            **metadata, "schema_version": SCHEMA_VERSION,
            "protocol_sha256": canonical_sha256(protocol), "artifacts": artifacts,
        }
        for name, document in (("protocol.json", protocol), ("manifest.json", manifest)):
            (staging / name).write_text(json.dumps(document, ensure_ascii=False, sort_keys=True,
                                                  indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.rename(staging, output)
    return manifest


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new directory.")
    for path in (args.dfg_checkpoint, args.drg_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    config_path = args.training_config
    if config_path is None:
        candidate = args.dfg_checkpoint.parent / "config.json"
        if candidate.is_file():
            config_path = candidate
    options, training = load_options(args.split_percent, config_path)
    if config_path is None:
        print("No training config found: using published AWA2 inference defaults; "
              "training percentage and DRG pairing are user-declared.", flush=True)
    dfg_hash = file_sha256(args.dfg_checkpoint)
    drg_hash = file_sha256(args.drg_checkpoint)
    if training.get("drg_sha256") and training["drg_sha256"] != drg_hash:
        raise ValueError("DRG checkpoint hash disagrees with the DFG training config.")
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("Only cpu and cuda devices are supported.")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    if resolved.type == "cuda":
        # cuBLAS requires this for deterministic matmul on supported CUDA builds.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = load_awa2(args.dataroot, options)
    protocol = make_protocol(data, args)
    if args.protocol is not None:
        validate_protocol(json.loads(args.protocol.read_text(encoding="utf-8")), protocol)
    runtime = FeatureRuntime(options, args.dfg_checkpoint, args.drg_checkpoint, device)
    print(f"Exporting {args.method}: AWA2 {args.split_percent}%, "
          f"10 unseen classes x {args.samples_per_class}, {args.sampling}, {device}", flush=True)
    fake_features, fake_labels = generate_features(
        runtime, data, args.samples_per_class, args.seed, args.batch_size, args.sampling,
    )
    packages = make_packages(data, fake_features, fake_labels, args.samples_per_class)
    if args.view != "both":
        name = "generated_only.npz" if args.view == "generated-only" else "real_vs_generated.npz"
        packages = {name: packages[name]}
    exporter_root = Path(__file__).resolve().parent
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "AWA2", "method": args.method, "split_percent": args.split_percent,
        "view": args.view, "feature_space": "V/ce_ce", "device": str(runtime.device),
        "inference_options": vars(options), "generation": protocol["generation"],
        "dfg_checkpoint": {"path": str(args.dfg_checkpoint.resolve()), "sha256": dfg_hash},
        "drg_checkpoint": {"path": str(args.drg_checkpoint.resolve()), "sha256": drg_hash},
        "training_config": ({"path": str(config_path.resolve()), "sha256": file_sha256(config_path)}
                            if config_path is not None else None),
        "training_config_status": "provided_or_auto_detected" if config_path else "published_defaults_user_declared",
        "checkpoint_method_metadata": runtime.method_metadata,
        "checkpoint_has_relation_state": runtime.has_relation_state,
        "data_files": data.files, "git": git_provenance(exporter_root.parent),
        "exporter_source_sha256": {path.name: file_sha256(path) for path in sorted(exporter_root.glob("*.py"))},
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    manifest = write_export(output, packages, protocol, metadata)
    print(f"Saved: {output}")
    for name, artifact in manifest["artifacts"].items():
        print(f"  {name}: {artifact['real_rows']} real + {artifact['generated_rows']} generated")
    return output


if __name__ == "__main__":
    main()
