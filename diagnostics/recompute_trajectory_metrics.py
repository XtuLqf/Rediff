"""Recompute expanded topology metrics from an existing trajectory archive."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from diagnostics.relation_metrics import timestep_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upgrade an existing trajectory.npz without loading a model."
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    metadata_path = args.metadata or args.trajectory.with_name("metadata.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    diagnostic_seed = metadata.get("seed", "unknown")
    arrays = np.load(args.trajectory)
    rows = []
    for episode_id in range(arrays["predictions"].shape[0]):
        predictions = [
            torch.from_numpy(prediction)
            for prediction in arrays["predictions"][episode_id]
        ]
        episode_rows = timestep_metrics(
            predictions,
            torch.from_numpy(arrays["attributes"][episode_id]),
            torch.from_numpy(arrays["contrastive"][episode_id]),
            torch.from_numpy(arrays["labels"][episode_id]),
        )
        episode_seed = (
            int(diagnostic_seed) + episode_id
            if diagnostic_seed != "unknown"
            else "unknown"
        )
        for row in episode_rows:
            row.update(
                episode=episode_id,
                episode_seed=episode_seed,
                diagnostic_seed=diagnostic_seed,
                episode_uid=f"{diagnostic_seed}:{episode_id}",
            )
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
