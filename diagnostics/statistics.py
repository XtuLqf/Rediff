"""Paired bootstrap summaries for repeated timestep diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def load_metric_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for run_index, path in enumerate(paths):
        with path.open("r", newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                copied = dict(row)
                copied["_run_index"] = str(run_index)
                copied["_source"] = str(path)
                rows.append(copied)
    if not rows:
        raise ValueError("No diagnostic metric rows were loaded.")
    return rows


def evaluation_unit(row: Dict[str, str]) -> Tuple[str, str]:
    return row["_run_index"], row["episode"]


def bootstrap_mean(
    values: Iterable[float],
    rng: np.random.Generator,
    samples: int = 2000,
) -> Tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return np.nan, np.nan, np.nan, 0
    mean = float(array.mean())
    if array.size == 1 or samples <= 0:
        return mean, mean, mean, int(array.size)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    bootstrap = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return mean, float(low), float(high), int(array.size)


def aggregate_by_timestep(
    rows: Sequence[Dict[str, str]],
    key: str,
    rng: np.random.Generator,
    samples: int = 2000,
) -> List[Dict[str, float]]:
    timesteps = sorted({int(row["timestep"]) for row in rows})
    summaries = []
    for timestep in timesteps:
        values = (
            float(row[key])
            for row in rows
            if int(row["timestep"]) == timestep and key in row
        )
        mean, low, high, count = bootstrap_mean(values, rng, samples)
        summaries.append(
            {
                "timestep": timestep,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n": count,
            }
        )
    return summaries


def paired_endpoint_summary(
    rows: Sequence[Dict[str, str]],
    key: str,
    rng: np.random.Generator,
    samples: int = 5000,
) -> Dict[str, float]:
    timesteps = sorted({int(row["timestep"]) for row in rows})
    first, last = timesteps[0], timesteps[-1]
    paired: Dict[Tuple[str, str], Dict[int, float]] = {}
    for row in rows:
        if key not in row:
            continue
        value = float(row[key])
        if np.isfinite(value):
            paired.setdefault(evaluation_unit(row), {})[int(row["timestep"])] = value
    deltas = np.asarray(
        [values[last] - values[first] for values in paired.values() if first in values and last in values],
        dtype=float,
    )
    mean, low, high, count = bootstrap_mean(deltas, rng, samples)
    if count == 0:
        p_value = np.nan
    elif np.allclose(deltas, 0.0):
        p_value = 1.0
    else:
        signs = rng.choice((-1.0, 1.0), size=(samples, count))
        null_means = (signs * deltas[None, :]).mean(axis=1)
        p_value = float((np.count_nonzero(np.abs(null_means) >= abs(mean)) + 1) / (samples + 1))
    return {
        "timestep_start": first,
        "timestep_end": last,
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "n": count,
        "paired_permutation_p": p_value,
    }


def write_statistics(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
