"""Small helpers for averaging repeated diagnostic episodes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

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


def aggregate_by_timestep(
    rows: Sequence[Dict[str, str]], key: str
) -> List[Dict[str, float]]:
    summaries = []
    for timestep in sorted({int(row["timestep"]) for row in rows}):
        values = np.asarray(
            [
                float(row[key])
                for row in rows
                if int(row["timestep"]) == timestep and key in row
            ],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        summaries.append(
            {
                "timestep": timestep,
                "mean": float(values.mean()) if values.size else np.nan,
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "n": int(values.size),
            }
        )
    return summaries
