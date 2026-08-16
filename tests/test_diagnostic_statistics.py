import numpy as np
import pytest

from diagnostics.statistics import (
    aggregate_by_timestep,
    paired_endpoint_summary,
)


def synthetic_rows():
    rows = []
    for run in range(2):
        for episode in range(5):
            for timestep in range(4):
                rows.append(
                    {
                        "_run_index": str(run),
                        "episode": str(episode),
                        "timestep": str(timestep),
                        "metric": str(1.0 - 0.1 * timestep + 0.01 * episode),
                    }
                )
    return rows


def test_timestep_aggregation_preserves_monotonic_mean():
    summary = aggregate_by_timestep(
        synthetic_rows(), "metric", np.random.default_rng(7), samples=200
    )
    assert [item["timestep"] for item in summary] == [0, 1, 2, 3]
    assert summary[0]["mean"] > summary[-1]["mean"]
    assert all(item["n"] == 10 for item in summary)


def test_paired_endpoint_summary_detects_known_decline():
    summary = paired_endpoint_summary(
        synthetic_rows(), "metric", np.random.default_rng(11), samples=1000
    )
    assert summary["n"] == 10
    assert summary["mean"] == pytest.approx(-0.3)
    assert summary["ci_high"] < 0.0
