from diagnostics.statistics import aggregate_by_timestep


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
    summary = aggregate_by_timestep(synthetic_rows(), "metric")
    assert [item["timestep"] for item in summary] == [0, 1, 2, 3]
    assert summary[0]["mean"] > summary[-1]["mean"]
    assert all(item["n"] == 10 for item in summary)
    assert all(item["std"] > 0 for item in summary)
