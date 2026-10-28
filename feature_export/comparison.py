"""Compare saved Rediff runs only after their ZeroDiff references pass replay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np

from feature_export.diagnostic import write_csv, write_json
from feature_export.reference import VCS_SCHEMA, read_result, triplet_signature, verify_baseline
from feature_export.relation_report import percent


def comparison_row(rediff, reference):
    verify_baseline(reference)
    if rediff.summary.get("schema_version") == VCS_SCHEMA:
        raise ValueError("Use compare_vcs_relations.py for VCS results; V and VCS rates cannot be mixed.")
    if rediff.summary.get("method") != "Rediff":
        raise ValueError("Expected a Rediff result package.")
    for key in ("dataset", "split", "sampling", "seeds", "samples_per_class", "min_gap", "tie_gap", "distance"):
        if rediff.summary[key] != reference.summary[key]:
            raise ValueError(f"Comparison settings differ: {key}.")
    if rediff.manifest.get("reference", {}).get("sha256") != reference.sha256:
        raise ValueError("Rediff result was generated against a different reference package.")
    if rediff.manifest.get("batch_size") != reference.manifest["batch_size"]:
        raise ValueError("Comparison batch sizes differ.")
    if rediff.manifest.get("data_alignment") != reference.manifest.get("data_alignment"):
        raise ValueError("Comparison data or preprocessing differ.")
    for key in ("class_ids", "class_names", "semantics", "real_centers", "real_counts", "semantic_distances", "real_distances"):
        if not np.array_equal(rediff.arrays[key], reference.arrays[key]):
            raise ValueError(f"Comparison reference arrays differ: {key}.")
    signature = triplet_signature(reference.rows)
    if triplet_signature(rediff.rows) != signature:
        raise ValueError("Comparison eligible sets or triplet indices differ.")
    baseline, actual = reference.records[0], rediff.records[0]
    return {"dataset": rediff.summary["dataset"], "eligible_triplets": actual["eligible_triplets"],
            "zerodiff_reversed_triplets": baseline["reversed_triplets"],
            "zerodiff_reversal_rate": baseline["reversal_rate"], "zerodiff_tied_triplets": baseline["tied_triplets"],
            "zerodiff_tie_rate": baseline["tie_rate"], "rediff_reversed_triplets": actual["reversed_triplets"],
            "rediff_reversal_rate": actual["reversal_rate"], "rediff_tied_triplets": actual["tied_triplets"],
            "rediff_tie_rate": actual["tie_rate"],
            "delta_percentage_points": 100 * (actual["reversal_rate"] - baseline["reversal_rate"]),
            "triplet_signature": signature, "reference_sha256": reference.sha256, "rediff_sha256": rediff.sha256}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rediff-results", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("out/relation_diagnostic/comparison"))
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new --output-dir.")
    references = {}
    for path in args.reference_results:
        result = read_result(path)
        verify_baseline(result)
        dataset = result.summary["dataset"]
        if dataset in references:
            raise ValueError(f"Duplicate reference dataset: {dataset}.")
        references[dataset] = result
    rows = []
    datasets = set()
    for path in args.rediff_results:
        result = read_result(path)
        dataset = result.summary["dataset"]
        if dataset in datasets or dataset not in references:
            raise ValueError(f"Duplicate Rediff dataset or missing reference: {dataset}.")
        datasets.add(dataset)
        rows.append(comparison_row(result, references[dataset]))
    if datasets != set(references):
        raise ValueError("Provide exactly one reference for each Rediff dataset.")
    order = {"AWA2": 0, "CUB": 1, "SUN": 2}
    rows.sort(key=lambda r: order[r["dataset"]])
    lines = ["# Rediff / ZeroDiff class relation comparison", "",
             "| Dataset | Eligible | ZeroDiff reversed / rate | Rediff reversed / rate | Change (pp) | ZeroDiff tie rate | Rediff tie rate |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['eligible_triplets']} | {row['zerodiff_reversed_triplets']} / "
                     f"{percent(row['zerodiff_reversal_rate'])} | {row['rediff_reversed_triplets']} / "
                     f"{percent(row['rediff_reversal_rate'])} | {row['delta_percentage_points']:+.6f} | "
                     f"{percent(row['zerodiff_tie_rate'])} | {percent(row['rediff_tie_rate'])} |")
    lines += ["", "Change = 100 × (Rediff reversal rate − ZeroDiff reversal rate), in percentage points.",
              "Negative values mean fewer reversals. Ties are counted separately.",
              "Each row uses the same original visual space and the same eligible triplet indices for both methods."]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".relation-comparison-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        write_csv(stage / "comparison.csv", rows)
        write_json(stage / "summary.json", {"schema_version": "class-relations-v1", "comparison": rows})
        (stage / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.rename(stage, output)
    print(f"Comparison: {output / 'report.md'}")
    return rows
