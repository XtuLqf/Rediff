"""Compare classifier-native VCS and its paired V control on fixed triplets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from feature_export.diagnostic import write_csv, write_json
from feature_export.reference import VCS_SCHEMA, read_result, verify_baseline, with_preservation
from feature_export.relation_report import percent
from feature_export.vcs import check_visual_replay, validate_shared_reference, validate_visual_source


def comparison_rows(rediff, zerodiff, reference):
    verify_baseline(reference)
    for result, method in ((rediff, "Rediff"), (zerodiff, "ZeroDiff")):
        if result.summary.get("schema_version") != VCS_SCHEMA or result.summary.get("method") != method:
            raise ValueError(f"Expected a {method} VCS result package.")
        validate_shared_reference(result, reference)
        if result.manifest.get("reference", {}).get("sha256") != reference.sha256:
            raise ValueError("VCS runs used a different original V reference package.")
        if result.manifest.get("batch_size") != 100 or result.manifest.get("decoder", {}).get("strict") is not True:
            raise ValueError("VCS comparison requires batch size 100 and strict decoder loading.")
        if result.summary.get("block_scaling") != "classifier_native_no_extra_normalization":
            raise ValueError("VCS blocks must retain classifier-native scaling.")
        if result.summary["visual_control"]["source_results_sha256"] != result.manifest.get("visual_source", {}).get("sha256"):
            raise ValueError("VCS visual source fingerprints disagree.")
    for key in ("feature_blocks", "generated_feature_dim", "decoder_hidden_dim"):
        if rediff.summary[key] != zerodiff.summary[key]:
            raise ValueError(f"VCS layouts differ: {key}.")
    validate_visual_source(reference, reference, "ZeroDiff", zerodiff.manifest["dfg_checkpoint"]["sha256"],
                           zerodiff.manifest["drg_checkpoint"]["sha256"])
    if zerodiff.summary["visual_control"]["source_results_sha256"] != reference.sha256:
        raise ValueError("ZeroDiff VCS did not replay the original ZeroDiff V result.")
    control = zerodiff.summary["visual_control"]["per_seed"][0]
    check_visual_replay(zerodiff.arrays["generated_centers"][0, :, :2048],
                        zerodiff.arrays["generated_visual_distances"][0], reference, 0, control)
    rows = []
    for space in ("V", "VCS"):
        baseline = (with_preservation(zerodiff.records[0]) if space == "VCS" else
                    zerodiff.summary["visual_control"]["per_seed"][0])
        actual = (with_preservation(rediff.records[0]) if space == "VCS" else
                  rediff.summary["visual_control"]["per_seed"][0])
        row = {"dataset": rediff.summary["dataset"], "generated_space": space,
               "seed": actual["seed"], "eligible_triplets": actual["eligible_triplets"]}
        for method, record in (("zerodiff", baseline), ("rediff", actual)):
            for key in ("preserved_triplets", "preservation_rate", "reversed_triplets", "reversal_rate", "tied_triplets", "tie_rate"):
                row[method + "_" + key] = record[key]
        row.update(delta_preservation_percentage_points=100 * (actual["preservation_rate"] - baseline["preservation_rate"]),
                   delta_reversal_percentage_points=100 * (actual["reversal_rate"] - baseline["reversal_rate"]),
                   reference_sha256=reference.sha256, rediff_vcs_sha256=rediff.sha256, zerodiff_vcs_sha256=zerodiff.sha256)
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", type=Path, nargs="+", required=True, help="Original verified ZeroDiff V ZIPs.")
    parser.add_argument("--zerodiff-results", type=Path, nargs="+", required=True, help="New ZeroDiff VCS ZIPs.")
    parser.add_argument("--rediff-results", type=Path, nargs="+", required=True, help="New Rediff VCS ZIPs.")
    parser.add_argument("--output-dir", type=Path, default=Path("out/relation_diagnostic/comparison_vcs"))
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new --output-dir.")

    def by_dataset(paths):
        result = {}
        for path in paths:
            package = read_result(path)
            dataset = package.summary["dataset"]
            if dataset in result:
                raise ValueError(f"Duplicate dataset: {dataset}.")
            result[dataset] = package
        return result

    references, zerodiff, rediff = [by_dataset(paths) for paths in
                                   (args.reference_results, args.zerodiff_results, args.rediff_results)]
    if set(references) != set(zerodiff) or set(references) != set(rediff):
        raise ValueError("Provide exactly one original V reference and two VCS results per dataset.")
    rows = []
    for dataset in ("AWA2", "CUB", "SUN"):
        if dataset in references:
            rows.extend(comparison_rows(rediff[dataset], zerodiff[dataset], references[dataset]))
    lines = ["# V / VCS relation preservation on fixed reference triplets", "",
             "| Dataset | Space | Eligible | ZeroDiff preserved | Rediff preserved | Preservation change (pp) | ZeroDiff reversed | Rediff reversed | ZeroDiff tied | Rediff tied |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['generated_space']} | {row['eligible_triplets']} | "
                     f"{percent(row['zerodiff_preservation_rate'])} | {percent(row['rediff_preservation_rate'])} | "
                     f"{row['delta_preservation_percentage_points']:+.6f} | {row['zerodiff_reversed_triplets']} | "
                     f"{row['rediff_reversed_triplets']} | {row['zerodiff_tied_triplets']} | {row['rediff_tied_triplets']} |")
    lines += ["", "Preservation change = 100 × (Rediff − ZeroDiff); positive means higher strict preservation.",
              "Both spaces use the same generated samples and the same eligibility from semantics and real V.",
              "VCS = [V, Dec(V), decoder hidden, generated C], using each method's own frozen decoder and native scaling.",
              "This is supplementary geometry evidence. It does not measure classification accuracy or preservation of real VCS geometry."]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vcs-comparison-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        write_csv(stage / "comparison.csv", rows)
        write_json(stage / "summary.json", {"schema_version": VCS_SCHEMA, "comparison": rows})
        (stage / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.rename(stage, output)
    print(f"VCS comparison: {output / 'report.md'}")
    return rows
