"""Generate and audit Rediff class relations against a verified ZeroDiff package."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import zipfile

import numpy as np

from feature_export.reference import SEEDS, read_result, sha256_file, triplet_signature, verify_baseline, validate_alignment
from feature_export.relations import class_centers, cosine_distances, score_triplets, choose_example
from feature_export.relation_report import figure_svg, report_markdown
from feature_export.specs import DATASET_SPECS


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-reference", type=Path, nargs="+", metavar="RESULTS_ZIP",
                        help="Replay fixed ZeroDiff results without data or checkpoints, then exit.")
    parser.add_argument("--dataset", choices=list(DATASET_SPECS))
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--feature-space", choices=["V", "VCS"], default="V")
    parser.add_argument("--method", choices=["Rediff", "ZeroDiff"], default="Rediff",
                        help="ZeroDiff is supported for the supplementary VCS diagnostic.")
    parser.add_argument("--visual-results", type=Path,
                        help="Original V results for these exact weights; required for VCS (ZeroDiff defaults to reference-results).")
    parser.add_argument("--dataroot", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--dfg-checkpoint", type=Path)
    parser.add_argument("--drg-checkpoint", type=Path, help="The actual DRG paired with this DFG during training.")
    parser.add_argument("--training-config", type=Path, help="Training JSON; otherwise config.json beside DFG, then embedded config.")
    parser.add_argument("--split-percent", type=int, choices=[100], default=100)
    parser.add_argument("--sampling", choices=["one-step"], default="one-step")
    parser.add_argument("--seeds", "--seed", type=int, nargs="+")
    parser.add_argument("--samples-per-class", type=int, choices=[100], default=100)
    parser.add_argument("--batch-size", type=int, choices=[100], default=100)
    parser.add_argument("--min-gap", type=float, choices=[0.05], default=0.05)
    parser.add_argument("--tie-gap", type=float, choices=[1e-6], default=1e-6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-inputs", action="store_true", help="Check input paths and replay the reference; do not load models.")
    args = parser.parse_args(argv)
    if args.verify_reference:
        if args.reference_results or args.dfg_checkpoint or args.drg_checkpoint or args.check_inputs:
            parser.error("--verify-reference is a separate replay-only mode")
        return args
    for name in ("dataset", "reference_results", "dfg_checkpoint", "drg_checkpoint"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for generation")
    args.seeds = args.seeds if args.seeds is not None else [SEEDS[args.dataset]]
    if args.seeds != [SEEDS[args.dataset]]:
        parser.error("This fixed comparison requires the dataset's single prescribed generation seed")
    if args.feature_space == "V" and (args.method != "Rediff" or args.visual_results is not None):
        parser.error("--method ZeroDiff and --visual-results are supported with --feature-space VCS")
    if args.feature_space == "VCS" and args.visual_results is None:
        args.visual_results = (args.reference_results if args.method == "ZeroDiff" else
                               ROOT / "out/relation_diagnostic" / f"{args.dataset}_rediff_{args.seeds[0]}" / "results.zip")
    if args.output_dir is None:
        name = (f"{args.dataset}_{args.method.lower()}_vcs_{args.seeds[0]}" if args.feature_space == "VCS" else
                f"{args.dataset}_rediff_{args.seeds[0]}")
        args.output_dir = ROOT / "out" / "relation_diagnostic" / name
    return args


def write_json(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    iterator = iter(rows)
    first = next(iterator)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def git_provenance():
    def query(*arguments):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True,
                                           encoding="utf-8", stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": query("rev-parse", "HEAD"), "working_tree_status": query("status", "--porcelain")}


def make_summary(dataset, seeds, rows, records, outcomes, reference):
    eligible = sum(row["eligible"] for row in rows)
    rates = [r["reversal_rate"] for r in records] if eligible else []
    aggregate = ({"mean_reversal_rate": float(np.mean(rates)),
                  "sample_sd_reversal_rate": float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
                  "min_reversal_rate": min(rates), "max_reversal_rate": max(rates),
                  "mean_tie_rate": float(np.mean([r["tie_rate"] for r in records]))} if eligible else None)
    summary = {"schema_version": "class-relations-v1", "method": "Rediff", "dataset": dataset,
               "split": "test_unseen_loc", "sampling": "one-step", "seeds": seeds,
               "samples_per_class": 100, "min_gap": 0.05, "tie_gap": 1e-6,
               "distance": "cosine_of_class_mean", "total_triplets": len(rows),
               "eligible_triplets": eligible, "coverage": eligible / len(rows),
               "per_seed": records, "aggregate": aggregate, "example": choose_example(rows, outcomes),
               "triplet_signature": triplet_signature(rows),
               "conclusion": ("undefined_no_eligible_triplets" if not eligible else
                              "reversals_observed" if any(r["reversed_triplets"] for r in records) else "no_reversals_observed")}
    if reference is not None and eligible:
        baseline = reference.records[0]
        summary["comparison"] = {
            "reference_sha256": reference.sha256, "zerodiff_seed": baseline["seed"],
            "zerodiff_reversed_triplets": baseline["reversed_triplets"],
            "zerodiff_reversal_rate": baseline["reversal_rate"], "zerodiff_tie_rate": baseline["tie_rate"],
            "delta_percentage_points": 100 * (records[0]["reversal_rate"] - baseline["reversal_rate"])}
    return summary


def audit_rows(rows, outcomes, seeds, arrays):
    for index, row in enumerate(rows):
        common = {"triplet_index": index, **row}
        for role in ("anchor", "near", "far"):
            common[role + "_class_id"] = int(arrays["class_ids"][row[role]])
            common[role + "_class_name"] = str(arrays["class_names"][row[role]])
        for seed, observations in zip(seeds, outcomes):
            yield {**common, "seed": seed, **observations[index]}


def write_results(output, summary, arrays, rows, outcomes, manifest):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new --output-dir.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".relation-diagnostic-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        write_json(stage / "summary.json", summary)
        write_csv(stage / "per_seed.csv", summary["per_seed"])
        write_csv(stage / "triplets.csv", audit_rows(rows, outcomes, summary["seeds"], arrays))
        np.savez_compressed(stage / "centers.npz", **arrays)
        (stage / "report.md").write_text(report_markdown(summary), encoding="utf-8")
        (stage / "figure1.svg").write_text(figure_svg(summary, rows, outcomes, arrays["class_names"]), encoding="utf-8")
        manifest = {**manifest, "artifacts": {p.name: sha256_file(p) for p in sorted(stage.iterdir())},
                    "integrity_policy": "manifest hashes the six payloads; manifest and results.zip are excluded to avoid self-reference"}
        write_json(stage / "manifest.json", manifest)
        with zipfile.ZipFile(stage / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.iterdir()):
                if path.name != "results.zip":
                    archive.write(path, path.name)
        os.rename(stage, output)
    return output


def configure_device(requested):
    import torch
    device = requested if requested != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda") or (resolved.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Unsupported or unavailable device: {device}.")
    if resolved.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return device


def make_manifest(args, adapter, data, reference, alignment):
    import torch
    import scipy
    import sklearn
    from feature_export.generation import RNG_POLICY
    source_paths = sorted((ROOT / "feature_export").glob("*.py")) + [
        ROOT / "scripts/diagnose_relations.py", ROOT / "scripts/compare_relations.py",
        ROOT / "scripts/compare_vcs_relations.py", ROOT / "scripts/run_vcs_diagnostics.py",
        ROOT / "zerodiff_tools.py", ROOT / "datasets/image_util.py", ROOT / "classifiers/classifier_images.py"]
    return {"schema_version": "class-relations-v1", "method": args.method, "dataset": args.dataset,
            "generated_space": args.feature_space,
            "created_utc": datetime.now(timezone.utc).isoformat(), "device": str(adapter.runtime.device),
            "options": vars(adapter.options), "batch_size": args.batch_size, "seeds": args.seeds,
            "rng_policy": RNG_POLICY, "git": git_provenance(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                         "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
            "dfg_checkpoint": {"path": str(args.dfg_checkpoint.resolve()), "sha256": adapter.dfg_sha256},
            "drg_checkpoint": {"path": str(args.drg_checkpoint.resolve()), "sha256": adapter.drg_sha256},
            "training_config": ({"path": str(adapter.config_path.resolve()), "sha256": sha256_file(adapter.config_path)}
                                if adapter.config_path else None),
            "training_config_status": adapter.config_status, "checkpoint_pairing": adapter.pairing_status,
            "checkpoint_method_metadata": adapter.runtime.method_metadata,
            "checkpoint_has_relation_state": adapter.runtime.has_relation_state,
            "data": data.files, "data_alignment": data.alignment,
            "reference": {"path": str(reference.path), "sha256": reference.sha256,
                          "acceptance": verify_baseline(reference), "alignment": alignment},
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in source_paths}}


def run(args, reference):
    from feature_export.data import load_dataset
    from feature_export.generation import VisualFeatureGenerator
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new --output-dir.")
    device = configure_device(args.device)
    adapter = VisualFeatureGenerator(dataset=args.dataset, device=device, dfg_checkpoint=args.dfg_checkpoint,
                                     drg_checkpoint=args.drg_checkpoint, training_config=args.training_config)
    data = load_dataset(args.dataroot, adapter.options)
    alignment = validate_alignment(reference, data)
    centers, distances, records, outcomes = [], [], [], []
    for seed in args.seeds:
        features, labels = adapter.generate(class_ids=data.class_ids, attributes=data.attributes,
                                            samples_per_class=args.samples_per_class, seed=seed,
                                            batch_size=args.batch_size, sampling=args.sampling)
        center, counts = class_centers(features, labels, data.class_ids)
        if not np.all(counts == args.samples_per_class):
            raise ValueError("Unexpected generated class counts.")
        distance = cosine_distances(center)
        scores, observations = score_triplets(reference.rows, distance, args.tie_gap)
        records.append({"seed": seed, **scores})
        outcomes.append(observations)
        centers.append(center)
        distances.append(distance)
        print(json.dumps(records[-1], allow_nan=False), flush=True)
    arrays = {key: value.copy() for key, value in reference.arrays.items()
              if key not in ("generated_centers", "generated_distances", "seeds")}
    arrays.update(generated_centers=np.stack(centers), generated_distances=np.stack(distances),
                  seeds=np.asarray(args.seeds, dtype=np.int64))
    summary = make_summary(args.dataset, args.seeds, reference.rows, records, outcomes, reference)
    manifest = make_manifest(args, adapter, data, reference, alignment)
    write_results(output, summary, arrays, reference.rows, outcomes, manifest)
    print(f"Results: {output / 'results.zip'}", flush=True)
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.verify_reference:
        records = [verify_baseline(read_result(path)) for path in args.verify_reference]
        print(json.dumps({"status": "passed", "references": records}, ensure_ascii=False, indent=2, allow_nan=False))
        return records
    paths = [args.reference_results, args.dfg_checkpoint, args.drg_checkpoint]
    split = DATASET_SPECS[args.dataset]["embedding"] + "_splits.mat"
    paths += [args.dataroot / args.dataset / name for name in ("res101.mat", "ce_ce.mat", split)]
    if args.training_config is not None:
        paths.append(args.training_config)
    if args.visual_results is not None:
        paths.append(args.visual_results)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = read_result(args.reference_results)
    verify_baseline(reference)
    if reference.summary["dataset"] != args.dataset:
        raise ValueError("--dataset disagrees with --reference-results.")
    if args.check_inputs:
        print(json.dumps({"status": "paths_and_reference_passed", "note": "Weights and local data alignment are checked during generation.",
                          "inputs": [str(p.resolve()) for p in paths]}, indent=2))
        return None
    if args.feature_space == "VCS":
        from feature_export.vcs import run_vcs
        return run_vcs(args, reference)
    return run(args, reference)
