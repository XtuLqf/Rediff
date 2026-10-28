#!/usr/bin/env python3
"""Run both methods' VCS diagnostics using checkpoint paths in existing V packages."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feature_export.reference import SEEDS, read_result, verify_baseline


REFERENCE_FOLDERS = {"AWA2": "AWA2_100pct_onestep", "CUB": "CUB_100pct_onestep_seed3483",
                     "SUN": "SUN_100pct_onestep_seed4115"}


def checkpoint_paths(package):
    """Use saved provenance; ambiguous legacy names are an error, never a guess."""
    paths = {}
    for role in ("dfg", "drg"):
        explicit = package.manifest.get(role + "_checkpoint", {}).get("path")
        if explicit:
            paths[role] = explicit
            continue
        candidates = [path for path in package.manifest.get("inputs", {})
                      if ("_DFG_" in Path(path).name if role == "dfg" else
                          "_DRG_" in Path(path).name or Path(path).name.startswith("diffzero_pretrain"))]
        if len(candidates) != 1:
            raise ValueError(f"Cannot resolve the actual {role.upper()} from {package.path}; use diagnose_relations.py with explicit paths.")
        paths[role] = candidates[0]
    config = package.manifest.get("training_config")
    paths["config"] = config.get("path") if isinstance(config, dict) else config
    visual_file = package.manifest.get("data", {}).get("res101.mat", {}).get("path")
    paths["dataroot"] = str(Path(visual_file).parent.parent) if visual_file else None
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=list(SEEDS), default=list(SEEDS))
    parser.add_argument("--reference-results", type=Path, nargs="+", help="Original ZeroDiff V ZIPs; defaults to the neighboring Zerodiff-LQF repository.")
    parser.add_argument("--visual-results", type=Path, nargs="+", help="Original Rediff V ZIPs; defaults to this repository's out/relation_diagnostic.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "out/relation_diagnostic")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Read V provenance and print commands without loading weights or generating results.")
    args = parser.parse_args(argv)
    if len(set(args.dataset)) != len(args.dataset):
        parser.error("datasets must be distinct")
    reference_paths = args.reference_results or [ROOT.parent / "Zerodiff-LQF/out/relation_diagnostic" /
                                                REFERENCE_FOLDERS[d] / "results.zip" for d in args.dataset]
    visual_paths = args.visual_results or [ROOT / "out/relation_diagnostic" / f"{d}_rediff_{SEEDS[d]}" /
                                          "results.zip" for d in args.dataset]

    def index(paths):
        packages = {}
        for path in paths:
            result = read_result(path)
            dataset = result.summary["dataset"]
            if dataset in packages:
                raise ValueError(f"Duplicate V package dataset: {dataset}.")
            packages[dataset] = result
        if set(packages) != set(args.dataset):
            raise ValueError("Supply one V package for each requested dataset.")
        return packages

    references, visuals = index(reference_paths), index(visual_paths)
    output_root = args.output_root.expanduser().resolve()
    commands, zerodiff_outputs, rediff_outputs = [], [], []
    for dataset in args.dataset:
        reference = references[dataset]
        verify_baseline(reference)
        for method, source, outputs in (("ZeroDiff", reference, zerodiff_outputs),
                                         ("Rediff", visuals[dataset], rediff_outputs)):
            paths = checkpoint_paths(source)
            output = output_root / f"{dataset}_{method.lower()}_vcs_{SEEDS[dataset]}"
            command = [sys.executable, str(ROOT / "scripts/diagnose_relations.py"),
                       "--dataset", dataset, "--feature-space", "VCS", "--method", method,
                       "--reference-results", str(reference.path), "--visual-results", str(source.path),
                       "--dfg-checkpoint", paths["dfg"], "--drg-checkpoint", paths["drg"],
                       "--device", args.device, "--output-dir", str(output)]
            if paths["config"]:
                command += ["--training-config", paths["config"]]
            if paths["dataroot"]:
                command += ["--dataroot", paths["dataroot"]]
            commands.append(command)
            outputs.append(str(output / "results.zip"))
    commands.append([sys.executable, str(ROOT / "scripts/compare_vcs_relations.py"),
                     "--reference-results", *[str(references[d].path) for d in args.dataset],
                     "--zerodiff-results", *zerodiff_outputs, "--rediff-results", *rediff_outputs,
                     "--output-dir", str(output_root / "comparison_vcs")])
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    return commands


if __name__ == "__main__":
    main()
