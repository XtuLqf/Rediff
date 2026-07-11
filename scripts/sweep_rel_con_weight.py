#!/usr/bin/env python3
"""Run and summarize the fixed-weight VSRA rel_con_weight grid."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("CUB", "AWA2", "SUN")
DEFAULT_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
MODALITIES = ("V", "VS", "C", "VC", "VCS")
RUN_SCRIPTS = {
    "CUB": ROOT / "scripts" / "run_cub_zerodiff_DFG_train.py",
    "AWA2": ROOT / "scripts" / "run_awa2_zerodiff_DFG_train.py",
    "SUN": ROOT / "scripts" / "run_sun_zerodiff_DFG_train.py",
}
FIELDNAMES = (
    "dataset",
    "rel_con_weight",
    "mode",
    "modality",
    "gzsl_unseen",
    "gzsl_seen",
    "gzsl_h",
    "zsl_acc",
    "status",
    "log_path",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--weights", nargs="+", type=float, default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "vsra_rel_con_sweep.csv")
    args = parser.parse_args()
    if any(weight < 0 for weight in args.weights):
        parser.error("all weights must be non-negative")
    return args


def build_command(dataset, weight, mode):
    return [
        sys.executable,
        str(RUN_SCRIPTS[dataset]),
        "--rel-con-weight",
        str(weight),
        "--vsra-weight-mode",
        mode,
    ]


def load_completed(output_path):
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "completed":
                completed.add((row["dataset"], float(row["rel_con_weight"]), row["mode"]))
    return completed


def find_latest_log(dataset, started_at):
    log_dir = ROOT / "log" / dataset
    candidates = [
        path for path in log_dir.glob("train_zerodiff_DFG_*.log")
        if path.stat().st_mtime >= started_at - 2.0
    ]
    if not candidates:
        raise FileNotFoundError(f"No DFG log produced for {dataset}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_log(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows = {}
    for modality, unseen, seen, harmonic in re.findall(
        r"best GZSL \((VCS|VS|VC|V|C)\): U: ([0-9.eE+-]+), S: ([0-9.eE+-]+), H: ([0-9.eE+-]+)",
        text,
    ):
        rows.setdefault(modality, {}).update(
            gzsl_unseen=unseen,
            gzsl_seen=seen,
            gzsl_h=harmonic,
        )
    for modality, accuracy in re.findall(
        r"best ZSL \((VCS|VS|VC|V|C)\): ([0-9.eE+-]+)",
        text,
    ):
        rows.setdefault(modality, {})["zsl_acc"] = accuracy
    missing = [
        modality for modality in MODALITIES
        if modality not in rows or not {"gzsl_unseen", "gzsl_seen", "gzsl_h", "zsl_acc"}.issubset(rows[modality])
    ]
    if missing:
        raise ValueError(f"Incomplete final metrics in {log_path}: {', '.join(missing)}")
    return rows


def append_rows(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def result_rows(dataset, weight, mode, status, log_path=None, metrics=None):
    metrics = metrics or {}
    modalities = MODALITIES if status == "completed" else ("",)
    return [
        {
            "dataset": dataset,
            "rel_con_weight": weight,
            "mode": mode,
            "modality": modality,
            "gzsl_unseen": metrics.get(modality, {}).get("gzsl_unseen", ""),
            "gzsl_seen": metrics.get(modality, {}).get("gzsl_seen", ""),
            "gzsl_h": metrics.get(modality, {}).get("gzsl_h", ""),
            "zsl_acc": metrics.get(modality, {}).get("zsl_acc", ""),
            "status": status,
            "log_path": "" if log_path is None else str(log_path.relative_to(ROOT)),
        }
        for modality in modalities
    ]


def main():
    args = parse_args()
    jobs = [(dataset, weight) for dataset in args.datasets for weight in args.weights]
    if args.dry_run:
        for dataset, weight in jobs:
            print(subprocess.list2cmdline(build_command(dataset, weight, args.mode)))
        print(f"Dry run: {len(jobs)} experiment(s).")
        return 0

    completed = load_completed(args.output) if args.skip_existing else set()
    failures = 0
    for dataset, weight in jobs:
        key = (dataset, float(weight), args.mode)
        if key in completed:
            print(f"Skipping completed experiment: {dataset} weight={weight} mode={args.mode}")
            continue
        command = build_command(dataset, weight, args.mode)
        print(f"Running: {subprocess.list2cmdline(command)}", flush=True)
        started_at = time.time()
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            log_path = find_latest_log(dataset, started_at)
            metrics = parse_log(log_path)
            append_rows(args.output, result_rows(dataset, weight, args.mode, "completed", log_path, metrics))
        except (subprocess.CalledProcessError, OSError, ValueError) as error:
            failures += 1
            print(f"Experiment failed: {error}", file=sys.stderr)
            append_rows(args.output, result_rows(dataset, weight, args.mode, "failed"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
