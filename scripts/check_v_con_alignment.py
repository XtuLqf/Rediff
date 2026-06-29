#!/usr/bin/env python3
"""Check whether ZeroDiff visual and contrastive feature mats are aligned.

This script is intentionally read-only. It validates feature shapes and optional
metadata, then reports C-prototype accuracy and V/C/S relation correlations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

np = None
sio = None
pdist = None
pearsonr = None
spearmanr = None


DEFAULT_CLASS_EMBEDDINGS = {
    "CUB": "sent",
    "AWA2": "att",
    "SUN": "att",
}

META_KEYS = (
    "labels",
    "label",
    "image_files",
    "image_file",
    "paths",
    "path",
    "filenames",
    "filename",
    "indices",
    "index",
    "original_index",
)

SPLIT_KEYS = ("trainval_loc", "test_seen_loc", "test_unseen_loc")


def load_dependencies() -> None:
    global np, sio, pdist, pearsonr, spearmanr
    try:
        import numpy as _np
        import scipy.io as _sio
        from scipy.spatial.distance import pdist as _pdist
        from scipy.stats import pearsonr as _pearsonr
        from scipy.stats import spearmanr as _spearmanr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python dependency: {exc.name}. Install the project dependencies "
            "(numpy and scipy are required) before running this diagnostic."
        ) from exc

    np = _np
    sio = _sio
    pdist = _pdist
    pearsonr = _pearsonr
    spearmanr = _spearmanr


@dataclass
class SplitCorr:
    split: str
    n_used: int
    pearson_sc: float
    spearman_sc: float
    pearson_vc: float
    spearman_vc: float
    pearson_sv: float
    spearman_sv: float


@dataclass
class DatasetReport:
    dataset: str
    status: str = "OK"
    messages: List[str] = field(default_factory=list)
    shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    dtypes: Dict[str, str] = field(default_factory=dict)
    health: Dict[str, str] = field(default_factory=dict)
    mat_keys: Dict[str, List[str]] = field(default_factory=dict)
    meta_keys: Dict[str, List[str]] = field(default_factory=dict)
    label_checks: List[str] = field(default_factory=list)
    proto_acc: Dict[str, float] = field(default_factory=dict)
    correlations: List[SplitCorr] = field(default_factory=list)
    recommendation: str = "unknown"

    def fail(self, message: str) -> None:
        self.status = "FAIL"
        self.messages.append(message)

    def warn(self, message: str) -> None:
        if self.status == "OK":
            self.status = "WARN"
        self.messages.append(message)


def parse_class_embeddings(values: Iterable[str]) -> Dict[str, str]:
    mapping = dict(DEFAULT_CLASS_EMBEDDINGS)
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError(
                "--class-embeddings entries must use DATASET:embedding, e.g. CUB:sent"
            )
        dataset, embedding = value.split(":", 1)
        mapping[dataset.strip()] = embedding.strip()
    return mapping


def mat_public_keys(mat: Dict[str, object]) -> List[str]:
    return sorted(k for k in mat.keys() if not k.startswith("__"))


def existing_meta_keys(mat: Dict[str, object]) -> List[str]:
    public = set(mat_public_keys(mat))
    return [key for key in META_KEYS if key in public]


def summarize_array(name: str, array: np.ndarray, report: DatasetReport) -> None:
    report.shapes[name] = tuple(array.shape)
    report.dtypes[name] = str(array.dtype)
    if np.issubdtype(array.dtype, np.number):
        nan_count = int(np.isnan(array).sum())
        inf_count = int(np.isinf(array).sum())
        report.health[name] = f"nan={nan_count}, inf={inf_count}"
        if nan_count or inf_count:
            report.warn(f"{name} contains nan/inf values.")
    else:
        report.health[name] = "non-numeric"


def loadmat(path: Path, report: DatasetReport, logical_name: str) -> Optional[Dict[str, object]]:
    if not path.exists():
        report.fail(f"Missing {logical_name}: {path}")
        return None
    mat = sio.loadmat(path)
    report.mat_keys[logical_name] = mat_public_keys(mat)
    report.meta_keys[logical_name] = existing_meta_keys(mat)
    return mat


def labels_zero_based(raw_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(raw_labels).astype(np.int64).squeeze()
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if labels.size and labels.min() == 1:
        labels = labels - 1
    return labels


def split_indices(raw_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(raw_indices).astype(np.int64).squeeze()
    if indices.ndim != 1:
        indices = indices.reshape(-1)
    return indices - 1


def features_samples_first(
    raw_features: np.ndarray,
    n_samples: int,
    logical_name: str,
    report: DatasetReport,
) -> Optional[np.ndarray]:
    features = np.asarray(raw_features)
    summarize_array(logical_name, features, report)
    if features.ndim != 2:
        report.fail(f"{logical_name} must be 2-D, got shape {features.shape}.")
        return None
    if features.shape[0] == n_samples:
        return features.astype(np.float32, copy=False)
    if features.shape[1] == n_samples:
        report.warn(
            f"{logical_name} appears transposed ({features.shape}); calculations use transpose, "
            "but project loader expects samples-first."
        )
        return features.T.astype(np.float32, copy=False)
    report.fail(f"{logical_name} sample count does not match labels: {features.shape} vs {n_samples}.")
    return None


def class_attributes(
    split_mat: Dict[str, object],
    labels: np.ndarray,
    class_embedding: str,
    normalize_semantic: bool,
    report: DatasetReport,
) -> Optional[np.ndarray]:
    if class_embedding == "att" and "original_att" in split_mat:
        raw = np.asarray(split_mat["original_att"])
    elif "att" in split_mat:
        raw = np.asarray(split_mat["att"])
    else:
        report.fail(f"{class_embedding}_splits.mat has no attribute matrix under 'att'/'original_att'.")
        return None

    n_classes = int(labels.max()) + 1
    if raw.shape[0] == n_classes:
        attrs = raw
    elif raw.shape[1] == n_classes:
        attrs = raw.T
    else:
        report.fail(f"Cannot align semantic matrix shape {raw.shape} to {n_classes} classes.")
        return None

    attrs = attrs.astype(np.float32, copy=False)
    if normalize_semantic:
        attrs = l2_normalize(attrs)
    summarize_array("S_class", attrs, report)
    return attrs


def compare_optional_labels(
    mat: Dict[str, object],
    logical_name: str,
    base_labels: np.ndarray,
    report: DatasetReport,
) -> None:
    label_key = "labels" if "labels" in mat else "label" if "label" in mat else None
    if label_key is None:
        report.label_checks.append(f"{logical_name}: no labels key; exact label alignment not directly provable.")
        return

    candidate = labels_zero_based(np.asarray(mat[label_key]))
    if candidate.shape != base_labels.shape:
        report.warn(f"{logical_name}:{label_key} shape {candidate.shape} differs from res101 labels {base_labels.shape}.")
        report.label_checks.append(f"{logical_name}: labels present but shape mismatch.")
        return

    same = bool(np.array_equal(candidate, base_labels))
    if same:
        report.label_checks.append(f"{logical_name}: labels match res101 labels exactly.")
        return

    one_based_same = bool(np.array_equal(candidate - 1, base_labels))
    if one_based_same:
        report.label_checks.append(f"{logical_name}: labels match res101 labels after one-based conversion.")
        return

    mismatch = int(np.count_nonzero(candidate != base_labels))
    report.fail(f"{logical_name}:{label_key} differs from res101 labels at {mismatch} positions.")
    report.label_checks.append(f"{logical_name}: labels DO NOT match res101 labels.")


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def prototypes(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    proto = np.zeros((classes.size, features.shape[1]), dtype=np.float32)
    for i, cls in enumerate(classes):
        proto[i] = features[labels == cls].mean(axis=0)
    return classes, proto


def nearest_prototype_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    normalize: bool,
) -> float:
    if normalize:
        train_features = l2_normalize(train_features)
        eval_features = l2_normalize(eval_features)
    classes, proto = prototypes(train_features, train_labels)
    if normalize:
        proto = l2_normalize(proto)
    distances = squared_euclidean(eval_features, proto)
    pred = classes[np.argmin(distances, axis=1)]
    return float(np.mean(pred == eval_labels))


def squared_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)


def safe_corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(a, b).statistic)
    if kind == "spearman":
        return float(spearmanr(a, b).statistic)
    raise ValueError(kind)


def relation_corr_for_split(
    split_name: str,
    indices: np.ndarray,
    v_features: np.ndarray,
    c_features: np.ndarray,
    labels: np.ndarray,
    attrs: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    normalize_features: bool,
) -> SplitCorr:
    if indices.size > sample_size:
        indices = np.sort(rng.choice(indices, size=sample_size, replace=False))

    v = v_features[indices]
    c = c_features[indices]
    s = attrs[labels[indices]]
    if normalize_features:
        v = l2_normalize(v)
        c = l2_normalize(c)
        s = l2_normalize(s)

    rv = pdist(v, metric="euclidean")
    rc = pdist(c, metric="euclidean")
    rs = pdist(s, metric="euclidean")
    return SplitCorr(
        split=split_name,
        n_used=int(indices.size),
        pearson_sc=safe_corr(rs, rc, "pearson"),
        spearman_sc=safe_corr(rs, rc, "spearman"),
        pearson_vc=safe_corr(rv, rc, "pearson"),
        spearman_vc=safe_corr(rv, rc, "spearman"),
        pearson_sv=safe_corr(rs, rv, "pearson"),
        spearman_sv=safe_corr(rs, rv, "spearman"),
    )


def fmt_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.4f}"


def make_recommendation(report: DatasetReport, cross_dataset_baseline: Optional[float]) -> str:
    if report.status == "FAIL":
        return "FAIL: fix data files/alignment before using C teacher."

    train_corrs = [c for c in report.correlations if c.split == "trainval"]
    if not train_corrs:
        return "unknown: relation correlations were not available."

    train = train_corrs[0]
    c_train = report.proto_acc.get("C trainval->trainval", float("nan"))
    c_seen = report.proto_acc.get("C trainval->test_seen", float("nan"))
    min_c_relation = min(train.spearman_sc, train.spearman_vc)

    if min_c_relation < 0.15 or c_train < 0.50:
        return "disable C teacher: C has weak class/relationship structure."
    if min_c_relation < 0.25 or c_seen < 0.40:
        return "lower rel_con_weight to 0.3/0.5: C teacher looks weak."
    if cross_dataset_baseline is not None and min_c_relation < cross_dataset_baseline - 0.15:
        return "lower rel_con_weight to 0.3/0.5: C relation quality is below the cross-dataset baseline."
    if report.status == "WARN":
        return "keep rel_con_weight=1 only if extraction provenance confirms shared image order."
    return "keep rel_con_weight=1: C teacher passed these diagnostics."


def check_dataset(
    dataroot: Path,
    dataset: str,
    class_embedding: str,
    sample_size: int,
    rng: np.random.Generator,
    normalize_features: bool,
    normalize_semantic: bool,
) -> DatasetReport:
    report = DatasetReport(dataset=dataset)
    ds_dir = dataroot / dataset
    if not ds_dir.exists():
        report.fail(f"Missing dataset directory: {ds_dir}")
        return report

    res_mat = loadmat(ds_dir / "res101.mat", report, "res101")
    split_mat = loadmat(ds_dir / f"{class_embedding}_splits.mat", report, f"{class_embedding}_splits")
    ce_mat = loadmat(ds_dir / "ce_ce.mat", report, "ce_ce")
    con_mat = loadmat(ds_dir / "con_paco.mat", report, "con_paco")
    if report.status == "FAIL" or res_mat is None or split_mat is None or ce_mat is None or con_mat is None:
        return report

    if "labels" not in res_mat:
        report.fail("res101.mat has no labels key.")
        return report
    if "features" not in ce_mat:
        report.fail("ce_ce.mat has no features key.")
        return report
    if "features" not in con_mat:
        report.fail("con_paco.mat has no features key.")
        return report

    labels = labels_zero_based(np.asarray(res_mat["labels"]))
    summarize_array("labels", labels, report)
    n_samples = labels.size

    v_features = features_samples_first(np.asarray(ce_mat["features"]), n_samples, "V_ce_ce", report)
    c_features = features_samples_first(np.asarray(con_mat["features"]), n_samples, "C_con_paco", report)
    compare_optional_labels(ce_mat, "ce_ce", labels, report)
    compare_optional_labels(con_mat, "con_paco", labels, report)

    if "image_files" in res_mat:
        report.label_checks.append("res101: image_files present; feature order can be tied to source images.")
    else:
        report.warn("res101.mat has no image_files key; provenance check is weaker.")

    for logical_name, mat in (("ce_ce", ce_mat), ("con_paco", con_mat)):
        if "image_files" in mat:
            report.label_checks.append(f"{logical_name}: image_files present.")
        else:
            report.label_checks.append(f"{logical_name}: no image_files key.")

    attrs = class_attributes(split_mat, labels, class_embedding, normalize_semantic, report)
    if report.status == "FAIL" or v_features is None or c_features is None or attrs is None:
        return report

    splits: Dict[str, np.ndarray] = {}
    for key in SPLIT_KEYS:
        if key not in split_mat:
            report.fail(f"{class_embedding}_splits.mat has no {key}.")
            return report
        splits[key] = split_indices(np.asarray(split_mat[key]))
        if np.any(splits[key] < 0) or np.any(splits[key] >= n_samples):
            report.fail(f"{key} contains indices outside [0, {n_samples}).")
            return report

    train_idx = splits["trainval_loc"]
    test_seen_idx = splits["test_seen_loc"]
    test_unseen_idx = splits["test_unseen_loc"]

    report.proto_acc["C trainval->trainval"] = nearest_prototype_accuracy(
        c_features[train_idx], labels[train_idx], c_features[train_idx], labels[train_idx], normalize_features
    )
    report.proto_acc["C trainval->test_seen"] = nearest_prototype_accuracy(
        c_features[train_idx], labels[train_idx], c_features[test_seen_idx], labels[test_seen_idx], normalize_features
    )
    report.proto_acc["C test_unseen self"] = nearest_prototype_accuracy(
        c_features[test_unseen_idx],
        labels[test_unseen_idx],
        c_features[test_unseen_idx],
        labels[test_unseen_idx],
        normalize_features,
    )

    for split_name, indices in (
        ("trainval", train_idx),
        ("test_seen", test_seen_idx),
        ("test_unseen", test_unseen_idx),
    ):
        if indices.size < 3:
            report.warn(f"{split_name} has fewer than 3 samples; skipping relation correlation.")
            continue
        report.correlations.append(
            relation_corr_for_split(
                split_name,
                indices,
                v_features,
                c_features,
                labels,
                attrs,
                sample_size,
                rng,
                normalize_features,
            )
        )

    return report


def print_report(report: DatasetReport) -> None:
    print(f"\n[{report.dataset}] status={report.status}")
    for message in report.messages:
        print(f"  - {message}")

    print("  shapes/dtypes/health:")
    for name in sorted(report.shapes):
        print(f"    {name}: shape={report.shapes[name]}, dtype={report.dtypes[name]}, {report.health[name]}")

    print("  mat keys:")
    for name, keys in report.mat_keys.items():
        meta = report.meta_keys.get(name, [])
        print(f"    {name}: keys={keys}; metadata_keys={meta or 'none'}")

    print("  label/provenance checks:")
    for line in report.label_checks:
        print(f"    {line}")

    if report.proto_acc:
        print("  C nearest-prototype accuracy:")
        for name, value in report.proto_acc.items():
            print(f"    {name}: {fmt_float(value)}")

    if report.correlations:
        print("  relation correlations:")
        header = "split       n      pearson(S,C)  spear(S,C)  pearson(V,C)  spear(V,C)  pearson(S,V)  spear(S,V)"
        print(f"    {header}")
        for corr in report.correlations:
            print(
                "    "
                f"{corr.split:<10} {corr.n_used:<6} "
                f"{fmt_float(corr.pearson_sc):>12} {fmt_float(corr.spearman_sc):>11} "
                f"{fmt_float(corr.pearson_vc):>12} {fmt_float(corr.spearman_vc):>11} "
                f"{fmt_float(corr.pearson_sv):>12} {fmt_float(corr.spearman_sv):>11}"
            )

    print(f"  recommendation: {report.recommendation}")


def compute_cross_dataset_baseline(reports: List[DatasetReport]) -> Optional[float]:
    values = []
    for report in reports:
        if report.status == "FAIL":
            continue
        train = next((c for c in report.correlations if c.split == "trainval"), None)
        if train is None:
            continue
        values.append(min(train.spearman_sc, train.spearman_vc))
    if len(values) < 2:
        return None
    return float(np.median(values))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check V/C feature alignment and VSRA C-teacher reliability for ZeroDiff datasets."
    )
    parser.add_argument("--dataroot", default="./Dataset", help="Dataset root containing CUB/AWA2/SUN directories.")
    parser.add_argument("--datasets", nargs="+", default=["CUB", "AWA2", "SUN"], help="Datasets to check.")
    parser.add_argument(
        "--class-embeddings",
        nargs="*",
        default=[],
        help="Per-dataset class embedding mapping, e.g. CUB:sent AWA2:att SUN:att.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=4096,
        help="Maximum samples per split for pairwise relation correlations.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Sampling seed for relation correlations.")
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="L2-normalize V/C/S before prototype and relation checks.",
    )
    parser.add_argument(
        "--normalize-semantic",
        action="store_true",
        help="L2-normalize class semantic vectors before assigning S to samples.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.sample_size < 3:
        parser.error("--sample-size must be at least 3.")

    load_dependencies()

    class_embeddings = parse_class_embeddings(args.class_embeddings)
    rng = np.random.default_rng(args.seed)
    dataroot = Path(args.dataroot)

    print(f"Dataset root: {dataroot}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Sample size per split: {args.sample_size}")
    print(f"Normalize features: {args.normalize_features}")
    print(f"Normalize semantic: {args.normalize_semantic}")
    print("Note: ce_ce.mat/con_paco.mat usually contain only features; exact image-level proof needs saved labels/image_files.")

    reports = []
    for dataset in args.datasets:
        class_embedding = class_embeddings.get(dataset, "att")
        reports.append(
            check_dataset(
                dataroot,
                dataset,
                class_embedding,
                args.sample_size,
                rng,
                args.normalize_features,
                args.normalize_semantic,
            )
        )

    baseline = compute_cross_dataset_baseline(reports)
    for report in reports:
        report.recommendation = make_recommendation(report, baseline)
        print_report(report)

    failed = [report.dataset for report in reports if report.status == "FAIL"]
    if failed:
        print(f"\nOverall: FAIL for {', '.join(failed)}")
        return 1

    warned = [report.dataset for report in reports if report.status == "WARN"]
    if warned:
        print(f"\nOverall: WARN for {', '.join(warned)}")
        return 0

    print("\nOverall: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
