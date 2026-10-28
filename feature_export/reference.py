"""Read auditable result packages and freeze the shared evaluation reference.

Replay needs only NumPy; it never imports the training program or PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from feature_export.relations import cosine_distances, reference_triplets, score_triplets
from feature_export.specs import DATASET_SPECS


SEEDS = {"AWA2": 9182, "CUB": 3483, "SUN": 4115}
BASELINE_COUNTS = {"AWA2": (137, 6), "CUB": (31898, 738), "SUN": (100191, 2742)}
VCS_SCHEMA = "class-relations-vcs-v1"
SCHEMAS = {"awa2-class-relations-v1", "class-relations-v1", VCS_SCHEMA}


def vcs_layout(semantic_dim, hidden_dim):
    return {"V": [0, 2048], "S": [2048, 2048 + semantic_dim],
            "S_hidden": [2048 + semantic_dim, 2048 + semantic_dim + hidden_dim],
            "C": [2048 + semantic_dim + hidden_dim, 4096 + semantic_dim + hidden_dim]}


def with_preservation(scores):
    kept = scores["eligible_triplets"] - scores["reversed_triplets"] - scores["tied_triplets"]
    return {**scores, "preserved_triplets": kept,
            "preservation_rate": kept / scores["eligible_triplets"] if scores["eligible_triplets"] else None}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def triplet_signature(rows):
    values = np.asarray([(r["anchor"], r["near"], r["far"], int(r["eligible"]))
                         for r in rows], dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass
class Result:
    path: Path
    sha256: str
    summary: dict
    manifest: dict
    arrays: dict
    rows: list
    records: list


def _same(value, expected, field):
    if value != expected:
        raise ValueError(f"Result mismatch at {field}: {value!r} != {expected!r}.")


def _matrix(actual, expected, field):
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise ValueError(f"Result mismatch at {field}.")


def read_result(path) -> Result:
    """Read ZIP members without extraction and verify all declared artifact hashes."""
    path = Path(path).expanduser().resolve()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members.")
        for required in ("summary.json", "manifest.json", "centers.npz"):
            if required not in names:
                raise ValueError(f"Missing result member: {required}.")
        manifest = json.loads(archive.read("manifest.json"))
        artifacts = manifest.get("artifacts", {})
        for required in ("summary.json", "centers.npz"):
            if required not in artifacts:
                raise ValueError(f"Missing artifact SHA256: {required}.")
        for name, expected in artifacts.items():
            if name not in names or Path(name).name != name:
                raise ValueError(f"Invalid artifact member: {name}.")
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            _same(digest.hexdigest(), expected, f"artifacts.{name}.sha256")
        summary = json.loads(archive.read("summary.json"))
        with np.load(io.BytesIO(archive.read("centers.npz")), allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
    if summary.get("schema_version") not in SCHEMAS:
        raise ValueError("Unsupported class relation schema.")
    dataset = summary.get("dataset")
    if dataset not in DATASET_SPECS:
        raise ValueError("Unsupported result dataset.")
    spec = DATASET_SPECS[dataset]
    c, a = spec["unseen"], spec["semantic_dim"]
    is_vcs = summary["schema_version"] == VCS_SCHEMA
    generated_dim = 2048
    if is_vcs:
        hidden_dim = summary.get("decoder_hidden_dim")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("Invalid VCS decoder hidden dimension.")
        generated_dim = 4096 + a + hidden_dim
        for key, value in {"generated_space": "VCS", "reference_space": "V",
                           "generated_feature_dim": generated_dim,
                           "feature_blocks": vcs_layout(a, hidden_dim),
                           "eligibility_policy": "fixed_semantics_and_real_V",
                           "block_scaling": "classifier_native_no_extra_normalization"}.items():
            _same(summary.get(key), value, key)
        for key in ("generated_space", "reference_space", "feature_blocks", "eligibility_policy", "block_scaling"):
            _same(manifest.get(key), summary[key], f"manifest.{key}")
    seeds = arrays.get("seeds")
    if seeds is None or seeds.dtype != np.dtype("int64") or seeds.ndim != 1 or not seeds.size:
        raise ValueError("Expected nonempty int64 seeds.")
    if len(set(seeds.tolist())) != len(seeds) or np.any(seeds < 0):
        raise ValueError("Invalid or duplicate result seeds.")
    shapes = {"class_ids": (c,), "class_names": (c,), "semantics": (c, a),
              "real_centers": (c, 2048), "real_counts": (c,),
              "generated_centers": (len(seeds), c, generated_dim),
              "semantic_distances": (c, c), "real_distances": (c, c),
              "generated_distances": (len(seeds), c, c)}
    if is_vcs:
        shapes["generated_visual_distances"] = (len(seeds), c, c)
    for name, shape in shapes.items():
        value = arrays.get(name)
        if value is None or value.shape != shape:
            raise ValueError(f"Invalid centers.npz shape: {name}, expected {shape}.")
        expected_dtype = ("int64" if name in ("class_ids", "real_counts") else
                          "float32" if name == "semantics" else "float64")
        if name == "class_names":
            if value.dtype.kind != "U":
                raise ValueError("class_names must use Unicode without pickle.")
        elif value.dtype != np.dtype(expected_dtype) or not np.isfinite(value).all():
            raise ValueError(f"Invalid centers.npz dtype or values: {name}.")
    ids = arrays["class_ids"]
    if np.any(np.diff(ids) <= 0) or ids.min() < 0 or ids.max() >= spec["classes"]:
        raise ValueError("Class IDs must retain sorted original zero-based IDs.")
    if np.any(arrays["real_counts"] <= 0):
        raise ValueError("Real class counts must be positive.")
    _same(summary.get("seeds"), seeds.tolist(), "seeds")
    _same(summary.get("split"), "test_unseen_loc", "split")
    _same(summary.get("distance"), "cosine_of_class_mean", "distance")
    for vectors, distances in (("semantics", "semantic_distances"), ("real_centers", "real_distances")):
        _matrix(arrays[distances], cosine_distances(arrays[vectors]), distances)
    for index, centers in enumerate(arrays["generated_centers"]):
        _matrix(arrays["generated_distances"][index], cosine_distances(centers), "generated_distances")
        if is_vcs:
            _matrix(arrays["generated_visual_distances"][index], cosine_distances(centers[:, :2048]),
                    "generated_visual_distances")
    rows = reference_triplets(arrays["semantic_distances"], arrays["real_distances"], summary["min_gap"])
    records = []
    for seed, distances in zip(seeds, arrays["generated_distances"]):
        scores, _ = score_triplets(rows, distances, summary["tie_gap"])
        if is_vcs:
            scores = with_preservation(scores)
        records.append({"seed": int(seed), **scores})
    _same(summary.get("total_triplets"), len(rows), "total_triplets")
    eligible = sum(row["eligible"] for row in rows)
    _same(summary.get("eligible_triplets"), eligible, "eligible_triplets")
    _same(summary.get("coverage"), eligible / len(rows), "coverage")
    saved = summary.get("per_seed", [])
    _same(len(saved), len(records), "per_seed length")
    for expected, actual in zip(saved, records):
        for key, value in actual.items():
            _same(expected.get(key), value, f"per_seed.{key}")
    if is_vcs:
        controls = summary.get("visual_control", {}).get("per_seed", [])
        _same(len(controls), len(seeds), "visual_control length")
        for control, seed, distances in zip(controls, seeds, arrays["generated_visual_distances"]):
            scores, _ = score_triplets(rows, distances, summary["tie_gap"])
            for key, value in {"seed": int(seed), **with_preservation(scores)}.items():
                _same(control.get(key), value, f"visual_control.{key}")
        _same(summary.get("triplet_signature"), triplet_signature(rows), "triplet_signature")
    return Result(path, sha256_file(path), summary, manifest, arrays, rows, records)


def verify_baseline(result):
    """Acceptance gate for the three fixed ZeroDiff runs in the experiment."""
    summary = result.summary
    if summary["schema_version"] == VCS_SCHEMA:
        raise ValueError("Baseline acceptance requires the original V reference, not a VCS result.")
    dataset = summary["dataset"]
    _same(summary.get("method", "ZeroDiff"), "ZeroDiff", "method")
    for key, value in {"sampling": "one-step", "seeds": [SEEDS[dataset]],
                       "samples_per_class": 100, "min_gap": 0.05, "tie_gap": 1e-6}.items():
        _same(summary.get(key), value, key)
    eligible, reversed_count = BASELINE_COUNTS[dataset]
    _same(result.records[0]["eligible_triplets"], eligible, "baseline eligible_triplets")
    _same(result.records[0]["reversed_triplets"], reversed_count, "baseline reversed_triplets")
    _same(result.records[0]["tied_triplets"], 0, "baseline tied_triplets")
    options = result.manifest.get("options", {})
    for key, value in {"split_percent": 100, "preprocessing": True, "standardization": False,
                       "class_embedding_norm": DATASET_SPECS[dataset]["normalize"], "n_T": 4}.items():
        _same(options.get(key), value, f"reference options.{key}")
    _same(result.manifest.get("batch_size"), 100, "reference batch_size")
    return {"dataset": dataset, "reference_sha256": result.sha256,
            "triplet_signature": triplet_signature(result.rows), **result.records[0]}


def validate_alignment(reference, data):
    """Fail closed if data bytes or the exact preprocessing fingerprint differ."""
    alignment = reference.manifest.get("data_alignment", {})
    for key, value in data.alignment.items():
        _same(alignment.get(key), value, f"data_alignment.{key}")
    for name, value in data.files.items():
        _same(reference.manifest.get("data", {}).get(name, {}).get("sha256"),
              value["sha256"], f"data.{name}.sha256")
    from feature_export.relations import class_centers
    centers, counts = class_centers(data.real_features, data.real_labels, data.class_ids)
    for name, values in (("class_ids", data.class_ids), ("class_names", data.class_names),
                         ("semantics", data.attributes[data.class_ids].cpu().numpy()), ("real_counts", counts)):
        if not np.array_equal(reference.arrays[name], values):
            raise ValueError(f"Local data disagree with reference {name}.")
    _matrix(reference.arrays["real_centers"], centers, "local real_centers")
    _matrix(reference.arrays["real_distances"], cosine_distances(centers), "local real_distances")
    return {"status": "exact_data_and_preprocessing_match", "center_atol": 1e-12, "center_rtol": 1e-12}
