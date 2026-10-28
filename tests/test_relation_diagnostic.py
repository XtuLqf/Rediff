"""Rule boundaries, training equivalence, reference locking and portable results."""

import ast
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import zipfile

import numpy as np
import pytest
import scipy.io as sio
import torch

from datasets.image_util import DATA_LOADER
from feature_export.comparison import comparison_row, main as compare_main
from feature_export.data import load_dataset
from feature_export.diagnostic import main, make_summary, write_results
from feature_export.generation import VisualFeatureGenerator, generate_visual_features
from feature_export.reference import BASELINE_COUNTS, SEEDS, read_result, sha256_file, triplet_signature, verify_baseline, validate_alignment
from feature_export.relations import class_centers, cosine_distances, relative_gap, reference_triplets, score_triplets
from feature_export.runtime import load_options
from feature_export.specs import DATASET_SPECS
import zerodiff_tools


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    torch.use_deterministic_algorithms(deterministic)


def make_data(tmp_path, dataset):
    spec = DATASET_SPECS[dataset]
    root = tmp_path / "Dataset" / dataset
    root.mkdir(parents=True)
    rng = np.random.default_rng(902)
    ids = np.sort(rng.choice(spec["classes"], spec["unseen"], replace=False))
    labels = np.repeat(np.arange(spec["classes"]), 2)
    unseen = np.flatnonzero(np.isin(labels, ids))
    train = np.flatnonzero(~np.isin(labels, ids))
    base = rng.uniform(0.1, 5, (spec["classes"], 8))
    features = (np.tile(base[labels], (1, 256)) + rng.uniform(0, .01, (len(labels), 2048))).astype(np.float32)
    names = np.empty((spec["classes"], 1), dtype=object)
    for i in range(len(names)):
        # Deliberately unrelated displayed numbers: labels must come from MAT IDs.
        names[i, 0] = f"{spec['classes'] - i:03d}.class_{i}"
    att = rng.uniform(.1, 1, (spec["semantic_dim"], spec["classes"])).astype(np.float32)
    sio.savemat(root / "res101.mat", {"labels": labels[:, None] + 1, "features": features.T})
    sio.savemat(root / "ce_ce.mat", {"features": features})
    sio.savemat(root / "con_paco.mat", {"features": features * .7})
    sio.savemat(root / f"{spec['embedding']}_splits.mat", {
        "trainval_loc": train[:, None] + 1, "train_loc": train[:, None] + 1,
        "test_seen_loc": train[:, None] + 1, "test_unseen_loc": unseen[:, None] + 1,
        "val_loc": unseen[:, None] + 1, "att": att, "original_att": att * 100,
        "allclasses_names": names})
    training = {"dataset": dataset, "split_percent": 100, "decoder_layer_sizes": [8, 2048]}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"training": training}))
    options, _ = load_options(100, config, dataset=dataset)
    data = load_dataset(root.parent, options)
    return SimpleNamespace(dataset=dataset, dataroot=root.parent, root=root, config=config,
                           options=options, data=data, class_ids=ids, training=training)


def make_weights(example):
    torch.manual_seed(330)
    dfg, drg = example.root.parent.parent / "dfg.tar", example.root.parent.parent / "drg.tar"
    torch.save({"state_dict_R": zerodiff_tools.DRG_Generator(example.options).state_dict()}, drg)
    example.training["drg_sha256"] = sha256_file(drg)
    example.config.write_text(json.dumps({"training": example.training}))
    torch.save({"state_dict_G": zerodiff_tools.DFG_Generator(example.options).state_dict(),
                "state_dict_VSRA": {"training_only": torch.ones(1)},
                "method_metadata": {"name": "ds_reg_sdga"}, "training_config": example.training}, dfg)
    example.dfg, example.drg = dfg, drg
    return example


@pytest.mark.parametrize("dataset", ["AWA2", "CUB", "SUN"])
def test_dataset_matches_training_and_preserves_mat_ids(tmp_path, dataset):
    example = make_data(tmp_path, dataset)
    options = deepcopy(example.options)
    options.dataroot = str(example.dataroot)
    expected = DATA_LOADER(options)
    np.testing.assert_array_equal(example.data.real_features, expected.test_unseen_feature.numpy())
    np.testing.assert_array_equal(example.data.real_labels, expected.test_unseen_label.numpy())
    torch.testing.assert_close(example.data.attributes, expected.attribute, rtol=0, atol=0)
    np.testing.assert_array_equal(example.data.class_ids, example.class_ids)
    assert example.data.class_names[0].endswith(f"class_{example.class_ids[0]}")
    assert example.data.attributes.shape[1] == DATASET_SPECS[dataset]["semantic_dim"]


def test_mean_precedes_cosine_and_invalid_vectors_fail():
    centers, counts = class_centers([[100, 0], [0, 1], [1, 1]], [5, 5, 8], [5, 8])
    np.testing.assert_array_equal(centers, [[50, .5], [1, 1]])
    np.testing.assert_array_equal(counts, [2, 1])
    assert centers.dtype == np.float64
    for bad in ([[0, 0]], [[np.nan, 1]], [[np.inf, 1]], [[1e-13, 0]]):
        with pytest.raises(ValueError):
            cosine_distances(bad)


def test_reversal_tie_thresholds_and_empty_eligible_set():
    semantic = np.array([[0, .2, .8], [.2, 0, .6], [.8, .6, 0]])
    rows = reference_triplets(semantic, semantic)
    assert len(rows) == 3
    selected = [rows[0]]
    reversed_distance = semantic.copy()
    reversed_distance[0, 1], reversed_distance[0, 2] = .8, .2
    scores, outcomes = score_triplets(selected, reversed_distance)
    assert scores["reversed_triplets"] == 1 and outcomes[0]["outcome"] == "reversed"
    same_centers = cosine_distances(np.ones((3, 2048)))
    scores, _ = score_triplets(selected, same_centers)
    assert scores["tied_triplets"] == 1 and scores["reversed_triplets"] == 0
    # Exact equality to either threshold must follow > for eligibility and <= for ties.
    gap = relative_gap(.2, .8)
    assert not reference_triplets(semantic, semantic, min_gap=gap)[0]["eligible"]
    assert score_triplets(selected, reversed_distance, tie_gap=gap)[0]["tied_triplets"] == 1
    disagreeing = reference_triplets(semantic, reversed_distance)
    assert not disagreeing[0]["eligible"]
    excluded = [{**rows[0], "eligible": False}]
    scores, outcomes = score_triplets(excluded, semantic)
    assert scores["reversal_rate"] is None and scores["tie_rate"] is None
    summary = make_summary("AWA2", [9182], excluded, [{"seed": 9182, **scores}], [outcomes], None)
    assert summary["aggregate"] is None
    assert summary["conclusion"] == "undefined_no_eligible_triplets"
    assert summary["example"] is None
    assert triplet_signature(rows) == triplet_signature(reference_triplets(semantic, semantic))


@pytest.mark.parametrize("count,total", [(10, 360), (50, 58800), (72, 178920)])
def test_candidate_counts(count, total):
    matrix = np.zeros((count, count))
    assert len(reference_triplets(matrix, matrix)) == total


@pytest.mark.parametrize("dataset", ["AWA2", "CUB", "SUN"])
def test_adapter_matches_actual_training_sampler_and_class_rng(tmp_path, dataset):
    example = make_weights(make_data(tmp_path, dataset))
    adapter = VisualFeatureGenerator(dataset=dataset, device="cpu", dfg_checkpoint=example.dfg,
                                     drg_checkpoint=example.drg, training_config=example.config)
    runtime = adapter.runtime
    path = Path(__file__).resolve().parents[1] / "zerodiff_DFG_train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ZERODIFF")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_from_model")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    original = SimpleNamespace(dim_v=2048, dim_noise=adapter.options.noiseSize, n_T=4, device="cpu",
                               netG=runtime.netG, netR=runtime.netR)
    sample = MethodType(namespace["sample_from_model"], original)
    seed = SEEDS[dataset]
    # Perturb global RNG before each generation: the local class streams must be isolated.
    torch.manual_seed(1)
    actual, labels = adapter.generate(class_ids=example.data.class_ids, attributes=example.data.attributes,
                                      samples_per_class=3, seed=seed, batch_size=2, sampling="one-step")
    expected = []
    for class_id in example.data.class_ids.tolist():
        torch.manual_seed((seed + 1000003 * class_id) % (2**63 - 1))
        for count in (2, 1):
            features, _ = sample(example.data.attributes[class_id].repeat(count, 1))
            expected.append(features)
    np.testing.assert_array_equal(actual, torch.cat(expected).numpy())
    np.testing.assert_array_equal(labels, np.repeat(example.data.class_ids, 3))
    torch.manual_seed(9876)
    repeated, _ = adapter.generate(class_ids=example.data.class_ids, attributes=example.data.attributes,
                                   samples_per_class=3, seed=seed, batch_size=2, sampling="one-step")
    np.testing.assert_array_equal(repeated, actual)
    public_features, public_labels = generate_visual_features(
        dataset=dataset, class_ids=example.data.class_ids, attributes=example.data.attributes,
        samples_per_class=3, seed=seed, batch_size=2, sampling="one-step", device="cpu",
        dfg_checkpoint=str(example.dfg), drg_checkpoint=str(example.drg), training_config=str(example.config))
    np.testing.assert_array_equal(public_features, actual)
    np.testing.assert_array_equal(public_labels, labels)
    assert actual.dtype == np.float32 and labels.dtype == np.int64
    assert adapter.pairing_status == "sha256_verified"
    assert runtime.has_relation_state and not runtime.netG.training and not runtime.netR.training
    assert all(not p.requires_grad for p in runtime.netG.parameters())


def synthetic_reference(example, monkeypatch):
    data = example.data
    real, counts = class_centers(data.real_features, data.real_labels, data.class_ids)
    semantics = data.attributes[data.class_ids].numpy()
    semantic, visual = cosine_distances(semantics), cosine_distances(real)
    rows = reference_triplets(semantic, visual)
    scores, outcomes = score_triplets(rows, visual)
    assert scores["eligible_triplets"] > 0 and scores["tied_triplets"] == 0
    seed = SEEDS[example.dataset]
    arrays = {"class_ids": data.class_ids, "class_names": data.class_names, "semantics": semantics,
              "real_centers": real, "real_counts": counts, "generated_centers": real[None],
              "seeds": np.array([seed], dtype=np.int64), "semantic_distances": semantic,
              "real_distances": visual, "generated_distances": visual[None]}
    summary = make_summary(example.dataset, [seed], rows, [{"seed": seed, **scores}], [outcomes], None)
    summary["method"] = "ZeroDiff"
    summary["schema_version"] = "awa2-class-relations-v1"
    manifest = {"options": vars(example.options), "data": data.files, "data_alignment": data.alignment, "batch_size": 100}
    output = example.root.parent.parent / "reference"
    write_results(output, summary, arrays, rows, [outcomes], manifest)
    # Synthetic fixtures verify the pipeline. Real reference acceptance is tested separately below.
    monkeypatch.setitem(BASELINE_COUNTS, example.dataset, (scores["eligible_triplets"], scores["reversed_triplets"]))
    return read_result(output / "results.zip")


def arguments(example, reference, output):
    return ["--dataset", example.dataset, "--dataroot", str(example.dataroot),
            "--dfg-checkpoint", str(example.dfg), "--drg-checkpoint", str(example.drg),
            "--training-config", str(example.config), "--reference-results", str(reference.path),
            "--device", "cpu", "--output-dir", str(output)]


@pytest.mark.parametrize("dataset", ["AWA2", "CUB", "SUN"])
def test_full_cli_package_comparison_and_no_overwrite(tmp_path, monkeypatch, dataset):
    example = make_weights(make_data(tmp_path, dataset))
    reference = synthetic_reference(example, monkeypatch)
    main(["--verify-reference", str(reference.path)])
    validate_alignment(reference, example.data)
    output = tmp_path / "diagnostic"
    summary = main(arguments(example, reference, output))
    result = read_result(output / "results.zip")
    assert summary == result.summary
    assert result.summary["method"] == "Rediff"
    assert result.summary["aggregate"]["sample_sd_reversal_rate"] is None
    assert set(p.name for p in output.iterdir()) == {"summary.json", "per_seed.csv", "triplets.csv", "centers.npz",
                                                   "manifest.json", "report.md", "figure1.svg", "results.zip"}
    with zipfile.ZipFile(output / "results.zip") as archive:
        assert len(archive.namelist()) == 7 and "results.zip" not in archive.namelist()
    with (output / "triplets.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    count = DATASET_SPECS[dataset]["unseen"]
    total = count * (count - 1) * (count - 2) // 2
    assert len(rows) == total and [int(r["triplet_index"]) for r in rows] == list(range(total))
    assert triplet_signature(result.rows) == triplet_signature(reference.rows)
    compare_dir = tmp_path / "comparison"
    comparison = compare_main(["--rediff-results", str(output / "results.zip"), "--reference-results",
                               str(reference.path), "--output-dir", str(compare_dir)])
    assert comparison[0]["delta_percentage_points"] == 100 * result.records[0]["reversal_rate"]
    assert (compare_dir / "report.md").is_file()
    assert result.manifest["checkpoint_pairing"] == "sha256_verified"
    assert result.manifest["rng_policy"]["extra_random_operations"] == []
    assert result.manifest["source_sha256"]["feature_export/generation.py"]
    with pytest.raises(FileExistsError):
        main(arguments(example, reference, output))
    with pytest.raises(FileExistsError):
        compare_main(["--rediff-results", str(output / "results.zip"), "--reference-results",
                      str(reference.path), "--output-dir", str(compare_dir)])
    changed = deepcopy(result)
    changed.arrays["class_ids"][0] += 1
    with pytest.raises(ValueError, match="arrays differ"):
        comparison_row(changed, reference)


def test_reference_hash_tampering_and_data_mismatch_stop_generation(tmp_path, monkeypatch):
    example = make_weights(make_data(tmp_path, "AWA2"))
    reference = synthetic_reference(example, monkeypatch)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(reference.path) as source, zipfile.ZipFile(tampered, "w") as target:
        for name in source.namelist():
            value = source.read(name)
            target.writestr(name, value + b" " if name == "summary.json" else value)
    with pytest.raises(ValueError, match="sha256"):
        read_result(tampered)
    content = sio.loadmat(example.root / "ce_ce.mat")["features"]
    content[0, 0] += 1
    sio.savemat(example.root / "ce_ce.mat", {"features": content})
    def forbidden(*args, **kwargs):
        pytest.fail("Generation must not run on mismatched data")
    monkeypatch.setattr(VisualFeatureGenerator, "generate", forbidden)
    with pytest.raises(ValueError, match="data_alignment"):
        main(arguments(example, reference, tmp_path / "bad_run"))
    assert not (tmp_path / "bad_run").exists()


def test_embedded_config_pairing_and_strict_generator_loading(tmp_path):
    example = make_weights(make_data(tmp_path, "AWA2"))
    example.config.unlink()
    adapter = VisualFeatureGenerator(dataset="AWA2", device="cpu", dfg_checkpoint=example.dfg, drg_checkpoint=example.drg)
    assert adapter.config_status == "embedded_checkpoint"
    checkpoint = torch.load(example.dfg, weights_only=True)
    checkpoint["training_config"]["drg_sha256"] = "0" * 64
    torch.save(checkpoint, example.dfg)
    with pytest.raises(ValueError, match="DRG checkpoint hash"):
        VisualFeatureGenerator(dataset="AWA2", device="cpu", dfg_checkpoint=example.dfg, drg_checkpoint=example.drg)
    checkpoint["training_config"]["drg_sha256"] = sha256_file(example.drg)
    checkpoint["state_dict_G"].pop(next(iter(checkpoint["state_dict_G"])))
    torch.save(checkpoint, example.dfg)
    adapter = VisualFeatureGenerator(dataset="AWA2", device="cpu", dfg_checkpoint=example.dfg, drg_checkpoint=example.drg)
    with pytest.raises(RuntimeError, match="Missing key"):
        _ = adapter.runtime


@pytest.mark.parametrize("dataset,folder", [("AWA2", "AWA2_100pct_onestep"),
                                          ("CUB", "CUB_100pct_onestep_seed3483"),
                                          ("SUN", "SUN_100pct_onestep_seed4115")])
def test_real_zerodiff_reference_replay(dataset, folder):
    root = os.environ.get("REDIFF_ZERODIFF_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set REDIFF_ZERODIFF_REFERENCE_ROOT to the directory containing the three ZeroDiff result folders")
    result = read_result(Path(root) / folder / "results.zip")
    record = verify_baseline(result)
    assert (record["eligible_triplets"], record["reversed_triplets"]) == BASELINE_COUNTS[dataset]
