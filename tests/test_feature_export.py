"""AWA2 export contracts and equivalence to the actual training sampler."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import scipy.io as sio
import torch

from datasets.image_util import DATA_LOADER
from feature_export import SCHEMA_VERSION
from feature_export.data import load_awa2
from feature_export.export import main, make_packages, make_protocol, validate_protocol
from feature_export.runtime import FeatureRuntime, generate_features, load_options
import zerodiff_tools


@pytest.fixture
def example(tmp_path):
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    root = tmp_path / "Dataset" / "AWA2"
    root.mkdir(parents=True)
    labels = np.concatenate([np.arange(1, 41), np.arange(1, 41), np.repeat(np.arange(41, 51), 2)])
    rng = np.random.default_rng(12)
    features = rng.uniform(0, 5, (100, 2048)).astype(np.float32)
    sio.savemat(root / "res101.mat", {"labels": labels[:, None], "features": features.T})
    sio.savemat(root / "ce_ce.mat", {"features": features})
    sio.savemat(root / "con_paco.mat", {"features": features * 0.7})
    names = np.empty((50, 1), dtype=object)
    for index in range(50):
        names[index, 0] = f"class_{index}"
    attributes = rng.uniform(0, 1, (85, 50)).astype(np.float32)
    sio.savemat(root / "att_splits.mat", {
        "trainval_loc": np.arange(1, 41)[:, None], "train_loc": np.arange(1, 41)[:, None],
        "val_loc": np.arange(81, 101)[:, None], "test_seen_loc": np.arange(41, 81)[:, None],
        "test_unseen_loc": np.arange(81, 101)[:, None],
        "att": attributes, "original_att": attributes * 100, "allclasses_names": names,
    })
    for percent, number in ((10, 4), (30, 12)):
        sio.savemat(root / f"split_{percent}percent.mat", {"trainval_loc": np.arange(1, number + 1)[:, None]})
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"training": {"dataset": "AWA2", "split_percent": 100,
                                               "decoder_layer_sizes": [8, 8192], "n_T": 3}}))
    options, _ = load_options(100, config)
    torch.manual_seed(123)
    dfg = tmp_path / "dfg.tar"
    drg = tmp_path / "drg.tar"
    torch.save({"state_dict_G": zerodiff_tools.DFG_Generator(options).state_dict(),
                "state_dict_VSRA": {"unused_training_parameter": torch.ones(1)},
                "method_metadata": {"name": "test_relation_method"}}, dfg)
    torch.save({"state_dict_R": zerodiff_tools.DRG_Generator(options).state_dict()}, drg)
    yield SimpleNamespace(root=root, dataroot=root.parent, config=config, options=options,
                          dfg=dfg, drg=drg, output=tmp_path / "exports")
    torch.set_num_threads(old_threads)
    torch.use_deterministic_algorithms(old_determinism)


@pytest.mark.parametrize("percent", [10, 30, 100])
@pytest.mark.parametrize("preprocessing,standardization", [(True, False), (True, True), (False, False)])
def test_real_features_and_attributes_match_training_loader(example, percent, preprocessing, standardization):
    options = deepcopy(example.options)
    options.split_percent = percent
    options.dataroot = str(example.dataroot)
    options.preprocessing = preprocessing
    options.standardization = standardization
    options.class_embedding_norm = preprocessing
    options.image_embedding = "res101"
    expected = DATA_LOADER(options)
    actual = load_awa2(example.dataroot, options)
    np.testing.assert_array_equal(actual.real_features, expected.test_unseen_feature.numpy())
    np.testing.assert_array_equal(actual.real_labels, expected.test_unseen_label.numpy())
    torch.testing.assert_close(actual.attributes, expected.attribute, rtol=0, atol=0)
    assert actual.class_ids.tolist() == list(range(40, 50))
    assert actual.class_names.tolist() == [f"class_{index}" for index in range(40, 50)]
    assert actual.real_indices.tolist() == list(range(80, 100))


def reference_sampler(runtime):
    path = Path(__file__).resolve().parents[1] / "zerodiff_DFG_train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ZERODIFF")
    methods = [node for node in trainer.body if isinstance(node, ast.FunctionDef)
               and node.name in ("sample_from_model", "sample_posterior")]
    namespace = {"torch": torch, "zerodiff_tools": zerodiff_tools}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    reference = SimpleNamespace(dim_v=2048, dim_noise=85, n_T=runtime.options.n_T,
                                device=runtime.device, netG=runtime.netG, netR=runtime.netR,
                                posterior_coefficients=runtime.posterior)
    for name in ("sample_from_model", "sample_posterior"):
        setattr(reference, name, MethodType(namespace[name], reference))
    return reference


@pytest.mark.parametrize("sampling", ["one-step", "progressive"])
def test_sampler_matches_original_training_methods_exactly_on_cpu(example, sampling):
    runtime = FeatureRuntime(example.options, example.dfg, example.drg, "cpu")
    reference = reference_sampler(runtime)
    attributes = load_awa2(example.dataroot, example.options).attributes[40].repeat(3, 1)
    torch.manual_seed(71)
    expected, _ = reference.sample_from_model(attributes, progressive=sampling == "progressive")
    generator = torch.Generator().manual_seed(71)
    actual = runtime.sample(attributes, sampling, generator, generator)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad
    assert runtime.has_relation_state


def arguments(example, output, *extra):
    return ["--dfg-checkpoint", str(example.dfg), "--drg-checkpoint", str(example.drg),
            "--dataroot", str(example.dataroot), "--method", "Rediff", "--split-percent", "100",
            "--sampling", "one-step", "--samples-per-class", "3", "--batch-size", "2",
            "--device", "cpu", "--output-dir", str(output), *extra]


@pytest.mark.parametrize("view,filenames", [
    ("both", ["generated_only.npz", "real_vs_generated.npz"]),
    ("generated-only", ["generated_only.npz"]),
    ("real-vs-generated", ["real_vs_generated.npz"]),
])
def test_cli_writes_portable_arrays_and_reuses_generated_samples(example, view, filenames):
    main(arguments(example, example.output, "--view", view))
    manifest = json.loads((example.output / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["artifacts"]) == filenames
    assert manifest["checkpoint_has_relation_state"]
    assert manifest["training_config_status"] == "provided_or_auto_detected"
    assert manifest["split_percent"] == 100
    assert manifest["inference_options"]["decoder_layer_sizes"] == [8, 2048]
    loaded = {}
    for name in filenames:
        with np.load(example.output / name, allow_pickle=False) as package:
            arrays = {key: package[key] for key in package.files}
        loaded[name] = arrays
        assert arrays["features"].dtype == np.float32
        assert arrays["labels"].dtype == np.int64
        assert arrays["source"].dtype == np.uint8
        assert arrays["schema_version"].item() == SCHEMA_VERSION
        assert np.isfinite(arrays["features"]).all()
        assert np.unique(arrays["sample_ids"]).size == arrays["features"].shape[0]
        assert np.bincount(arrays["labels"][arrays["source"] == 1], minlength=50)[40:].tolist() == [3] * 10
    if view == "both":
        generated, combined = loaded[filenames[0]], loaded[filenames[1]]
        for key in ("features", "labels", "source", "sample_ids", "dataset_indices"):
            np.testing.assert_array_equal(generated[key], combined[key][combined["source"] == 1])
        assert combined["features"].shape == (50, 2048)
        assert combined["dataset_indices"][:20].tolist() == list(range(80, 100))


def test_repeat_export_and_protocol_reuse_across_method_names(example):
    main(arguments(example, example.output))
    second = example.output.parent / "baseline"
    main(arguments(example, second, "--method", "ZeroDiff", "--protocol", str(example.output / "protocol.json")))
    for name in ("generated_only.npz", "real_vs_generated.npz"):
        with np.load(example.output / name, allow_pickle=False) as first, np.load(second / name, allow_pickle=False) as other:
            for key in first.files:
                np.testing.assert_array_equal(first[key], other[key])
    with pytest.raises(FileExistsError, match="already exists"):
        main(arguments(example, example.output))


def test_protocol_rejects_changed_seed_before_loading_models(example, monkeypatch):
    main(arguments(example, example.output))
    import feature_export.export as exporter
    monkeypatch.setattr(exporter, "FeatureRuntime", lambda *args: pytest.fail("Must reject before inference"))
    second = example.output.parent / "mismatch"
    with pytest.raises(ValueError, match="generation.seed"):
        main(arguments(example, second, "--seed", "12", "--protocol", str(example.output / "protocol.json")))
    assert not second.exists()


def test_protocol_rejects_different_feature_space_but_allows_training_percent(example):
    data100 = load_awa2(example.dataroot, example.options)
    options30 = deepcopy(example.options)
    options30.split_percent = 30
    data30 = load_awa2(example.dataroot, options30)
    args = SimpleNamespace(samples_per_class=3, sampling="one-step", seed=9182, batch_size=2)
    protocol = make_protocol(data100, args)
    validate_protocol(protocol, make_protocol(data30, args))
    changed = deepcopy(protocol)
    changed["alignment"]["data_sha256"]["ce_ce.mat"] = "different-feature-extractor"
    with pytest.raises(ValueError, match="ce_ce.mat"):
        validate_protocol(protocol, changed)


def test_drg_pairing_and_training_percentage_are_checked(example):
    with pytest.raises(ValueError, match="split-percent"):
        main(arguments(example, example.output, "--split-percent", "30"))
    config = json.loads(example.config.read_text())
    config["training"]["drg_sha256"] = "wrong-drg"
    example.config.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="DRG checkpoint hash"):
        main(arguments(example, example.output))
    assert not example.output.exists()


def test_frozen_baseline_with_original_drg_key_is_supported(example):
    checkpoint = torch.load(example.dfg, weights_only=True)
    checkpoint.pop("state_dict_VSRA")
    checkpoint.pop("method_metadata")
    torch.save(checkpoint, example.dfg)
    drg = torch.load(example.drg, weights_only=True)
    torch.save({"state_dict_G_con": drg["state_dict_R"]}, example.drg)
    runtime = FeatureRuntime(example.options, example.dfg, example.drg, "cpu")
    data = load_awa2(example.dataroot, example.options)
    generated, labels = generate_features(runtime, data, 2, 19, 2, "one-step")
    assert generated.shape == (20, 2048)
    assert not runtime.has_relation_state
    invalid = generated.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        make_packages(data, invalid, labels, 2)


def test_export_package_runs_in_minimal_baseline_checkout(example):
    import os
    import shutil
    import subprocess
    import sys

    source = Path(__file__).resolve().parents[1]
    baseline = example.output.parent / "minimal_zerodiff"
    baseline.mkdir()
    package = baseline / "feature_export"
    package.mkdir()
    for path in (source / "feature_export").glob("*.py"):
        shutil.copy2(path, package / path.name)
    shutil.copy2(source / "zerodiff_tools.py", baseline / "zerodiff_tools.py")
    (baseline / "datasets").mkdir()
    shutil.copy2(source / "datasets" / "image_util.py", baseline / "datasets" / "image_util.py")
    (baseline / "datasets" / "__init__.py").write_text("")
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    output = example.output.parent / "portable_export"
    result = subprocess.run(
        [sys.executable, "-m", "feature_export.export", *arguments(example, output)],
        cwd=baseline, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["generated_only.npz"]["generated_rows"] == 30
    assert manifest["git"]["commit"] is None
