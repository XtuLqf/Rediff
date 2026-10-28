"""VCS composition matches the classifier and keeps V samples/reference fixed."""

from copy import deepcopy
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from classifiers.classifier_images import CLASSIFIER
from feature_export.comparison import comparison_row
from feature_export.diagnostic import main, make_summary, write_results
from feature_export.generation import VisualFeatureGenerator
from feature_export.reference import (
    BASELINE_COUNTS, SEEDS, VCS_SCHEMA, read_result, sha256_file, triplet_signature, verify_baseline,
)
from feature_export.relations import class_centers, cosine_distances, reference_triplets, score_triplets
from feature_export.runtime import FeatureRuntime
from feature_export.vcs_comparison import main as compare_vcs, comparison_rows
from scripts.run_vcs_diagnostics import main as launch_vcs, checkpoint_paths
import zerodiff_tools
from test_relation_diagnostic import make_data, make_weights, arguments, single_thread


def decoder_example(tmp_path, dataset):
    example = make_data(tmp_path, dataset)
    example.training["ngh"] = 8
    example.options.ngh = 8
    make_weights(example)
    checkpoint = torch.load(example.dfg, weights_only=True)
    checkpoint.pop("state_dict_VSRA")
    checkpoint.pop("method_metadata")
    checkpoint["state_dict_Dec"] = zerodiff_tools.V2S_mapping(example.options, example.options.attSize).state_dict()
    torch.save(checkpoint, example.dfg)
    return example


def original_reference(example, monkeypatch):
    adapter = VisualFeatureGenerator(dataset=example.dataset, device="cpu", dfg_checkpoint=example.dfg,
                                     drg_checkpoint=example.drg, training_config=example.config)
    seed = SEEDS[example.dataset]
    fake, labels = adapter.generate(class_ids=example.data.class_ids, attributes=example.data.attributes,
                                    samples_per_class=100, seed=seed, batch_size=100, sampling="one-step")
    real, counts = class_centers(example.data.real_features, example.data.real_labels, example.data.class_ids)
    centers, _ = class_centers(fake, labels, example.data.class_ids)
    semantics = example.data.attributes[example.data.class_ids].numpy()
    semantic, real_distance, generated = [cosine_distances(v) for v in (semantics, real, centers)]
    rows = reference_triplets(semantic, real_distance)
    scores, outcomes = score_triplets(rows, generated)
    assert scores["eligible_triplets"] > 0 and scores["tied_triplets"] == 0
    arrays = {"class_ids": example.data.class_ids, "class_names": example.data.class_names, "semantics": semantics,
              "real_centers": real, "real_counts": counts, "generated_centers": centers[None],
              "seeds": np.array([seed], dtype=np.int64), "semantic_distances": semantic,
              "real_distances": real_distance, "generated_distances": generated[None]}
    summary = make_summary(example.dataset, [seed], rows, [{"seed": seed, **scores}], [outcomes], None)
    summary["method"] = "ZeroDiff"
    manifest = {"options": vars(example.options), "data": example.data.files, "data_alignment": example.data.alignment,
                "batch_size": 100, "inputs": {str(example.dfg): sha256_file(example.dfg), str(example.drg): sha256_file(example.drg)},
                "dfg_checkpoint": {"path": str(example.dfg), "sha256": sha256_file(example.dfg)},
                "drg_checkpoint": {"path": str(example.drg), "sha256": sha256_file(example.drg)}}
    output = example.root.parent.parent / "original_zerodiff_v"
    write_results(output, summary, arrays, rows, [outcomes], manifest)
    monkeypatch.setitem(BASELINE_COUNTS, example.dataset, (scores["eligible_triplets"], scores["reversed_triplets"]))
    return read_result(output / "results.zip")


@pytest.mark.parametrize("dataset", ["AWA2", "CUB", "SUN"])
def test_vcs_matches_classifier_input_and_paired_visual_rng(tmp_path, dataset):
    example = decoder_example(tmp_path, dataset)
    visual = VisualFeatureGenerator(dataset=dataset, device="cpu", dfg_checkpoint=example.dfg,
                                    drg_checkpoint=example.drg, training_config=example.config)
    vcs = VisualFeatureGenerator(dataset=dataset, device="cpu", dfg_checkpoint=example.dfg,
                                 drg_checkpoint=example.drg, training_config=example.config, feature_space="VCS")
    kwargs = dict(class_ids=example.data.class_ids, attributes=example.data.attributes,
                  samples_per_class=3, seed=SEEDS[dataset], batch_size=2, sampling="one-step")
    expected_v, labels = visual.generate(**kwargs)
    actual, vcs_labels = vcs.generate(**kwargs)
    np.testing.assert_array_equal(actual[:, :2048], expected_v)
    np.testing.assert_array_equal(labels, vcs_labels)
    assert actual.shape[1] == 4096 + example.options.attSize + 8
    # Call the production classifier's feature composition directly, without training it.
    stub = SimpleNamespace(batch_size=2, cuda=False, useV=True, useS=True, useC=True, netDec=vcs.runtime.netDec)
    expected = []
    # Preserve the per-class final short batch: changing GEMV/GEMM shapes can
    # introduce float32 rounding even for the identical frozen decoder.
    for start in range(0, len(actual), 3):
        expected.append(CLASSIFIER.compute_dec_out(
            stub, torch.from_numpy(actual[start:start + 3, :2048].copy()), actual.shape[1],
            test_C=torch.from_numpy(actual[start:start + 3, -2048:].copy())))
    np.testing.assert_array_equal(actual, torch.cat(expected).numpy())
    rng1 = torch.Generator(device="cpu").manual_seed(122)
    rng2 = torch.Generator(device="cpu").manual_seed(122)
    att = example.data.attributes[example.data.class_ids[0]].repeat(2, 1)
    old_v = visual.runtime.sample(att, "one-step", rng1, rng1)
    new_v, new_c = vcs.runtime.sample(att, "one-step", rng2, rng2, return_contrastive=True)
    torch.testing.assert_close(old_v, new_v, rtol=0, atol=0)
    assert new_c.shape == (2, 2048)
    assert not vcs.runtime.netDec.training and all(not p.requires_grad for p in vcs.runtime.netDec.parameters())


def test_decoder_is_applied_before_class_averaging_and_loads_strictly(tmp_path):
    example = decoder_example(tmp_path, "AWA2")
    runtime = FeatureRuntime(example.options, example.dfg, example.drg, "cpu", with_decoder=True)
    with torch.no_grad():
        runtime.netDec.fc1.weight.zero_()
        runtime.netDec.fc1.bias.zero_()
        runtime.netDec.fc1.weight[:, 0] = 1
    visual, contrastive = torch.zeros(2, 2048), torch.ones(2, 2048)
    visual[:, 0] = torch.tensor([-1., 1.])
    transformed_mean = runtime.compose_vcs(visual, contrastive).double().mean(0)
    mean_transformed = runtime.compose_vcs(visual.mean(0, keepdim=True), contrastive.mean(0, keepdim=True))[0].double()
    assert not torch.allclose(transformed_mean, mean_transformed)
    checkpoint = torch.load(example.dfg, weights_only=True)
    checkpoint["state_dict_Dec"].pop("fc1.weight")
    torch.save(checkpoint, example.dfg)
    with pytest.raises(RuntimeError, match="Missing key"):
        FeatureRuntime(example.options, example.dfg, example.drg, "cpu", with_decoder=True)
    checkpoint.pop("state_dict_Dec")
    torch.save(checkpoint, example.dfg)
    with pytest.raises(KeyError, match="state_dict_Dec"):
        FeatureRuntime(example.options, example.dfg, example.drg, "cpu", with_decoder=True)
    # Existing V inference does not acquire a new decoder dependency.
    assert FeatureRuntime(example.options, example.dfg, example.drg, "cpu").netDec is None


@pytest.mark.parametrize("dataset", ["AWA2", "CUB", "SUN"])
def test_two_method_vcs_cli_keeps_reference_and_replays_v(tmp_path, monkeypatch, dataset):
    baseline = decoder_example(tmp_path, dataset)
    reference = original_reference(baseline, monkeypatch)
    rediff = deepcopy(baseline)
    folder = tmp_path / "rediff"
    folder.mkdir()
    rediff.config, rediff.dfg = folder / "config.json", folder / "dfg.tar"
    rediff.config.write_text(json.dumps({"training": rediff.training}))
    torch.manual_seed(880)
    torch.save({"state_dict_G": zerodiff_tools.DFG_Generator(rediff.options).state_dict(),
                "state_dict_Dec": zerodiff_tools.V2S_mapping(rediff.options, rediff.options.attSize).state_dict(),
                "state_dict_VSRA": {"training_only": torch.ones(1)}, "training_config": rediff.training}, rediff.dfg)
    v_out, rediff_out, baseline_out = tmp_path / "rediff_v", tmp_path / "rediff_vcs", tmp_path / "zerodiff_vcs"
    main(arguments(rediff, reference, v_out))
    rediff_args = arguments(rediff, reference, rediff_out) + ["--feature-space", "VCS", "--visual-results", str(v_out / "results.zip")]
    baseline_args = arguments(baseline, reference, baseline_out) + ["--feature-space", "VCS", "--method", "ZeroDiff"]
    main(rediff_args)
    main(baseline_args)
    r, z = read_result(rediff_out / "results.zip"), read_result(baseline_out / "results.zip")
    for result in (r, z):
        assert result.summary["schema_version"] == VCS_SCHEMA
        assert result.summary["reference_space"] == "V"
        assert result.summary["generated_space"] == "VCS"
        assert triplet_signature(result.rows) == triplet_signature(reference.rows)
        assert result.summary["visual_control"]["replay_checks"][0]["counts"] == "exact_match"
        assert result.summary["visual_control"]["replay_checks"][0]["centers_bitwise_equal"]
        assert result.manifest["decoder"]["strict"] is True
        ET.parse(result.path.parent / "figure1.svg")
    assert z.summary["visual_control"]["per_seed"][0]["reversed_triplets"] == reference.records[0]["reversed_triplets"]
    assert r.records[0]["preserved_triplets"] + r.records[0]["reversed_triplets"] + r.records[0]["tied_triplets"] == r.records[0]["eligible_triplets"]
    rows = compare_vcs(["--reference-results", str(reference.path), "--zerodiff-results", str(z.path),
                        "--rediff-results", str(r.path), "--output-dir", str(tmp_path / "comparison")])
    assert [row["generated_space"] for row in rows] == ["V", "VCS"]
    assert rows[0]["rediff_reversed_triplets"] == read_result(v_out / "results.zip").records[0]["reversed_triplets"]
    commands = launch_vcs(["--dataset", dataset, "--reference-results", str(reference.path),
                           "--visual-results", str(v_out / "results.zip"), "--output-root", str(tmp_path / "launcher"),
                           "--device", "cpu", "--dry-run"])
    assert len(commands) == 3
    assert commands[0][commands[0].index("--dfg-checkpoint") + 1] == str(baseline.dfg)
    assert commands[1][commands[1].index("--dfg-checkpoint") + 1] == str(rediff.dfg)
    assert not (tmp_path / "launcher").exists()
    with pytest.raises(ValueError, match="cannot be mixed"):
        comparison_row(r, reference)
    with pytest.raises(ValueError, match="original V reference"):
        verify_baseline(z)
    altered = deepcopy(r)
    altered.summary["feature_blocks"]["S_hidden"][1] += 1
    with pytest.raises(ValueError, match="layouts differ"):
        comparison_rows(altered, z, reference)
    with pytest.raises(FileExistsError):
        main(rediff_args)
    # A modified decoder changes the DFG hash and cannot be paired with old V results.
    checkpoint = torch.load(rediff.dfg, weights_only=True)
    checkpoint["state_dict_Dec"]["fc1.bias"][0] += 1
    torch.save(checkpoint, rediff.dfg)
    bad_args = arguments(rediff, reference, tmp_path / "bad") + ["--feature-space", "VCS", "--visual-results", str(v_out / "results.zip")]
    with pytest.raises(ValueError, match="DFG checkpoint hash"):
        main(bad_args)
    assert not (tmp_path / "bad").exists()


def test_legacy_manifest_checkpoint_paths_are_unambiguous():
    package = SimpleNamespace(path="original/results.zip", manifest={
        "inputs": {"/server/out/zerodiff_DFG_100_gzsl_VCS.tar": "dfg",
                   "/server/out/diffzero_pretrain_100_gzsl.tar": "drg", "/server/config.json": "config"},
        "training_config": "/server/config.json"})
    paths = checkpoint_paths(package)
    assert paths["dfg"].endswith("gzsl_VCS.tar") and paths["drg"].endswith("pretrain_100_gzsl.tar")
    assert paths["config"] == "/server/config.json"
    package.manifest["inputs"]["/other/zerodiff_DFG_100_gzsl_VCS.tar"] = "another"
    with pytest.raises(ValueError, match="Cannot resolve"):
        checkpoint_paths(package)
