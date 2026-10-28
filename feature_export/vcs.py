"""Classifier-native VCS geometry on the original, fixed V-reference triplets.

Each method uses its own frozen decoder. This measures preservation of the
shared reference ordering, not reconstruction of method-specific real VCS.
"""

from __future__ import annotations

import json

import numpy as np

from feature_export.reference import (
    VCS_SCHEMA, read_result, triplet_signature, validate_alignment, verify_baseline,
    vcs_layout, with_preservation,
)
from feature_export.relations import class_centers, cosine_distances, score_triplets


REFERENCE_ARRAYS = ("class_ids", "class_names", "semantics", "real_centers", "real_counts",
                    "semantic_distances", "real_distances")
PROTOCOL_FIELDS = ("dataset", "split", "sampling", "seeds", "samples_per_class", "min_gap", "tie_gap", "distance")


def validate_shared_reference(result, reference):
    for key in PROTOCOL_FIELDS:
        if result.summary[key] != reference.summary[key]:
            raise ValueError(f"VCS reference protocol mismatch: {key}.")
    for key in REFERENCE_ARRAYS:
        if not np.array_equal(result.arrays[key], reference.arrays[key]):
            raise ValueError(f"VCS reference arrays differ: {key}.")
    if result.manifest.get("data_alignment") != reference.manifest.get("data_alignment"):
        raise ValueError("VCS reference data or preprocessing differ.")
    if triplet_signature(result.rows) != triplet_signature(reference.rows):
        raise ValueError("VCS eligible sets or triplet indices differ.")


def validate_visual_source(source, reference, method, dfg_hash, drg_hash, options=None):
    """Bind the supplementary run to the exact original V checkpoint pair."""
    if source.summary["schema_version"] == VCS_SCHEMA:
        raise ValueError("--visual-results must contain original V results.")
    if source.summary.get("method", "ZeroDiff") != method:
        raise ValueError("--method disagrees with --visual-results.")
    validate_shared_reference(source, reference)
    if method == "ZeroDiff":
        verify_baseline(source)
        if source.sha256 != reference.sha256:
            raise ValueError("ZeroDiff VCS must use the original verified ZeroDiff V package.")
    elif source.manifest.get("reference", {}).get("sha256") != reference.sha256:
        raise ValueError("Original Rediff V results used a different ZeroDiff reference.")
    if source.manifest.get("batch_size") != 100:
        raise ValueError("Original V results must use batch size 100.")
    for role, actual_hash in (("dfg", dfg_hash), ("drg", drg_hash)):
        stored = source.manifest.get(role + "_checkpoint", {}).get("sha256")
        if stored is not None:
            matched = stored == actual_hash
        else:
            # Original ZeroDiff manifests store the two input hashes without role keys.
            matched = actual_hash in source.manifest.get("inputs", {}).values()
        if not matched:
            raise ValueError(f"{role.upper()} checkpoint hash differs from the original V results.")
    if options is not None:
        for key in ("dataset", "resSize", "attSize", "noiseSize", "dim_t", "n_T", "decoder_layer_sizes",
                    "class_embedding", "class_embedding_norm", "preprocessing", "standardization", "split_percent"):
            if source.manifest.get("options", {}).get(key) != getattr(options, key):
                raise ValueError(f"Inference option differs from the original V results: {key}.")


def check_visual_replay(centers, distances, source, index, scores):
    # This small numeric tolerance permits floating point differences across devices;
    # original checkpoint hashes, generation settings and reversal/tie counts remain exact.
    expected = source.arrays["generated_centers"][index]
    if not np.allclose(centers, expected, rtol=1e-6, atol=1e-7):
        raise ValueError("Regenerated V centers disagree with --visual-results; check the environment and sampling.")
    if not np.allclose(distances, source.arrays["generated_distances"][index], rtol=1e-6, atol=1e-7):
        raise ValueError("Regenerated V distances disagree with --visual-results.")
    for key in ("eligible_triplets", "reversed_triplets", "tied_triplets"):
        if scores[key] != source.records[index][key]:
            raise ValueError(f"Regenerated V statistics disagree with --visual-results: {key}.")
    return {"max_abs_center_difference": float(np.max(np.abs(centers - expected))),
            "centers_bitwise_equal": bool(np.array_equal(centers, expected)),
            "rtol": 1e-6, "atol": 1e-7, "counts": "exact_match"}


def run_vcs(args, reference):
    from feature_export.data import load_dataset
    from feature_export.diagnostic import configure_device, make_manifest, make_summary, write_results
    from feature_export.generation import VisualFeatureGenerator

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new --output-dir.")
    source = read_result(args.visual_results)
    adapter = VisualFeatureGenerator(dataset=args.dataset, device=configure_device(args.device),
                                     dfg_checkpoint=args.dfg_checkpoint, drg_checkpoint=args.drg_checkpoint,
                                     training_config=args.training_config, feature_space="VCS")
    validate_visual_source(source, reference, args.method, adapter.dfg_sha256, adapter.drg_sha256, adapter.options)
    data = load_dataset(args.dataroot, adapter.options)
    alignment = validate_alignment(reference, data)
    centers, distances, visual_distances, controls, replay_checks, records, outcomes = [], [], [], [], [], [], []
    for index, seed in enumerate(args.seeds):
        features, labels = adapter.generate(class_ids=data.class_ids, attributes=data.attributes,
                                            samples_per_class=args.samples_per_class, seed=seed,
                                            batch_size=args.batch_size, sampling=args.sampling)
        center, counts = class_centers(features, labels, data.class_ids)
        if not np.all(counts == args.samples_per_class):
            raise ValueError("Unexpected VCS generated class counts.")
        visual_distance = cosine_distances(center[:, :2048])
        visual_scores, _ = score_triplets(reference.rows, visual_distance, args.tie_gap)
        replay_checks.append(check_visual_replay(center[:, :2048], visual_distance, source, index, visual_scores))
        controls.append({"seed": seed, **with_preservation(visual_scores)})
        distance = cosine_distances(center)
        scores, observations = score_triplets(reference.rows, distance, args.tie_gap)
        records.append({"seed": seed, **with_preservation(scores)})
        centers.append(center)
        distances.append(distance)
        visual_distances.append(visual_distance)
        outcomes.append(observations)
        print(json.dumps({"method": args.method, "generated_space": "VCS", **records[-1]}, allow_nan=False), flush=True)
    arrays = {key: reference.arrays[key].copy() for key in REFERENCE_ARRAYS}
    arrays.update(generated_centers=np.stack(centers), generated_distances=np.stack(distances),
                  generated_visual_distances=np.stack(visual_distances), seeds=np.asarray(args.seeds, dtype=np.int64))
    summary = make_summary(args.dataset, args.seeds, reference.rows, records, outcomes, None)
    layout = vcs_layout(adapter.options.attSize, adapter.options.ngh)
    summary.update(schema_version=VCS_SCHEMA, method=args.method, generated_space="VCS", reference_space="V",
                   generated_feature_dim=adapter.runtime.vcs_dim, decoder_hidden_dim=adapter.options.ngh,
                   feature_blocks=layout, eligibility_policy="fixed_semantics_and_real_V",
                   block_scaling="classifier_native_no_extra_normalization",
                   visual_control={"source_results_sha256": source.sha256, "per_seed": controls, "replay_checks": replay_checks})
    if summary["aggregate"] is not None:
        rates = [r["preservation_rate"] for r in records]
        summary["aggregate"].update(mean_preservation_rate=float(np.mean(rates)),
                                    sample_sd_preservation_rate=float(np.std(rates, ddof=1)) if len(rates) > 1 else None)
    manifest = make_manifest(args, adapter, data, reference, alignment)
    manifest.update(schema_version=VCS_SCHEMA, reference_space="V", feature_blocks=layout,
                    block_scaling=summary["block_scaling"], eligibility_policy=summary["eligibility_policy"],
                    visual_source={"path": str(source.path), "sha256": source.sha256},
                    decoder={"source": "state_dict_Dec in dfg_checkpoint", "strict": True,
                             "hidden_dim": adapter.options.ngh, "method_specific": True})
    write_results(output, summary, arrays, reference.rows, outcomes, manifest)
    print(f"VCS results: {output / 'results.zip'}", flush=True)
    return summary
