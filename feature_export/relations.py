"""Class-centre relation diagnostics; no learned evaluation projection."""

from itertools import combinations

import numpy as np


def class_centers(features, labels, class_ids):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    class_ids = np.asarray(class_ids)
    if (features.ndim != 2 or labels.shape != (len(features),)
            or class_ids.ndim != 1 or len(np.unique(class_ids)) != len(class_ids)
            or not np.isfinite(features).all()):
        raise ValueError("Invalid features, labels or class IDs.")
    if set(np.unique(labels)) != set(class_ids):
        raise ValueError("Feature labels must cover exactly the requested classes.")
    counts = np.asarray([(labels == c).sum() for c in class_ids], dtype=np.int64)
    return np.stack([features[labels == c].mean(axis=0) for c in class_ids]), counts


def cosine_distances(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite matrix of class representations.")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Cosine distance is undefined for a zero class representation.")
    unit = vectors / norms[:, None]
    distances = np.clip(1 - unit @ unit.T, 0, 2)
    np.fill_diagonal(distances, 0)
    return distances


def relative_gap(near, far):
    """Positive means near < far; invariant to uniform distance scaling."""
    return (far - near) / max(far + near, 1e-12)


def reference_triplets(semantic, real, min_gap=0.05):
    """Orient each unordered candidate pair using semantics only, then filter."""
    if not 0 <= min_gap < 1:
        raise ValueError("min_gap must be in [0, 1).")
    semantic, real = np.asarray(semantic), np.asarray(real)
    if (semantic.ndim != 2 or semantic.shape[0] != semantic.shape[1]
            or real.shape != semantic.shape
            or not np.isfinite(semantic).all() or not np.isfinite(real).all()):
        raise ValueError("Reference distance matrices must be finite, square and aligned.")
    rows = []
    for anchor in range(len(semantic)):
        for left, right in combinations([i for i in range(len(semantic)) if i != anchor], 2):
            near, far = sorted((left, right), key=lambda i: (semantic[anchor, i], i))
            sem_gap = relative_gap(semantic[anchor, near], semantic[anchor, far])
            vis_gap = relative_gap(real[anchor, near], real[anchor, far])
            rows.append({
                "anchor": anchor, "near": near, "far": far,
                "semantic_near": float(semantic[anchor, near]),
                "semantic_far": float(semantic[anchor, far]),
                "real_near": float(real[anchor, near]),
                "real_far": float(real[anchor, far]),
                "semantic_gap": float(sem_gap), "real_gap": float(vis_gap),
                "eligible": bool(sem_gap > min_gap and vis_gap > min_gap),
            })
    return rows


def score_triplets(rows, generated, tie_gap=1e-6):
    if not 0 <= tie_gap < 1:
        raise ValueError("tie_gap must be in [0, 1).")
    generated = np.asarray(generated, dtype=np.float64)
    if (generated.ndim != 2 or generated.shape[0] != generated.shape[1]
            or not np.isfinite(generated).all()):
        raise ValueError("Generated distances must be a finite square matrix.")
    outcomes = []
    for row in rows:
        a, b, c = row["anchor"], row["near"], row["far"]
        near, far = float(generated[a, b]), float(generated[a, c])
        gap = relative_gap(near, far)
        outcome = "tie" if abs(gap) <= tie_gap else ("reversed" if gap < 0 else "preserved")
        outcomes.append({"generated_near": near, "generated_far": far,
                         "generated_gap": float(gap), "outcome": outcome})
    selected = [o for r, o in zip(rows, outcomes) if r["eligible"]]
    count = len(selected)
    reversed_count = sum(o["outcome"] == "reversed" for o in selected)
    tied_count = sum(o["outcome"] == "tie" for o in selected)
    return {
        "eligible_triplets": count, "reversed_triplets": reversed_count,
        "tied_triplets": tied_count,
        "reversal_rate": reversed_count / count if count else None,
        "tie_rate": tied_count / count if count else None,
    }, outcomes


def choose_example(rows, outcomes_by_seed):
    """Prefer majority-repeat reversals; choose median frequency, never the worst gap."""
    candidates = []
    for index, row in enumerate(rows):
        frequency = sum(outcomes[index]["outcome"] == "reversed" for outcomes in outcomes_by_seed)
        if row["eligible"] and frequency:
            candidates.append((index, frequency))
    if not candidates:
        return None
    majority = [c for c in candidates if c[1] * 2 > len(outcomes_by_seed)]
    pool = majority or candidates
    median = float(np.median([c[1] for c in pool]))
    index, frequency = min(pool, key=lambda c: (abs(c[1] - median), c[0]))
    reversed_seeds = [s for s, outcomes in enumerate(outcomes_by_seed)
                      if outcomes[index]["outcome"] == "reversed"]
    median_gap = float(np.median([outcomes_by_seed[s][index]["generated_gap"] for s in reversed_seeds]))
    seed_index = min(reversed_seeds,
                     key=lambda s: (abs(outcomes_by_seed[s][index]["generated_gap"] - median_gap), s))
    return {"triplet_index": index, "seed_index": seed_index, "reversed_seed_count": frequency,
            "seed_count": len(outcomes_by_seed), "majority_repeat": bool(majority)}
