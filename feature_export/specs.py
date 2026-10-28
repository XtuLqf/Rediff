"""Dataset settings from the repository's ZeroDiff DFG training scripts."""

DATASET_SPECS = {
    "AWA2": {"classes": 50, "unseen": 10, "embedding": "att", "semantic_dim": 85, "normalize": True},
    "CUB": {"classes": 200, "unseen": 50, "embedding": "sent", "semantic_dim": 1024, "normalize": False},
    "SUN": {"classes": 717, "unseen": 72, "embedding": "att", "semantic_dim": 102, "normalize": True},
}
