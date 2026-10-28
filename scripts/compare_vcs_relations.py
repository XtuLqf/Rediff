#!/usr/bin/env python3
"""Compare new ZeroDiff/Rediff VCS runs and their paired V controls."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_export.vcs_comparison import main


if __name__ == "__main__":
    main()
