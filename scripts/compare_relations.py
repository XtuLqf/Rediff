#!/usr/bin/env python3
"""Build the AWA2/CUB/SUN comparison table from verified result ZIPs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_export.comparison import main


if __name__ == "__main__":
    main()
