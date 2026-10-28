#!/usr/bin/env python3
"""Rediff class-relations-v1 generation and ZeroDiff reference replay."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_export.diagnostic import main


if __name__ == "__main__":
    main()
