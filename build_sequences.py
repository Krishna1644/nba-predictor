"""
Backward-compatible entrypoint.

Prefer: python build_datasets.py

This wrapper rebuilds leak-safe sequences with a true chronological /
season holdout split (train through 2024-25, test 2025-26).
"""

from build_datasets import main

if __name__ == "__main__":
    main()
