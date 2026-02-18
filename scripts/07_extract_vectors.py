#!/usr/bin/env python3
"""
Extract task vectors from trained checkpoints.

Extracts LoRA parameters, computes materialized weight deltas,
pairwise cosine similarity matrix, and summary statistics.

Usage:
    python scripts/07_extract_vectors.py --config configs/train_config.yaml
    python scripts/07_extract_vectors.py --property stability
"""

import argparse
import logging
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROPERTIES = ["stability", "binding", "expression", "activity"]


def main():
    parser = argparse.ArgumentParser(
        description="Extract task vectors from trained checkpoints"
    )
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--property", type=str, choices=PROPERTIES,
                        help="Extract for a specific property only")
    parser.add_argument("--resume", action="store_true",
                        help="Skip if task vector already exists")
    parser.add_argument("--smoke-test", action="store_true",
                        help="No effect (included for CLI consistency)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Delegate to the extraction module
    sys.argv = ["extract", "--config", args.config]
    if args.property:
        sys.argv.extend(["--property", args.property])

    from src.vectors.extract import main as extract_main

    extract_main()


if __name__ == "__main__":
    main()
