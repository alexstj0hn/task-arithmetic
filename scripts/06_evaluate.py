#!/usr/bin/env python3
"""
Evaluate trained models and compute the cross-property evaluation matrix.

Usage:
    python scripts/06_evaluate.py --cross-property --config configs/train_config.yaml
    python scripts/06_evaluate.py --property stability --smoke-test
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
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--property", type=str, choices=PROPERTIES,
                        help="Evaluate a specific property model")
    parser.add_argument("--cross-property", action="store_true",
                        help="Compute the cross-property evaluation matrix")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip evaluation if results already exist")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Delegate to the evaluation module
    sys.argv = ["evaluate"]  # Reset argv for the module's argparse
    if args.config:
        sys.argv.extend(["--config", args.config])
    if args.property:
        sys.argv.extend(["--property", args.property])
    if args.cross_property:
        sys.argv.append("--cross-property")
    if args.smoke_test:
        sys.argv.append("--smoke-test")

    from src.evaluation.evaluate import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
