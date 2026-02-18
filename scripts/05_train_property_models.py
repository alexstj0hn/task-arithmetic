#!/usr/bin/env python3
"""
Train a property-specific LoRA model on pooled DMS assays.

Trains one property at a time. Run once per property, or let the
orchestrator (deploy/run_training.sh) call this for each.

Usage:
    python scripts/05_train_property_models.py --property stability --config configs/train_config.yaml
    python scripts/05_train_property_models.py --property binding --resume --smoke-test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROPERTIES = ["stability", "binding", "expression", "activity"]


def main():
    parser = argparse.ArgumentParser(
        description="Train a property-specific LoRA model"
    )
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--property", required=True, choices=PROPERTIES,
                        help="Which property to train")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick run with tiny data subset")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply smoke test overrides
    if args.smoke_test:
        config["smoke_test"]["enabled"] = True
        config["training"]["num_epochs"] = config["smoke_test"]["max_epochs"]
        config["training"]["list_size"] = config["smoke_test"].get(
            "list_size", config["training"]["list_size"]
        )
        logger.info("Smoke test mode: reduced epochs and data")

    # Check if already complete
    best_model_dir = (
        Path(config["checkpointing"]["checkpoint_dir"])
        / args.property
        / "best_model"
    )
    if best_model_dir.exists() and not args.resume:
        logger.info(
            f"Best model already exists at {best_model_dir}. "
            f"Use --resume to continue training or delete to retrain."
        )
        return

    # Set seed
    seed = config.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    from transformers import EsmTokenizer

    from src.data.categorize import load_categorization
    from src.data.dataset import PropertyCategoryDataset, load_assays_for_category
    from src.data.download import load_reference_file

    reference_df = load_reference_file(config)
    categorization = load_categorization(
        Path(config["data"]["processed_dir"]) / "category_assignments.json"
    )

    # Load split assignments
    with open(Path(config["data"]["splits_dir"]) / "train_assays.json") as f:
        train_dms_ids = set(json.load(f))
    with open(Path(config["data"]["splits_dir"]) / "test_assays.json") as f:
        test_dms_ids = set(json.load(f))

    # Get DMS IDs for this property, split into train/val
    property_dms_ids = categorization[args.property]
    train_ids = [d for d in property_dms_ids if d in train_dms_ids]
    val_ids = [d for d in property_dms_ids if d in test_dms_ids]

    if args.smoke_test:
        n = config["smoke_test"]["assays_per_category"]
        train_ids = train_ids[:n]
        val_ids = val_ids[:min(n, len(val_ids))]

    logger.info(f"Property: {args.property}")
    logger.info(f"Train assays: {len(train_ids)}")
    logger.info(f"Val assays: {len(val_ids)}")

    if not train_ids:
        logger.error(f"No training assays found for {args.property}!")
        sys.exit(1)

    # Load tokenizer and assays
    tokenizer = EsmTokenizer.from_pretrained(config["model"]["base_model"])
    raw_data_dir = Path(config["data"]["raw_dir"]) / "DMS_ProteinGym_substitutions"

    max_length = config["data"]["max_sequence_length"]
    normalize = config["data"]["score_normalization"]

    if args.smoke_test:
        max_variants = config["smoke_test"].get("max_variants_per_assay")
    else:
        max_variants = None

    train_assays = load_assays_for_category(
        args.property, train_ids, reference_df, raw_data_dir,
        tokenizer, max_length, normalize,
    )
    val_assays = load_assays_for_category(
        args.property, val_ids, reference_df, raw_data_dir,
        tokenizer, max_length, normalize,
    )

    if not train_assays:
        logger.error(f"Failed to load any training assays for {args.property}!")
        sys.exit(1)

    # Create training dataset
    train_dataset = PropertyCategoryDataset(
        train_assays,
        list_size=config["training"]["list_size"],
    )

    # Create trainer and run
    from src.training.trainer import PropertyTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    trainer = PropertyTrainer(
        config=config,
        property_name=args.property,
        train_dataset=train_dataset,
        val_assays=val_assays,
        device=device,
        resume=args.resume,
    )

    results = trainer.train()

    logger.info(f"\nTraining complete for {args.property}")
    logger.info(f"Best {results['metric_name']}: {results['best_metric']:.4f}")
    logger.info(f"Checkpoint dir: {results['checkpoint_dir']}")


if __name__ == "__main__":
    main()
