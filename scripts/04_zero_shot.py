#!/usr/bin/env python3
"""
Compute ESM-2 zero-shot baseline scores for DMS assays.

Wraps src.models.baseline with property filtering and idempotency.
Each assay's output is checked before scoring — safe to re-run.

Usage:
    python scripts/04_zero_shot.py --config configs/train_config.yaml
    python scripts/04_zero_shot.py --property stability --smoke-test
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Compute ESM-2 zero-shot baseline scores"
    )
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--property", type=str,
                        choices=["stability", "binding", "expression", "activity"],
                        help="Score only assays in this property category")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run on 2 assays per category only")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-scored assays (default behavior)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.data.download import load_reference_file

    reference_df = load_reference_file(config)

    # Determine which assays to score
    if args.property:
        from src.data.categorize import load_categorization

        categorization = load_categorization(
            Path(config["data"]["processed_dir"]) / "category_assignments.json"
        )
        assays_to_score = categorization[args.property]
        logger.info(f"Filtering to {args.property}: {len(assays_to_score)} assays")
    else:
        assays_to_score = reference_df["DMS_id"].tolist()

    if args.smoke_test:
        n = config["smoke_test"].get("assays_per_category", 2)
        assays_to_score = assays_to_score[:n]
        logger.info(f"Smoke test: limiting to {len(assays_to_score)} assays")

    # Check which are already done
    output_dir = Path(config["zero_shot"]["output_dir"])
    remaining = []
    for dms_id in assays_to_score:
        output_path = output_dir / f"{dms_id}_zero_shot.csv"
        if output_path.exists():
            logger.info(f"  {dms_id}: already scored, skipping")
        else:
            remaining.append(dms_id)

    if not remaining:
        logger.info("All assays already scored. Nothing to do.")
        return

    logger.info(f"Scoring {len(remaining)} assays...")

    # Load model (heavy — only if we have work to do)
    import torch
    from transformers import EsmForMaskedLM, EsmTokenizer

    from src.models.baseline import compute_zero_shot_for_assay

    model_name = config["model"]["base_model"]
    logger.info(f"Loading {model_name}...")
    model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    raw_dir = Path(config["data"]["raw_dir"]) / "DMS_ProteinGym_substitutions"

    for dms_id in remaining:
        row = reference_df[reference_df["DMS_id"] == dms_id]
        if row.empty:
            logger.warning(f"  {dms_id}: not found in reference, skipping")
            continue

        row = row.iloc[0]
        csv_path = raw_dir / row["DMS_filename"]
        wildtype_seq = row["target_seq"]
        output_path = output_dir / f"{dms_id}_zero_shot.csv"

        logger.info(f"\n{dms_id}:")
        try:
            compute_zero_shot_for_assay(
                model, tokenizer, csv_path, wildtype_seq, output_path, device
            )
        except Exception as e:
            logger.error(f"  Failed to score {dms_id}: {e}")
            continue

    # Generate summary CSV
    _generate_summary(output_dir, assays_to_score, reference_df)
    logger.info("Zero-shot scoring complete!")


def _generate_summary(output_dir: Path, assays: list, reference_df):
    """Generate a summary CSV with Spearman per assay."""
    import pandas as pd
    from scipy.stats import spearmanr

    rows = []
    for dms_id in assays:
        path = output_dir / f"{dms_id}_zero_shot.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "zero_shot_score" not in df.columns:
            continue
        rho, _ = spearmanr(df["DMS_score"], df["zero_shot_score"])
        rows.append({"dms_id": dms_id, "spearman": rho, "n_variants": len(df)})

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(output_dir / "zero_shot_summary.csv", index=False)
        logger.info(
            f"Summary: {len(rows)} assays, "
            f"mean Spearman={summary['spearman'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
