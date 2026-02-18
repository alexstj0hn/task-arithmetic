"""
Evaluation metrics for protein fitness prediction.

Computes Spearman, NDCG, top-k recall, and AUC for individual assays.
Generates the cross-property evaluation matrix (the key Phase 1 result).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, roc_auc_score

from src.data.dataset import ProteinGymAssay, load_assays_for_category
from src.models.esm_lora import ESMLoRAForRanking, create_model
from src.training.utils import load_model_weights

logger = logging.getLogger(__name__)

PROPERTIES = ["stability", "binding", "expression", "activity"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    binary_labels: Optional[np.ndarray] = None,
    top_k_fraction: float = 0.1,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for one assay.

    Args:
        y_true: Ground truth DMS scores
        y_pred: Predicted scores
        binary_labels: Optional binary fitness labels for AUC
        top_k_fraction: Fraction for top-k recall (default 10%)

    Returns:
        metrics: Dict with spearman, ndcg, top_k_recall, auc
    """
    metrics = {}

    # Spearman correlation
    if len(y_true) < 2:
        metrics["spearman"] = float("nan")
        metrics["spearman_pval"] = float("nan")
    else:
        rho, pval = spearmanr(y_true, y_pred)
        metrics["spearman"] = float(rho) if not np.isnan(rho) else 0.0
        metrics["spearman_pval"] = float(pval) if not np.isnan(pval) else 1.0

    # NDCG (scikit-learn requires 2D arrays)
    try:
        metrics["ndcg"] = float(ndcg_score(
            y_true.reshape(1, -1), y_pred.reshape(1, -1)
        ))
    except (ValueError, IndexError):
        metrics["ndcg"] = float("nan")

    # NDCG@10%
    k = max(1, int(len(y_true) * top_k_fraction))
    try:
        metrics["ndcg_10"] = float(ndcg_score(
            y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k
        ))
    except (ValueError, IndexError):
        metrics["ndcg_10"] = float("nan")

    # Top-k recall
    true_top_k = set(np.argsort(y_true)[-k:])
    pred_top_k = set(np.argsort(y_pred)[-k:])
    metrics["top_10_recall"] = len(true_top_k & pred_top_k) / k

    # AUC (if binary labels available)
    if binary_labels is not None and len(np.unique(binary_labels)) == 2:
        try:
            metrics["auc"] = float(roc_auc_score(binary_labels, y_pred))
        except ValueError:
            metrics["auc"] = float("nan")

    return metrics


def evaluate_model_on_assay(
    model: ESMLoRAForRanking,
    assay: ProteinGymAssay,
    device: torch.device,
    batch_size: int = 32,
    mixed_precision: str = "bf16",
) -> Dict[str, float]:
    """
    Run inference on all variants in an assay and compute metrics.

    Args:
        model: Trained ESMLoRAForRanking
        assay: ProteinGymAssay with variants and ground truth
        device: Torch device
        batch_size: Inference batch size
        mixed_precision: "bf16", "fp16", or "no"

    Returns:
        metrics: Dict with all metrics for this assay
    """
    model.eval()
    all_predictions = []

    use_amp = mixed_precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16

    with torch.no_grad():
        for start_idx in range(0, len(assay), batch_size):
            end_idx = min(start_idx + batch_size, len(assay))
            sequences = [assay.sequences[i] for i in range(start_idx, end_idx)]

            encoded = model.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=assay.max_length + 2,
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    preds = model(input_ids, attention_mask)
            else:
                preds = model(input_ids, attention_mask)

            all_predictions.extend(preds.cpu().float().numpy())

    all_predictions = np.array(all_predictions)

    metrics = compute_metrics(
        y_true=assay.scores,
        y_pred=all_predictions,
        binary_labels=assay.binary_labels,
    )

    metrics["dms_id"] = assay.dms_id
    metrics["category"] = assay.category
    metrics["n_variants"] = len(assay)

    return metrics


def evaluate_all_assays(
    model: ESMLoRAForRanking,
    assays: List[ProteinGymAssay],
    device: torch.device,
    batch_size: int = 32,
    mixed_precision: str = "bf16",
) -> pd.DataFrame:
    """
    Evaluate model across multiple assays.

    Returns:
        results_df: DataFrame with one row per assay
    """
    results = []

    for assay in assays:
        logger.info(f"  Evaluating {assay.dms_id} ({len(assay)} variants)...")
        metrics = evaluate_model_on_assay(
            model, assay, device, batch_size, mixed_precision
        )
        results.append(metrics)

    return pd.DataFrame(results)


def evaluate_zero_shot_for_assays(
    assays: List[ProteinGymAssay],
    zero_shot_dir: Path,
    top_k_fraction: float = 0.1,
) -> pd.DataFrame:
    """
    Compute metrics from pre-computed zero-shot scores.

    Args:
        assays: List of assays to evaluate
        zero_shot_dir: Directory containing {dms_id}_zero_shot.csv files
        top_k_fraction: Fraction for top-k metrics

    Returns:
        results_df: DataFrame with one row per assay
    """
    results = []

    for assay in assays:
        zs_path = zero_shot_dir / f"{assay.dms_id}_zero_shot.csv"
        if not zs_path.exists():
            logger.warning(f"No zero-shot scores for {assay.dms_id}, skipping")
            continue

        zs_df = pd.read_csv(zs_path)

        if "zero_shot_score" not in zs_df.columns:
            logger.warning(f"No zero_shot_score column in {zs_path}")
            continue

        y_true = zs_df["DMS_score"].values
        y_pred = zs_df["zero_shot_score"].values

        binary = None
        if "DMS_score_bin" in zs_df.columns:
            binary = zs_df["DMS_score_bin"].values

        metrics = compute_metrics(y_true, y_pred, binary, top_k_fraction)
        metrics["dms_id"] = assay.dms_id
        metrics["category"] = assay.category
        metrics["n_variants"] = len(zs_df)
        results.append(metrics)

    return pd.DataFrame(results)


def compute_cross_property_matrix(
    config: Dict,
    device: torch.device,
) -> pd.DataFrame:
    """
    Evaluate each property model on ALL test assays across ALL categories.

    This is THE key Phase 1 result. Returns a matrix where:
      rows = models (stability, binding, expression, activity, zero-shot)
      cols = property categories
      values = mean Spearman across test assays in that category

    Diagonal dominance proves property specificity.
    """
    from src.data.categorize import load_categorization
    from src.data.download import load_reference_file
    from transformers import EsmTokenizer

    reference_df = load_reference_file(config)
    categorization = load_categorization(
        Path(config["data"]["processed_dir"]) / "category_assignments.json"
    )

    # Load test assays
    with open(Path(config["data"]["splits_dir"]) / "test_assays.json") as f:
        test_dms_ids = set(json.load(f))

    tokenizer = EsmTokenizer.from_pretrained(config["model"]["base_model"])
    raw_data_dir = Path(config["data"]["raw_dir"]) / "DMS_ProteinGym_substitutions"

    # Load test assays per category
    test_assays_by_category = {}
    for prop in PROPERTIES:
        prop_test_ids = [d for d in categorization[prop] if d in test_dms_ids]
        if prop_test_ids:
            assays = load_assays_for_category(
                prop, prop_test_ids, reference_df, raw_data_dir,
                tokenizer, config["data"]["max_sequence_length"],
                config["data"]["score_normalization"],
            )
            test_assays_by_category[prop] = assays
            logger.info(f"  {prop}: {len(assays)} test assays")

    batch_size = config["evaluation"].get("eval_batch_size", 32)
    mixed_precision = config["training"].get("mixed_precision", "bf16")
    checkpoint_dir = Path(config["checkpointing"]["checkpoint_dir"])

    # Results matrix: rows = models, cols = categories
    matrix_data = {}

    # Evaluate each property model on all categories
    for model_prop in PROPERTIES:
        best_dir = checkpoint_dir / model_prop / "best_model"
        if not best_dir.exists():
            logger.warning(f"No best model for {model_prop}, skipping")
            continue

        logger.info(f"\nLoading {model_prop} model from {best_dir}")
        model = create_model(config, device)
        load_model_weights(best_dir, model)
        model.eval()

        row = {}
        for eval_category, assays in test_assays_by_category.items():
            results_df = evaluate_all_assays(
                model, assays, device, batch_size, mixed_precision
            )
            mean_spearman = results_df["spearman"].mean()
            row[eval_category] = float(mean_spearman)
            logger.info(
                f"  {model_prop} model on {eval_category} test: "
                f"Spearman={mean_spearman:.4f}"
            )

        matrix_data[model_prop] = row

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Add zero-shot baseline row
    zero_shot_dir = Path(config["zero_shot"]["output_dir"])
    if zero_shot_dir.exists():
        zs_row = {}
        for eval_category, assays in test_assays_by_category.items():
            zs_df = evaluate_zero_shot_for_assays(
                assays, zero_shot_dir,
                config["evaluation"].get("top_k_fraction", 0.1),
            )
            if not zs_df.empty:
                zs_row[eval_category] = float(zs_df["spearman"].mean())
        matrix_data["zero_shot"] = zs_row

    # Build DataFrame
    matrix_df = pd.DataFrame(matrix_data).T
    matrix_df.index.name = "model"
    matrix_df.columns.name = "eval_category"

    logger.info(f"\n{'='*60}")
    logger.info("Cross-Property Evaluation Matrix (Spearman)")
    logger.info(f"{'='*60}")
    logger.info(f"\n{matrix_df.to_string()}")

    return matrix_df


def main():
    """CLI entry point for evaluation."""
    import argparse

    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--property", type=str, choices=PROPERTIES,
                        help="Evaluate a specific property model")
    parser.add_argument("--cross-property", action="store_true",
                        help="Compute the cross-property evaluation matrix")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cross_property:
        logger.info("Computing cross-property evaluation matrix...")
        matrix_df = compute_cross_property_matrix(config, device)
        matrix_path = output_dir / "cross_property_matrix.csv"
        matrix_df.to_csv(matrix_path)
        logger.info(f"Saved cross-property matrix to {matrix_path}")

        # Also save as JSON for programmatic access
        summary = {
            "matrix": matrix_df.to_dict(),
            "notes": {
                "overlap_clusters": ["CP2C9_HUMAN", "PTEN_HUMAN"],
                "overlap_context": (
                    "Same proteins appear in train (expression) and test (activity). "
                    "This is expected for cross-property evaluation and does NOT "
                    "constitute data leakage for within-property metrics."
                ),
            },
        }
        with open(output_dir / "phase1_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    elif args.property:
        logger.info(f"Evaluating {args.property} model...")
        from src.data.categorize import load_categorization
        from src.data.download import load_reference_file
        from transformers import EsmTokenizer

        reference_df = load_reference_file(config)
        categorization = load_categorization(
            Path(config["data"]["processed_dir"]) / "category_assignments.json"
        )

        with open(Path(config["data"]["splits_dir"]) / "test_assays.json") as f:
            test_dms_ids = set(json.load(f))

        tokenizer = EsmTokenizer.from_pretrained(config["model"]["base_model"])
        raw_data_dir = Path(config["data"]["raw_dir"]) / "DMS_ProteinGym_substitutions"
        prop_test_ids = [d for d in categorization[args.property] if d in test_dms_ids]

        if args.smoke_test:
            prop_test_ids = prop_test_ids[:2]

        assays = load_assays_for_category(
            args.property, prop_test_ids, reference_df, raw_data_dir,
            tokenizer, config["data"]["max_sequence_length"],
            config["data"]["score_normalization"],
        )

        # Load model
        best_dir = Path(config["checkpointing"]["checkpoint_dir"]) / args.property / "best_model"
        if not best_dir.exists():
            raise FileNotFoundError(f"No best model found at {best_dir}")

        model = create_model(config, device)
        load_model_weights(best_dir, model)

        results_df = evaluate_all_assays(
            model, assays, device,
            config["evaluation"].get("eval_batch_size", 32),
            config["training"].get("mixed_precision", "bf16"),
        )

        results_path = output_dir / f"per_assay_results_{args.property}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Mean Spearman: {results_df['spearman'].mean():.4f}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
