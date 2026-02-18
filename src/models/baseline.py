"""
Zero-shot ESM-2 fitness prediction via masked marginal likelihood.

Implementation of Meier et al. (2021) scoring method.
Reference: https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1
"""

import torch
import torch.nn.functional as F
from transformers import EsmForMaskedLM, EsmTokenizer
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def parse_mutations(mutant_str: str) -> List[Tuple[str, int, str]]:
    """
    Parse mutation string into list of (wt_aa, position, mut_aa).

    Format: "A1P:D2N" -> [("A", 0, "P"), ("D", 1, "N")]

    Args:
        mutant_str: Colon-separated mutation string

    Returns:
        mutations: List of (wt_aa, position_0indexed, mut_aa)
    """
    if pd.isna(mutant_str) or mutant_str == "":
        return []

    mutations = []
    for mut in mutant_str.split(":"):
        # Format: <WT><Position><MUT>
        # Example: "A125P" -> WT=A, Position=125, MUT=P
        wt_aa = mut[0]
        mut_aa = mut[-1]

        # Extract position (all digits in between)
        pos_str = ""
        for char in mut[1:-1]:
            if char.isdigit():
                pos_str += char

        if pos_str:
            position = int(pos_str) - 1  # Convert to 0-indexed
            mutations.append((wt_aa, position, mut_aa))

    return mutations


def compute_masked_marginal_score(
    model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    wildtype_seq: str,
    mutations: List[Tuple[str, int, str]],
    device: torch.device,
) -> float:
    """
    Compute masked marginal likelihood score for a single variant.

    For each mutation, mask that position in the wildtype sequence
    and compute log P(mut_aa | context) - log P(wt_aa | context).

    Args:
        model: ESM-2 for masked language modeling
        tokenizer: ESM tokenizer
        wildtype_seq: Wild-type protein sequence
        mutations: List of (wt_aa, position_0indexed, mut_aa)
        device: Device to run on

    Returns:
        score: Sum of log-likelihood ratios across all mutation sites
    """
    if not mutations:
        return 0.0

    # Tokenize wildtype (includes BOS at position 0)
    inputs = tokenizer(wildtype_seq, return_tensors="pt", padding=False)
    input_ids = inputs["input_ids"].to(device)  # [1, L+1]

    score = 0.0

    for wt_aa, pos, mut_aa in mutations:
        # Create masked version
        # Position in tokenized sequence is pos + 1 (due to BOS token at 0)
        masked_ids = input_ids.clone()
        token_pos = pos + 1

        # Check bounds
        if token_pos >= masked_ids.shape[1]:
            # Mutation position beyond sequence (should have been filtered)
            continue

        masked_ids[0, token_pos] = tokenizer.mask_token_id

        # Forward pass
        with torch.no_grad():
            outputs = model(masked_ids)
            logits = outputs.logits  # [1, L+1, vocab_size]

        # Get log probabilities at masked position
        log_probs = F.log_softmax(logits[0, token_pos], dim=-1)

        # Get token IDs for wildtype and mutant amino acids
        wt_token_id = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_token_id = tokenizer.convert_tokens_to_ids(mut_aa)

        # Compute score: log P(mut) - log P(wt)
        score += (log_probs[mut_token_id] - log_probs[wt_token_id]).item()

    return score


def compute_all_masked_marginals(
    model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    wildtype_seq: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Precompute masked marginal log probabilities for ALL positions.

    This optimization computes one forward pass per position (rather than
    per variant), then we can index into the result for each variant.

    Args:
        model: ESM-2 for masked language modeling
        tokenizer: ESM tokenizer
        wildtype_seq: Wild-type sequence
        device: Device

    Returns:
        all_log_probs: [seq_len+1, vocab_size] - log probs at each position
    """
    inputs = tokenizer(wildtype_seq, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    all_log_probs = torch.zeros(seq_len, tokenizer.vocab_size, device=device)

    # For each position (skip BOS at 0)
    for i in range(1, seq_len):
        masked = input_ids.clone()
        masked[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked)
            logits = outputs.logits

        all_log_probs[i] = F.log_softmax(logits[0, i], dim=-1)

    return all_log_probs


def score_variants_for_assay(
    model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    assay_df: pd.DataFrame,
    wildtype_seq: str,
    device: torch.device,
    use_precompute: bool = True,
) -> np.ndarray:
    """
    Score all variants in a DMS assay using masked marginal likelihood.

    Args:
        model: ESM-2 model
        tokenizer: ESM tokenizer
        assay_df: Assay DataFrame with 'mutant' column
        wildtype_seq: Wild-type sequence
        device: Device
        use_precompute: If True, precompute all position scores (faster for single-mutants)

    Returns:
        scores: Array of zero-shot fitness predictions
    """
    scores = []

    if use_precompute and len(wildtype_seq) < 1024:
        # Precompute all positions
        print(f"  Precomputing masked marginals for {len(wildtype_seq)} positions...")
        all_log_probs = compute_all_masked_marginals(model, tokenizer, wildtype_seq, device)

        # Score each variant by indexing
        for mutant_str in tqdm(assay_df["mutant"], desc="Scoring variants"):
            mutations = parse_mutations(mutant_str)

            if not mutations:
                scores.append(0.0)
                continue

            score = 0.0
            for wt_aa, pos, mut_aa in mutations:
                token_pos = pos + 1  # Adjust for BOS

                if token_pos >= all_log_probs.shape[0]:
                    continue

                wt_token_id = tokenizer.convert_tokens_to_ids(wt_aa)
                mut_token_id = tokenizer.convert_tokens_to_ids(mut_aa)

                score += (
                    all_log_probs[token_pos, mut_token_id].item()
                    - all_log_probs[token_pos, wt_token_id].item()
                )

            scores.append(score)

    else:
        # Score each variant independently (slower, but handles long sequences)
        for mutant_str in tqdm(assay_df["mutant"], desc="Scoring variants"):
            mutations = parse_mutations(mutant_str)
            score = compute_masked_marginal_score(
                model, tokenizer, wildtype_seq, mutations, device
            )
            scores.append(score)

    return np.array(scores)


def compute_zero_shot_for_assay(
    model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    assay_csv_path: Path,
    wildtype_seq: str,
    output_path: Path,
    device: torch.device,
):
    """
    Compute zero-shot scores for one assay and save results.

    Args:
        model: ESM-2 model
        tokenizer: ESM tokenizer
        assay_csv_path: Path to assay CSV
        wildtype_seq: Wild-type sequence
        output_path: Path to save results CSV
        device: Device
    """
    # Load assay
    assay_df = pd.read_csv(assay_csv_path)

    # Score variants
    zero_shot_scores = score_variants_for_assay(
        model, tokenizer, assay_df, wildtype_seq, device
    )

    # Add to dataframe
    result_df = assay_df.copy()
    result_df["zero_shot_score"] = zero_shot_scores

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"  Saved zero-shot scores to: {output_path}")

    # Compute correlation with DMS scores
    from scipy.stats import spearmanr
    rho = spearmanr(result_df["DMS_score"], result_df["zero_shot_score"]).statistic
    print(f"  Spearman correlation: {rho:.4f}")


def main():
    """CLI for computing zero-shot scores."""
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Compute ESM-2 zero-shot baseline scores")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--assay", type=str, help="Specific DMS_id to score (optional)")
    parser.add_argument("--smoke-test", action="store_true", help="Run on 2 assays only")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    model_name = config["model"]["base_model"]
    print(f"Loading {model_name}...")
    model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    # Load reference
    from src.data.download import load_reference_file
    reference_df = load_reference_file(config)

    # Determine which assays to score
    if args.assay:
        assays_to_score = [args.assay]
    elif args.smoke_test:
        assays_to_score = reference_df["DMS_id"].head(2).tolist()
    else:
        assays_to_score = reference_df["DMS_id"].tolist()

    raw_dir = Path(config["data"]["raw_dir"]) / "DMS_ProteinGym_substitutions"
    output_dir = Path(config["zero_shot"]["output_dir"])

    print(f"\nScoring {len(assays_to_score)} assays...")

    for dms_id in assays_to_score:
        row = reference_df[reference_df["DMS_id"] == dms_id].iloc[0]
        csv_path = raw_dir / row["DMS_filename"]
        wildtype_seq = row["target_seq"]

        print(f"\n{dms_id}:")
        output_path = output_dir / f"{dms_id}_zero_shot.csv"

        if output_path.exists():
            print(f"  Already scored, skipping")
            continue

        compute_zero_shot_for_assay(
            model, tokenizer, csv_path, wildtype_seq, output_path, device
        )

    print("\n✓ Zero-shot scoring complete!")


if __name__ == "__main__":
    main()
