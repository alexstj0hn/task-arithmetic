"""
PyTorch Dataset classes for protein fitness prediction.

Supports:
- Loading individual DMS assays
- Aggregating assays by property category
- Sampling ranking lists for ListMLE training
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize DMS scores.

    Args:
        scores: Array of fitness scores
        method: One of ["minmax", "zscore", "rank"]

    Returns:
        normalized_scores: Normalized to comparable scale
    """
    if method == "minmax":
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-8:  # Degenerate case: all scores identical
            return np.full_like(scores, 0.5)
        return (scores - s_min) / (s_max - s_min)

    elif method == "zscore":
        mean, std = scores.mean(), scores.std()
        if std < 1e-8:
            return np.zeros_like(scores)
        return (scores - mean) / std

    elif method == "rank":
        from scipy.stats import rankdata
        return rankdata(scores) / len(scores)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


class ProteinGymAssay(Dataset):
    """
    Single DMS assay dataset.

    Loads one assay CSV with variants and their fitness scores.
    Handles tokenization, score normalization, and sequence truncation.
    """

    def __init__(
        self,
        csv_path: Path,
        dms_id: str,
        category: str,
        tokenizer,
        max_length: int = 1022,
        normalize: str = "minmax",
    ):
        """
        Args:
            csv_path: Path to assay CSV file
            dms_id: Unique assay identifier
            category: Property category (stability, binding, expression, activity)
            tokenizer: ESM tokenizer
            max_length: Maximum sequence length (ESM-2 max is 1022)
            normalize: Score normalization method
        """
        self.csv_path = csv_path
        self.dms_id = dms_id
        self.category = category
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load CSV
        df = pd.read_csv(csv_path)

        # Extract sequences and scores
        self.sequences = df["mutated_sequence"].values
        self.scores = df["DMS_score"].values
        self.mutants = df["mutant"].values

        # Get wildtype
        if "target_seq" in df.columns:
            self.wildtype = df["target_seq"].iloc[0]
        else:
            self.wildtype = None

        # Normalize scores
        self.scores_normalized = normalize_scores(self.scores, method=normalize)

        # Binary labels if available
        if "DMS_score_bin" in df.columns:
            self.binary_labels = df["DMS_score_bin"].values
        else:
            self.binary_labels = None

        # Handle long sequences
        self.truncated_indices = []
        self.excluded_indices = []

        for i, seq in enumerate(self.sequences):
            if len(seq) > max_length:
                # Check if mutation is beyond max_length
                # Parse mutant string to get positions
                mutation_positions = self._parse_mutation_positions(self.mutants[i])

                if mutation_positions and max(mutation_positions) >= max_length:
                    # Mutation beyond truncation point - exclude
                    self.excluded_indices.append(i)
                else:
                    # Mutation within bounds - truncate sequence
                    self.truncated_indices.append(i)
                    self.sequences[i] = seq[:max_length]

        # Remove excluded variants
        if self.excluded_indices:
            print(f"  [{dms_id}] Excluded {len(self.excluded_indices)} variants (mutations beyond position {max_length})")
            valid_mask = np.ones(len(self.sequences), dtype=bool)
            valid_mask[self.excluded_indices] = False

            self.sequences = self.sequences[valid_mask]
            self.scores = self.scores[valid_mask]
            self.scores_normalized = self.scores_normalized[valid_mask]
            self.mutants = self.mutants[valid_mask]
            if self.binary_labels is not None:
                self.binary_labels = self.binary_labels[valid_mask]

        if self.truncated_indices:
            print(f"  [{dms_id}] Truncated {len(self.truncated_indices)} variants to {max_length} residues")

    def _parse_mutation_positions(self, mutant_str: str) -> List[int]:
        """
        Parse mutation positions from mutant string.

        Format: "A1P:D2N" -> positions [1, 2]

        Returns:
            List of 0-indexed mutation positions
        """
        if pd.isna(mutant_str) or mutant_str == "":
            return []

        positions = []
        for mut in mutant_str.split(":"):
            # Extract position (number in the middle)
            # Format: <WT_AA><Position><MUT_AA>
            pos_str = ""
            for char in mut[1:]:  # Skip first char (WT amino acid)
                if char.isdigit():
                    pos_str += char
                else:
                    break
            if pos_str:
                positions.append(int(pos_str) - 1)  # Convert to 0-indexed

        return positions

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - sequence: str
                - score: float (raw DMS score)
                - score_normalized: float (normalized [0,1])
                - mutant: str
                - dms_id: str
                - category: str
        """
        return {
            "sequence": self.sequences[idx],
            "score": self.scores[idx],
            "score_normalized": self.scores_normalized[idx],
            "mutant": self.mutants[idx],
            "dms_id": self.dms_id,
            "category": self.category,
        }


class PropertyCategoryDataset:
    """
    Aggregates all assays in one property category.

    Provides sampling of ranking lists for ListMLE training.
    """

    def __init__(
        self,
        assays: List[ProteinGymAssay],
        list_size: int = 32,
    ):
        """
        Args:
            assays: List of ProteinGymAssay instances
            list_size: Number of variants per ranking list
        """
        self.assays = assays
        self.list_size = list_size

        print(f"PropertyCategoryDataset with {len(assays)} assays")
        total_variants = sum(len(a) for a in assays)
        print(f"  Total variants: {total_variants}")

    def sample_ranking_list(self, tokenizer, device: torch.device) -> Dict:
        """
        Sample a ranking list from one assay.

        Strategy:
            1. Randomly select an assay
            2. Sample list_size variants from that assay
            3. Tokenize sequences
            4. Return batch ready for model

        Returns:
            batch: Dict with keys:
                - input_ids: [list_size, seq_len]
                - attention_mask: [list_size, seq_len]
                - labels: [list_size] - normalized scores
                - dms_id: str
        """
        # Randomly select an assay
        assay = np.random.choice(self.assays)

        # Sample variants
        n = len(assay)
        if n <= self.list_size:
            # Use all variants if assay is small
            indices = list(range(n))
        else:
            # Random sample
            indices = np.random.choice(n, size=self.list_size, replace=False)

        # Get data
        sequences = [assay.sequences[i] for i in indices]
        labels = torch.tensor([assay.scores_normalized[i] for i in indices], dtype=torch.float32)

        # Tokenize
        encoded = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=assay.max_length + 2,  # +2 for BOS/EOS
        )

        return {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
            "labels": labels.to(device),
            "dms_id": assay.dms_id,
        }


def load_assays_for_category(
    category: str,
    dms_ids: List[str],
    reference_df: pd.DataFrame,
    raw_data_dir: Path,
    tokenizer,
    max_length: int = 1022,
    normalize: str = "minmax",
) -> List[ProteinGymAssay]:
    """
    Load all assays for a property category.

    Args:
        category: Property category name
        dms_ids: List of DMS_ids in this category
        reference_df: ProteinGym reference DataFrame
        raw_data_dir: Directory containing assay CSV files
        tokenizer: ESM tokenizer
        max_length: Maximum sequence length
        normalize: Score normalization method

    Returns:
        assays: List of ProteinGymAssay instances
    """
    assays = []

    for dms_id in dms_ids:
        # Get filename from reference
        row = reference_df[reference_df["DMS_id"] == dms_id]
        if row.empty:
            print(f"Warning: {dms_id} not found in reference, skipping")
            continue

        filename = row.iloc[0]["DMS_filename"]
        csv_path = raw_data_dir / filename

        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {dms_id}")
            continue

        # Load assay
        assay = ProteinGymAssay(
            csv_path=csv_path,
            dms_id=dms_id,
            category=category,
            tokenizer=tokenizer,
            max_length=max_length,
            normalize=normalize,
        )

        assays.append(assay)

    print(f"\nLoaded {len(assays)} assays for {category}")

    return assays


def main():
    """Test dataset loading."""
    import yaml
    from transformers import EsmTokenizer

    # Load config
    with open("configs/train_config.yaml") as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tokenizer = EsmTokenizer.from_pretrained(config["model"]["base_model"])

    # Load one assay as a test
    from .download import load_reference_file
    from .categorize import load_categorization

    reference_df = load_reference_file(config)
    categorization = load_categorization(
        Path(config["data"]["processed_dir"]) / "category_assignments.json"
    )

    # Load stability assays
    stability_dms_ids = categorization["stability"][:5]  # First 5 for testing
    raw_data_dir = Path(config["data"]["raw_dir"]) / "DMS_ProteinGym_substitutions"

    assays = load_assays_for_category(
        category="stability",
        dms_ids=stability_dms_ids,
        reference_df=reference_df,
        raw_data_dir=raw_data_dir,
        tokenizer=tokenizer,
    )

    # Create category dataset
    category_dataset = PropertyCategoryDataset(assays, list_size=16)

    # Sample a ranking list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = category_dataset.sample_ranking_list(tokenizer, device)

    print(f"\nSampled ranking list:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  DMS_id: {batch['dms_id']}")


if __name__ == "__main__":
    main()
