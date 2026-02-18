"""
ESM-2 + LoRA model for protein fitness ranking.

Architecture:
    Protein sequence → ESM-2 (frozen) + LoRA (trainable) → Mean pool → RankingHead → Scalar score
"""

import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from peft import LoraConfig, get_peft_model
from typing import Optional, Dict


class RankingHead(nn.Module):
    """
    Regression/ranking head on top of pooled ESM-2 representations.

    Maps pooled embeddings [batch, hidden_dim] → scalar fitness scores [batch]
    """

    def __init__(self, input_dim: int = 1280, hidden_dim: int = 640, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [batch_size, hidden_dim]

        Returns:
            scores: [batch_size] - predicted fitness scores
        """
        x = self.dense(pooled_output)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.output_proj(x)  # [batch_size, 1]
        return x.squeeze(-1)     # [batch_size]


class ESMLoRAForRanking(nn.Module):
    """
    ESM-2 with LoRA adapters for property-specific fitness ranking.

    The base ESM-2 model is frozen except for LoRA adapters applied to
    query, key, value projection layers in all attention heads.

    Args:
        config: Dict with keys:
            - model: {base_model, hidden_size}
            - lora: {r, lora_alpha, lora_dropout, target_modules, bias}
            - ranking_head: {input_dim, hidden_dim, dropout, pooling}
    """

    def __init__(self, config: Dict):
        super().__init__()

        # Load base ESM-2 model
        model_name = config["model"]["base_model"]
        print(f"Loading ESM-2 model: {model_name}")
        self.esm = EsmModel.from_pretrained(model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)

        # Apply LoRA
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            target_modules=config["lora"]["target_modules"],
            bias=config["lora"]["bias"],
        )
        self.esm = get_peft_model(self.esm, lora_config)

        print(f"LoRA configuration:")
        print(f"  rank: {lora_config.r}")
        print(f"  alpha: {lora_config.lora_alpha}")
        print(f"  target_modules: {lora_config.target_modules}")

        # Ranking head (trained from scratch)
        self.ranking_head = RankingHead(
            input_dim=config["ranking_head"]["input_dim"],
            hidden_dim=config["ranking_head"]["hidden_dim"],
            dropout=config["ranking_head"]["dropout"],
        )

        self.pooling = config["ranking_head"]["pooling"]
        assert self.pooling in ["mean", "cls"], f"Pooling must be 'mean' or 'cls', got {self.pooling}"

    def mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pool over non-padding, non-special-token positions.

        ESM-2 tokenizer adds BOS (cls token) at position 0.
        Padding token is 1. We mask out both for mean pooling.

        Args:
            last_hidden_state: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding

        Returns:
            pooled: [batch, hidden_dim]
        """
        # Create mask: 1 for real amino acid positions, 0 for BOS and PAD
        # attention_mask already handles PAD; additionally mask position 0 (BOS)
        pool_mask = attention_mask.clone()
        pool_mask[:, 0] = 0  # Mask BOS token at position 0

        # Expand for broadcasting: [batch, seq_len, 1]
        mask_expanded = pool_mask.unsqueeze(-1).float()

        # Sum and normalize
        summed = (last_hidden_state * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-8)
        return summed / counts  # [batch, hidden_dim]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through ESM-2 + LoRA + ranking head.

        Args:
            input_ids: [batch_size, seq_len] - tokenized protein sequences
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding

        Returns:
            scores: [batch_size] - predicted fitness scores
        """
        # ESM-2 forward pass
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, 1280]

        # Pooling
        if self.pooling == "mean":
            pooled = self.mean_pool(last_hidden, attention_mask)
        else:  # cls
            pooled = last_hidden[:, 0, :]  # BOS token at position 0

        # Ranking head
        scores = self.ranking_head(pooled)  # [batch]
        return scores

    def get_trainable_params_count(self) -> tuple[int, int]:
        """
        Returns:
            (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def print_trainable_params(self):
        """Print trainable parameter counts."""
        total, trainable = self.get_trainable_params_count()
        percentage = 100 * trainable / total if total > 0 else 0
        print(f"Trainable parameters: {trainable:,} / {total:,} ({percentage:.2f}%)")

        # LoRA-specific stats
        self.esm.print_trainable_parameters()

    def save_lora_adapter(self, save_path: str):
        """Save only the LoRA adapter weights."""
        self.esm.save_pretrained(save_path)
        print(f"Saved LoRA adapter to {save_path}")

    def load_lora_adapter(self, load_path: str):
        """Load LoRA adapter weights from checkpoint."""
        from peft import PeftModel
        self.esm = PeftModel.from_pretrained(self.esm.get_base_model(), load_path)
        print(f"Loaded LoRA adapter from {load_path}")


def create_model(config: Dict, device: Optional[torch.device] = None) -> ESMLoRAForRanking:
    """
    Factory function to create ESM-2 + LoRA model.

    Args:
        config: Configuration dictionary
        device: Device to load model on (default: cuda if available)

    Returns:
        model: ESMLoRAForRanking instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ESMLoRAForRanking(config)
    model = model.to(device)

    print(f"\nModel loaded on device: {device}")
    model.print_trainable_params()

    return model
