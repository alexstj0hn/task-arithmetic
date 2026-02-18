"""
Ranking losses for protein fitness prediction.

Primary loss: ListMLE (Xia et al., 2008)
Alternatives for ablation: MSE regression, pairwise hinge
"""

import torch
import torch.nn as nn
from typing import Optional


class ListMLELoss(nn.Module):
    """
    ListMLE (List-wise Maximum Likelihood Estimation) ranking loss.

    Based on the Plackett-Luce model. Computes the negative log-likelihood
    of the ground-truth permutation given predicted scores.

    Mathematical formula:
        L = -∑ᵢ [s_πᵢ - log(∑ⱼ≥ᵢ exp(s_πⱼ))]

    where π is the permutation that sorts ground truth scores in descending order.

    Reference:
        Xia et al. (2008). "Listwise Approach to Learning to Rank - Theory and Algorithm"
        ICML 2008. https://icml.cc/Conferences/2008/papers/167.pdf
    """

    def __init__(self, eps: float = 1e-10):
        """
        Args:
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ListMLE loss.

        Args:
            y_pred: Predicted scores [batch_size, list_size]
            y_true: Ground truth relevance labels [batch_size, list_size]
            mask: Optional mask for padded positions [batch_size, list_size]
                  1 = valid, 0 = padding. If None, all positions are valid.

        Returns:
            loss: Scalar loss value (mean over batch)
        """
        # Sort by ground truth in descending order
        _, indices = y_true.sort(descending=True, dim=-1)

        # Rearrange predictions according to ground-truth ranking
        y_pred_sorted = y_pred.gather(1, indices)

        if mask is not None:
            mask_sorted = mask.gather(1, indices)
        else:
            mask_sorted = torch.ones_like(y_pred_sorted)

        # Numerically stable computation
        # For each position i, compute: pred_i - log(sum_{j>=i} exp(pred_j))

        # Subtract max for numerical stability (prevents overflow in exp)
        max_pred = y_pred_sorted.max(dim=-1, keepdim=True).values
        y_pred_stable = y_pred_sorted - max_pred

        # Compute cumulative sum from the end: sum_{j>=i} exp(pred_j - max)
        # This gives the normalization constant for each position
        exp_pred = torch.exp(y_pred_stable) * mask_sorted
        cumsums = torch.cumsum(exp_pred.flip(dims=[-1]), dim=-1).flip(dims=[-1])

        # Loss per position: log(cumsum) - (pred - max)
        # After cancellation, this equals: log(sum_{j>=i} exp(pred_j)) - pred_i
        observation_loss = torch.log(cumsums + self.eps) - y_pred_stable

        # Apply mask and aggregate
        observation_loss = observation_loss * mask_sorted

        # Sum over list positions, mean over batch
        # Normalize by number of valid items per list
        per_list_loss = observation_loss.sum(dim=-1) / mask_sorted.sum(dim=-1).clamp(min=1)
        return per_list_loss.mean()


class MSERegressionLoss(nn.Module):
    """
    Simple mean squared error regression loss.

    Used as a baseline for ablation comparisons.
    Treats fitness prediction as a regression problem rather than ranking.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted scores [batch_size, list_size] or [batch_size]
            y_true: Ground truth scores [batch_size, list_size] or [batch_size]
            mask: Optional mask (ignored for simplicity in MSE)

        Returns:
            loss: MSE loss
        """
        if mask is not None:
            # Masked MSE
            valid = mask.bool()
            return self.mse(y_pred[valid], y_true[valid])
        else:
            return self.mse(y_pred, y_true)


class PairwiseHingeLoss(nn.Module):
    """
    Pairwise margin-based ranking loss.

    For all pairs (i, j) where y_true[i] > y_true[j]:
        loss += max(0, margin - (y_pred[i] - y_pred[j]))

    Used as an alternative for ablation studies.
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Minimum required score difference between pairs
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted scores [batch_size, list_size]
            y_true: Ground truth scores [batch_size, list_size]
            mask: Optional mask for valid pairs

        Returns:
            loss: Pairwise hinge loss
        """
        batch_size, list_size = y_pred.shape

        # Expand to all pairs
        # pred_i: [batch, list_size, 1]
        # pred_j: [batch, 1, list_size]
        pred_i = y_pred.unsqueeze(2)
        pred_j = y_pred.unsqueeze(1)

        true_i = y_true.unsqueeze(2)
        true_j = y_true.unsqueeze(1)

        # Pairwise differences
        pred_diff = pred_i - pred_j  # [batch, list_size, list_size]
        true_diff = true_i - true_j  # [batch, list_size, list_size]

        # Loss for pairs where true_i > true_j (should have pred_i > pred_j)
        # Only penalize violations: pred_i - pred_j < margin
        violations = (true_diff > 0).float()
        hinge_loss = torch.clamp(self.margin - pred_diff, min=0)

        # Apply violation mask and aggregate
        total_loss = (hinge_loss * violations).sum(dim=[1, 2])
        num_pairs = violations.sum(dim=[1, 2]).clamp(min=1)

        # Average over pairs and batch
        per_sample_loss = total_loss / num_pairs
        return per_sample_loss.mean()


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create loss function by name.

    Args:
        loss_type: One of ["listmle", "mse", "pairwise_hinge"]
        **kwargs: Additional arguments passed to loss constructor

    Returns:
        loss_fn: Loss function module

    Example:
        >>> loss_fn = get_loss_function("listmle", eps=1e-10)
        >>> loss_fn = get_loss_function("pairwise_hinge", margin=1.0)
    """
    loss_map = {
        "listmle": ListMLELoss,
        "mse": MSERegressionLoss,
        "pairwise_hinge": PairwiseHingeLoss,
    }

    if loss_type not in loss_map:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Choose from {list(loss_map.keys())}"
        )

    return loss_map[loss_type](**kwargs)
