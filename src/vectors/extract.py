"""
Task vector extraction and composition.

For LoRA-adapted models, the task vector IS the LoRA weights themselves
(the delta from the base model). This module handles extraction, saving,
and arithmetic composition of task vectors for Phase 2.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def extract_task_vector(
    checkpoint_dir: Path,
    output_path: Path,
) -> Dict[str, torch.Tensor]:
    """
    Extract task vector from a LoRA checkpoint.

    For LoRA, the task vector is the set of (A, B) matrices across all layers.
    The effective weight change is: delta_W = B @ A (scaled by alpha/r).

    Args:
        checkpoint_dir: Directory containing LoRA adapter checkpoint
        output_path: Path to save task vector .pt file

    Returns:
        task_vector: Dict mapping parameter names to tensors
    """
    print(f"Extracting task vector from: {checkpoint_dir}")

    # Load LoRA adapter weights
    # The checkpoint directory should contain adapter_model.bin or adapter_model.safetensors
    adapter_file = checkpoint_dir / "adapter_model.bin"
    if not adapter_file.exists():
        adapter_file = checkpoint_dir / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(
                f"No adapter weights found in {checkpoint_dir}\n"
                f"Expected adapter_model.bin or adapter_model.safetensors"
            )

    # Load state dict
    if adapter_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        adapter_state = load_file(adapter_file)
    else:
        adapter_state = torch.load(adapter_file, map_location="cpu")

    # Filter to LoRA parameters only (lora_A, lora_B)
    task_vector = {
        k: v.clone()
        for k, v in adapter_state.items()
        if "lora_" in k
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(task_vector, output_path)

    # Print statistics
    total_params = sum(p.numel() for p in task_vector.values())
    num_layers = len([k for k in task_vector.keys() if "lora_A" in k])

    print(f"Task vector extracted:")
    print(f"  Parameters: {total_params:,}")
    print(f"  LoRA layers: {num_layers}")
    print(f"  Saved to: {output_path}")

    return task_vector


def load_task_vector(task_vector_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load a saved task vector.

    Args:
        task_vector_path: Path to .pt file

    Returns:
        task_vector: Dict of LoRA parameters
    """
    task_vector = torch.load(task_vector_path, map_location="cpu")
    print(f"Loaded task vector from: {task_vector_path}")
    return task_vector


def compose_task_vectors(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    scaling_coefficients: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    Compose multiple task vectors via weighted addition.

    For LoRA task vectors, composition works at the level of weight deltas.
    We compute the effective delta for each layer:
        delta_W = sum_i (alpha_i * (B_i @ A_i))

    Args:
        task_vectors: Dict mapping property_name -> task_vector_dict
        scaling_coefficients: Dict mapping property_name -> scalar weight

    Returns:
        composed_delta: Dict mapping layer_prefix -> composed weight delta

    Example:
        >>> stability_vec = load_task_vector("stability_task_vector.pt")
        >>> binding_vec = load_task_vector("binding_task_vector.pt")
        >>> composed = compose_task_vectors(
        ...     {"stability": stability_vec, "binding": binding_vec},
        ...     {"stability": 0.5, "binding": 0.5}
        ... )
    """
    # Identify all unique layer/module prefixes
    # LoRA params are named like:
    #   base_model.model.esm.encoder.layer.0.attention.self.query.lora_A.default.weight
    #   base_model.model.esm.encoder.layer.0.attention.self.query.lora_B.default.weight

    param_prefixes = set()
    first_vector = list(task_vectors.values())[0]
    for k in first_vector.keys():
        if "lora_A" in k:
            prefix = k.replace(".lora_A.default.weight", "")
            param_prefixes.add(prefix)

    composed = {}

    for prefix in param_prefixes:
        a_key = f"{prefix}.lora_A.default.weight"
        b_key = f"{prefix}.lora_B.default.weight"

        # Compute composed delta: sum_i alpha_i * (B_i @ A_i)
        composed_delta = None

        for prop_name, tv in task_vectors.items():
            alpha = scaling_coefficients[prop_name]

            if a_key not in tv or b_key not in tv:
                print(f"Warning: {prefix} not found in {prop_name} task vector")
                continue

            A = tv[a_key]  # [r, in_dim]
            B = tv[b_key]  # [out_dim, r]

            # Compute effective weight delta
            delta = alpha * (B @ A)  # [out_dim, in_dim]

            if composed_delta is None:
                composed_delta = delta
            else:
                composed_delta = composed_delta + delta

        if composed_delta is not None:
            composed[prefix] = composed_delta

    print(f"Composed {len(composed)} layer deltas from {len(task_vectors)} task vectors")
    for prop, alpha in scaling_coefficients.items():
        print(f"  {prop}: {alpha}")

    return composed


def apply_task_vector_to_model(
    model,
    task_vector: Dict[str, torch.Tensor],
    scaling: float = 1.0,
):
    """
    Apply a task vector to a base model by adding weight deltas.

    This modifies the model in-place.

    For LoRA task vectors, we compute delta_W = B @ A for each layer
    and add it to the corresponding base model weight.

    Args:
        model: Base ESM model (without LoRA)
        task_vector: Task vector from extract_task_vector()
        scaling: Optional scaling factor to apply to the task vector

    Example:
        >>> from transformers import EsmModel
        >>> base_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        >>> task_vec = load_task_vector("stability_task_vector.pt")
        >>> apply_task_vector_to_model(base_model, task_vec, scaling=0.5)
    """
    # Group LoRA A and B matrices by prefix
    lora_pairs = {}
    for k, v in task_vector.items():
        if "lora_A" in k:
            prefix = k.replace(".lora_A.default.weight", "")
            if prefix not in lora_pairs:
                lora_pairs[prefix] = {}
            lora_pairs[prefix]["A"] = v
        elif "lora_B" in k:
            prefix = k.replace(".lora_B.default.weight", "")
            if prefix not in lora_pairs:
                lora_pairs[prefix] = {}
            lora_pairs[prefix]["B"] = v

    # Apply deltas
    with torch.no_grad():
        for prefix, matrices in lora_pairs.items():
            if "A" not in matrices or "B" not in matrices:
                print(f"Warning: Incomplete LoRA pair for {prefix}, skipping")
                continue

            A = matrices["A"]  # [r, in_dim]
            B = matrices["B"]  # [out_dim, r]

            # Compute delta
            delta_W = scaling * (B @ A)  # [out_dim, in_dim]

            # Find corresponding parameter in base model
            # Convert prefix from LoRA naming to base model naming
            # "base_model.model.esm.encoder.layer.X.attention.self.query"
            # -> "esm.encoder.layer.X.attention.self.query.weight"

            param_path = prefix.replace("base_model.model.", "") + ".weight"

            # Navigate to parameter
            try:
                param_dict = dict(model.named_parameters())
                if param_path in param_dict:
                    param = param_dict[param_path]
                    param.add_(delta_W.to(param.device))
                else:
                    print(f"Warning: Parameter {param_path} not found in base model")
            except Exception as e:
                print(f"Error applying delta to {param_path}: {e}")

    print(f"Applied task vector to model (scaling={scaling})")


def compute_materialized_deltas(
    task_vector: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute B @ A for each LoRA layer pair to get effective weight deltas.

    Returns:
        deltas: Dict mapping layer_prefix -> weight delta [out_dim, in_dim]
    """
    lora_pairs = {}
    for k, v in task_vector.items():
        if "lora_A" in k:
            prefix = k.replace(".lora_A.default.weight", "")
            if prefix not in lora_pairs:
                lora_pairs[prefix] = {}
            lora_pairs[prefix]["A"] = v
        elif "lora_B" in k:
            prefix = k.replace(".lora_B.default.weight", "")
            if prefix not in lora_pairs:
                lora_pairs[prefix] = {}
            lora_pairs[prefix]["B"] = v

    deltas = {}
    for prefix, matrices in lora_pairs.items():
        if "A" in matrices and "B" in matrices:
            A = matrices["A"]  # [r, in_dim]
            B = matrices["B"]  # [out_dim, r]
            deltas[prefix] = B @ A  # [out_dim, in_dim]

    return deltas


def compute_pairwise_cosine_similarity(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity between flattened task vectors.

    Each property's task vector is flattened into a single 1D vector by
    concatenating all LoRA parameters, then cosine similarity is computed
    for all pairs.

    Returns:
        similarity_df: N x N DataFrame with cosine similarities
    """
    # Flatten each task vector
    flat_vectors = {}
    for prop_name, tv in task_vectors.items():
        params = []
        for k in sorted(tv.keys()):
            params.append(tv[k].flatten())
        flat_vectors[prop_name] = torch.cat(params)

    # Compute pairwise cosine similarity
    prop_names = list(flat_vectors.keys())
    n = len(prop_names)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            v_i = flat_vectors[prop_names[i]]
            v_j = flat_vectors[prop_names[j]]
            cos_sim = torch.nn.functional.cosine_similarity(
                v_i.unsqueeze(0), v_j.unsqueeze(0)
            ).item()
            sim_matrix[i, j] = cos_sim

    return pd.DataFrame(sim_matrix, index=prop_names, columns=prop_names)


def generate_summary_json(
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    output_path: Path,
) -> Dict:
    """
    Generate summary statistics for extracted task vectors.

    Includes param count, L2 norm, per-layer norms, pairwise cosine sim.
    """
    summary = {"properties": {}}

    for prop_name, tv in task_vectors.items():
        total_params = sum(p.numel() for p in tv.values())
        all_params = torch.cat([p.flatten() for p in tv.values()])
        l2_norm = float(torch.norm(all_params, p=2))
        num_lora_layers = len([k for k in tv.keys() if "lora_A" in k])

        # Compute materialized deltas and their norms
        deltas = compute_materialized_deltas(tv)
        delta_norms = {
            prefix: float(torch.norm(delta, p="fro"))
            for prefix, delta in deltas.items()
        }

        summary["properties"][prop_name] = {
            "total_parameters": total_params,
            "l2_norm": l2_norm,
            "num_lora_layers": num_lora_layers,
            "delta_frobenius_norms": delta_norms,
        }

    # Pairwise cosine similarity
    sim_df = compute_pairwise_cosine_similarity(task_vectors)
    summary["pairwise_cosine_similarity"] = sim_df.to_dict()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved task vector summary to {output_path}")
    return summary


PROPERTIES = ["stability", "binding", "expression", "activity"]


def main():
    """CLI for extracting task vectors."""
    import argparse

    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Extract task vectors from checkpoints")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--property", type=str, choices=PROPERTIES,
                        help="Extract for a specific property only")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    checkpoint_dir = Path(config["checkpointing"]["checkpoint_dir"])
    output_dir = Path(config["task_vectors"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    properties = [args.property] if args.property else PROPERTIES
    all_task_vectors = {}

    for prop in properties:
        prop_checkpoint_dir = checkpoint_dir / prop

        if not prop_checkpoint_dir.exists():
            logger.warning(f"No checkpoint found for {prop}, skipping")
            continue

        # Look for best_model first, then latest step checkpoint
        best_model_dir = prop_checkpoint_dir / "best_model" / "lora_adapter"
        if not best_model_dir.exists():
            best_model_dir = prop_checkpoint_dir / "best_model"
        if not best_model_dir.exists():
            from src.training.utils import find_latest_checkpoint

            latest = find_latest_checkpoint(prop_checkpoint_dir)
            if latest is None:
                logger.warning(f"No checkpoints in {prop_checkpoint_dir}, skipping")
                continue
            best_model_dir = latest / "lora_adapter"

        output_path = output_dir / f"{prop}_task_vector.pt"
        tv = extract_task_vector(best_model_dir, output_path)
        all_task_vectors[prop] = tv

    # Save materialized deltas for each property
    for prop, tv in all_task_vectors.items():
        deltas = compute_materialized_deltas(tv)
        deltas_path = output_dir / f"{prop}_materialized_deltas.pt"
        torch.save(deltas, deltas_path)
        logger.info(f"Saved materialized deltas for {prop} to {deltas_path}")

    # Compute pairwise cosine similarity if we have multiple properties
    if len(all_task_vectors) > 1:
        sim_df = compute_pairwise_cosine_similarity(all_task_vectors)
        sim_path = output_dir / "cosine_similarity_matrix.csv"
        sim_df.to_csv(sim_path)
        logger.info(f"Saved cosine similarity matrix to {sim_path}")
        logger.info(f"\n{sim_df.to_string()}")

    # Generate summary JSON
    if all_task_vectors:
        summary = generate_summary_json(
            all_task_vectors,
            output_dir / "task_vector_summary.json",
        )

    logger.info("Task vector extraction complete!")


if __name__ == "__main__":
    main()
