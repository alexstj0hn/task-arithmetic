"""
Checkpoint management and training utilities.

Handles saving/loading full training state for preemption recovery,
GCS synchronization, and status tracking.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    metric_name: str,
) -> Path:
    """
    Save a full training checkpoint with atomic write for preemption safety.

    Directory layout:
      checkpoint_dir/step_{global_step}/
        lora_adapter/           # PEFT adapter weights
        ranking_head.pt         # RankingHead state_dict
        optimizer.pt            # Optimizer state_dict
        scheduler.pt            # Scheduler state_dict
        trainer_state.json      # epoch, step, best_metric

    Returns:
        save_path: Path to the created checkpoint directory
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_path = checkpoint_dir / f"step_{global_step}"
    tmp_path = checkpoint_dir / f"_tmp_step_{global_step}"

    # Write to temp directory first, then rename atomically
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    try:
        # Save LoRA adapter via PEFT
        adapter_path = tmp_path / "lora_adapter"
        model.esm.save_pretrained(str(adapter_path))

        # Save ranking head separately (not part of PEFT)
        torch.save(
            model.ranking_head.state_dict(),
            tmp_path / "ranking_head.pt",
        )

        # Save optimizer state
        torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")

        # Save scheduler state
        torch.save(scheduler.state_dict(), tmp_path / "scheduler.pt")

        # Save trainer metadata
        trainer_state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "metric_name": metric_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(tmp_path / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=2)

        # Atomic rename
        if final_path.exists():
            shutil.rmtree(final_path)
        tmp_path.rename(final_path)

        logger.info(f"Saved checkpoint to {final_path}")
        return final_path

    except Exception:
        # Clean up failed write
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        raise


def save_best_checkpoint(
    checkpoint_dir: Path,
    model,
    best_metric: float,
    metric_name: str,
) -> Path:
    """
    Save only model weights as the best checkpoint (no optimizer state needed).
    """
    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / "best_model"
    tmp_path = checkpoint_dir / "_tmp_best_model"

    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    try:
        # Save LoRA adapter
        model.esm.save_pretrained(str(tmp_path / "lora_adapter"))

        # Save ranking head
        torch.save(
            model.ranking_head.state_dict(),
            tmp_path / "ranking_head.pt",
        )

        # Save metadata
        with open(tmp_path / "best_info.json", "w") as f:
            json.dump({
                "metric_name": metric_name,
                "best_metric": best_metric,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

        if best_path.exists():
            shutil.rmtree(best_path)
        tmp_path.rename(best_path)

        logger.info(f"Saved best model ({metric_name}={best_metric:.4f}) to {best_path}")
        return best_path

    except Exception:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        raise


def load_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer=None,
    scheduler=None,
) -> Dict:
    """
    Load a full checkpoint and restore model/optimizer/scheduler state.

    Uses direct state_dict loading for the LoRA adapter (since the model
    is already a PeftModel — calling PeftModel.from_pretrained would fail).

    Returns:
        trainer_state: Dict with epoch, global_step, best_metric, etc.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load LoRA adapter weights directly into existing PeftModel
    adapter_dir = checkpoint_path / "lora_adapter"
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        adapter_file = adapter_dir / "adapter_model.bin"

    if adapter_file.exists():
        if adapter_file.suffix == ".safetensors":
            from safetensors.torch import load_file
            adapter_state = load_file(str(adapter_file))
        else:
            adapter_state = torch.load(adapter_file, map_location="cpu", weights_only=True)

        model.esm.load_state_dict(adapter_state, strict=False)
        logger.info(f"Loaded LoRA adapter from {adapter_dir}")
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    # Load ranking head
    head_path = checkpoint_path / "ranking_head.pt"
    if head_path.exists():
        model.ranking_head.load_state_dict(
            torch.load(head_path, map_location="cpu", weights_only=True)
        )
        logger.info("Loaded ranking head weights")

    # Load optimizer
    if optimizer is not None:
        opt_path = checkpoint_path / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(
                torch.load(opt_path, map_location="cpu", weights_only=True)
            )
            logger.info("Loaded optimizer state")

    # Load scheduler
    if scheduler is not None:
        sched_path = checkpoint_path / "scheduler.pt"
        if sched_path.exists():
            scheduler.load_state_dict(
                torch.load(sched_path, map_location="cpu", weights_only=True)
            )
            logger.info("Loaded scheduler state")

    # Load trainer state
    state_path = checkpoint_path / "trainer_state.json"
    if state_path.exists():
        with open(state_path) as f:
            trainer_state = json.load(f)
        logger.info(
            f"Resuming from epoch {trainer_state['epoch']}, "
            f"step {trainer_state['global_step']}"
        )
        return trainer_state

    return {"epoch": 0, "global_step": 0, "best_metric": -float("inf")}


def load_model_weights(
    checkpoint_path: Path,
    model,
) -> None:
    """
    Load only model weights from a checkpoint (no optimizer/scheduler).
    Used for evaluation and inference.
    """
    checkpoint_path = Path(checkpoint_path)

    # Load LoRA adapter
    adapter_dir = checkpoint_path / "lora_adapter"
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        adapter_file = adapter_dir / "adapter_model.bin"

    if adapter_file.exists():
        if adapter_file.suffix == ".safetensors":
            from safetensors.torch import load_file
            adapter_state = load_file(str(adapter_file))
        else:
            adapter_state = torch.load(adapter_file, map_location="cpu", weights_only=True)

        model.esm.load_state_dict(adapter_state, strict=False)
    else:
        raise FileNotFoundError(f"No adapter weights in {adapter_dir}")

    # Load ranking head
    head_path = checkpoint_path / "ranking_head.pt"
    if head_path.exists():
        model.ranking_head.load_state_dict(
            torch.load(head_path, map_location="cpu", weights_only=True)
        )

    logger.info(f"Loaded model weights from {checkpoint_path}")


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the most recent checkpoint by step number.

    Looks for directories named step_{N} and returns the one with highest N.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    step_dirs = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step_num = int(d.name.split("_")[1])
                step_dirs.append((step_num, d))
            except (ValueError, IndexError):
                continue

    if not step_dirs:
        return None

    step_dirs.sort(key=lambda x: x[0])
    latest = step_dirs[-1][1]
    logger.info(f"Found latest checkpoint: {latest}")
    return latest


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    keep_last_n: int = 3,
) -> None:
    """
    Remove old checkpoints, keeping the N most recent and best_model.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    step_dirs = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                step_num = int(d.name.split("_")[1])
                step_dirs.append((step_num, d))
            except (ValueError, IndexError):
                continue

    if len(step_dirs) <= keep_last_n:
        return

    step_dirs.sort(key=lambda x: x[0])
    to_remove = step_dirs[:-keep_last_n]

    for _, path in to_remove:
        logger.info(f"Removing old checkpoint: {path}")
        shutil.rmtree(path)


def sync_checkpoint_to_gcs(
    local_path: Path,
    gcs_bucket: str,
    property_name: str,
    timeout: int = 300,
) -> bool:
    """
    Sync a checkpoint directory to GCS using gsutil.

    Non-blocking on failure: logs warning and returns False.
    """
    gcs_dest = f"{gcs_bucket}/checkpoints/{property_name}/{local_path.name}/"

    cmd = [
        "gsutil", "-m", "rsync", "-r",
        str(local_path),
        gcs_dest,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.info(f"Synced checkpoint to {gcs_dest}")
            return True
        else:
            logger.warning(f"GCS sync failed (rc={result.returncode}): {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning(f"GCS sync timed out after {timeout}s")
        return False
    except FileNotFoundError:
        logger.warning("gsutil not found — skipping GCS sync")
        return False


def update_status_file(
    status_path: Path,
    current_step: str,
    property_name: Optional[str] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    train_loss: Optional[float] = None,
    val_metric: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Update results/status.json with current pipeline state.
    """
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    status = {
        "current_step": current_step,
        "property": property_name,
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_spearman": val_metric,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "errors": [error] if error else [],
    }

    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
