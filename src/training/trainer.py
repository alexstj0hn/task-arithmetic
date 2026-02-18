"""
Property-specific trainer for ESM-2 + LoRA fine-tuning.

Trains a single property model using ListMLE ranking loss on sampled
ranking lists from PropertyCategoryDataset. Supports:
- Resume from checkpoint (full optimizer + scheduler state)
- WandB logging
- Mixed precision (bf16 on A100)
- Gradient accumulation
- Early stopping on validation Spearman
- Periodic checkpoint saving with GCS sync hooks
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import get_cosine_schedule_with_warmup

from src.data.dataset import PropertyCategoryDataset, ProteinGymAssay
from src.models.esm_lora import ESMLoRAForRanking, create_model
from src.training.losses import get_loss_function
from src.training.utils import (
    cleanup_old_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
    sync_checkpoint_to_gcs,
    update_status_file,
)

logger = logging.getLogger(__name__)

PROPERTIES = ["stability", "binding", "expression", "activity"]


class PropertyTrainer:
    """
    Trainer for property-specific LoRA fine-tuning on protein fitness data.

    Uses PropertyCategoryDataset's sampling interface: each training step
    samples a ranking list of `list_size` variants from a random assay.
    An "epoch" is defined as `steps_per_epoch` sampling steps.
    """

    def __init__(
        self,
        config: Dict,
        property_name: str,
        train_dataset: PropertyCategoryDataset,
        val_assays: List[ProteinGymAssay],
        device: torch.device,
        resume: bool = False,
    ):
        self.config = config
        self.property_name = property_name
        self.train_dataset = train_dataset
        self.val_assays = val_assays
        self.device = device

        # Training hyperparameters from config
        self.num_epochs = config["training"]["num_epochs"]
        self.list_size = config["training"]["list_size"]
        self.gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
        self.max_grad_norm = config["training"]["max_grad_norm"]
        self.mixed_precision = config["training"]["mixed_precision"]
        self.early_stopping_patience = config["training"].get("early_stopping_patience", 5)

        # Logging intervals
        self.log_every = config["logging"]["log_every_n_steps"]
        self.eval_every = config["logging"]["eval_every_n_steps"]

        # Checkpointing
        self.checkpoint_dir = Path(config["checkpointing"]["checkpoint_dir"]) / property_name
        self.save_every_steps = config["checkpointing"]["save_every_n_steps"]
        self.save_every_epochs = config["checkpointing"]["save_every_n_epochs"]
        self.keep_last_n = config["checkpointing"]["keep_last_n"]
        self.sync_gcs = config["checkpointing"].get("sync_to_gcs", False)
        self.gcs_bucket = config.get("paths", {}).get("gcs_bucket", "")

        # Status tracking
        self.status_path = Path(config.get("status", {}).get(
            "status_file", "results/status.json"
        ))

        # Compute steps per epoch
        self.steps_per_epoch = config["training"].get("steps_per_epoch")
        if not self.steps_per_epoch:
            total_variants = sum(len(a) for a in train_dataset.assays)
            self.steps_per_epoch = max(1, total_variants // self.list_size)
        logger.info(f"Steps per epoch: {self.steps_per_epoch}")

        # Total optimizer steps for scheduler
        self.total_optimizer_steps = (
            (self.steps_per_epoch * self.num_epochs)
            // self.gradient_accumulation_steps
        )

        # Create model
        self.model = create_model(config, device)

        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.loss_fn = self._setup_loss()

        # Tracking state
        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = -float("inf")
        self.metric_name = config["checkpointing"].get("best_metric", "spearman_avg")
        self.epochs_without_improvement = 0

        # Resume from checkpoint if requested
        if resume:
            self._resume_from_checkpoint()

        # Initialize WandB
        self._init_wandb()

    def _setup_optimizer(self) -> torch.optim.AdamW:
        """Create AdamW optimizer for trainable parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        logger.info(
            f"Optimizer: AdamW (lr={self.config['training']['learning_rate']}, "
            f"wd={self.config['training']['weight_decay']})"
        )
        return optimizer

    def _setup_scheduler(self):
        """Create LR scheduler based on config."""
        sched_type = self.config["training"]["lr_scheduler"]
        warmup_ratio = self.config["training"]["warmup_ratio"]
        num_warmup = int(self.total_optimizer_steps * warmup_ratio)

        if sched_type in ("cosine_with_warmup", "cosine"):
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup,
                num_training_steps=self.total_optimizer_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")

        logger.info(
            f"Scheduler: {sched_type} (warmup={num_warmup}/{self.total_optimizer_steps} steps)"
        )
        return scheduler

    def _setup_loss(self):
        """Create loss function from config."""
        loss_type = self.config["training"]["loss"]
        loss_fn = get_loss_function(loss_type)
        logger.info(f"Loss function: {loss_type}")
        return loss_fn

    def _init_wandb(self):
        """Initialize WandB logging."""
        self.use_wandb = False
        wandb_project = self.config["logging"].get("wandb_project")
        if wandb_project:
            try:
                import wandb

                experiment_name = self.config.get("experiment", {}).get(
                    "name", "phase1"
                )
                wandb.init(
                    project=wandb_project,
                    entity=self.config["logging"].get("wandb_entity"),
                    name=f"{experiment_name}_{self.property_name}",
                    config=self.config,
                    resume="allow",
                )
                self.use_wandb = True
                logger.info(f"WandB initialized: {wandb_project}")
            except Exception as e:
                logger.warning(f"WandB init failed: {e}. Continuing without WandB.")

    def _resume_from_checkpoint(self):
        """Find and load the latest checkpoint if it exists."""
        latest = find_latest_checkpoint(self.checkpoint_dir)
        if latest is None:
            logger.info("No checkpoint found, starting from scratch")
            return

        trainer_state = load_checkpoint(
            latest, self.model, self.optimizer, self.scheduler
        )
        self.start_epoch = trainer_state.get("epoch", 0)
        self.global_step = trainer_state.get("global_step", 0)
        self.best_metric = trainer_state.get("best_metric", -float("inf"))
        logger.info(
            f"Resumed from epoch {self.start_epoch}, step {self.global_step}, "
            f"best {self.metric_name}={self.best_metric:.4f}"
        )

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            results: Dict with final metrics, best checkpoint path, etc.
        """
        logger.info(f"Starting training for property: {self.property_name}")
        logger.info(f"  Epochs: {self.num_epochs}, Steps/epoch: {self.steps_per_epoch}")
        logger.info(f"  Grad accum: {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimizer steps: {self.total_optimizer_steps}")

        seed = self.config.get("experiment", {}).get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        for epoch in range(self.start_epoch, self.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"{'='*60}")

            # Train one epoch
            avg_loss = self._train_one_epoch(epoch)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

            # Evaluate
            val_metrics = self._evaluate()
            val_spearman = val_metrics.get("spearman_avg", 0.0)
            logger.info(f"Epoch {epoch + 1} val Spearman: {val_spearman:.4f}")

            if self.use_wandb:
                import wandb

                wandb.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_loss,
                    "val/spearman_avg": val_spearman,
                    **{
                        f"val/spearman_{k}": v
                        for k, v in val_metrics.get("spearman_per_assay", {}).items()
                    },
                }, step=self.global_step)

            # Check if best model
            is_best = val_spearman > self.best_metric
            if is_best:
                self.best_metric = val_spearman
                self.epochs_without_improvement = 0
                save_best_checkpoint(
                    self.checkpoint_dir,
                    self.model,
                    self.best_metric,
                    self.metric_name,
                )
                if self.sync_gcs and self.gcs_bucket:
                    sync_checkpoint_to_gcs(
                        self.checkpoint_dir / "best_model",
                        self.gcs_bucket,
                        self.property_name,
                    )
            else:
                self.epochs_without_improvement += 1

            # Epoch checkpoint
            if self.save_every_epochs and (epoch + 1) % self.save_every_epochs == 0:
                ckpt_path = save_checkpoint(
                    self.checkpoint_dir,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    self.global_step,
                    self.best_metric,
                    self.metric_name,
                )
                cleanup_old_checkpoints(self.checkpoint_dir, self.keep_last_n)
                if self.sync_gcs and self.gcs_bucket:
                    sync_checkpoint_to_gcs(
                        ckpt_path, self.gcs_bucket, self.property_name
                    )

            # Update status
            update_status_file(
                self.status_path,
                current_step=f"training_{self.property_name}",
                property_name=self.property_name,
                epoch=epoch + 1,
                step=self.global_step,
                train_loss=avg_loss,
                val_metric=val_spearman,
            )

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping: no improvement for "
                    f"{self.early_stopping_patience} epochs"
                )
                break

        # Finalize
        if self.use_wandb:
            import wandb

            wandb.finish()

        results = {
            "property": self.property_name,
            "best_metric": self.best_metric,
            "metric_name": self.metric_name,
            "final_epoch": epoch + 1,
            "final_step": self.global_step,
            "checkpoint_dir": str(self.checkpoint_dir),
        }

        update_status_file(
            self.status_path,
            current_step=f"completed_{self.property_name}",
            property_name=self.property_name,
            epoch=epoch + 1,
            step=self.global_step,
            val_metric=self.best_metric,
        )

        logger.info(f"Training complete. Best {self.metric_name}: {self.best_metric:.4f}")
        return results

    def _train_one_epoch(self, epoch: int) -> float:
        """
        Run one epoch of training (steps_per_epoch sampling steps).

        Returns:
            avg_loss: Average training loss for this epoch
        """
        self.model.train()
        total_loss = 0.0
        num_loss_updates = 0
        self.optimizer.zero_grad()

        use_bf16 = self.mixed_precision == "bf16"
        use_fp16 = self.mixed_precision == "fp16"
        use_amp = use_bf16 or use_fp16
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        # GradScaler only for fp16, not bf16
        scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

        for step in range(self.steps_per_epoch):
            # Sample a ranking list from the training dataset
            batch = self.train_dataset.sample_ranking_list(
                self.model.tokenizer, self.device
            )

            # Forward pass with mixed precision
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    predictions = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    # Reshape [list_size] -> [1, list_size] for loss function
                    pred_2d = predictions.unsqueeze(0)
                    label_2d = batch["labels"].unsqueeze(0)
                    loss = self.loss_fn(pred_2d, label_2d)
                    loss = loss / self.gradient_accumulation_steps
            else:
                predictions = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                pred_2d = predictions.unsqueeze(0)
                label_2d = batch["labels"].unsqueeze(0)
                loss = self.loss_fn(pred_2d, label_2d)
                loss = loss / self.gradient_accumulation_steps

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_loss_updates += 1

            # Optimizer step every gradient_accumulation_steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Periodic logging
                if self.global_step % self.log_every == 0:
                    recent_loss = total_loss / max(num_loss_updates, 1)
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"  Step {self.global_step} | "
                        f"Loss: {recent_loss:.4f} | LR: {lr:.2e}"
                    )
                    if self.use_wandb:
                        import wandb

                        wandb.log({
                            "train/loss": recent_loss,
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                        }, step=self.global_step)

                # Periodic evaluation
                if (
                    self.eval_every > 0
                    and self.global_step % self.eval_every == 0
                    and self.val_assays
                ):
                    val_metrics = self._evaluate()
                    val_spearman = val_metrics.get("spearman_avg", 0.0)
                    logger.info(
                        f"  Step {self.global_step} | Val Spearman: {val_spearman:.4f}"
                    )
                    if self.use_wandb:
                        import wandb

                        wandb.log({
                            "val/spearman_avg": val_spearman,
                        }, step=self.global_step)
                    self.model.train()  # Back to training mode

                # Periodic checkpoint save
                if (
                    self.save_every_steps > 0
                    and self.global_step % self.save_every_steps == 0
                ):
                    ckpt_path = save_checkpoint(
                        self.checkpoint_dir,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        self.global_step,
                        self.best_metric,
                        self.metric_name,
                    )
                    cleanup_old_checkpoints(self.checkpoint_dir, self.keep_last_n)
                    if self.sync_gcs and self.gcs_bucket:
                        sync_checkpoint_to_gcs(
                            ckpt_path, self.gcs_bucket, self.property_name
                        )

        avg_loss = total_loss / max(num_loss_updates, 1)
        return avg_loss

    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation assays.

        Computes Spearman correlation on each validation assay using ALL variants
        (not sampled), then averages across assays.

        Returns:
            metrics: Dict with spearman_avg, spearman_per_assay
        """
        if not self.val_assays:
            return {"spearman_avg": 0.0, "spearman_per_assay": {}}

        self.model.eval()
        spearman_scores = {}

        for assay in self.val_assays:
            rho = self._evaluate_single_assay(assay)
            spearman_scores[assay.dms_id] = rho

        # Filter NaN values for averaging
        valid_scores = [v for v in spearman_scores.values() if not np.isnan(v)]
        avg_spearman = np.mean(valid_scores) if valid_scores else 0.0

        return {
            "spearman_avg": float(avg_spearman),
            "spearman_per_assay": spearman_scores,
        }

    def _evaluate_single_assay(self, assay: ProteinGymAssay) -> float:
        """
        Evaluate model on a single assay.

        Processes ALL variants in mini-batches, collects predictions,
        computes Spearman correlation with ground truth.

        Returns:
            spearman_rho: Spearman correlation coefficient
        """
        all_predictions = []
        all_scores = []
        batch_size = self.config["evaluation"].get("eval_batch_size", 32)

        use_bf16 = self.mixed_precision == "bf16"
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        use_amp = self.mixed_precision in ("bf16", "fp16")

        with torch.no_grad():
            for start_idx in range(0, len(assay), batch_size):
                end_idx = min(start_idx + batch_size, len(assay))
                sequences = [assay.sequences[i] for i in range(start_idx, end_idx)]

                encoded = self.model.tokenizer(
                    sequences,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=assay.max_length + 2,
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        preds = self.model(input_ids, attention_mask)
                else:
                    preds = self.model(input_ids, attention_mask)

                all_predictions.extend(preds.cpu().float().numpy())
                all_scores.extend(assay.scores[start_idx:end_idx])

        if len(all_predictions) < 2:
            return float("nan")

        rho, _ = spearmanr(all_scores, all_predictions)
        return float(rho) if not np.isnan(rho) else 0.0
