"""
train.py — Multi-Task Training Loop
=====================================
Trains the TemporalGRU to simultaneously predict:
  - In-Hospital Mortality (binary)
  - Sepsis Diagnosis      (binary)

Loss:
  L = w_mort × BCEWithLogitsLoss(mortality, pos_weight) +
      w_sepsis × BCEWithLogitsLoss(sepsis,   pos_weight)

Metrics logged per epoch (train & val):
  - AUROC   (Area Under ROC Curve)
  - AUPRC   (Area Under Precision-Recall Curve)
  - F1 @ threshold=0.5

Training features:
  - Early stopping (based on mean val AUROC across both tasks)
  - Best checkpoint saved when val AUROC improves
  - Progress bars via tqdm

Usage:
    # Quick smoke test on 10 hospitals, 2 epochs
    python -m src.GRU_timeseries.train --sample-hospitals 10 --epochs 2

    # Full training run
    python -m src.GRU_timeseries.train
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# sklearn for AUROC / AUPRC / F1
try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
    )
except ImportError:
    raise ImportError("Install scikit-learn: pip install scikit-learn")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.GRU_timeseries.config import DataConfig, ModelConfig, TrainConfig
from src.GRU_timeseries.dataset import (
    EICUTimeseriesDataset,
    pad_collate_fn,
    build_dataloaders,
)
from src.GRU_timeseries.model import TemporalGRU
from src.GRU_timeseries.preprocessing import (
    get_canonical_feature_names,
    extract_and_cache_targets,
    make_hospital_splits,
)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_multitask_loss(
    outputs: dict,
    targets: torch.Tensor,
    mort_loss_fn: nn.BCEWithLogitsLoss,
    seps_loss_fn: nn.BCEWithLogitsLoss,
    w_mort: float,
    w_seps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute weighted multi-task loss.

    Args:
        outputs   : dict from model.forward() with keys 'mortality_logit', 'sepsis_logit'
        targets   : (B, 2) tensor — [:, 0] = mortality, [:, 1] = sepsis
        mort_loss_fn, seps_loss_fn: pre-built BCEWithLogitsLoss with pos_weights
        w_mort, w_seps: task loss weights

    Returns:
        (total_loss, mort_loss, seps_loss)
    """
    y_mort = targets[:, 0].unsqueeze(1)   # (B, 1)
    y_seps = targets[:, 1].unsqueeze(1)   # (B, 1)

    mort_loss = mort_loss_fn(outputs["mortality_logit"], y_mort)
    seps_loss = seps_loss_fn(outputs["sepsis_logit"],   y_seps)

    total_loss = w_mort * mort_loss + w_seps * seps_loss
    return total_loss, mort_loss, seps_loss


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    logits: np.ndarray,    # (N,)
    labels: np.ndarray,    # (N,)
    task_name: str,
) -> Dict[str, float]:
    """
    Compute AUROC, AUPRC, and F1@0.5 for a single binary task.

    Returns dict: {'auroc': ..., 'auprc': ..., 'f1': ...}
    """
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    metrics = {}

    # AUROC — needs at least one positive label
    if labels.sum() > 0 and labels.sum() < len(labels):
        metrics[f"{task_name}_auroc"] = roc_auc_score(labels, probs)
        metrics[f"{task_name}_auprc"] = average_precision_score(labels, probs)
    else:
        metrics[f"{task_name}_auroc"] = 0.5  # degenerate
        metrics[f"{task_name}_auprc"] = labels.mean()

    metrics[f"{task_name}_f1"] = f1_score(labels, preds, zero_division=0)
    return metrics


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: TemporalGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mort_loss_fn: nn.BCEWithLogitsLoss,
    seps_loss_fn: nn.BCEWithLogitsLoss,
    train_cfg: TrainConfig,
) -> Dict[str, float]:
    """Run one training epoch. Returns average loss metrics."""
    model.train()
    device = train_cfg.device

    total_loss_sum = 0.0
    mort_loss_sum  = 0.0
    seps_loss_sum  = 0.0
    n_batches = 0

    for x, lengths, targets, _pids in tqdm(loader, desc="  [train]", leave=False):
        x       = x.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(x, lengths)

        total_loss, mort_loss, seps_loss = compute_multitask_loss(
            outputs, targets, mort_loss_fn, seps_loss_fn,
            train_cfg.loss_w_mort, train_cfg.loss_w_sepsis,
        )

        total_loss.backward()
        # Clip gradients — prevents explosions during early training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss_sum += total_loss.item()
        mort_loss_sum  += mort_loss.item()
        seps_loss_sum  += seps_loss.item()
        n_batches += 1

    return {
        "train_loss":      total_loss_sum / max(n_batches, 1),
        "train_mort_loss": mort_loss_sum  / max(n_batches, 1),
        "train_seps_loss": seps_loss_sum  / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# One epoch of validation / evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: TemporalGRU,
    loader: DataLoader,
    mort_loss_fn: nn.BCEWithLogitsLoss,
    seps_loss_fn: nn.BCEWithLogitsLoss,
    train_cfg: TrainConfig,
    split_name: str = "val",
) -> Dict[str, float]:
    """Run inference on val/test. Returns loss + AUROC + AUPRC + F1."""
    model.eval()
    device = train_cfg.device

    all_mort_logits = []
    all_seps_logits = []
    all_mort_labels = []
    all_seps_labels = []

    total_loss_sum = 0.0
    n_batches = 0

    for x, lengths, targets, _pids in tqdm(loader, desc=f"  [{split_name}]", leave=False):
        x       = x.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        outputs = model(x, lengths)

        total_loss, _, _ = compute_multitask_loss(
            outputs, targets, mort_loss_fn, seps_loss_fn,
            train_cfg.loss_w_mort, train_cfg.loss_w_sepsis,
        )
        total_loss_sum += total_loss.item()
        n_batches += 1

        all_mort_logits.extend(outputs["mortality_logit"].squeeze(1).cpu().numpy())
        all_seps_logits.extend(outputs["sepsis_logit"].squeeze(1).cpu().numpy())
        all_mort_labels.extend(targets[:, 0].cpu().numpy())
        all_seps_labels.extend(targets[:, 1].cpu().numpy())

    metrics = {f"{split_name}_loss": total_loss_sum / max(n_batches, 1)}
    metrics.update(
        compute_metrics(np.array(all_mort_logits), np.array(all_mort_labels), f"{split_name}_mort")
    )
    metrics.update(
        compute_metrics(np.array(all_seps_logits), np.array(all_seps_labels), f"{split_name}_seps")
    )
    metrics[f"{split_name}_mean_auroc"] = (
        metrics[f"{split_name}_mort_auroc"] + metrics[f"{split_name}_seps_auroc"]
    ) / 2

    return metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_training(
    data_cfg: DataConfig   = None,
    model_cfg: ModelConfig = None,
    train_cfg: TrainConfig = None,
    sample_hospitals: int  = None,   # if set, use only N hospitals (for smoke test)
) -> Dict[str, float]:
    """
    Full training pipeline.

    1. Build dataloaders
    2. Instantiate model
    3. Train with early stopping
    4. Save best checkpoint
    5. Return best validation metrics

    Args:
        sample_hospitals: if not None, use only this many hospitals for a fast smoke test.

    Returns:
        dict of best validation metrics.
    """
    data_cfg  = data_cfg  or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    device = train_cfg.device
    print(f"\n{'='*60}")
    print(f"GRU Time-Series Training")
    print(f"{'='*60}")
    print(f"Device          : {device}")
    print(f"Observation win : {data_cfg.max_seq_len} bins = {data_cfg.max_seq_len * data_cfg.bin_size_hours}h")
    print(f"GRU hidden_dim  : {model_cfg.hidden_dim}")
    print(f"Num layers      : {model_cfg.num_layers}")
    print(f"Pooling         : {model_cfg.pooling}")
    print(f"Batch size      : {train_cfg.batch_size}")
    print(f"Learning rate   : {train_cfg.learning_rate}")
    print(f"Max epochs      : {train_cfg.epochs}")

    # ── Step 1: Ensure targets manifest exists ────────────────────────────
    if not data_cfg.target_manifest_path.exists():
        print("\n[PREPROCESSING] Building target manifest...")
        extract_and_cache_targets(data_cfg.hospitals_dir, data_cfg.target_manifest_path)

    # ── Step 2: Ensure hospital splits exist ─────────────────────────────
    if not data_cfg.hospital_splits_path.exists():
        print("[PREPROCESSING] Creating hospital splits...")
        make_hospital_splits(
            data_cfg.hospitals_dir,
            data_cfg.hospital_splits_path,
            data_cfg.train_frac,
            data_cfg.val_frac,
            data_cfg.random_seed,
        )

    # ── Step 3: Build DataLoaders ─────────────────────────────────────────
    print("\n[DATA] Building DataLoaders...")
    canonical_cols = get_canonical_feature_names(data_cfg.feature_schema_path)
    model_cfg.input_dim = len(canonical_cols)
    print(f"Input features  : {model_cfg.input_dim}")

    with open(data_cfg.hospital_splits_path) as f:
        splits = json.load(f)

    # Apply sample_hospitals limit if in smoke-test mode
    if sample_hospitals is not None:
        for key in splits:
            splits[key] = splits[key][: max(1, sample_hospitals // 3)]
        print(f"[SMOKE TEST] Using only {sample_hospitals} hospitals")

    loaders = {}
    for split_name, hosp_ids in splits.items():
        ds = EICUTimeseriesDataset(hosp_ids, data_cfg, canonical_cols)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=train_cfg.batch_size,
            shuffle=(split_name == "train"),
            collate_fn=pad_collate_fn,
            num_workers=train_cfg.num_workers,
        )
        print(f"  {split_name:5s}: {len(ds):>7,} patients")

    # ── Step 4: Instantiate model ─────────────────────────────────────────
    model = TemporalGRU(model_cfg).to(device)
    print(f"\n[MODEL] Trainable parameters: {model.count_parameters():,}")

    # ── Step 5: Loss functions with class imbalance correction ────────────
    mort_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([train_cfg.pos_weight_mort], device=device)
    )
    seps_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([train_cfg.pos_weight_sepsis], device=device)
    )

    # ── Step 6: Optimizer ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    # Reduce LR when validation metric plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # ── Step 7: Training loop with early stopping ─────────────────────────
    best_val_auroc  = 0.0
    best_val_metrics = {}
    patience_counter = 0

    history = []

    print(f"\n[TRAINING] Starting for {train_cfg.epochs} epochs...")
    for epoch in range(1, train_cfg.epochs + 1):
        t_start = time.time()

        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, mort_loss_fn, seps_loss_fn, train_cfg
        )
        val_metrics = evaluate(
            model, loaders["val"], mort_loss_fn, seps_loss_fn, train_cfg, split_name="val"
        )

        elapsed = time.time() - t_start
        combined = {**train_metrics, **val_metrics, "epoch": epoch, "time_s": elapsed}
        history.append(combined)

        val_mean_auroc = val_metrics["val_mean_auroc"]
        scheduler.step(val_mean_auroc)

        print(
            f"Epoch {epoch:02d}/{train_cfg.epochs:02d} | "
            f"train_loss={train_metrics['train_loss']:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"mort_auroc={val_metrics['val_mort_auroc']:.3f} | "
            f"seps_auroc={val_metrics['val_seps_auroc']:.3f} | "
            f"mean_auroc={val_mean_auroc:.3f} | "
            f"{elapsed:.0f}s"
        )

        # ── Early stopping / checkpointing ─────────────────────────────────
        if val_mean_auroc > best_val_auroc:
            best_val_auroc   = val_mean_auroc
            best_val_metrics = val_metrics
            patience_counter = 0

            # Save checkpoint
            ckpt_path = Path(train_cfg.checkpoint_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_cfg":  model_cfg.__dict__,
                    "train_cfg":  train_cfg.__dict__,
                    "val_metrics": val_metrics,
                    "canonical_cols": canonical_cols,
                },
                ckpt_path,
            )
            print(f"  [OK] Best checkpoint saved (mean_auroc={best_val_auroc:.3f})")

        else:
            patience_counter += 1
            if patience_counter >= train_cfg.early_stopping_patience:
                print(f"\n[STOP] Early stopping triggered after {epoch} epochs.")
                break

    print(f"\n{'='*60}")
    print(f"Best Validation Results:")
    print(f"  Mortality AUROC : {best_val_metrics.get('val_mort_auroc', 0):.4f}")
    print(f"  Mortality AUPRC : {best_val_metrics.get('val_mort_auprc', 0):.4f}")
    print(f"  Sepsis AUROC    : {best_val_metrics.get('val_seps_auroc', 0):.4f}")
    print(f"  Sepsis AUPRC    : {best_val_metrics.get('val_seps_auprc', 0):.4f}")
    print(f"  Mean AUROC      : {best_val_metrics.get('val_mean_auroc', 0):.4f}")
    print(f"{'='*60}\n")

    return best_val_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TemporalGRU model.")
    parser.add_argument("--epochs",            type=int,   default=20)
    parser.add_argument("--hidden-dim",        type=int,   default=64)
    parser.add_argument("--num-layers",        type=int,   default=1)
    parser.add_argument("--dropout",           type=float, default=0.2)
    parser.add_argument("--batch-size",        type=int,   default=32)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--pooling",           type=str,   default="last",
                        choices=["last", "mean", "attention"])
    parser.add_argument("--bidirectional",     action="store_true", default=False)
    parser.add_argument("--sample-hospitals",  type=int,   default=None,
                        help="Use only N hospitals (for fast smoke test).")
    args = parser.parse_args()

    data_cfg  = DataConfig()
    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        pooling=args.pooling,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    run_training(data_cfg, model_cfg, train_cfg, sample_hospitals=args.sample_hospitals)
