#!/usr/bin/env python3
"""
Step 6: End-to-End Multi-Task Multimodal GNN Training
====================================================
Jointly trains:
  TemporalEncoder (LSTM) + TabularAutoencoder + AttentionFusion + PatientGraphGNN + Dual Heads
On:
  - Task 1: In-Hospital Mortality Prediction
  - Task 2: ICU Sepsis Prediction
  - Auxiliary: Tabular Feature Reconstruction

Hospital-level split from processed/hospital_split.json:
  - Train: 145 hospitals (143,419 patients)
  - Val:    31 hospitals (30,047 patients) [unseen]
  - Test:   31 hospitals (27,360 patients) [unseen]

Exact within-hospital k-NN graphs (zero cross-hospital edges).
"""

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.models.architecture import MultimodalGNNPipeline, MultiTaskLoss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "processed"
CACHE_DIR = PROCESSED_ROOT / "multitask_cache"
OUT_DIR = PROJECT_ROOT / "checkpoints" / "fusion" / "multitask"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_SEPSIS = 1.0
LAMBDA_RECON = 0.1
K_NEIGHBORS = 5
SEED = 42
BETA_MORTALITY = 1.5          # Clinical sensitivity weighting for mortality (ensures recall >= 60% without precision collapse)
BETA_SEPSIS = 2.0             # Clinical sensitivity weighting for sepsis (early detection priority)
MIN_MORTALITY_RECALL = 0.65   # Target validation recall constraint for mortality triage


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def fbeta(p, r, beta=2.0):
    return (1 + beta**2) * p * r / (beta**2 * p + r + 1e-8)


def compute_metrics(y_true, prob, thresh, beta=1.0):
    pred = (prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "threshold": round(float(thresh), 4),
        "auroc": round(float(roc_auc_score(y_true, prob)), 4),
        "auprc": round(float(average_precision_score(y_true, prob)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "f2": round(float(fbeta(prec, rec, beta=2.0)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_split(model, hospital_names, device):
    model.eval()
    all_mort_probs, all_mort_y = [], []
    all_seps_probs, all_seps_y = [], []
    all_pids = []

    with torch.no_grad():
        for h in hospital_names:
            p = CACHE_DIR / f"{h}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            x_seq = torch.tensor(d["x_seq"], dtype=torch.float32, device=device)
            mask = torch.tensor(d["mask"], dtype=torch.bool, device=device)
            x_tab = torch.tensor(d["x_tab"], dtype=torch.float32, device=device)
            x_seq = torch.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)
            x_tab = torch.nan_to_num(x_tab, nan=0.0, posinf=0.0, neginf=0.0)
            y_m = d["y_mort"]
            y_s = d["y_seps"]
            pids = d["pids"]

            mort_l, seps_l, _, _ = model(x_seq, x_tab, seq_mask=mask)
            mort_p = torch.sigmoid(mort_l).cpu().numpy()
            seps_p = torch.sigmoid(seps_l).cpu().numpy()

            all_mort_probs.append(mort_p)
            all_mort_y.append(y_m)
            all_seps_probs.append(seps_p)
            all_seps_y.append(y_s)
            all_pids.append(pids)

    all_mort_probs = np.concatenate(all_mort_probs)
    all_mort_y = np.concatenate(all_mort_y)
    all_seps_probs = np.concatenate(all_seps_probs)
    all_seps_y = np.concatenate(all_seps_y)
    all_pids = np.concatenate(all_pids)

    auc_mort = float(roc_auc_score(all_mort_y, all_mort_probs))
    ap_mort = float(average_precision_score(all_mort_y, all_mort_probs))
    auc_seps = float(roc_auc_score(all_seps_y, all_seps_probs))
    ap_seps = float(average_precision_score(all_seps_y, all_seps_probs))

    return {
        "mort_probs": all_mort_probs,
        "mort_y": all_mort_y,
        "seps_probs": all_seps_probs,
        "seps_y": all_seps_y,
        "pids": all_pids,
        "auc_mort": auc_mort,
        "ap_mort": ap_mort,
        "auc_seps": auc_seps,
        "ap_seps": ap_seps,
        "mean_ap": 0.5 * (ap_mort + ap_seps),
    }


def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device for End-to-End Multi-Task Training: {device}")

    with open(OUT_DIR / "split_summary.json") as f:
        summary = json.load(f)

    train_hospitals = summary["train"]["hospital_names"]
    val_hospitals = summary["val"]["hospital_names"]
    test_hospitals = summary["test"]["hospital_names"]

    w_mort = torch.tensor([summary["train"]["mortality_pos_weight"]], dtype=torch.float32, device=device)
    w_seps = torch.tensor([summary["train"]["sepsis_pos_weight"]], dtype=torch.float32, device=device)
    print(f"Training pos_weight: Mortality={w_mort.item():.2f}, Sepsis={w_seps.item():.2f}")

    # Build End-to-End Model
    model = MultimodalGNNPipeline(
        temporal_input_size=92,
        tabular_input_dim=67,
        lstm_hidden=120,
        temporal_emb_dim=64,
        tabular_emb_dim=64,
        fusion_type="attention",
        gnn_hidden=128,
        gnn_out=64,
        k_neighbors=K_NEIGHBORS,
        dropout=0.25,
        use_sepsis_head=True,
    ).to(device)

    criterion = MultiTaskLoss(
        pos_weight_mortality=w_mort,
        pos_weight_sepsis=w_seps,
        lambda_sepsis=LAMBDA_SEPSIS,
        lambda_recon=LAMBDA_RECON,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-4)

    best_mean_ap = -1.0
    best_epoch = 0
    best_state = None
    best_val_eval = None

    print(f"\nStarting End-to-End Multi-Task Training ({NUM_EPOCHS} epochs across {len(train_hospitals)} hospitals)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        
        # Shuffle hospitals each epoch
        np.random.seed(SEED + epoch)
        shuffled_train = list(train_hospitals)
        np.random.shuffle(shuffled_train)

        total_loss, total_mort_loss, total_seps_loss, total_recon_loss = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        pbar = tqdm(shuffled_train, desc=f"Epoch {epoch}/{NUM_EPOCHS}", unit="hosp")
        for h in pbar:
            p = CACHE_DIR / f"{h}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            x_seq = torch.tensor(d["x_seq"], dtype=torch.float32, device=device)
            mask = torch.tensor(d["mask"], dtype=torch.bool, device=device)
            x_tab = torch.tensor(d["x_tab"], dtype=torch.float32, device=device)
            x_seq = torch.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)
            x_tab = torch.nan_to_num(x_tab, nan=0.0, posinf=0.0, neginf=0.0)
            y_m = torch.tensor(d["y_mort"], dtype=torch.float32, device=device)
            y_s = torch.tensor(d["y_seps"], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            mort_l, seps_l, recon, _ = model(x_seq, x_tab, seq_mask=mask)
            loss, l_m, l_s, l_r = criterion(mort_l, seps_l, recon, y_m, y_s, x_tab)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_mort_loss += l_m.item()
            total_seps_loss += l_s.item()
            total_recon_loss += l_r.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "m_l": f"{l_m.item():.3f}", "s_l": f"{l_s.item():.3f}"})

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(n_batches, 1)

        # Validation evaluation on all 31 unseen hospitals
        val_eval = evaluate_split(model, val_hospitals, device)
        print(
            f"Epoch {epoch:2d} ({epoch_time/60:.1f}m) | Train Loss: {avg_loss:.4f} | "
            f"Val Mort AUPRC: {val_eval['ap_mort']:.4f} (AUC: {val_eval['auc_mort']:.4f}) | "
            f"Val Seps AUPRC: {val_eval['ap_seps']:.4f} (AUC: {val_eval['auc_seps']:.4f}) | "
            f"Mean AUPRC: {val_eval['mean_ap']:.4f}"
        )

        if val_eval["mean_ap"] > best_mean_ap:
            best_mean_ap = val_eval["mean_ap"]
            best_epoch = epoch
            best_val_eval = val_eval
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New Best Checkpoint at Epoch {epoch} (Mean AUPRC: {best_mean_ap:.4f})")

    # Load best checkpoint
    print(f"\nTraining Complete! Best Epoch: {best_epoch} with Mean Validation AUPRC: {best_mean_ap:.4f}")
    model.load_state_dict(best_state)

    # Threshold Selection on Validation Set Only
    val_mort_probs = best_val_eval["mort_probs"]
    val_mort_y = best_val_eval["mort_y"]
    val_seps_probs = best_val_eval["seps_probs"]
    val_seps_y = best_val_eval["seps_y"]

    thresholds = np.linspace(0.05, 0.95, 181)
    
    # Mortality: optimize F_beta (clinical sensitivity weighting with beta=BETA_MORTALITY,
    # ensuring validation recall >= MIN_MORTALITY_RECALL to surpass the 60% requirement on unseen test hospitals)
    mort_scores = []
    for t in thresholds:
        pred_m = (val_mort_probs >= t).astype(int)
        r = recall_score(val_mort_y, pred_m, zero_division=0)
        p = precision_score(val_mort_y, pred_m, zero_division=0)
        fb = fbeta(p, r, beta=BETA_MORTALITY)
        score = fb if (r >= MIN_MORTALITY_RECALL) else (-1.0 + r)
        mort_scores.append(score)
    best_t_mort = float(thresholds[np.argmax(mort_scores)])

    # Sepsis: optimize F2 (clinical sensitivity)
    seps_f2s = [
        fbeta(
            precision_score(val_seps_y, (val_seps_probs >= t).astype(int), zero_division=0),
            recall_score(val_seps_y, (val_seps_probs >= t).astype(int), zero_division=0),
            beta=BETA_SEPSIS
        )
        for t in thresholds
    ]
    best_t_seps = float(thresholds[np.argmax(seps_f2s)])

    print(f"\nValidation Selected Thresholds:")
    print(f"  Mortality (Clinical F_{BETA_MORTALITY} / Sensitivity Target >= {MIN_MORTALITY_RECALL*100:.0f}%): {best_t_mort:.4f}")
    print(f"  Sepsis (F2-clinical sensitivity):                                       {best_t_seps:.4f}")

    # Evaluate Once on 31 Unseen Test Hospitals
    print("\nEvaluating on 31 Unseen Test Hospitals...")
    test_eval = evaluate_split(model, test_hospitals, device)

    mort_metrics = compute_metrics(test_eval["mort_y"], test_eval["mort_probs"], best_t_mort)
    seps_metrics = compute_metrics(test_eval["seps_y"], test_eval["seps_probs"], best_t_seps)

    print("\n================ TEST RESULTS: MORTALITY ================")
    print(json.dumps(mort_metrics, indent=2))

    print("\n================ TEST RESULTS: SEPSIS ================")
    print(json.dumps(seps_metrics, indent=2))

    # Save Checkpoint, Metrics, and Predictions
    ckpt_path = OUT_DIR / "multitask_pipeline.pt"
    torch.save({
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_mean_val_auprc": best_mean_ap,
        "threshold_mortality": best_t_mort,
        "threshold_sepsis": best_t_seps,
        "model_config": {
            "temporal_input_size": 92,
            "tabular_input_dim": 67,
            "lstm_hidden": 120,
            "temporal_emb_dim": 64,
            "tabular_emb_dim": 64,
            "fusion_type": "attention",
            "gnn_hidden": 128,
            "gnn_out": 64,
            "k_neighbors": K_NEIGHBORS,
            "dropout": 0.25,
        }
    }, ckpt_path)

    results = {
        "model": "end_to_end_multimodal_multitask_pipeline",
        "best_epoch": best_epoch,
        "val_mean_auprc": best_mean_ap,
        "threshold_mortality": best_t_mort,
        "threshold_sepsis": best_t_seps,
        "test_mortality_metrics": mort_metrics,
        "test_sepsis_metrics": seps_metrics,
        "counts": {
            "train": summary["train"],
            "val": summary["val"],
            "test": summary["test"],
        }
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        OUT_DIR / "test_predictions.npz",
        pids=test_eval["pids"],
        mortality_true=test_eval["mort_y"],
        mortality_prob=test_eval["mort_probs"],
        sepsis_true=test_eval["seps_y"],
        sepsis_prob=test_eval["seps_probs"],
    )

    print(f"\nAll Multi-Task Artifacts Successfully Saved to {OUT_DIR}!")


if __name__ == "__main__":
    main()
