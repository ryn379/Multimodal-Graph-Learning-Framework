#!/usr/bin/env python3
"""
Train & Evaluate Dual-Task LSTM Temporal Encoder (Mortality + Sepsis)
===================================================================
Produces:
  1. checkpoints/lstm_temporal_encoder.pt (dual-task weights)
  2. checkpoints/lstm_baseline_metrics.json (dual-task test metrics)
  3. checkpoints/lstm_dual_task_evaluation.png (visual diagnostic plots)
  4. checkpoints/training_history.json (training curve history)
  5. checkpoints/temporal_embeddings_{train,val,test}.npy
  6. checkpoints/temporal_labels_{train,val,test}.npy
  7. checkpoints/temporal_sepsis_labels_{train,val,test}.npy
  8. checkpoints/temporal_patient_ids_{train,val,test}.npy
"""

from pathlib import Path
import copy
import json
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ICUSequenceDataset, get_hospital_splits
from src.models.architecture import TemporalEncoder

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 5
PATIENCE = 3
LR = 5e-4
LSTM_HIDDEN = 120
EMB_DIM = 64
DROPOUT = 0.238

PROCESSED_ROOT = PROJECT_ROOT / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = PROJECT_ROOT / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
print(f"Device: {device}")


def load_dataset(name, hospital_dirs):
    t0 = perf_counter()
    ds = ICUSequenceDataset(
        hospital_dirs,
        return_pid=True,
        include_sepsis=True,
        use_cache=True,
    )
    print(f"{name}: {len(ds):,} patients loaded in {perf_counter() - t0:.2f}s")
    return ds


train_dirs, val_dirs, test_dirs = get_hospital_splits(str(PROCESSED_ROOT))
print(f"Hospitals: train={len(train_dirs)}, val={len(val_dirs)}, test={len(test_dirs)}")

train_ds = load_dataset("Train", train_dirs)
val_ds = load_dataset("Validation", val_dirs)
test_ds = load_dataset("Test", test_dirs)

# Prevalences
for split_name, ds in [("Train", train_ds), ("Validation", val_ds), ("Test", test_ds)]:
    n_tot = len(ds)
    n_mort = sum(int(s[2]) for s in ds.samples)
    n_seps = sum(int(s[3]) for s in ds.samples)
    print(
        f"  {split_name}: Mortality={n_mort:,} ({n_mort/n_tot*100:.2f}%), "
        f"Sepsis={n_seps:,} ({n_seps/n_tot*100:.2f}%)"
    )

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

train_mort_labels = np.asarray([sample[2] for sample in train_ds.samples])
train_seps_labels = np.asarray([sample[3] for sample in train_ds.samples])

pos_weight_mort = torch.tensor(
    (train_mort_labels == 0).sum() / max((train_mort_labels == 1).sum(), 1),
    dtype=torch.float32,
)
pos_weight_seps = torch.tensor(
    (train_seps_labels == 0).sum() / max((train_seps_labels == 1).sum(), 1),
    dtype=torch.float32,
)

print(f"pos_weight Mortality: {pos_weight_mort.item():.4f}")
print(f"pos_weight Sepsis:    {pos_weight_seps.item():.4f}")

criterion_mort = nn.BCEWithLogitsLoss(pos_weight=pos_weight_mort.to(device))
criterion_seps = nn.BCEWithLogitsLoss(pos_weight=pos_weight_seps.to(device))

model = TemporalEncoder(
    input_size=92,
    lstm_hidden=LSTM_HIDDEN,
    emb_dim=EMB_DIM,
    dropout=DROPOUT,
    use_sepsis_head=True,
).to(device)

ckpt_path = OUT_DIR / "lstm_temporal_encoder.pt"
if ckpt_path.exists():
    try:
        old_state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(old_state, strict=False)
        print(f"Pre-loaded backbone from {ckpt_path} (new heads: {missing})")
    except Exception as e:
        print(f"Could not warm-start from {ckpt_path}: {e}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def run_epoch(model, loader, criterion_mort, criterion_seps, device, optimizer=None, return_preds=False):
    training = optimizer is not None
    model.train(training)

    total_loss, total_loss_mort, total_loss_seps = 0.0, 0.0, 0.0
    all_mort_probs, all_mort_labels = [], []
    all_seps_probs, all_seps_labels = [], []

    for batch in loader:
        x, mask, y_mort, y_seps = batch[0], batch[1], batch[2], batch[3]
        x, mask = x.to(device), mask.to(device)
        y_mort = y_mort.to(device).float().view(-1)
        y_seps = y_seps.to(device).float().view(-1)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            mort_logits, seps_logits = model(x, mask)
            mort_logits = mort_logits.view(-1)
            seps_logits = seps_logits.view(-1)

            loss_mort = criterion_mort(mort_logits, y_mort)
            loss_seps = criterion_seps(seps_logits, y_seps)
            loss = loss_mort + loss_seps

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        mort_prob = torch.sigmoid(mort_logits)
        seps_prob = torch.sigmoid(seps_logits)

        b_size = len(y_mort)
        total_loss += loss.item() * b_size
        total_loss_mort += loss_mort.item() * b_size
        total_loss_seps += loss_seps.item() * b_size

        all_mort_probs.extend(mort_prob.detach().cpu().numpy().tolist())
        all_mort_labels.extend(y_mort.detach().cpu().numpy().tolist())
        all_seps_probs.extend(seps_prob.detach().cpu().numpy().tolist())
        all_seps_labels.extend(y_seps.detach().cpu().numpy().tolist())

    n_samples = max(len(loader.dataset), 1)
    avg_loss = total_loss / n_samples
    avg_loss_mort = total_loss_mort / n_samples
    avg_loss_seps = total_loss_seps / n_samples

    all_mort_labels = np.asarray(all_mort_labels)
    all_mort_probs = np.asarray(all_mort_probs)
    all_seps_labels = np.asarray(all_seps_labels)
    all_seps_probs = np.asarray(all_seps_probs)

    mort_auc = roc_auc_score(all_mort_labels, all_mort_probs) if len(np.unique(all_mort_labels)) == 2 else float("nan")
    mort_ap = average_precision_score(all_mort_labels, all_mort_probs) if np.sum(all_mort_labels) > 0 else float("nan")

    seps_auc = roc_auc_score(all_seps_labels, all_seps_probs) if len(np.unique(all_seps_labels)) == 2 else float("nan")
    seps_ap = average_precision_score(all_seps_labels, all_seps_probs) if np.sum(all_seps_labels) > 0 else float("nan")

    metrics_dict = {
        "loss": avg_loss,
        "loss_mort": avg_loss_mort,
        "loss_seps": avg_loss_seps,
        "mort_auc": mort_auc,
        "mort_ap": mort_ap,
        "seps_auc": seps_auc,
        "seps_ap": seps_ap,
        "mean_ap": (mort_ap + seps_ap) / 2.0 if np.isfinite(mort_ap) and np.isfinite(seps_ap) else float("nan"),
    }

    if return_preds:
        return metrics_dict, (all_mort_labels, all_mort_probs), (all_seps_labels, all_seps_probs)
    return metrics_dict


print("\n--- Starting Multi-Task LSTM Training ---")
best_mean_ap = -np.inf
best_state = None
wait = 0
history = []

for epoch in range(MAX_EPOCHS):
    t_start = perf_counter()
    train_m = run_epoch(model, train_loader, criterion_mort, criterion_seps, device, optimizer=optimizer)
    val_m = run_epoch(model, val_loader, criterion_mort, criterion_seps, device)
    ep_time = perf_counter() - t_start

    epoch_record = {
        "epoch": epoch + 1,
        "train_loss": train_m["loss"],
        "train_loss_mort": train_m["loss_mort"],
        "train_loss_seps": train_m["loss_seps"],
        "train_mort_auc": train_m["mort_auc"],
        "train_mort_ap": train_m["mort_ap"],
        "train_seps_auc": train_m["seps_auc"],
        "train_seps_ap": train_m["seps_ap"],
        "val_loss": val_m["loss"],
        "val_loss_mort": val_m["loss_mort"],
        "val_loss_seps": val_m["loss_seps"],
        "val_mort_auc": val_m["mort_auc"],
        "val_mort_ap": val_m["mort_ap"],
        "val_seps_auc": val_m["seps_auc"],
        "val_seps_ap": val_m["seps_ap"],
        "val_mean_ap": val_m["mean_ap"],
        "epoch_seconds": ep_time,
    }
    history.append(epoch_record)

    print(
        f"Epoch {epoch + 1:02d}/{MAX_EPOCHS} ({ep_time:.1f}s) | "
        f"Train Loss: {train_m['loss']:.4f} (Mort={train_m['loss_mort']:.4f}, Seps={train_m['loss_seps']:.4f}) | "
        f"Val Loss: {val_m['loss']:.4f}\n"
        f"   Val Mortality: AUROC={val_m['mort_auc']:.4f}, AUPRC={val_m['mort_ap']:.4f} | "
        f"Val Sepsis: AUROC={val_m['seps_auc']:.4f}, AUPRC={val_m['seps_ap']:.4f} | "
        f"Mean AUPRC={val_m['mean_ap']:.4f}"
    )

    if np.isfinite(val_m["mean_ap"]) and val_m["mean_ap"] > best_mean_ap:
        best_mean_ap = val_m["mean_ap"]
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)
    print(f"Saved best dual-task weights to {ckpt_path}")

with open(OUT_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2, allow_nan=True)


# Validation Threshold Calibration
print("\n--- Calibrating Validation Thresholds ---")
val_m, (y_val_mort, val_prob_mort), (y_val_seps, val_prob_seps) = run_epoch(
    model, val_loader, criterion_mort, criterion_seps, device, return_preds=True
)


def find_best_threshold(y_true, y_prob, metric="f1", beta=1.0):
    best_thresh = 0.5
    best_score = -1.0
    records = []

    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f_beta = ((1 + beta**2) * prec * rec) / (beta**2 * prec + rec) if (beta**2 * prec + rec) > 0 else 0.0

        records.append({
            "threshold": round(float(threshold), 2),
            "F1": f1,
            "F_beta": f_beta,
            "Precision": prec,
            "Recall": rec,
            "Accuracy": acc,
            "Balanced_accuracy": bal_acc,
        })
        score_val = f_beta if metric == "fbeta" else f1
        if score_val > best_score:
            best_score = score_val
            best_thresh = round(float(threshold), 2)

    return best_thresh, pd.DataFrame(records)


best_thresh_mort, mort_val_df = find_best_threshold(y_val_mort, val_prob_mort, metric="f1")
best_thresh_seps, seps_val_df = find_best_threshold(y_val_seps, val_prob_seps, metric="fbeta", beta=2.0)

print(f"Optimal Mortality Threshold: {best_thresh_mort:.2f} (max F1)")
print(f"Optimal Sepsis Threshold:    {best_thresh_seps:.2f} (max F2)")

# Test Evaluation
print("\n--- Evaluating on Held-Out Test Set ---")
test_m, (y_test_mort, test_prob_mort), (y_test_seps, test_prob_seps) = run_epoch(
    model, test_loader, criterion_mort, criterion_seps, device, return_preds=True
)

y_test_pred_mort = (test_prob_mort >= best_thresh_mort).astype(int)
y_test_pred_seps = (test_prob_seps >= best_thresh_seps).astype(int)


def compute_task_metrics(y_true, y_prob, y_pred, threshold, task_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

    return {
        "Task": task_name,
        "Optimal Threshold": round(float(threshold), 2),
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Accuracy (%)": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "Balanced Accuracy (%)": round(float(balanced_accuracy_score(y_true, y_pred)) * 100, 2),
        "Sensitivity / Recall (%)": round(float(rec) * 100, 2),
        "Specificity (%)": round(float(spec) * 100, 2),
        "Precision (%)": round(float(prec) * 100, 2),
        "F1-Score": round(float(f1), 4),
        "F2-Score": round(float(f2), 4),
        "True Positives (TP)": int(tp),
        "False Positives (FP)": int(fp),
        "True Negatives (TN)": int(tn),
        "False Negatives (FN)": int(fn),
        "Confusion Matrix": cm.tolist(),
    }


mort_test_metrics = compute_task_metrics(y_test_mort, test_prob_mort, y_test_pred_mort, best_thresh_mort, "In-Hospital Mortality")
seps_test_metrics = compute_task_metrics(y_test_seps, test_prob_seps, y_test_pred_seps, best_thresh_seps, "Sepsis")

comparison_rows = []
for k in [
    "Optimal Threshold", "AUROC", "AUPRC", "Accuracy (%)", "Balanced Accuracy (%)",
    "Sensitivity / Recall (%)", "Specificity (%)", "Precision (%)", "F1-Score", "F2-Score",
    "True Positives (TP)", "False Positives (FP)", "True Negatives (TN)", "False Negatives (FN)",
]:
    comparison_rows.append({
        "Evaluation Metric": k,
        "Mortality Prediction": mort_test_metrics[k],
        "Sepsis Prediction": seps_test_metrics[k],
    })

comparison_df = pd.DataFrame(comparison_rows)
print("=" * 78)
print("   PHASE 1: DUAL-TASK TEMPORAL LSTM PERFORMANCE ON HELD-OUT TEST COHORT   ")
print("=" * 78)
print(comparison_df.to_string(index=False))
print("=" * 78)

# Save baseline metrics
metrics_export = {
    "stage": "Phase1_TemporalEncoder_DualTask",
    "test_loss_total": float(test_m["loss"]),
    "test_loss_mortality": float(test_m["loss_mort"]),
    "test_loss_sepsis": float(test_m["loss_seps"]),
    "mortality_metrics": mort_test_metrics,
    "sepsis_metrics": seps_test_metrics,
}
with open(OUT_DIR / "lstm_baseline_metrics.json", "w") as f:
    json.dump(metrics_export, f, indent=2, allow_nan=True)
print(f"Saved metrics to: {OUT_DIR / 'lstm_baseline_metrics.json'}")

# Generate publication diagnostic figures
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Dual-Task Temporal LSTM: Comprehensive Test Cohort Evaluation", fontsize=16, fontweight="bold", y=0.98)

# 1. Confusion Matrix: Mortality
cm_m = np.array(mort_test_metrics["Confusion Matrix"])
im0 = axes[0, 0].imshow(cm_m, cmap="Blues", interpolation="nearest")
axes[0, 0].set_title(f"Mortality Confusion Matrix\n(Threshold = {best_thresh_mort:.2f})", fontsize=13, fontweight="semibold")
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_xticklabels(["Surviving (0)", "Expired (1)"])
axes[0, 0].set_yticklabels(["Surviving (0)", "Expired (1)"])
axes[0, 0].set_xlabel("Predicted Class", fontsize=11)
axes[0, 0].set_ylabel("True Class", fontsize=11)
for i in range(2):
    for j in range(2):
        count = cm_m[i, j]
        pct = count / cm_m.sum() * 100
        axes[0, 0].text(
            j, i, f"{count:,}\n({pct:.1f}%)", ha="center", va="center",
            color="white" if cm_m[i, j] > cm_m.max() / 2 else "black", fontsize=11, fontweight="bold"
        )
fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

# 2. Confusion Matrix: Sepsis
cm_s = np.array(seps_test_metrics["Confusion Matrix"])
im1 = axes[0, 1].imshow(cm_s, cmap="Purples", interpolation="nearest")
axes[0, 1].set_title(f"Sepsis Confusion Matrix\n(Threshold = {best_thresh_seps:.2f})", fontsize=13, fontweight="semibold")
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(["Non-Sepsis (0)", "Sepsis (1)"])
axes[0, 1].set_yticklabels(["Non-Sepsis (0)", "Sepsis (1)"])
axes[0, 1].set_xlabel("Predicted Class", fontsize=11)
axes[0, 1].set_ylabel("True Class", fontsize=11)
for i in range(2):
    for j in range(2):
        count = cm_s[i, j]
        pct = count / cm_s.sum() * 100
        axes[0, 1].text(
            j, i, f"{count:,}\n({pct:.1f}%)", ha="center", va="center",
            color="white" if cm_s[i, j] > cm_s.max() / 2 else "black", fontsize=11, fontweight="bold"
        )
fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

# 3. Metric Comparison Bar Chart
metric_labels = ["Accuracy", "Bal. Acc", "Sensitivity", "Specificity", "F1 x 100", "F2 x 100"]
m_vals = [
    mort_test_metrics["Accuracy (%)"], mort_test_metrics["Balanced Accuracy (%)"],
    mort_test_metrics["Sensitivity / Recall (%)"], mort_test_metrics["Specificity (%)"],
    mort_test_metrics["F1-Score"] * 100, mort_test_metrics["F2-Score"] * 100,
]
s_vals = [
    seps_test_metrics["Accuracy (%)"], seps_test_metrics["Balanced Accuracy (%)"],
    seps_test_metrics["Sensitivity / Recall (%)"], seps_test_metrics["Specificity (%)"],
    seps_test_metrics["F1-Score"] * 100, seps_test_metrics["F2-Score"] * 100,
]

x_idx = np.arange(len(metric_labels))
width = 0.35
axes[0, 2].bar(x_idx - width/2, m_vals, width, label="Mortality", color="#1f77b4", alpha=0.85)
axes[0, 2].bar(x_idx + width/2, s_vals, width, label="Sepsis", color="#9467bd", alpha=0.85)
axes[0, 2].set_title("Test Cohort Metric Comparison", fontsize=13, fontweight="semibold")
axes[0, 2].set_xticks(x_idx)
axes[0, 2].set_xticklabels(metric_labels, rotation=25, ha="right", fontsize=10)
axes[0, 2].set_ylabel("Score (%)", fontsize=11)
axes[0, 2].set_ylim(0, 105)
axes[0, 2].legend(frameon=True)
axes[0, 2].grid(axis="y", linestyle="--", alpha=0.4)

# 4. ROC Curves
fpr_m, tpr_m, _ = roc_curve(y_test_mort, test_prob_mort)
fpr_s, tpr_s, _ = roc_curve(y_test_seps, test_prob_seps)
axes[1, 0].plot(fpr_m, tpr_m, color="#1f77b4", lw=2.5, label=f"Mortality (AUROC = {mort_test_metrics['AUROC']:.4f})")
axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance Baseline")
axes[1, 0].set_title("Mortality ROC Curve", fontsize=13, fontweight="semibold")
axes[1, 0].set_xlabel("False Positive Rate", fontsize=11)
axes[1, 0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
axes[1, 0].legend(loc="lower right", frameon=True)
axes[1, 0].grid(linestyle="--", alpha=0.4)

axes[1, 1].plot(fpr_s, tpr_s, color="#9467bd", lw=2.5, label=f"Sepsis (AUROC = {seps_test_metrics['AUROC']:.4f})")
axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance Baseline")
axes[1, 1].set_title("Sepsis ROC Curve", fontsize=13, fontweight="semibold")
axes[1, 1].set_xlabel("False Positive Rate", fontsize=11)
axes[1, 1].set_ylabel("True Positive Rate (Recall)", fontsize=11)
axes[1, 1].legend(loc="lower right", frameon=True)
axes[1, 1].grid(linestyle="--", alpha=0.4)

# 5. Precision-Recall Curves
prec_m, rec_m, _ = precision_recall_curve(y_test_mort, test_prob_mort)
prec_s, rec_s, _ = precision_recall_curve(y_test_seps, test_prob_seps)
prev_m = float(np.mean(y_test_mort))
prev_s = float(np.mean(y_test_seps))

axes[1, 2].plot(rec_m, prec_m, color="#1f77b4", lw=2.5, label=f"Mortality (AUPRC = {mort_test_metrics['AUPRC']:.4f})")
axes[1, 2].axhline(prev_m, color="#1f77b4", linestyle=":", alpha=0.6, label=f"Mortality Prev ({prev_m*100:.1f}%)")
axes[1, 2].plot(rec_s, prec_s, color="#9467bd", lw=2.5, label=f"Sepsis (AUPRC = {seps_test_metrics['AUPRC']:.4f})")
axes[1, 2].axhline(prev_s, color="#9467bd", linestyle=":", alpha=0.6, label=f"Sepsis Prev ({prev_s*100:.1f}%)")
axes[1, 2].set_title("Precision-Recall Curves", fontsize=13, fontweight="semibold")
axes[1, 2].set_xlabel("Recall (Sensitivity)", fontsize=11)
axes[1, 2].set_ylabel("Precision (PPV)", fontsize=11)
axes[1, 2].legend(loc="upper right", frameon=True)
axes[1, 2].grid(linestyle="--", alpha=0.4)

plt.tight_layout()
plot_path = FIGURES_DIR / "lstm_dual_task_evaluation.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved diagnostic plots to: {plot_path}")

# Embedding extraction
print("\n--- Extracting Temporal Embeddings ---")
train_embed_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_embed_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_embed_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def extract_embeddings(model, loader, device):
    model.eval()
    embeddings, mort_labels, seps_labels, patient_ids = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x, mask = batch[0], batch[1]
            y_mort = batch[2]
            y_seps = batch[3] if len(batch) >= 4 else None
            pid = batch[-1]

            z = model.embed(x.to(device), mask.to(device))

            embeddings.append(z.detach().cpu().numpy())
            mort_labels.append(y_mort.cpu().numpy().reshape(-1))
            if y_seps is not None and len(batch) >= 5:
                seps_labels.append(y_seps.cpu().numpy().reshape(-1))
            patient_ids.append(pid.cpu().numpy().reshape(-1))

    if not embeddings:
        raise RuntimeError("Cannot extract embeddings from an empty loader.")

    res = [
        np.concatenate(embeddings, axis=0),
        np.concatenate(mort_labels, axis=0),
        np.concatenate(patient_ids, axis=0),
    ]
    if seps_labels:
        res.append(np.concatenate(seps_labels, axis=0))
    return tuple(res)


def save_split_embeddings(split, loader):
    res = extract_embeddings(model, loader, device)
    z, y_mort, pids = res[0], res[1], res[2]

    np.save(OUT_DIR / f"temporal_embeddings_{split}.npy", z)
    np.save(OUT_DIR / f"temporal_labels_{split}.npy", y_mort)
    np.save(OUT_DIR / f"temporal_patient_ids_{split}.npy", pids)

    if len(res) > 3:
        y_seps = res[3]
        np.save(OUT_DIR / f"temporal_sepsis_labels_{split}.npy", y_seps)
        print(f"{split}: embeddings={z.shape}, mort={y_mort.shape}, seps={y_seps.shape}, pids={pids.shape}")
    else:
        print(f"{split}: embeddings={z.shape}, labels={y_mort.shape}, pids={pids.shape}")


save_split_embeddings("train", train_embed_loader)
save_split_embeddings("val", val_embed_loader)
save_split_embeddings("test", test_embed_loader)
print("\nDual-task training, evaluation, plot generation, and embedding export complete!")
