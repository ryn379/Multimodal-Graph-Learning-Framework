#!/usr/bin/env python3
"""
Evaluate Multi-Task Multimodal GNN Pipeline & Generate Diagnostic Figures
=========================================================================
Uses:
  1. checkpoints/fusion/multitask/multitask_pipeline.pt
  2. checkpoints/fusion/multitask/split_summary.json
  3. processed/multitask_cache/*.npz
Produces:
  1. checkpoints/fusion/multitask/multitask_evaluation_plots.png
  2. checkpoints/fusion/multitask/metrics.json
  3. checkpoints/fusion/multitask/test_predictions.npz
"""

from pathlib import Path
import json
import sys
import time

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architecture import MultimodalGNNPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = PROJECT_ROOT / "checkpoints" / "fusion" / "multitask"
CACHE_DIR = PROJECT_ROOT / "processed" / "multitask_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load split summary
with open(OUT_DIR / "split_summary.json") as f:
    summary = json.load(f)

val_hospitals = summary["val"]["hospital_names"]
test_hospitals = summary["test"]["hospital_names"]

# 2. Load model
ckpt_path = OUT_DIR / "multitask_pipeline.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = MultimodalGNNPipeline(k_neighbors=5, use_sepsis_head=True).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded model from epoch {ckpt['best_epoch']} (Val mean AUPRC: {ckpt['best_mean_val_auprc']:.4f})")


def evaluate_split(model, hospital_names):
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

            mort_l, seps_l, _, _ = model(x_seq, x_tab, seq_mask=mask)
            all_mort_probs.extend(torch.sigmoid(mort_l).cpu().numpy())
            all_mort_y.extend(d["y_mort"])
            all_seps_probs.extend(torch.sigmoid(seps_l).cpu().numpy())
            all_seps_y.extend(d["y_seps"])
            all_pids.extend(d["pids"])

    return {
        "mort_probs": np.asarray(all_mort_probs),
        "mort_y": np.asarray(all_mort_y),
        "seps_probs": np.asarray(all_seps_probs),
        "seps_y": np.asarray(all_seps_y),
        "pids": np.asarray(all_pids),
    }


# 3. Calibrate on validation set
print("\n--- Evaluating Validation Set (31 Unseen Hospitals) ---")
val_data = evaluate_split(model, val_hospitals)


def find_best_threshold(y_true, y_prob, metric="f1", beta=1.0, min_recall=None):
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
        if min_recall is not None:
            # Constrained optimization: guarantee target clinical recall (e.g. >= 60-65%), then maximize F_beta / F1
            score_val = f_beta if (rec >= min_recall) else (-1.0 + rec)
        elif metric == "fbeta":
            score_val = f_beta
        elif metric == "balanced_accuracy":
            score_val = bal_acc
        else:
            score_val = f1

        if score_val > best_score:
            best_score = score_val
            best_thresh = round(float(threshold), 2)

    return best_thresh, pd.DataFrame(records)


# Clinical calibration: In ICU mortality prediction, missing a critical patient (false negative) is fatal.
# Standard F1 optimization under 10:1 class imbalance picks an overly conservative cutoff (~0.69-0.75),
# suppressing recall to ~49-58%. Calibrating with F_beta (beta=1.5 and min_recall=0.65) safely boosts recall to >= 64%.
thresh_mort, df_val_mort = find_best_threshold(val_data["mort_y"], val_data["mort_probs"], metric="fbeta", beta=1.5, min_recall=0.65)
thresh_seps, df_val_seps = find_best_threshold(val_data["seps_y"], val_data["seps_probs"], metric="fbeta", beta=2.0)

print(f"Optimal Mortality Threshold: {thresh_mort:.2f} (clinical F1.5 / sensitivity target >= 60%)")
print(f"Optimal Sepsis Threshold:    {thresh_seps:.2f} (max F2 clinical early warning)")

# 4. Evaluate on held-out test set
print("\n--- Evaluating Test Set (31 Unseen Hospitals) ---")
test_data = evaluate_split(model, test_hospitals)


def compute_metrics(y_true, y_prob, threshold, task_name):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

    return {
        "task": task_name,
        "threshold": round(float(threshold), 2),
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "auprc": round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)) * 100, 2),
        "recall_sensitivity": round(float(rec) * 100, 2),
        "specificity": round(float(spec) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "f1": round(float(f1), 4),
        "f2": round(float(f2), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
    }


test_mort_metrics = compute_metrics(test_data["mort_y"], test_data["mort_probs"], thresh_mort, "In-Hospital Mortality")
test_seps_metrics = compute_metrics(test_data["seps_y"], test_data["seps_probs"], thresh_seps, "Sepsis")

# 5. Build comparison DataFrame
comparison_rows = []
for k, label in [
    ("threshold", "Optimal Threshold"),
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("accuracy", "Accuracy (%)"),
    ("balanced_accuracy", "Balanced Accuracy (%)"),
    ("recall_sensitivity", "Sensitivity / Recall (%)"),
    ("specificity", "Specificity (%)"),
    ("precision", "Precision (%)"),
    ("f1", "F1-Score"),
    ("f2", "F2-Score"),
    ("tp", "True Positives (TP)"),
    ("fp", "False Positives (FP)"),
    ("tn", "True Negatives (TN)"),
    ("fn", "False Negatives (FN)"),
]:
    comparison_rows.append({
        "Metric": label,
        "Mortality Prediction": test_mort_metrics[k],
        "Sepsis Prediction": test_seps_metrics[k],
    })

comparison_df = pd.DataFrame(comparison_rows)
print("=" * 80)
print("   STEP 6: END-TO-END MULTI-TASK MULTIMODAL GNN PERFORMANCE (TEST COHORT)   ")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)

# Cross-phase comparison table
phase_rows = [
    {"Model / Pipeline": "Stage 1: Standalone Temporal LSTM", "Mortality AUROC": 0.8474, "Mortality AUPRC": 0.4590, "Sepsis AUROC": 0.8668, "Sepsis AUPRC": 0.6105},
    {"Model / Pipeline": "Stage 2: Tabular Autoencoder", "Mortality AUROC": 0.8350, "Mortality AUPRC": 0.4410, "Sepsis AUROC": 0.7850, "Sepsis AUPRC": 0.4120},
    {"Model / Pipeline": "Stage 3: Cross-Attention Fusion Baseline", "Mortality AUROC": 0.8793, "Mortality AUPRC": 0.5293, "Sepsis AUROC": 0.8850, "Sepsis AUPRC": 0.5840},
    {"Model / Pipeline": "Stage 6: Multi-Task Multimodal GNN", "Mortality AUROC": test_mort_metrics["auroc"], "Mortality AUPRC": test_mort_metrics["auprc"], "Sepsis AUROC": test_seps_metrics["auroc"], "Sepsis AUPRC": test_seps_metrics["auprc"]},
]
phase_df = pd.DataFrame(phase_rows)
print("\n" + "=" * 80)
print("   CROSS-PHASE PROGRESSION: FROM STANDALONE TEMPORAL LSTM TO MULTI-TASK GNN   ")
print("=" * 80)
print(phase_df.to_string(index=False))
print("=" * 80)

# 6. Save metrics json & predictions
metrics_dict = {
    "model": "end_to_end_multimodal_multitask_pipeline",
    "best_epoch": int(ckpt["best_epoch"]),
    "val_mean_auprc": float(ckpt["best_mean_val_auprc"]),
    "threshold_mortality": float(thresh_mort),
    "threshold_sepsis": float(thresh_seps),
    "test_mortality_metrics": test_mort_metrics,
    "test_sepsis_metrics": test_seps_metrics,
    "cross_phase_comparison": phase_rows,
}

with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)
print(f"Saved updated metrics to {OUT_DIR / 'metrics.json'}")

np.savez_compressed(
    OUT_DIR / "test_predictions.npz",
    mort_probs=test_data["mort_probs"],
    mort_y=test_data["mort_y"],
    seps_probs=test_data["seps_probs"],
    seps_y=test_data["seps_y"],
    pids=test_data["pids"],
)

# 7. Generate publication diagnostic figure
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Multi-Task Multimodal GNN Pipeline: Held-Out Test Cohort Evaluation", fontsize=16, fontweight="bold", y=0.98)

# 1. Confusion Matrix: Mortality
cm_m = np.array(test_mort_metrics["confusion_matrix"])
im0 = axes[0, 0].imshow(cm_m, cmap="Blues", interpolation="nearest")
axes[0, 0].set_title(f"Mortality Confusion Matrix\n(Threshold = {thresh_mort:.2f})", fontsize=13, fontweight="semibold")
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
cm_s = np.array(test_seps_metrics["confusion_matrix"])
im1 = axes[0, 1].imshow(cm_s, cmap="Purples", interpolation="nearest")
axes[0, 1].set_title(f"Sepsis Confusion Matrix\n(Threshold = {thresh_seps:.2f})", fontsize=13, fontweight="semibold")
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

# 3. Metric Comparison Bar Chart across tasks
metric_labels = ["Accuracy", "Bal. Acc", "Sensitivity", "Specificity", "F1 x 100", "F2 x 100"]
m_vals = [
    test_mort_metrics["accuracy"], test_mort_metrics["balanced_accuracy"],
    test_mort_metrics["recall_sensitivity"], test_mort_metrics["specificity"],
    test_mort_metrics["f1"] * 100, test_mort_metrics["f2"] * 100,
]
s_vals = [
    test_seps_metrics["accuracy"], test_seps_metrics["balanced_accuracy"],
    test_seps_metrics["recall_sensitivity"], test_seps_metrics["specificity"],
    test_seps_metrics["f1"] * 100, test_seps_metrics["f2"] * 100,
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
fpr_m, tpr_m, _ = roc_curve(test_data["mort_y"], test_data["mort_probs"])
fpr_s, tpr_s, _ = roc_curve(test_data["seps_y"], test_data["seps_probs"])
axes[1, 0].plot(fpr_m, tpr_m, color="#1f77b4", lw=2.5, label=f"Mortality (AUROC = {test_mort_metrics['auroc']:.4f})")
axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance Baseline")
axes[1, 0].set_title("Mortality ROC Curve", fontsize=13, fontweight="semibold")
axes[1, 0].set_xlabel("False Positive Rate", fontsize=11)
axes[1, 0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
axes[1, 0].legend(loc="lower right", frameon=True)
axes[1, 0].grid(linestyle="--", alpha=0.4)

axes[1, 1].plot(fpr_s, tpr_s, color="#9467bd", lw=2.5, label=f"Sepsis (AUROC = {test_seps_metrics['auroc']:.4f})")
axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance Baseline")
axes[1, 1].set_title("Sepsis ROC Curve", fontsize=13, fontweight="semibold")
axes[1, 1].set_xlabel("False Positive Rate", fontsize=11)
axes[1, 1].set_ylabel("True Positive Rate (Recall)", fontsize=11)
axes[1, 1].legend(loc="lower right", frameon=True)
axes[1, 1].grid(linestyle="--", alpha=0.4)

# 5. Precision-Recall Curves
prec_m, rec_m, _ = precision_recall_curve(test_data["mort_y"], test_data["mort_probs"])
prec_s, rec_s, _ = precision_recall_curve(test_data["seps_y"], test_data["seps_probs"])
prev_m = float(np.mean(test_data["mort_y"]))
prev_s = float(np.mean(test_data["seps_y"]))

axes[1, 2].plot(rec_m, prec_m, color="#1f77b4", lw=2.5, label=f"Mortality (AUPRC = {test_mort_metrics['auprc']:.4f})")
axes[1, 2].axhline(prev_m, color="#1f77b4", linestyle=":", alpha=0.6, label=f"Mortality Prev ({prev_m*100:.1f}%)")
axes[1, 2].plot(rec_s, prec_s, color="#9467bd", lw=2.5, label=f"Sepsis (AUPRC = {test_seps_metrics['auprc']:.4f})")
axes[1, 2].axhline(prev_s, color="#9467bd", linestyle=":", alpha=0.6, label=f"Sepsis Prev ({prev_s*100:.1f}%)")
axes[1, 2].set_title("Precision-Recall Curves", fontsize=13, fontweight="semibold")
axes[1, 2].set_xlabel("Recall (Sensitivity)", fontsize=11)
axes[1, 2].set_ylabel("Precision (PPV)", fontsize=11)
axes[1, 2].legend(loc="upper right", frameon=True)
axes[1, 2].grid(linestyle="--", alpha=0.4)

plt.tight_layout()
plot_path = FIGURES_DIR / "multitask_evaluation_plots.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved diagnostic plots to {plot_path}")
print("\nMulti-task evaluation and plot generation complete!")
