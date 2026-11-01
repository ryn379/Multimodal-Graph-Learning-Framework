#!/opt/homebrew/Cellar/jupyterlab/4.6.1/libexec/bin/python3
"""
1. GCN Training and Validation Loss (Total + Mortality + Sepsis)
2. GCN Validation AUROC and AUPRC across Epochs
3. Graph Smoothing across GCN Layers (Connected Patient Neighbor Cosine Similarity)
4. GCN Performance vs Number of Neighbours (k in [1, 3, 5, 10, 15])
"""

import gc
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "fusion"
MULTITASK_DIR = CHECKPOINTS_DIR / "multitask"
ATTN_DIR = CHECKPOINTS_DIR / "attention_embeddings"
GRAPH_DIR = CHECKPOINTS_DIR / "patient_graphs_knn"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def scipy_to_torch_sparse(adj: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device)


class MultiTaskPatientGCN(nn.Module):
    """
    Two-layer sparse Graph Convolutional Network predicting both Mortality and Sepsis.
    Exposes intermediate representations H_gcn1 and H_gcn2 for graph smoothing analysis.
    """

    def __init__(self, in_features=64, hidden_dim=128, out_features=64, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Linear(in_features, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_features)
        self.dropout = nn.Dropout(dropout)
        self.mortality_head = nn.Sequential(
            nn.Linear(out_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.sepsis_head = nn.Sequential(
            nn.Linear(out_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, adj_norm):
        # Layer 1
        h1 = F.relu(self.conv1(torch.sparse.mm(adj_norm, x)))
        h1_drop = self.dropout(h1)
        # Layer 2
        h2 = F.relu(self.conv2(torch.sparse.mm(adj_norm, h1_drop)))
        h2_drop = self.dropout(h2)
        # Heads
        mort_logits = self.mortality_head(h2_drop).squeeze(-1)
        seps_logits = self.sepsis_head(h2_drop).squeeze(-1)
        return mort_logits, seps_logits, h1, h2


def precompute_hospital_restricted_neighbors(
    embeddings: np.ndarray,
    hospital_ids: np.ndarray,
    max_k: int,
) -> np.ndarray:
    """Precompute up to max_k nearest cosine neighbors restricted strictly to the same hospital."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    hospital_ids = np.asarray(hospital_ids).reshape(-1).astype(str)
    n_nodes = len(embeddings)
    neighbors = np.full((n_nodes, max_k), -1, dtype=np.int64)

    for hospital_name in np.unique(hospital_ids):
        global_idx = np.flatnonzero(hospital_ids == hospital_name)
        n_hospital = len(global_idx)
        if n_hospital <= 1:
            continue
        local_k = min(max_k, n_hospital - 1)
        nn_model = NearestNeighbors(
            n_neighbors=local_k + 1,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn_model.fit(embeddings[global_idx])
        local_nn = nn_model.kneighbors(embeddings[global_idx], return_distance=False)
        for local_row, candidates in enumerate(local_nn):
            candidates = candidates[candidates != local_row][:local_k]
            neighbors[global_idx[local_row], :len(candidates)] = global_idx[candidates]

    return neighbors


def normalized_adjacency_for_k(neighbor_indices: np.ndarray, k: int) -> sp.coo_matrix:
    """Build union-symmetrized binary k-NN adjacency with self-loops and Kipf-Welling normalization."""
    n_nodes = neighbor_indices.shape[0]
    selected = neighbor_indices[:, :k]
    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), selected.shape[1])
    cols = selected.reshape(-1)
    valid = cols >= 0

    adj = sp.coo_matrix(
        (np.ones(int(valid.sum()), dtype=np.float32), (rows[valid], cols[valid])),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    ).tocsr()
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    adj = adj + sp.eye(n_nodes, dtype=np.float32, format="csr")

    degree = np.asarray(adj.sum(axis=1)).reshape(-1)
    degree[degree <= 0] = 1.0
    inv_sqrt = np.power(degree, -0.5).astype(np.float32)
    d_inv = sp.diags(inv_sqrt, format="csr")
    return (d_inv @ adj @ d_inv).astype(np.float32).tocoo()


def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing GCN analysis on device: {device}")

    print("Loading multi-task fused representations and cohort labels...")
    train_data = np.load(MULTITASK_DIR / "multitask_fused_train.npz")
    X_train_raw = train_data["embeddings"]
    y_train_mort = train_data["mortality_labels"].astype(np.float32)
    y_train_seps = train_data["sepsis_labels"].astype(np.float32)
    hosp_train = train_data["hospital_ids"]

    val_data = np.load(MULTITASK_DIR / "multitask_fused_val.npz")
    X_val_raw = val_data["embeddings"]
    y_val_mort = val_data["mortality_labels"].astype(np.float32)
    y_val_seps = val_data["sepsis_labels"].astype(np.float32)
    hosp_val = val_data["hospital_ids"]

    # Standardize embeddings using training statistics
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    # Class positive weights
    w_mort = float((len(y_train_mort) - y_train_mort.sum()) / max(y_train_mort.sum(), 1.0))
    w_seps = float((len(y_train_seps) - y_train_seps.sum()) / max(y_train_seps.sum(), 1.0))
    print(f"Positive weights: Mortality={w_mort:.2f}, Sepsis={w_seps:.2f}")

    pos_w_mort_t = torch.tensor([w_mort], dtype=torch.float32, device=device)
    pos_w_seps_t = torch.tensor([w_seps], dtype=torch.float32, device=device)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_mort_t = torch.from_numpy(y_train_mort).to(device)
    y_train_seps_t = torch.from_numpy(y_train_seps).to(device)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_mort_t = torch.from_numpy(y_val_mort).to(device)
    y_val_seps_t = torch.from_numpy(y_val_seps).to(device)

    # Load precomputed normalized block-diagonal intra-hospital adjacency for k=5
    adj_norm_train = sp.load_npz(MULTITASK_DIR / "multitask_adj_norm_train.npz")
    adj_norm_val = sp.load_npz(MULTITASK_DIR / "multitask_adj_norm_val.npz")
    A_train_t = scipy_to_torch_sparse(adj_norm_train, device)
    A_val_t = scipy_to_torch_sparse(adj_norm_val, device)

    print("\nTraining Multi-Task GCN (50 epochs) to record loss curves and validation metrics...")
    model = MultiTaskPatientGCN(in_features=64, hidden_dim=128, out_features=64, dropout=0.25).to(device)
    crit_mort = nn.BCEWithLogitsLoss(pos_weight=pos_w_mort_t)
    crit_seps = nn.BCEWithLogitsLoss(pos_weight=pos_w_seps_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-4)

    history = []
    best_mean_ap = -1.0
    best_state = None

    for epoch in range(1, 51):
        # --- TRAIN ---
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mort_logits_tr, seps_logits_tr, _, _ = model(X_train_t, A_train_t)
        l_mort_tr = crit_mort(mort_logits_tr, y_train_mort_t)
        l_seps_tr = crit_seps(seps_logits_tr, y_train_seps_t)
        l_total_tr = l_mort_tr + l_seps_tr
        l_total_tr.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            mort_logits_val, seps_logits_val, _, _ = model(X_val_t, A_val_t)
            l_mort_val = crit_mort(mort_logits_val, y_val_mort_t)
            l_seps_val = crit_seps(seps_logits_val, y_val_seps_t)
            l_total_val = l_mort_val + l_seps_val

            prob_mort_val = torch.sigmoid(mort_logits_val).cpu().numpy()
            prob_seps_val = torch.sigmoid(seps_logits_val).cpu().numpy()

        val_mort_auroc = float(roc_auc_score(y_val_mort, prob_mort_val))
        val_mort_auprc = float(average_precision_score(y_val_mort, prob_mort_val))
        val_sepsis_auroc = float(roc_auc_score(y_val_seps, prob_seps_val))
        val_sepsis_auprc = float(average_precision_score(y_val_seps, prob_seps_val))

        epoch_record = {
            "train_total_loss": float(l_total_tr.item()),
            "val_total_loss": float(l_total_val.item()),
            "train_mortality_loss": float(l_mort_tr.item()),
            "val_mortality_loss": float(l_mort_val.item()),
            "train_sepsis_loss": float(l_seps_tr.item()),
            "val_sepsis_loss": float(l_seps_val.item()),
            "val_mortality_auroc": val_mort_auroc,
            "val_mortality_auprc": val_mort_auprc,
            "val_sepsis_auroc": val_sepsis_auroc,
            "val_sepsis_auprc": val_sepsis_auprc,
        }
        history.append(epoch_record)
        scheduler.step()

        mean_ap = 0.5 * (val_mort_auprc + val_sepsis_auprc)
        if mean_ap > best_mean_ap:
            best_mean_ap = mean_ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 50:
            print(
                f"Epoch {epoch:2d}/50 | Train Loss: {l_total_tr.item():.4f} | "
                f"Val Mort AUPRC: {val_mort_auprc:.4f} (AUC: {val_mort_auroc:.4f}) | "
                f"Val Seps AUPRC: {val_sepsis_auprc:.4f} (AUC: {val_sepsis_auroc:.4f})"
            )

    # Save training history
    with open(CHECKPOINTS_DIR / "gcn_multitask_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot 1: GCN Training and Validation Loss
    print("\nGenerating Plot 1: GCN Training and Validation Loss...")
    hist = pd.DataFrame(history)
    epochs = range(1, len(hist) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, hist["train_total_loss"], label="Train", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, hist["val_total_loss"], label="Validation", color="#ff7f0e", linewidth=2)
    axes[0].set_title("GCN Total Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, hist["train_mortality_loss"], label="Mortality train", color="#2ca02c", linestyle="--")
    axes[1].plot(epochs, hist["val_mortality_loss"], label="Mortality validation", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, hist["train_sepsis_loss"], label="Sepsis train", color="#d62728", linestyle="--")
    axes[1].plot(epochs, hist["val_sepsis_loss"], label="Sepsis validation", color="#d62728", linewidth=2)
    axes[1].set_title("GCN Task-Specific Losses", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gcn_training_validation_loss.png", dpi=300)
    plt.close(fig)
    print(f"  -> Saved {FIGURES_DIR / 'gcn_training_validation_loss.png'}")

    # Plot 2: GCN Validation AUROC and AUPRC across Epochs
    print("\nGenerating Plot 2: GCN Validation AUROC and AUPRC...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        epochs, hist["val_mortality_auroc"], marker="o",
        markersize=3, label="Mortality AUROC", color="#1f77b4"
    )
    axes[0].plot(
        epochs, hist["val_sepsis_auroc"], marker="o",
        markersize=3, label="Sepsis AUROC", color="#e377c2"
    )
    axes[0].set_title("GCN Validation AUROC", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("AUROC", fontsize=11)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs, hist["val_mortality_auprc"], marker="o",
        markersize=3, label="Mortality AUPRC", color="#2ca02c"
    )
    axes[1].plot(
        epochs, hist["val_sepsis_auprc"], marker="o",
        markersize=3, label="Sepsis AUPRC", color="#d62728"
    )
    axes[1].set_title("GCN Validation AUPRC", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("AUPRC", fontsize=11)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gcn_validation_metrics.png", dpi=300)
    plt.close(fig)
    print(f"  -> Saved {FIGURES_DIR / 'gcn_validation_metrics.png'}")

    # 3. Graph Smoothing across GCN Layers (Plot 3)
    print("\nComputing Graph Smoothing across GCN Layers (Plot 3)...")
    # Load best model checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        _, _, H_gcn1_t, H_gcn2_t = model(X_val_t, A_val_t)

    H_fused = X_val_t.cpu().numpy()
    H_gcn1 = H_gcn1_t.cpu().numpy()
    H_gcn2 = H_gcn2_t.cpu().numpy()

    rows, cols = adj_norm_val.tocsr().nonzero()
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]

    def mean_neighbor_similarity(H, rows, cols):
        if hasattr(H, "detach"):
            H = H.detach().cpu().numpy()
        H = np.asarray(H)
        similarities = np.sum(H[rows] * H[cols], axis=1)
        norms = (
            np.linalg.norm(H[rows], axis=1)
            * np.linalg.norm(H[cols], axis=1)
        )
        similarities = similarities / np.maximum(norms, 1e-12)
        return float(similarities.mean())

    stages = ["Before GCN", "After GCN layer 1", "After GCN layer 2"]
    embeddings = [H_fused, H_gcn1, H_gcn2]
    scores = [
        mean_neighbor_similarity(H, rows, cols)
        for H in embeddings
    ]

    for stage, score in zip(stages, scores):
        print(f"  {stage:22s}: Mean Neighbor Cosine Similarity = {score:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#3498db", "#2ecc71", "#e67e22"]
    bars = ax.bar(stages, scores, color=bar_colors, width=0.5, edgecolor="black", linewidth=1.2, alpha=0.9)
    ax.plot(stages, scores, color="#2c3e50", marker="o", linewidth=2.5, markersize=8)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, max(scores) * 1.25)
    ax.set_ylabel("Mean Neighbor Cosine Similarity", fontsize=12)
    ax.set_title("Graph Smoothing Across GCN Layers (Connected Patients)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gcn_graph_smoothing.png", dpi=300)
    plt.close(fig)
    print(f"  -> Saved {FIGURES_DIR / 'gcn_graph_smoothing.png'}")

    with open(CHECKPOINTS_DIR / "gcn_graph_smoothing_results.json", "w") as f:
        json.dump(dict(zip(stages, scores)), f, indent=2)

    # 4. GCN Performance vs Number of Neighbours k (Plot 4)
    print("\nRunning Graph Ablation across k in [1, 3, 5, 10, 15] (Plot 4)...")
    K_VALUES = [1, 3, 5, 10, 15]
    MAX_K = 15

    print(f"Precomputing intra-hospital neighbors up to k={MAX_K} for Train and Val...")
    train_neighbors = precompute_hospital_restricted_neighbors(X_train, hosp_train, MAX_K)
    val_neighbors = precompute_hospital_restricted_neighbors(X_val, hosp_val, MAX_K)

    k_mortality_auprcs = []
    k_sepsis_auprcs = []

    for k in K_VALUES:
        print(f"\n--- Training GCN with k={k} ---")
        adj_k_train = normalized_adjacency_for_k(train_neighbors, k)
        adj_k_val = normalized_adjacency_for_k(val_neighbors, k)

        A_k_train_t = scipy_to_torch_sparse(adj_k_train, device)
        A_k_val_t = scipy_to_torch_sparse(adj_k_val, device)

        set_seed(SEED)
        k_model = MultiTaskPatientGCN(in_features=64, hidden_dim=128, out_features=64, dropout=0.25).to(device)
        k_optimizer = torch.optim.Adam(k_model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_k_mort_ap = 0.0
        best_k_seps_ap = 0.0
        best_k_mean_ap = -1.0

        for epoch in range(1, 36):
            k_model.train()
            k_optimizer.zero_grad(set_to_none=True)
            m_l_tr, s_l_tr, _, _ = k_model(X_train_t, A_k_train_t)
            loss_tr = crit_mort(m_l_tr, y_train_mort_t) + crit_seps(s_l_tr, y_train_seps_t)
            loss_tr.backward()
            torch.nn.utils.clip_grad_norm_(k_model.parameters(), max_norm=5.0)
            k_optimizer.step()

            k_model.eval()
            with torch.no_grad():
                m_l_val, s_l_val, _, _ = k_model(X_val_t, A_k_val_t)
                prob_m = torch.sigmoid(m_l_val).cpu().numpy()
                prob_s = torch.sigmoid(s_l_val).cpu().numpy()

            m_ap = float(average_precision_score(y_val_mort, prob_m))
            s_ap = float(average_precision_score(y_val_seps, prob_s))
            mean_ap = 0.5 * (m_ap + s_ap)

            if mean_ap > best_k_mean_ap:
                best_k_mean_ap = mean_ap
                best_k_mort_ap = m_ap
                best_k_seps_ap = s_ap

        k_mortality_auprcs.append(round(best_k_mort_ap, 4))
        k_sepsis_auprcs.append(round(best_k_seps_ap, 4))
        print(f"k={k:2d} -> Best Val Mortality AUPRC: {best_k_mort_ap:.4f} | Sepsis AUPRC: {best_k_seps_ap:.4f} (Mean: {best_k_mean_ap:.4f})")

        del k_model, k_optimizer, A_k_train_t, A_k_val_t, adj_k_train, adj_k_val
        gc.collect()

    graph_results = pd.DataFrame({
        "k": K_VALUES,
        "mortality_auprc": k_mortality_auprcs,
        "sepsis_auprc": k_sepsis_auprcs,
    })
    print("\nFinal k-Ablation Table:")
    print(graph_results.to_string(index=False))

    graph_results.to_json(CHECKPOINTS_DIR / "gcn_k_sensitivity_results.json", orient="records", indent=2)

    # Plot layout
    plt.figure(figsize=(8, 5))
    plt.plot(
        graph_results["k"], graph_results["mortality_auprc"],
        marker="o", linewidth=2, label="Mortality", color="#1f77b4"
    )
    plt.plot(
        graph_results["k"], graph_results["sepsis_auprc"],
        marker="s", linewidth=2, label="Sepsis", color="#d62728"
    )
    plt.xlabel("Number of neighbours (k)", fontsize=11)
    plt.ylabel("Validation AUPRC", fontsize=11)
    plt.title("Sensitivity to Patient Graph Neighbourhood Size", fontsize=13, fontweight="bold")
    plt.xticks(K_VALUES)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gcn_k_sensitivity.png", dpi=300)
    plt.close()
    print(f"  -> Saved {FIGURES_DIR / 'gcn_k_sensitivity.png'}")

    print("\nAll 4 GCN analysis plots generated successfully!")


if __name__ == "__main__":
    main()
