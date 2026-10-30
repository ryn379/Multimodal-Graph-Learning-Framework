#!/opt/homebrew/Cellar/jupyterlab/4.6.1/libexec/bin/python3
"""
Attention-Based Multimodal Fusion Visualizations & Diagnostics:
================================================================
1. Cross-Modal Attention Heatmaps (4 Attention Heads + Ensemble Mean)
2. Clinical Outcome Stratification Heatmaps (Survived vs Deceased)
3. Patient-Level Modality Attention Weight Distributions (Violin/Box Plots)
4. Fused Multimodal Representation Manifold (PCA 2D Projection)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
FUSION_DIR = PROJECT_ROOT / "checkpoints" / "fusion"
ATTN_DIR = FUSION_DIR / "attention_baseline"
EMBED_DIR = FUSION_DIR / "attention_embeddings"


class AttentionFusionClassifier(nn.Module):
    """
    Multi-head cross-attention fusion classifier.
    Stacks temporal and tabular tokens, applies MultiheadAttention, LayerNorm, and FFN.
    """

    def __init__(self, embedding_dim=64, num_heads=4, dropout=0.25):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, temporal, tabular, return_attention=False):
        # Shape: (B, 2, embed_dim) [Token 0 = Temporal, Token 1 = Tabular]
        modalities = torch.stack([temporal, tabular], dim=1)
        attended, weights = self.attention(
            modalities,
            modalities,
            modalities,
            need_weights=True,
            average_attn_weights=False,  # Returns per-head weights: (B, num_heads, 2, 2)
        )
        x = self.norm1(modalities + attended)
        x = self.norm2(x + self.ffn(x))
        fused = x.mean(dim=1)  # (B, embed_dim)
        logits = self.classifier(fused).squeeze(-1)
        if return_attention:
            return logits, weights, fused
        return logits


def main():
    print("Loading Attention Fusion model and validation embeddings...")
    device = torch.device("cpu")  # CPU is deterministic and rapid for inference

    # 1. Load trained attention model
    model = AttentionFusionClassifier()
    model_path = ATTN_DIR / "attention_classifier.pt"
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load scalers and embeddings (Validation Split: 24,186 patients)
    scaler = np.load(ATTN_DIR / "input_scaler.npz")
    val_data = np.load(FUSION_DIR / "fused_embeddings_val.npz")
    temporal_raw = val_data["temporal_embeddings"]
    tabular_raw = val_data["tabular_embeddings"]
    y_val = val_data["labels"].astype(int)

    # Standardize according to training statistics
    temp_norm = (temporal_raw - scaler["temporal_mean"]) / scaler["temporal_std"]
    tab_norm = (tabular_raw - scaler["tabular_mean"]) / scaler["tabular_std"]

    # 3. Extract attention weights and fused embeddings
    with torch.no_grad():
        logits, weights, fused = model(
            torch.from_numpy(temp_norm).float(),
            torch.from_numpy(tab_norm).float(),
            return_attention=True,
        )

    # weights shape: (N_patients, 4 heads, 2 tokens, 2 tokens)
    # Token index 0: Temporal, Token index 1: Tabular
    weights_np = weights.numpy()
    fused_np = fused.numpy()
    modalities = ["Temporal", "Tabular"]

    print(f"Extracted attention weights for {len(weights_np):,} validation patients.")
    print(f"Outcome breakdown: Survived (y=0) = {(y_val == 0).sum():,}, Deceased (y=1) = {(y_val == 1).sum():,}")

    # =========================================================================
    # Plot 1: 4 Attention Heads + Overall Mean Matrix
    # =========================================================================
    print("\nGenerating Plot 1: Attention Head Heatmaps...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))

    for h in range(4):
        head_matrix = weights_np[:, h, :, :].mean(axis=0)
        sns.heatmap(
            head_matrix,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=modalities,
            yticklabels=modalities,
            ax=axes[h],
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
        )
        axes[h].set_title(f"Head {h + 1}", fontsize=13, fontweight="bold")
        axes[h].set_ylabel("Query Token" if h == 0 else "", fontsize=11)
        axes[h].set_xlabel("Key Token", fontsize=11)

    # Ensemble Mean across all 4 heads
    mean_matrix = weights_np.mean(axis=(0, 1))
    sns.heatmap(
        mean_matrix,
        annot=True,
        fmt=".3f",
        cmap="Purples",
        vmin=0.0,
        vmax=1.0,
        xticklabels=modalities,
        yticklabels=modalities,
        ax=axes[4],
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[4].set_title("Overall Mean (4 Heads)", fontsize=13, fontweight="bold")
    axes[4].set_xlabel("Key Token", fontsize=11)

    fig.suptitle(
        "Cross-Modal Attention Heatmaps: Multi-Head Self-Attention Interaction",
        fontsize=15,
        fontweight="bold",
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "attention_fusion_head_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {FIGURES_DIR / 'attention_fusion_head_matrices.png'}")

    # =========================================================================
    # Plot 2: Clinical Outcome Stratification (Survived vs Deceased)
    # =========================================================================
    print("\nGenerating Plot 2: Clinical Outcome Stratification Heatmaps...")
    mean_survived = weights_np[y_val == 0].mean(axis=(0, 1))
    mean_deceased = weights_np[y_val == 1].mean(axis=(0, 1))
    diff_matrix = mean_deceased - mean_survived

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sns.heatmap(
        mean_survived,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        vmin=0.2,
        vmax=0.8,
        xticklabels=modalities,
        yticklabels=modalities,
        ax=axes[0],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[0].set_title("Survived Patients (y=0)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Query Token", fontsize=11)
    axes[0].set_xlabel("Key Token", fontsize=11)

    sns.heatmap(
        mean_deceased,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        vmin=0.2,
        vmax=0.8,
        xticklabels=modalities,
        yticklabels=modalities,
        ax=axes[1],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[1].set_title("Deceased Patients (y=1)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Key Token", fontsize=11)

    sns.heatmap(
        diff_matrix,
        annot=True,
        fmt="+.3f",
        cmap="coolwarm",
        center=0.0,
        xticklabels=modalities,
        yticklabels=modalities,
        ax=axes[2],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[2].set_title("Difference: Deceased - Survived", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("")
    axes[2].set_xlabel("Key Token", fontsize=11)

    fig.suptitle(
        "Clinical Stratification of Cross-Modal Attention (Mortality Outcome)",
        fontsize=15,
        fontweight="bold",
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "attention_fusion_clinical_stratification.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {FIGURES_DIR / 'attention_fusion_clinical_stratification.png'}")

    # =========================================================================
    # Plot 3: Distribution of Patient Attention Weights (Violin / Box Plot)
    # =========================================================================
    print("\nGenerating Plot 3: Attention Weight Distributions...")
    # Average across the 4 heads for each patient
    # weights_np: (N, 4, 2, 2) -> patient_weights: (N, 2, 2)
    patient_weights = weights_np.mean(axis=1)

    # Focus on Temporal Query: how much it attends to Temporal vs Tabular
    # patient_weights[:, 0, 0] = Temporal -> Temporal
    # patient_weights[:, 0, 1] = Temporal -> Tabular
    attn_data = []
    for i in range(len(y_val)):
        outcome_str = "Deceased" if y_val[i] == 1 else "Survived"
        attn_data.append({
            "Patient": i,
            "Outcome": outcome_str,
            "Temporal -> Temporal": patient_weights[i, 0, 0],
            "Temporal -> Tabular": patient_weights[i, 0, 1],
            "Tabular -> Temporal": patient_weights[i, 1, 0],
            "Tabular -> Tabular": patient_weights[i, 1, 1],
        })

    import pandas as pd
    df_attn = pd.DataFrame(attn_data)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Temporal Query Attention Allocation
    sns.violinplot(
        data=df_attn,
        x="Outcome",
        y="Temporal -> Temporal",
        palette={"Survived": "#2ecc71", "Deceased": "#e74c3c"},
        inner="quartile",
        ax=axes[0],
    )
    axes[0].set_title("Temporal Query Focus on Temporal Key", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Attention Weight (Temporal -> Temporal)", fontsize=11)
    axes[0].set_xlabel("Clinical Outcome", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: Cross-Modal Interaction (Temporal -> Tabular)
    sns.violinplot(
        data=df_attn,
        x="Outcome",
        y="Temporal -> Tabular",
        palette={"Survived": "#2ecc71", "Deceased": "#e74c3c"},
        inner="quartile",
        ax=axes[1],
    )
    axes[1].set_title("Temporal Query Focus on Tabular Key (Cross-Modal)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Attention Weight (Temporal -> Tabular)", fontsize=11)
    axes[1].set_xlabel("Clinical Outcome", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Patient-Level Modality Weighting Distributions Stratified by Survival",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "attention_fusion_weight_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {FIGURES_DIR / 'attention_fusion_weight_distributions.png'}")

    # =========================================================================
    # Plot 4: Fused Representation Latent Space (PCA Projection)
    # =========================================================================
    print("\nGenerating Plot 4: Fused Multimodal Representation PCA Projection...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(fused_np)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_surv = ax.scatter(
        coords[y_val == 0, 0],
        coords[y_val == 0, 1],
        c="#3498db",
        alpha=0.25,
        s=12,
        label=f"Survived (n={(y_val == 0).sum():,})",
    )
    scatter_dec = ax.scatter(
        coords[y_val == 1, 0],
        coords[y_val == 1, 1],
        c="#e74c3c",
        alpha=0.6,
        s=18,
        label=f"Deceased (n={(y_val == 1).sum():,})",
    )
    ax.set_title(
        f"PCA of Attention-Fused Patient Embeddings (R^64 -> R^2)\n"
        f"Explained Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "attention_fusion_latent_pca.png", dpi=300)
    plt.close(fig)
    print(f"  -> Saved {FIGURES_DIR / 'attention_fusion_latent_pca.png'}")

    # Save numeric summary
    summary = {
        "head_1": weights_np[:, 0].mean(axis=0).tolist(),
        "head_2": weights_np[:, 1].mean(axis=0).tolist(),
        "head_3": weights_np[:, 2].mean(axis=0).tolist(),
        "head_4": weights_np[:, 3].mean(axis=0).tolist(),
        "overall_mean": mean_matrix.tolist(),
        "survived_mean": mean_survived.tolist(),
        "deceased_mean": mean_deceased.tolist(),
        "difference_deceased_minus_survived": diff_matrix.tolist(),
    }
    with open(FUSION_DIR / "attention_weights_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll Attention Fusion plots and numeric summaries successfully generated!")


if __name__ == "__main__":
    main()
