"""
eICU Multimodal Architecture  –  Model Definitions
====================================================

Stage 1 – Temporal Encoder (LSTM)
    Input  : (batch, 12, 92)    92 = 46 feature values + 46 missingness indicators
    Output : (batch, 64)        temporal_embedding

Stage 2 – Tabular Autoencoder
    Encoder Input  : (batch, 67)
    Latent         : (batch, 64)  tabular_embedding
    Decoder Output : (batch, 67)  reconstruction

Stage 3 – Multimodal Fusion
    Input  : concat([temporal_emb, tabular_emb])  →  (batch, 128)
    Output : (batch, fusion_dim)   default 128

Stage 4 – GNN (Graph Convolutional Network)
    Input  : (N_patients, fusion_dim)  node features
    Output : (N_patients, gnn_out_dim) graph-aware embeddings

Stage 5 – Prediction Heads
    Mortality head : Linear(gnn_out_dim) → 1 logit
    Sepsis head    : Linear(gnn_out_dim) → 1 logit  (optional)

All components are implemented as standalone nn.Module subclasses so they
can be trained individually (as called for in the phased plan) and then
composed into the full MultimodalGNNPipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 : Temporal Encoder (LSTM)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    LSTM-based temporal encoder.

    Architecture (default):
        Input (batch, 12, 92)
            → LSTM(hidden=128, layers=1)
            → last-valid hidden state  (batch, 128)
            → Linear(128 → 64) + ReLU
            → temporal_embedding (batch, 64)

    The encoder exposes an ``embed()`` method that returns the 64-d embedding
    without the classification head, so downstream fusion can call it directly.

    Parameters
    ----------
    input_size    : number of features per timestep  (default 92 = 46×2)
    lstm_hidden   : LSTM hidden units
    emb_dim       : output embedding dimension
    num_layers    : stacked LSTM layers (dropout applied between layers if >1)
    dropout       : dropout rate
    use_sepsis_head: if True, instantiates a secondary head and returns (mort_logits, seps_logits)
    """

    def __init__(
        self,
        input_size: int = 92,
        lstm_hidden: int = 128,
        emb_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        use_sepsis_head: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(lstm_hidden, emb_dim),
            nn.ReLU(),
        )
        # Mortality classification head (used during standalone LSTM training)
        self.head = nn.Linear(emb_dim, 1)
        self.use_sepsis_head = use_sepsis_head
        if use_sepsis_head:
            self.sepsis_head = nn.Linear(emb_dim, 1)
        else:
            self.sepsis_head = None

    # ------------------------------------------------------------------
    def _last_valid_hidden(
        self,
        lstm_out: torch.Tensor,
        mask: torch.Tensor,
        hidden_last: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the LSTM hidden state at the last *valid* timestep.

        Parameters
        ----------
        lstm_out   : (batch, T, hidden)  full LSTM output sequence
        mask       : (batch, T)  bool, True = valid timestep
        hidden_last: (batch, hidden)  final hidden state (fallback)
        """
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1) - 1   # (batch,)
            batch_idx = torch.arange(lstm_out.size(0), device=lstm_out.device)
            return lstm_out[batch_idx, lengths]           # (batch, hidden)
        return hidden_last

    # ------------------------------------------------------------------
    def embed(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass – returns temporal embedding (batch, emb_dim).
        This is what downstream fusion components should call.
        """
        lstm_out, (h_n, _) = self.lstm(x)               # (batch, T, hidden)
        last_h = self._last_valid_hidden(lstm_out, mask, h_n[-1])
        emb = self.proj(self.dropout(last_h))            # (batch, emb_dim)
        return emb

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass with classification head(s).
        Returns:
            mortality_logit of shape (batch,) if use_sepsis_head is False
            (mortality_logit, sepsis_logit) of shape ((batch,), (batch,)) if use_sepsis_head is True
        """
        emb = self.embed(x, mask)                        # (batch, emb_dim)
        mort_logit = self.head(emb).squeeze(-1)          # (batch,)
        if self.use_sepsis_head and self.sepsis_head is not None:
            seps_logit = self.sepsis_head(emb).squeeze(-1) # (batch,)
            return mort_logit, seps_logit
        return mort_logit


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 : Tabular Denoising Autoencoder
# ──────────────────────────────────────────────────────────────────────────────

class TabularAutoencoder(nn.Module):
    """
    Symmetric denoising autoencoder for static clinical features.

    Architecture:
        Encoder: Linear(input_dim) → BN → ReLU → Linear(128) → ReLU → Linear(64)
        Decoder: Linear(64) → ReLU → Linear(128) → ReLU → Linear(input_dim)
        Head   : Linear(64) → 1 logit  (mortality classification)

    Training objective:
        total_loss = MSE(reconstruction, clean_input) + BCE(logit, label)

    The encoder exposes an ``encode()`` method for downstream fusion.

    Parameters
    ----------
    input_dim  : number of static features  (default 67)
    noise_std  : Gaussian noise σ added to input during training
    """

    def __init__(
        self,
        input_dim: int = 67,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.noise_std = noise_std

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        # Downstream classification head
        self.head = nn.Linear(64, 1)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return tabular embedding (batch, 64)."""
        return self.encoder(x)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, training: bool = False):
        """
        Returns
        -------
        logit   : (batch,)  mortality prediction
        recon   : (batch, input_dim)  reconstruction of clean input
        latent  : (batch, 64)  latent embedding
        """
        if training and self.noise_std > 0:
            noisy = x + torch.randn_like(x) * self.noise_std
        else:
            noisy = x
        latent = self.encoder(noisy)        # (batch, 64)
        recon  = self.decoder(latent)       # (batch, input_dim)
        logit  = self.head(latent).squeeze(-1)  # (batch,)
        return logit, recon, latent


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 : Multimodal Fusion
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalFusion(nn.Module):
    """
    Fuses temporal and tabular embeddings into a single patient embedding via concatenation.

    Architecture:
        concat([temporal, tabular])  →  Linear(128 → fusion_dim) → ReLU → Dropout
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        tabular_dim: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        combined_dim = temporal_dim + tabular_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        temporal_emb: torch.Tensor,
        tabular_emb: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([temporal_emb, tabular_emb], dim=-1)
        return self.net(combined)


class AttentionFusion(nn.Module):
    """
    Multi-head cross-attention fusion between temporal and tabular representations.
    Stack modalities -> MultiheadAttention -> Residual + LayerNorm -> FFN -> LayerNorm.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        temporal_emb: torch.Tensor,
        tabular_emb: torch.Tensor,
    ) -> torch.Tensor:
        # (B, 2, embed_dim)
        modalities = torch.stack([temporal_emb, tabular_emb], dim=1)
        attended, _ = self.attention(modalities, modalities, modalities, need_weights=False)
        x = self.norm1(modalities + attended)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1)  # (B, embed_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 : Patient Similarity Graph + GNN
# ──────────────────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """
    Simple symmetric graph convolution:
        H' = σ( D^{-1/2} A D^{-1/2} H W )

    where A is the adjacency matrix (with self-loops added).

    This is the GCN formulation from Kipf & Welling 2017.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(
        self, x: torch.Tensor, adj_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (N, in_features)
        adj_norm : (N, N)  symmetrically normalised adjacency (with self-loops)

        Returns
        -------
        (N, out_features)
        """
        return F.relu(self.linear(adj_norm @ x))


class PatientGraphGNN(nn.Module):
    """
    Two-layer GCN that operates on the patient similarity graph.

    Graph construction (at inference time, call ``build_adj``):
        - Compute cosine similarity between all pairs of fused embeddings.
        - Connect patient i to its k nearest neighbours.
        - Add self-loops and symmetrically normalise.

    The GNN is trained jointly with the prediction heads.

    Parameters
    ----------
    in_features  : input node feature size  (= fusion_dim, default 128)
    hidden       : GCN hidden size
    out_features : GCN output size
    k_neighbors  : number of k-NN edges per node
    dropout      : dropout between layers
    """

    def __init__(
        self,
        in_features: int = 128,
        hidden: int = 128,
        out_features: int = 64,
        k_neighbors: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.k = k_neighbors
        self.conv1 = GraphConvLayer(in_features, hidden)
        self.conv2 = GraphConvLayer(hidden, out_features)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    @staticmethod
    def build_adj(
        embeddings: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Build a symmetrically normalised adjacency matrix from patient embeddings.

        Parameters
        ----------
        embeddings : (N, D)  patient embeddings (e.g. fused vectors)
        k          : number of nearest neighbours per patient

        Returns
        -------
        adj_norm : (N, N)  normalised adjacency  (same device as embeddings)
        """
        N = embeddings.size(0)
        # Cosine similarity
        norm = F.normalize(embeddings, p=2, dim=-1)        # (N, D)
        sim  = norm @ norm.t()                             # (N, N)

        # k-NN: for each row keep the top-k values, zero out the rest
        topk_vals, topk_idx = sim.topk(k=min(k, N), dim=-1)
        adj = torch.zeros_like(sim)
        adj.scatter_(1, topk_idx, topk_vals.clamp(min=0.0))

        # Symmetrise + self-loop
        adj = (adj + adj.t()) / 2.0
        adj = adj + torch.eye(N, device=adj.device)

        # D^{-1/2} A D^{-1/2}
        deg  = adj.sum(dim=-1).clamp(min=1e-8)
        d_inv_sqrt = deg.pow(-0.5)
        adj_norm = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(0)
        return adj_norm

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        adj_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (N, in_features)
        adj_norm : (N, N)

        Returns
        -------
        (N, out_features)
        """
        h = self.conv1(x, adj_norm)        # (N, hidden)
        h = self.dropout(h)
        h = self.conv2(h, adj_norm)        # (N, out_features)
        return h


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 : Prediction Heads
# ──────────────────────────────────────────────────────────────────────────────

class MortalityHead(nn.Module):
    """Binary mortality prediction head.  Returns a scalar logit."""

    def __init__(self, in_features: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SepsisHead(nn.Module):
    """
    Binary sepsis prediction head.
    Uses the same architecture as MortalityHead.
    Attach only when sepsis labels are available.
    """

    def __init__(self, in_features: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Full Pipeline (Stages 1–5 composed)
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalGNNPipeline(nn.Module):
    """
    End-to-end multimodal pipeline:
        temporal sequence  →  TemporalEncoder  →  temporal_emb (64)
        static features    →  TabularAutoencoder  →  tabular_emb (64) + reconstruction (67)
        modalities         →  AttentionFusion (or Concat)  →  fused_emb (64 or 128)
        patient graph      →  PatientGraphGNN   →  graph_emb  (64)
        graph_emb          →  MortalityHead     →  mortality logit
                           →  SepsisHead        →  sepsis logit

    Notes
    -----
    * Supports both within-hospital exact graph batching (precomputed / block-diagonal adj_norm)
      and on-the-fly dynamic graph construction.
    * Both tasks and tabular reconstruction are trained end-to-end jointly.
    """

    def __init__(
        self,
        temporal_input_size: int = 92,
        tabular_input_dim: int = 67,
        lstm_hidden: int = 120,
        temporal_emb_dim: int = 64,
        tabular_emb_dim: int = 64,
        fusion_type: str = "attention",
        gnn_hidden: int = 128,
        gnn_out: int = 64,
        k_neighbors: int = 10,
        dropout: float = 0.25,
        use_sepsis_head: bool = True,
    ) -> None:
        super().__init__()
        self.temporal_enc = TemporalEncoder(
            input_size=temporal_input_size,
            lstm_hidden=lstm_hidden,
            emb_dim=temporal_emb_dim,
            dropout=dropout,
        )
        self.tabular_ae = TabularAutoencoder(
            input_dim=tabular_input_dim,
        )
        self.fusion_type = fusion_type
        if fusion_type == "attention":
            self.fusion = AttentionFusion(
                embed_dim=temporal_emb_dim,
                num_heads=4,
                dropout=dropout,
            )
            gnn_in_features = temporal_emb_dim
        else:
            self.fusion = MultimodalFusion(
                temporal_dim=temporal_emb_dim,
                tabular_dim=tabular_emb_dim,
                fusion_dim=128,
                dropout=dropout,
            )
            gnn_in_features = 128

        self.gnn = PatientGraphGNN(
            in_features=gnn_in_features,
            hidden=gnn_hidden,
            out_features=gnn_out,
            k_neighbors=k_neighbors,
            dropout=dropout,
        )
        self.mortality_head = MortalityHead(in_features=gnn_out)
        self.sepsis_head = SepsisHead(in_features=gnn_out) if use_sepsis_head else None

    # ------------------------------------------------------------------
    def forward(
        self,
        x_seq: torch.Tensor,
        x_tab: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        adj_norm: torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        x_seq    : (N, T, input_size)   temporal sequences
        x_tab    : (N, tabular_dim)     static features
        seq_mask : (N, T) bool          valid-timestep mask
        adj_norm : (N, N)               precomputed normalized adjacency (optional)

        Returns
        -------
        mortality_logit : (N,)
        sepsis_logit    : (N,) or None
        tab_recon       : (N, tabular_dim)  for reconstruction loss
        graph_emb       : (N, gnn_out)      for downstream embedding / analysis
        """
        # Stage 1 – temporal embedding
        t_emb = self.temporal_enc.embed(x_seq, seq_mask)

        # Stage 2 – tabular embedding + reconstruction
        _, tab_recon, tab_latent = self.tabular_ae(x_tab, training=self.training)

        # Stage 3 – fusion
        fused = self.fusion(t_emb, tab_latent)

        # Stage 4 – GNN (use provided intra-hospital adj_norm or build on-the-fly)
        if adj_norm is None:
            adj_norm = PatientGraphGNN.build_adj(fused.detach(), self.gnn.k)
        graph_emb = self.gnn(fused, adj_norm)

        # Stage 5 – prediction heads
        mort_logit = self.mortality_head(graph_emb)
        sep_logit  = self.sepsis_head(graph_emb) if self.sepsis_head is not None else None

        return mort_logit, sep_logit, tab_recon, graph_emb


class MultiTaskLoss(nn.Module):
    """
    Joint multi-task loss for mortality and sepsis prediction with auxiliary tabular reconstruction.
    L_total = L_mortality + lambda_sepsis * L_sepsis + lambda_recon * L_recon
    """

    def __init__(
        self,
        pos_weight_mortality: torch.Tensor,
        pos_weight_sepsis: torch.Tensor,
        lambda_sepsis: float = 1.0,
        lambda_recon: float = 0.1,
    ) -> None:
        super().__init__()
        self.mort_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_mortality)
        self.sepsis_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_sepsis)
        self.recon_crit = nn.MSELoss()
        self.lambda_sepsis = lambda_sepsis
        self.lambda_recon = lambda_recon

    def forward(
        self,
        mort_logits: torch.Tensor,
        sepsis_logits: torch.Tensor,
        tab_recon: torch.Tensor,
        y_mort: torch.Tensor,
        y_seps: torch.Tensor,
        x_tab: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_mort = self.mort_crit(mort_logits, y_mort)
        loss_seps = self.sepsis_crit(sepsis_logits, y_seps)
        loss_recon = self.recon_crit(tab_recon, x_tab)
        total_loss = loss_mort + (self.lambda_sepsis * loss_seps) + (self.lambda_recon * loss_recon)
        return total_loss, loss_mort, loss_seps, loss_recon

