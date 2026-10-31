"""
model.py — TemporalGRU Model Architecture
==========================================
A modular, multi-task GRU for ICU time-series that:
  - Encodes variable-length physiological sequences -> fixed patient embedding
  - Predicts In-Hospital Mortality (binary classification head)
  - Predicts Sepsis Diagnosis    (binary classification head)

Architecture overview:
  Input (B, T, D_in)
      │
      ▼
  [GRU Core]  <- packed_sequence handles variable lengths efficiently
      │
      ▼
  [Temporal Pooling]  <- 'last', 'mean', or 'attention' (configurable)
      │
      ▼
  Patient Embedding (B, hidden_dim)  <- exported for multimodal fusion
      │
      ├──► [Mortality Head]  -> logit (B, 1)
      └──► [Sepsis Head]     -> logit (B, 1)

Run module directly for a quick forward-pass sanity check:
    python -m src.GRU_timeseries.model
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.GRU_timeseries.config import ModelConfig


# ---------------------------------------------------------------------------
# Attention Pooling (learnable, used when pooling='attention')
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """
    Learnable attention over the time dimension.

    Given hidden states H of shape (B, T, H_dim), learns a scalar weight
    for each timestep and returns a weighted sum: (B, H_dim).

    This allows the model to focus on the most clinically relevant time windows
    (e.g., a spike in creatinine at hour 24 might be more informative than baseline).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Single linear layer maps each hidden state to a scalar attention score
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, H_dim)
            padding_mask : (B, T) — True where the timestep is PADDING (should be ignored)

        Returns:
            context: (B, H_dim) — attention-weighted patient summary
        """
        # Compute raw attention scores: (B, T, 1) -> squeeze to (B, T)
        scores = self.attention(hidden_states).squeeze(-1)

        # Mask out padded positions with -inf before softmax
        scores = scores.masked_fill(padding_mask, float("-inf"))

        # Normalize: (B, T)
        weights = torch.softmax(scores, dim=1)

        # Weighted sum: (B, H_dim)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context


# ---------------------------------------------------------------------------
# Classification Head (shared for both tasks)
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Two-layer MLP: hidden_dim -> head_hidden_dim -> 1 (logit).

    Raw logit is returned — sigmoid is NOT applied here.
    Use BCEWithLogitsLoss during training (numerically more stable).
    Apply sigmoid only at inference if you need probabilities.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            logit: (B, 1)
        """
        return self.layers(x)


# ---------------------------------------------------------------------------
# Main GRU Model
# ---------------------------------------------------------------------------

class TemporalGRU(nn.Module):
    """
    Multi-task GRU for ICU patient temporal modeling.

    Parameters (all configured via ModelConfig):
        input_dim     : number of numeric time-series features (e.g., 36)
        hidden_dim    : GRU hidden state size (e.g., 64)
        num_layers    : number of stacked GRU layers (1 or 2 for CPU)
        dropout       : dropout rate between layers and inside heads
        bidirectional : False = causal (left-to-right only); True = retrospective
        pooling       : 'last' | 'mean' | 'attention'
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim    = cfg.hidden_dim
        self.bidirectional = cfg.bidirectional
        self.pooling       = cfg.pooling

        # Effective output dimension of the GRU
        # Bidirectional doubles the hidden state (forward + backward concatenated)
        self.gru_out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)

        # ── GRU Core ──────────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,            # input is (B, T, D_in)
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,  # only between layers
            bidirectional=cfg.bidirectional,
        )

        # ── Dropout after GRU output ──────────────────────────────────────
        self.gru_dropout = nn.Dropout(cfg.dropout)

        # ── Temporal Pooling ──────────────────────────────────────────────
        if cfg.pooling == "attention":
            self.attention_pooling = AttentionPooling(self.gru_out_dim)

        # ── Task-Specific Classification Heads ────────────────────────────
        self.mortality_head = ClassificationHead(
            input_dim=self.gru_out_dim,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.dropout,
        )
        self.sepsis_head = ClassificationHead(
            input_dim=self.gru_out_dim,
            hidden_dim=cfg.head_hidden_dim,
            dropout=cfg.dropout,
        )

    def get_embedding(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract the fixed-size temporal patient embedding.

        This is the core representation vector that can be exported for
        downstream multimodal fusion with the tabular encoder embeddings.

        Args:
            x       : (B, T, D_in) — padded sequence tensor
            lengths : (B,) — actual valid sequence lengths per patient

        Returns:
            embedding: (B, gru_out_dim) — one vector per patient
        """
        # Pack: tell PyTorch to skip padded timesteps for efficiency
        packed_input = pack_padded_sequence(
            x,
            lengths.cpu(),             # lengths must be on CPU
            batch_first=True,
            enforce_sorted=False,      # we don't require pre-sorted sequences
        )

        # GRU forward pass
        packed_output, _ = self.gru(packed_input)

        # Unpack: get full output tensor (B, T, gru_out_dim)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout to the output
        output = self.gru_dropout(output)

        # ── Pool across time dimension ─────────────────────────────────────
        embedding = self._pool(output, lengths)

        return embedding  # (B, gru_out_dim)

    def _pool(
        self, output: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Apply the selected pooling strategy to GRU outputs."""
        if self.pooling == "last":
            # Take the hidden state at the last VALID timestep for each sample
            # (not the padded position!)
            idx = (lengths - 1).clamp(min=0)  # shape: (B,)
            # Gather: output[:, idx, :] — need per-sample indexing
            idx_expanded = idx.view(-1, 1, 1).expand(-1, 1, output.size(-1))
            embedding = output.gather(1, idx_expanded).squeeze(1)  # (B, H)

        elif self.pooling == "mean":
            # Average over valid timesteps only (ignore padding)
            mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) < lengths.unsqueeze(1)
            # mask: (B, T) — True for valid positions
            mask_expanded = mask.unsqueeze(-1).float()             # (B, T, 1)
            sum_hidden = (output * mask_expanded).sum(dim=1)       # (B, H)
            embedding = sum_hidden / lengths.float().unsqueeze(1).clamp(min=1)

        elif self.pooling == "attention":
            # Build padding mask: True where positions are PADDED
            padding_mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) >= lengths.unsqueeze(1)
            embedding = self.attention_pooling(output, padding_mask)  # (B, H)

        else:
            raise ValueError(f"Unknown pooling strategy: '{self.pooling}'. "
                             "Choose 'last', 'mean', or 'attention'.")

        return embedding

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> dict:
        """
        Full forward pass: encode -> embed -> predict both tasks.

        Args:
            x       : (B, T, D_in) — padded sequence tensor
            lengths : (B,) — actual sequence lengths per patient

        Returns:
            dict with keys:
                'mortality_logit' : (B, 1) — raw logit for mortality prediction
                'sepsis_logit'    : (B, 1) — raw logit for sepsis prediction
                'embedding'       : (B, gru_out_dim) — patient temporal embedding
        """
        # Step 1: Get patient temporal embedding
        embedding = self.get_embedding(x, lengths)

        # Step 2: Task-specific predictions
        mortality_logit = self.mortality_head(embedding)  # (B, 1)
        sepsis_logit    = self.sepsis_head(embedding)     # (B, 1)

        return {
            "mortality_logit": mortality_logit,
            "sepsis_logit":    sepsis_logit,
            "embedding":       embedding,
        }

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Standalone forward-pass sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # ── Config ────────────────────────────────────────────────────────────
    cfg = ModelConfig(input_dim=36, hidden_dim=64, num_layers=1, dropout=0.2,
                      bidirectional=False, pooling="last")
    model = TemporalGRU(cfg)
    print(f"Model parameters : {model.count_parameters():,}")
    print(f"GRU output dim   : {model.gru_out_dim}")

    # ── Synthetic batch ────────────────────────────────────────────────────
    B, T_max, D = 8, 18, 36       # batch=8, up to 18 timesteps (72h), 36 features
    x = torch.randn(B, T_max, D)
    lengths = torch.randint(1, T_max + 1, (B,))  # variable lengths

    # ── Forward pass ──────────────────────────────────────────────────────
    out = model(x, lengths)
    print(f"\nForward pass output shapes:")
    for k, v in out.items():
        print(f"  {k:20s}: {tuple(v.shape)}")

    # ── Embedding extraction ───────────────────────────────────────────────
    emb = model.get_embedding(x, lengths)
    print(f"\nEmbedding shape  : {tuple(emb.shape)}  <- export this for multimodal fusion")

    # ── Test all pooling strategies ───────────────────────────────────────
    print("\nTesting all pooling strategies:")
    for pooling in ["last", "mean", "attention"]:
        cfg_p = ModelConfig(input_dim=D, hidden_dim=64, pooling=pooling)
        m = TemporalGRU(cfg_p)
        o = m(x, lengths)
        print(f"  pooling='{pooling}'  ->  mortality_logit: {tuple(o['mortality_logit'].shape)}, "
              f"embedding: {tuple(o['embedding'].shape)}")

    print("\n[OK] Model forward pass OK.")
