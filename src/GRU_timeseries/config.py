"""
config.py — Configuration Dataclasses for the GRU Time-Series Pipeline
=======================================================================
All model, training, and data settings live here.
To start a new experiment, simply create a different config object.

72-hour observation window is the default (18 bins of 4 hours each).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Paths – adjust these if your project root changes
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # minor_project/
_PROCESSED   = _PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Controls which data to load, how to window it, and where to cache targets."""

    # ---- paths ----
    hospitals_dir: Path   = _PROCESSED / "hospitals"
    feature_schema_path: Path = _PROCESSED / "feature_schema.json"
    target_manifest_path: Path = _PROCESSED / "patient_targets.parquet"
    hospital_splits_path: Path = _PROCESSED / "hospital_splits.json"

    # ---- observation window ----
    # 18 bins × 4 h/bin = 72 hours  (edit this one number to change the window)
    bin_size_hours: int   = 4
    max_seq_len: int      = 18   # <- 72 hours

    # ---- data-split (hospital-level, not patient-level) ----
    train_frac: float     = 0.70
    val_frac: float       = 0.15
    test_frac: float      = 0.15
    random_seed: int      = 42


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Controls the TemporalGRU architecture."""

    # ---- input ----
    # Will be auto-detected from the canonical feature list at runtime,
    # but you can hard-code it here for documentation clarity.
    input_dim: int         = 36   # 20 labs + 3 infusions + 11 treatments + 2 meds (numeric-only)

    # ---- GRU core ----
    hidden_dim: int        = 64   # <- PRIMARY TUNING TARGET
    num_layers: int        = 1    # 1 or 2 on CPU
    dropout: float         = 0.2  # between GRU layers and inside heads
    bidirectional: bool    = False # False = causal (real-time); True = retrospective

    # ---- temporal pooling ----
    # 'last'      -> use the hidden state at the final valid timestep
    # 'mean'      -> average hidden states over all valid timesteps
    # 'attention' -> learnable attention-weighted sum
    pooling: str           = "last"

    # ---- classification heads (shared bottleneck then task-specific) ----
    head_hidden_dim: int   = 32  # hidden layer inside each prediction head


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Controls the training loop, loss, and optimizer."""

    # ---- batching ----
    batch_size: int         = 32    # 16–64 depending on RAM
    num_workers: int        = 0     # 0 = main process only (safe on Windows)

    # ---- optimizer ----
    learning_rate: float    = 1e-3
    weight_decay: float     = 1e-4  # L2 regularization
    epochs: int             = 20
    early_stopping_patience: int = 5  # stop if val loss doesn't improve

    # ---- multi-task loss weights ----
    # Combined loss = loss_w_mort * L_mortality + loss_w_sepsis * L_sepsis
    loss_w_mort: float      = 1.0
    loss_w_sepsis: float    = 1.0

    # ---- class imbalance correction (pos_weight = neg_count / pos_count) ----
    # ~10% mortality  -> pos_weight ~= 9.0
    # ~15% sepsis     -> pos_weight ~= 5.7
    pos_weight_mort: float  = 9.0
    pos_weight_sepsis: float = 5.7

    # ---- device ----
    device: str             = "cpu"  # no CUDA on this machine

    # ---- checkpointing ----
    checkpoint_path: Path   = _PROCESSED / ".." / ".." / "models" / "best_gru.pt"


# ---------------------------------------------------------------------------
# Hyperparameter tuning grid
# ---------------------------------------------------------------------------

@dataclass
class TuningGrid:
    """
    Defines the search space for grid / random search.
    Each list contains the candidate values for that hyperparameter.
    """

    # --- Architecture ---
    hidden_dim: List[int]   = field(default_factory=lambda: [32, 64, 128])
    num_layers: List[int]   = field(default_factory=lambda: [1, 2])
    dropout: List[float]    = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    bidirectional: List[bool] = field(default_factory=lambda: [False, True])
    pooling: List[str]      = field(default_factory=lambda: ["last", "mean", "attention"])

    # --- Optimization ---
    learning_rate: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 3e-3])
    weight_decay: List[float]  = field(default_factory=lambda: [0.0, 1e-5, 1e-4, 1e-3])
    batch_size: List[int]      = field(default_factory=lambda: [16, 32, 64])

    # --- Multi-task loss ---
    loss_weights: List[Tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 1.0), (1.0, 0.5), (0.7, 0.3)]
    )
