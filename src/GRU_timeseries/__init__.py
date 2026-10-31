# GRU_timeseries package
# ========================
# A modular PyTorch GRU pipeline for the eICU time-series data.
#
# Predicts:
#   1. In-Hospital Mortality (binary classification)
#   2. Sepsis Diagnosis      (binary classification)
#
# Also produces:
#   - Patient Temporal Embeddings (dense fixed-size vector per ICU stay)
#     for downstream multimodal fusion with the tabular encoder.
#
# Modules:
#   config.py              -- All hyperparameters and path configs (edit here first!)
#   preprocessing.py       -- Target extraction (mortality + sepsis) and schema alignment
#   dataset.py             -- PyTorch Dataset and batch collation
#   model.py               -- TemporalGRU model definition
#   train.py               -- Training loop with metrics logging and checkpointing
#   tune.py                -- Hyperparameter grid search
#   extract_embeddings.py  -- Export temporal embeddings for all patients

from .model import TemporalGRU
from .config import DataConfig, ModelConfig, TrainConfig, TuningGrid
