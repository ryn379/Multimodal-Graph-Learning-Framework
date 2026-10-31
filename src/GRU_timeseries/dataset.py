"""
dataset.py — PyTorch Dataset and Batch Collation
=================================================
EICUTimeseriesDataset:
  - Loads pre-built timeseries_features.parquet and patient_targets.parquet.
  - Truncates sequences to max_seq_len (default = 18 bins = 72 hours).
  - Returns per-patient (sequence_tensor, seq_len, targets, patient_id).

pad_collate_fn:
  - Pads variable-length sequences to the max length in the batch.
  - Returns a packed-sequence-friendly (x, lengths, targets) tuple.

Run module directly for a shape-check on a small sample:
    python -m src.GRU_timeseries.dataset
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.GRU_timeseries.config import DataConfig
from src.GRU_timeseries.preprocessing import (
    get_canonical_feature_names,
    align_timeseries_df,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EICUTimeseriesDataset(Dataset):
    """
    Loads eICU time-series data for a specific set of hospitals (train/val/test).

    Each sample = one ICU patient stay.

    __getitem__ returns:
        sequence  (torch.Tensor)  shape: (T, D_in)  where T ≤ max_seq_len
        seq_len   (int)           actual number of valid timesteps
        targets   (torch.Tensor)  shape: (2,)  [mortality, has_sepsis]
        pid       (int)           patientunitstayid
    """

    def __init__(
        self,
        hospital_ids: List[int],
        cfg: DataConfig,
        canonical_cols: Optional[List[str]] = None,
    ):
        """
        Args:
            hospital_ids:   IDs of hospitals in this split (train, val, or test).
            cfg:            DataConfig with paths and max_seq_len.
            canonical_cols: Pre-computed canonical feature list (avoids reloading
                            the schema repeatedly across train/val/test datasets).
        """
        super().__init__()
        self.cfg = cfg
        self.max_seq_len = cfg.max_seq_len

        # Load canonical feature order once
        if canonical_cols is None:
            canonical_cols = get_canonical_feature_names(cfg.feature_schema_path)
        self.canonical_cols = canonical_cols

        # Load the global target manifest (all hospitals, all patients)
        if not cfg.target_manifest_path.exists():
            raise FileNotFoundError(
                f"Target manifest not found at {cfg.target_manifest_path}.\n"
                "Run: python -m src.GRU_timeseries.preprocessing"
            )
        targets_df = pd.read_parquet(cfg.target_manifest_path)

        # Filter to only patients from the specified hospitals
        targets_df = targets_df[targets_df["hospital_id"].isin(hospital_ids)].copy()
        targets_df = targets_df.reset_index(drop=True)

        # Build per-patient index: store (hospital_id, patientunitstayid, mortality, has_sepsis)
        self.records = targets_df[
            ["patientunitstayid", "hospital_id", "mortality", "has_sepsis"]
        ].to_dict("records")

        # Cache timeseries dataframes per hospital to avoid re-reading parquet on every __getitem__
        # Only load when first accessed (lazy loading per hospital)
        self._hospital_cache: dict = {}

    def _load_hospital_ts(self, hospital_id: int) -> pd.DataFrame:
        """
        Load and cache the timeseries DataFrame for one hospital.
        Columns: patientunitstayid, hour_bin, <canonical_cols>
        """
        if hospital_id in self._hospital_cache:
            return self._hospital_cache[hospital_id]

        ts_path = (
            self.cfg.hospitals_dir
            / f"hospital_{hospital_id}"
            / "timeseries"
            / "timeseries_features.parquet"
        )
        if not ts_path.exists():
            self._hospital_cache[hospital_id] = pd.DataFrame()
            return self._hospital_cache[hospital_id]

        df = pd.read_parquet(ts_path)
        # Keep only identifier columns + numeric features
        df = df[["patientunitstayid", "hour_bin"]].join(
            align_timeseries_df(df, self.canonical_cols)
        )
        self._hospital_cache[hospital_id] = df
        return df

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
        rec = self.records[idx]
        pid        = int(rec["patientunitstayid"])
        hosp_id    = int(rec["hospital_id"])
        mortality  = float(rec["mortality"])
        has_sepsis = float(rec["has_sepsis"])

        # ── Fetch this patient's rows from the hospital timeseries ─────────
        hosp_df = self._load_hospital_ts(hosp_id)

        if hosp_df.empty:
            # No timeseries data -> return a zero sequence of length 1
            seq_arr = np.zeros((1, len(self.canonical_cols)), dtype=np.float32)
        else:
            pat_df = hosp_df[hosp_df["patientunitstayid"] == pid].sort_values("hour_bin")

            # ── Truncate to observation window (first 72 hours = 18 bins) ──
            pat_df = pat_df[pat_df["hour_bin"] < self.max_seq_len]

            if pat_df.empty:
                seq_arr = np.zeros((1, len(self.canonical_cols)), dtype=np.float32)
            else:
                seq_arr = pat_df[self.canonical_cols].values.astype(np.float32)

        # Replace any remaining NaNs (shouldn't happen after preprocessing, but defensive)
        seq_arr = np.nan_to_num(seq_arr, nan=0.0)

        seq_len = seq_arr.shape[0]  # actual number of valid timesteps

        sequence = torch.tensor(seq_arr, dtype=torch.float32)               # (T, D)
        targets  = torch.tensor([mortality, has_sepsis], dtype=torch.float32)  # (2,)

        return sequence, seq_len, targets, pid


# ---------------------------------------------------------------------------
# Collation function
# ---------------------------------------------------------------------------

def pad_collate_fn(
    batch: List[Tuple[torch.Tensor, int, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Pads sequences of different lengths to the longest in the batch,
    then packs them via pack_padded_sequence in the model.

    Args:
        batch: list of (sequence, seq_len, targets, pid) tuples.

    Returns:
        x        (Tensor) shape: (B, T_max, D_in) — padded input sequences
        lengths  (Tensor) shape: (B,) — actual sequence lengths (unpadded)
        targets  (Tensor) shape: (B, 2) — [mortality, sepsis] labels
        pids     (Tensor) shape: (B,) — patient IDs for traceability
    """
    sequences, lengths, targets, pids = zip(*batch)

    # Pad all sequences to the max length in this batch
    # pad_sequence expects list of (T_i, D) tensors, returns (T_max, B, D)
    x_padded = pad_sequence(list(sequences), batch_first=True, padding_value=0.0)
    # -> shape: (B, T_max, D_in)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    targets_tensor = torch.stack(list(targets), dim=0)    # (B, 2)
    pids_tensor    = torch.tensor(pids, dtype=torch.long)

    return x_padded, lengths_tensor, targets_tensor, pids_tensor


# ---------------------------------------------------------------------------
# Helper: build train / val / test DataLoaders in one call
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: DataConfig, train_cfg) -> dict:
    """
    Read hospital_splits.json and build DataLoaders for train, val, and test.

    Args:
        cfg:       DataConfig
        train_cfg: TrainConfig (needs batch_size and num_workers)

    Returns:
        dict with keys 'train', 'val', 'test' -> DataLoader objects.
    """
    if not cfg.hospital_splits_path.exists():
        from src.GRU_timeseries.preprocessing import make_hospital_splits
        make_hospital_splits(
            cfg.hospitals_dir,
            cfg.hospital_splits_path,
            cfg.train_frac,
            cfg.val_frac,
            cfg.random_seed,
        )

    with open(cfg.hospital_splits_path) as f:
        splits = json.load(f)

    # Pre-compute canonical cols once and share across all splits
    canonical_cols = get_canonical_feature_names(cfg.feature_schema_path)

    loaders = {}
    for split_name, hospital_ids in splits.items():
        dataset = EICUTimeseriesDataset(hospital_ids, cfg, canonical_cols)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=(split_name == "train"),
            collate_fn=pad_collate_fn,
            num_workers=train_cfg.num_workers,
        )
        print(f"  {split_name:5s}: {len(dataset):>7,} patients | {len(loaders[split_name]):>4} batches")

    return loaders


# ---------------------------------------------------------------------------
# Standalone shape check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    cfg = DataConfig()
    canonical_cols = get_canonical_feature_names(cfg.feature_schema_path)
    print(f"Canonical feature count: {len(canonical_cols)}")

    # Use just one small hospital for the shape check
    hospital_dirs = sorted(
        [d for d in cfg.hospitals_dir.iterdir() if d.is_dir() and d.name.startswith("hospital_")]
    )
    sample_hosp_id = int(hospital_dirs[0].name.split("_")[1])

    # Targets manifest must exist
    if not cfg.target_manifest_path.exists():
        print("Target manifest missing. Running preprocessing first...")
        from src.GRU_timeseries.preprocessing import extract_and_cache_targets
        extract_and_cache_targets(cfg.hospitals_dir, cfg.target_manifest_path)

    dataset = EICUTimeseriesDataset([sample_hosp_id], cfg, canonical_cols)
    print(f"Dataset size (1 hospital): {len(dataset)}")

    # Peek at one sample
    seq, seq_len, tgt, pid = dataset[0]
    print(f"\nSample 0:")
    print(f"  patientunitstayid : {pid}")
    print(f"  sequence shape    : {seq.shape}   (T={seq_len}, D={seq.shape[-1]})")
    print(f"  targets           : mortality={tgt[0].item()}, sepsis={tgt[1].item()}")

    # Build a mini DataLoader and check batch shapes
    from torch.utils.data import DataLoader
    from src.GRU_timeseries.config import TrainConfig
    train_cfg = TrainConfig(batch_size=4)
    loader = DataLoader(dataset, batch_size=4, collate_fn=pad_collate_fn, shuffle=False)
    x, lengths, targets, pids = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  x       : {tuple(x.shape)}   (B, T_max, D_in)")
    print(f"  lengths : {tuple(lengths.shape)}")
    print(f"  targets : {tuple(targets.shape)}")
    print(f"  pids    : {tuple(pids.shape)}")
    print("\n[OK] Dataset and collation working correctly.")
