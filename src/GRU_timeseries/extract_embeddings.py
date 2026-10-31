"""
extract_embeddings.py — Export Patient Temporal Embeddings
==========================================================
Loads the best trained GRU checkpoint and runs inference across all
patient stays (or a specified subset of hospitals), saving a unified
patient embedding file for downstream multimodal fusion.

Output file: data/processed/patient_temporal_embeddings.parquet
  Columns: patientunitstayid, emb_0, emb_1, ..., emb_{H-1}

Usage:
    # Export embeddings for all hospitals
    python -m src.GRU_timeseries.extract_embeddings

    # Export for a subset of hospitals (faster for testing)
    python -m src.GRU_timeseries.extract_embeddings --sample-hospitals 20

    # Specify a custom checkpoint
    python -m src.GRU_timeseries.extract_embeddings --checkpoint models/best_gru.pt
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.GRU_timeseries.config import DataConfig, ModelConfig, TrainConfig
from src.GRU_timeseries.dataset import EICUTimeseriesDataset, pad_collate_fn
from src.GRU_timeseries.model import TemporalGRU
from src.GRU_timeseries.preprocessing import (
    get_canonical_feature_names,
    extract_and_cache_targets,
)


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cpu") -> tuple:
    """
    Load a TemporalGRU from a saved checkpoint file.

    Returns:
        (model, model_cfg, canonical_cols)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}\n"
            "Run training first: python -m src.GRU_timeseries.train"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model config from saved dict
    model_cfg_dict = ckpt["model_cfg"]
    model_cfg = ModelConfig(**{k: v for k, v in model_cfg_dict.items()
                               if hasattr(ModelConfig, k) or k in ModelConfig.__dataclass_fields__})

    # Rebuild model and load weights
    model = TemporalGRU(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)

    canonical_cols = ckpt.get("canonical_cols", None)

    print(f"[OK] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"   Validation mean AUROC at save: {ckpt.get('val_metrics', {}).get('val_mean_auroc', 'N/A')}")

    return model, model_cfg, canonical_cols


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: TemporalGRU,
    hospital_ids: List[int],
    data_cfg: DataConfig,
    canonical_cols: List[str],
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Run model in inference mode over all patient stays for the given hospitals.

    Returns a DataFrame with patientunitstayid and embedding columns.
    """
    dataset = EICUTimeseriesDataset(hospital_ids, data_cfg, canonical_cols)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=0,
    )

    all_pids   = []
    all_embeds = []

    for x, lengths, _targets, pids in tqdm(loader, desc="Extracting embeddings"):
        embeddings = model.get_embedding(x, lengths)  # (B, H)
        all_pids.extend(pids.numpy().tolist())
        all_embeds.append(embeddings.cpu().numpy())

    all_embeds_arr = np.concatenate(all_embeds, axis=0)   # (N, H)
    emb_dim = all_embeds_arr.shape[1]

    # Build DataFrame: pid + one column per embedding dimension
    emb_cols = {f"emb_{i}": all_embeds_arr[:, i] for i in range(emb_dim)}
    df = pd.DataFrame({"patientunitstayid": all_pids, **emb_cols})

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_extraction(
    checkpoint_path: Path   = None,
    output_path: Path       = None,
    sample_hospitals: int   = None,
):
    """
    Full pipeline:
      1. Load checkpoint
      2. Discover all hospital IDs (or sample subset)
      3. Extract embeddings
      4. Save to parquet
    """
    data_cfg = DataConfig()

    checkpoint_path = checkpoint_path or Path(
        __file__
    ).resolve().parent.parent.parent / "models" / "best_gru.pt"

    output_path = output_path or (
        data_cfg.hospitals_dir.parent / "patient_temporal_embeddings.parquet"
    )

    device = "cpu"

    print(f"\n{'='*60}")
    print(f"Patient Temporal Embedding Extraction")
    print(f"{'='*60}")

    # ── Ensure targets manifest exists (needed for Dataset __init__) ──────
    if not data_cfg.target_manifest_path.exists():
        print("Target manifest missing — building it now...")
        extract_and_cache_targets(data_cfg.hospitals_dir, data_cfg.target_manifest_path)

    # ── Load model ────────────────────────────────────────────────────────
    model, model_cfg, canonical_cols = load_model_from_checkpoint(checkpoint_path, device)

    if canonical_cols is None:
        canonical_cols = get_canonical_feature_names(data_cfg.feature_schema_path)

    print(f"Embedding dimension : {model.gru_out_dim}")
    print(f"Canonical features  : {len(canonical_cols)}")

    # ── Discover hospital IDs ─────────────────────────────────────────────
    hospital_dirs = sorted(
        [d for d in data_cfg.hospitals_dir.iterdir()
         if d.is_dir() and d.name.startswith("hospital_")]
    )
    all_hospital_ids = [int(d.name.split("_")[1]) for d in hospital_dirs]

    if sample_hospitals is not None:
        all_hospital_ids = all_hospital_ids[:sample_hospitals]

    print(f"Processing {len(all_hospital_ids)} hospitals...")

    # ── Extract ────────────────────────────────────────────────────────────
    df = extract_embeddings(model, all_hospital_ids, data_cfg, canonical_cols)

    # ── Save ───────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\n[OK] Embeddings saved to: {output_path}")
    print(f"   Shape : {df.shape}  ({len(df):,} patients × {model.gru_out_dim} dimensions)")
    print(f"\nThese embeddings are ready for multimodal fusion with the tabular encoder!")
    print(f"Merge on 'patientunitstayid' with your tabular encoder output.")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract patient temporal embeddings from a trained GRU."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pt checkpoint (default: models/best_gru.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save output parquet (default: data/processed/patient_temporal_embeddings.parquet)",
    )
    parser.add_argument(
        "--sample-hospitals",
        type=int,
        default=None,
        help="Only process N hospitals (for faster testing).",
    )
    args = parser.parse_args()

    run_extraction(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        sample_hospitals=args.sample_hospitals,
    )
