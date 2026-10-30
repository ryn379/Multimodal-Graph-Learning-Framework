#!/usr/bin/env python3
"""
Pre-extract and Cache Hospital-Level Multimodal Multi-Task Tensors
================================================================
Aligns:
  - Temporal sequences (12, 92) [46 features + 46 missingness indicators]
  - Tabular clinical features (67,)
  - Joint Targets:
      y_mortality: in-hospital mortality (0 or 1)
      y_sepsis: ICU sepsis from diagnosis / APACHE (0 or 1)
  - Split protocol strictly following processed/hospital_split.json:
      Train: 145 hospitals
      Val:   31 hospitals
      Test:  31 hospitals

Outputs:
  - processed/multitask_cache/hospital_*.npz
  - checkpoints/fusion/multitask/multitask_scaler.json
  - checkpoints/fusion/multitask/split_summary.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.dataset import (
    MAX_BINS,
    TABULAR_CONT_COLS,
    TABULAR_FEATURE_COLS,
    TEMPORAL_FEATURE_COLS,
    TARGET_COL,
    _LAB_COLS,
)
from src.data.extract_sepsis_labels import extract_sepsis_patient_ids

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "processed"
HOSPITALS_ROOT = PROCESSED_ROOT / "hospitals"
CACHE_DIR = PROCESSED_ROOT / "multitask_cache"
OUT_DIR = PROJECT_ROOT / "checkpoints" / "fusion" / "multitask"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_training_scaler(train_hospital_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std on continuous tabular features strictly on training hospitals."""
    print("Fitting tabular continuous scaler on 145 training hospitals...")
    frames = []
    for hname in train_hospital_names:
        p = HOSPITALS_ROOT / hname / "tabular" / "tabular_features.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["patientunitstayid", TARGET_COL] + [c for c in TABULAR_CONT_COLS])
            df = df.dropna(subset=[TARGET_COL]).drop_duplicates("patientunitstayid")
            frames.append(df[TABULAR_CONT_COLS])

    all_df = pd.concat(frames, ignore_index=True)
    scaler = {}
    for col in TABULAR_CONT_COLS:
        vals = pd.to_numeric(all_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        mu = float(vals.mean()) if len(vals) else 0.0
        sigma = float(vals.std()) if len(vals) else 1.0
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = 1.0
        scaler[col] = (mu, sigma)

    print(f"Fitted scaler for {len(scaler)} continuous columns.")
    return scaler


def process_hospital(
    hname: str,
    sepsis_pids: Set[int],
    scaler: Dict[str, Tuple[float, float]],
) -> Dict[str, np.ndarray] | None:
    """Extract and align multimodal features and dual labels for a single hospital."""
    hdir = HOSPITALS_ROOT / hname
    tab_p = hdir / "tabular" / "tabular_features.parquet"
    ts_p = hdir / "timeseries" / "timeseries_features.parquet"

    if not tab_p.exists() or not ts_p.exists():
        return None

    # 1. Tabular features & targets
    tab = pd.read_parquet(tab_p)
    tab = tab.dropna(subset=[TARGET_COL]).drop_duplicates("patientunitstayid")
    tab_pids = set(tab["patientunitstayid"].astype(int).unique())

    # 2. Timeseries filtering (0 to MAX_BINS - 1)
    ts = pd.read_parquet(ts_p)
    ts = ts[ts["hour_bin"].between(0, MAX_BINS - 1)].copy()
    ts = ts[ts["patientunitstayid"].isin(tab_pids)]
    
    # Standardize columns
    ts = ts.reindex(columns=["patientunitstayid", "hour_bin"] + TEMPORAL_FEATURE_COLS)
    ts[TEMPORAL_FEATURE_COLS] = (
        ts[TEMPORAL_FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    # 3. Align sequences
    seqs, masks, valid_pids = [], [], []
    for pid, grp in ts.groupby("patientunitstayid", sort=False):
        grp = grp.sort_values("hour_bin").drop_duplicates("hour_bin", keep="last").set_index("hour_bin")
        valid_bins = grp.index[grp.index.notna()]
        if len(valid_bins) == 0:
            continue
        last_bin = int(valid_bins.max())
        grp = grp.reindex(range(MAX_BINS))
        observed = grp[TEMPORAL_FEATURE_COLS].notna()
        grp[_LAB_COLS] = grp[_LAB_COLS].ffill()
        vals = grp[TEMPORAL_FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
        ind = observed.to_numpy(dtype=np.float32)
        x = np.concatenate([vals, ind], axis=1).astype(np.float32)
        mask = np.zeros(MAX_BINS, dtype=bool)
        mask[:last_bin + 1] = True
        seqs.append(x)
        masks.append(mask)
        valid_pids.append(int(pid))

    if not valid_pids:
        return None

    # Filter tabular to aligned pids in identical order
    tab = tab.set_index("patientunitstayid").reindex(valid_pids)
    y_mort = tab[TARGET_COL].to_numpy(dtype=np.float32)
    y_seps = np.array([1.0 if pid in sepsis_pids else 0.0 for pid in valid_pids], dtype=np.float32)

    # Standardize tabular features
    feat_df = tab.reindex(columns=TABULAR_FEATURE_COLS).fillna(0.0)
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for col, (mu, sigma) in scaler.items():
        if col in feat_df.columns and sigma > 0:
            feat_df[col] = (feat_df[col] - mu) / sigma

    x_tab = feat_df.to_numpy(dtype=np.float32)
    x_seq = np.asarray(seqs, dtype=np.float32)
    mask = np.asarray(masks, dtype=bool)
    pids = np.asarray(valid_pids, dtype=np.int64)

    return {
        "x_seq": x_seq,
        "mask": mask,
        "x_tab": x_tab,
        "y_mort": y_mort,
        "y_seps": y_seps,
        "pids": pids,
        "hospital_id": hname,
    }


def main():
    with open(PROCESSED_ROOT / "hospital_split.json") as f:
        splits = json.load(f)

    sepsis_pids = extract_sepsis_patient_ids(HOSPITALS_ROOT)
    print(f"Loaded {len(sepsis_pids):,} total unique sepsis patients.")

    # 1. Fit scaler strictly on training hospitals
    scaler = fit_training_scaler(splits["train"])
    with open(OUT_DIR / "multitask_scaler.json", "w") as f:
        json.dump({k: [float(v[0]), float(v[1])] for k, v in scaler.items()}, f, indent=2)

    # 2. Extract and cache each hospital
    summary = {}
    for split_name in ["train", "val", "test"]:
        h_list = splits[split_name]
        total_pts = 0
        total_mort = 0
        total_seps = 0
        cached_hospitals = []

        for hname in tqdm(h_list, desc=f"Caching {split_name.upper()} hospitals"):
            cache_file = CACHE_DIR / f"{hname}.npz"
            if cache_file.exists():
                data = np.load(cache_file)
                total_pts += len(data["pids"])
                total_mort += int(data["y_mort"].sum())
                total_seps += int(data["y_seps"].sum())
                cached_hospitals.append(hname)
                continue

            res = process_hospital(hname, sepsis_pids, scaler)
            if res is not None:
                np.savez_compressed(cache_file, **res)
                total_pts += len(res["pids"])
                total_mort += int(res["y_mort"].sum())
                total_seps += int(res["y_seps"].sum())
                cached_hospitals.append(hname)

        pos_w_mort = (total_pts - total_mort) / max(total_mort, 1)
        pos_w_seps = (total_pts - total_seps) / max(total_seps, 1)
        summary[split_name] = {
            "hospitals": len(cached_hospitals),
            "hospital_names": cached_hospitals,
            "total_patients": total_pts,
            "mortality_positives": total_mort,
            "mortality_prevalence": round(total_mort / total_pts * 100, 2),
            "mortality_pos_weight": round(pos_w_mort, 4),
            "sepsis_positives": total_seps,
            "sepsis_prevalence": round(total_seps / total_pts * 100, 2),
            "sepsis_pos_weight": round(pos_w_seps, 4),
        }
        print(
            f"[{split_name.upper()}] {len(cached_hospitals)} hospitals | {total_pts:,} patients | "
            f"Mort: {total_mort:,} ({summary[split_name]['mortality_prevalence']}%, w={pos_w_mort:.2f}) | "
            f"Seps: {total_seps:,} ({summary[split_name]['sepsis_prevalence']}%, w={pos_w_seps:.2f})"
        )

    with open(OUT_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Multi-task caching and split summary complete!")


if __name__ == "__main__":
    main()
