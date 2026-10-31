"""
preprocessing.py — Target Extraction & Feature Schema Alignment
================================================================
Two main responsibilities:

  1. BUILD THE TARGET MANIFEST
     Scan every hospital directory, extract:
       - hospital_expired  (mortality)    from tabular_features.parquet
       - has_sepsis        (sepsis)       from patient.csv + diagnosis.csv
     Save a single flat file -> data/processed/patient_targets.parquet

  2. CANONICAL FEATURE ALIGNMENT
     Different hospitals may have slightly different columns in their
     timeseries_features.parquet (sparse presence columns vary).
     get_canonical_feature_names() returns a fixed ordered list.
     align_timeseries_df() drops list-columns and pads missing numeric columns.

Run this module directly to regenerate the target manifest:
    python -m src.GRU_timeseries.preprocessing
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running as __main__ from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.GRU_timeseries.config import DataConfig


# ---------------------------------------------------------------------------
# 1. Target Extraction
# ---------------------------------------------------------------------------

def _extract_mortality(tabular_path: Path) -> Optional[pd.DataFrame]:
    """
    Load tabular_features.parquet and return a DataFrame with:
      patientunitstayid | hospital_expired (float 0/1)
    Returns None if the file doesn't exist.
    """
    if not tabular_path.exists():
        return None
    df = pd.read_parquet(tabular_path, columns=["patientunitstayid", "hospital_expired"])
    return df.rename(columns={"hospital_expired": "mortality"})


def _extract_sepsis(patient_csv: Path, diagnosis_csv: Path) -> pd.DataFrame:
    """
    Derive a sepsis label per patient from two sources:
      A) patient.csv  -> 'apacheadmissiondx' column (admission diagnosis string)
      B) diagnosis.csv -> 'diagnosisstring' column  (all diagnoses recorded during stay)

    A patient is considered sepsis-positive (label = 1) if the word 'sepsis'
    appears in ANY of their diagnosis records or their admission diagnosis.

    Returns a DataFrame: patientunitstayid | has_sepsis (float 0/1)
    """
    sepsis_pids = set()

    # ── Source A: admission diagnosis from patient.csv ────────────────────
    if patient_csv.exists():
        pt = pd.read_csv(patient_csv, usecols=lambda c: c in
                         ["patientunitstayid", "apacheadmissiondx"])
        if "apacheadmissiondx" in pt.columns:
            mask = pt["apacheadmissiondx"].astype(str).str.contains(
                "sepsis", case=False, na=False
            )
            sepsis_pids.update(pt.loc[mask, "patientunitstayid"].tolist())

    # ── Source B: all diagnosis records ──────────────────────────────────
    if diagnosis_csv.exists():
        try:
            diag = pd.read_csv(diagnosis_csv, usecols=lambda c: c in
                               ["patientunitstayid", "diagnosisstring"])
            if "diagnosisstring" in diag.columns:
                mask = diag["diagnosisstring"].astype(str).str.contains(
                    "sepsis", case=False, na=False
                )
                sepsis_pids.update(diag.loc[mask, "patientunitstayid"].tolist())
        except Exception:
            pass  # ignore empty / malformed diagnosis files

    return pd.DataFrame({"patientunitstayid": list(sepsis_pids), "has_sepsis": 1.0})


def extract_and_cache_targets(
    hospitals_dir: Path,
    output_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Walk every hospital_* directory, extract mortality + sepsis targets,
    and save a unified manifest to `output_path`.

    Returns the combined DataFrame.

    Columns in the manifest:
        patientunitstayid (int)
        hospital_id       (int)
        mortality         (float 0/1)
        has_sepsis        (float 0/1)
    """
    hospital_dirs = sorted(
        [d for d in hospitals_dir.iterdir() if d.is_dir() and d.name.startswith("hospital_")]
    )

    all_records: List[pd.DataFrame] = []

    it = tqdm(hospital_dirs, desc="Extracting targets", disable=not verbose)
    for h_dir in it:
        hospital_id = int(h_dir.name.split("_")[1])

        # ── mortality ─────────────────────────────────────────────────────
        tabular_path = h_dir / "tabular" / "tabular_features.parquet"
        mort_df = _extract_mortality(tabular_path)
        if mort_df is None:
            continue   # can't do anything without patient list

        # ── sepsis ────────────────────────────────────────────────────────
        patient_csv = h_dir / "tabular" / "patient.csv"
        diagnosis_csv = h_dir / "tabular" / "diagnosis.csv"
        sepsis_df = _extract_sepsis(patient_csv, diagnosis_csv)

        # ── merge ─────────────────────────────────────────────────────────
        merged = mort_df.merge(sepsis_df, on="patientunitstayid", how="left")
        merged["has_sepsis"] = merged["has_sepsis"].fillna(0.0)
        merged["hospital_id"] = hospital_id

        all_records.append(merged)

    if not all_records:
        raise RuntimeError("No hospital directories contained valid tabular data.")

    targets = pd.concat(all_records, ignore_index=True)
    targets = targets[["patientunitstayid", "hospital_id", "mortality", "has_sepsis"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(output_path, index=False)

    if verbose:
        n = len(targets)
        mort_rate   = targets["mortality"].mean()
        sepsis_rate = targets["has_sepsis"].mean()
        print(f"\n[OK] Target manifest saved -> {output_path}")
        print(f"   Total patients : {n:,}")
        print(f"   Mortality rate : {mort_rate:.2%}  (pos_weight ~= {(1 - mort_rate)/mort_rate:.1f})")
        print(f"   Sepsis rate    : {sepsis_rate:.2%}  (pos_weight ~= {(1 - sepsis_rate)/sepsis_rate:.1f})")

    return targets


# ---------------------------------------------------------------------------
# 2. Canonical Feature Schema
# ---------------------------------------------------------------------------

# These are the 3 "list" columns produced by build_timeseries.py.
# They contain Python lists of drug/lab names -> NOT usable as model inputs.
_LIST_COLUMNS = {"lab_other_names", "infusion_other_names", "med_other_names"}


def get_canonical_feature_names(feature_schema_path: Path) -> List[str]:
    """
    Build a fixed, alphabetically-sorted list of numeric time-series feature
    column names from the feature schema, excluding list columns.

    This guarantees that every patient's feature tensor has the same
    column ordering regardless of which hospital's parquet file is loaded.

    Returns a sorted list of column name strings.
    """
    with open(feature_schema_path, "r") as f:
        schema = json.load(f)

    canonical = set()

    # Lab features: prefixed with 'lab_'
    for lab in schema.get("lab_name", []):
        col = f"lab_{lab}"
        if col not in _LIST_COLUMNS:
            canonical.add(col)

    # Infusion features: prefixed with 'infusion_'
    for drug in schema.get("infusion_drug", []):
        col = f"infusion_{drug}"
        if col not in _LIST_COLUMNS:
            canonical.add(col)

    # Treatment features: prefixed with 'treatment_'
    for cat in schema.get("treatment_category", []):
        col = f"treatment_{cat}"
        if col not in _LIST_COLUMNS:
            canonical.add(col)

    # Medication features: prefixed with 'med_'
    for med in schema.get("medication_name", []):
        col = f"med_{med}"
        if col not in _LIST_COLUMNS:
            canonical.add(col)

    return sorted(canonical)   # stable alphabetical order


def align_timeseries_df(df: pd.DataFrame, canonical_cols: List[str]) -> pd.DataFrame:
    """
    Align a hospital-specific timeseries DataFrame to the canonical feature list:
      1. Drop any list-type columns ('*_other_names').
      2. Add any missing canonical columns as zeros.
      3. Reorder columns to match canonical_cols exactly.

    The output DataFrame has ONLY the canonical numeric columns
    (does NOT include patientunitstayid / hour_bin — those are indices).

    Args:
        df:             timeseries_features.parquet as loaded by pandas.
        canonical_cols: sorted list from get_canonical_feature_names().

    Returns:
        Aligned DataFrame with columns = canonical_cols.
    """
    # Drop list-type columns if present
    cols_to_drop = [c for c in df.columns if c in _LIST_COLUMNS]
    df = df.drop(columns=cols_to_drop)

    # Add any missing canonical columns with 0.0
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[canonical_cols]


# ---------------------------------------------------------------------------
# 3. Hospital-level train / val / test split  (hospital-level -> no leakage)
# ---------------------------------------------------------------------------

def make_hospital_splits(
    hospitals_dir: Path,
    output_path: Path,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    random_seed: int  = 42,
) -> dict:
    """
    Randomly assign each hospital to train / val / test by hospital_id.
    Saves and returns a dict: {'train': [...], 'val': [...], 'test': [...]}.

    Hospital-level splits prevent the model from memorizing hospital-specific
    equipment calibrations or coding practices.
    """
    import random

    hospital_dirs = sorted(
        [d for d in hospitals_dir.iterdir() if d.is_dir() and d.name.startswith("hospital_")]
    )
    hospital_ids = [int(d.name.split("_")[1]) for d in hospital_dirs]

    rng = random.Random(random_seed)
    rng.shuffle(hospital_ids)

    n = len(hospital_ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    splits = {
        "train": hospital_ids[:n_train],
        "val":   hospital_ids[n_train: n_train + n_val],
        "test":  hospital_ids[n_train + n_val:],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Hospital splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


# ---------------------------------------------------------------------------
# Entry-point: run preprocessing standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = DataConfig()

    print("=" * 60)
    print("Step 1 — Building patient target manifest")
    print("=" * 60)
    targets = extract_and_cache_targets(cfg.hospitals_dir, cfg.target_manifest_path)

    print("\n" + "=" * 60)
    print("Step 2 — Verifying canonical feature list")
    print("=" * 60)
    canonical = get_canonical_feature_names(cfg.feature_schema_path)
    print(f"   Canonical features : {len(canonical)}")
    print(f"   First 5            : {canonical[:5]}")
    print(f"   Last 5             : {canonical[-5:]}")

    print("\n" + "=" * 60)
    print("Step 3 — Generating hospital train/val/test splits")
    print("=" * 60)
    if not cfg.hospital_splits_path.exists():
        make_hospital_splits(
            cfg.hospitals_dir,
            cfg.hospital_splits_path,
            cfg.train_frac,
            cfg.val_frac,
            cfg.random_seed,
        )
    else:
        print(f"   Already exists: {cfg.hospital_splits_path}")
