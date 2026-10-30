#!/usr/bin/env python3
"""
Extract Sepsis Ground Truth Labels for eICU Cohort
=================================================
Identifies sepsis cases using:
1. diagnosis.csv: ICD-9 codes (038.*, 995.91, 995.92, 785.52) or text diagnosis ('sepsis', 'septic').
2. patient.csv: apacheadmissiondx containing 'sepsis' or 'septic'.

Aligns extracted labels exactly with existing split patient IDs:
  - checkpoints/temporal_patient_ids_train.npy (153,831)
  - checkpoints/temporal_patient_ids_val.npy (24,186)
  - checkpoints/temporal_patient_ids_test.npy (22,809)

Outputs:
  - checkpoints/fusion/sepsis_labels_{split}.npy
  - checkpoints/fusion/sepsis_metadata.json
"""

import glob
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "processed" / "hospitals"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FUSION_DIR = CHECKPOINTS_DIR / "fusion"
FUSION_DIR.mkdir(parents=True, exist_ok=True)


def extract_sepsis_patient_ids(hospitals_root: Path) -> set[int]:
    """Scans all hospital subdirectories for sepsis diagnoses."""
    hosp_dirs = sorted(glob.glob(str(hospitals_root / "hospital_*")))
    sepsis_pids = set()

    for h in hosp_dirs:
        # 1. diagnosis.csv
        diag_path = os.path.join(h, "tabular", "diagnosis.csv")
        if os.path.exists(diag_path):
            try:
                df_diag = pd.read_csv(
                    diag_path,
                    usecols=["patientunitstayid", "diagnosisstring", "icd9code"],
                    dtype={"patientunitstayid": "Int64", "diagnosisstring": "str", "icd9code": "str"},
                )
                s_diag = df_diag["diagnosisstring"].fillna("").astype(str)
                s_icd = df_diag["icd9code"].fillna("").astype(str)
                m_diag = s_diag.str.contains("seps|septic", case=False, na=False)
                m_icd = s_icd.str.startswith(("038", "995.91", "995.92", "785.52"))
                pids = df_diag.loc[m_diag | m_icd, "patientunitstayid"].dropna()
                sepsis_pids.update(pids.astype(int).tolist())
            except Exception:
                pass

        # 2. patient.csv
        pat_path = os.path.join(h, "tabular", "patient.csv")
        if os.path.exists(pat_path):
            try:
                df_pat = pd.read_csv(
                    pat_path,
                    usecols=["patientunitstayid", "apacheadmissiondx"],
                    dtype={"patientunitstayid": "Int64", "apacheadmissiondx": "str"},
                )
                s_pat = df_pat["apacheadmissiondx"].fillna("").astype(str)
                m_pat = s_pat.str.contains("seps|septic", case=False, na=False)
                pids = df_pat.loc[m_pat, "patientunitstayid"].dropna()
                sepsis_pids.update(pids.astype(int).tolist())
            except Exception:
                pass

    return sepsis_pids


def main():
    print(f"Scanning eICU hospitals under {PROCESSED_ROOT}...")
    sepsis_pids = extract_sepsis_patient_ids(PROCESSED_ROOT)
    print(f"Total unique sepsis patient IDs identified: {len(sepsis_pids):,}")

    stats = {}
    for split in ["train", "val", "test"]:
        pid_file = CHECKPOINTS_DIR / f"temporal_patient_ids_{split}.npy"
        if not pid_file.exists():
            raise FileNotFoundError(f"Missing {pid_file}")
        
        pids = np.load(pid_file)
        labels = np.array([1.0 if int(pid) in sepsis_pids else 0.0 for pid in pids], dtype=np.float32)
        
        out_file = FUSION_DIR / f"sepsis_labels_{split}.npy"
        np.save(out_file, labels)
        
        pos = int(labels.sum())
        total = len(labels)
        stats[split] = {
            "total_patients": total,
            "positive_sepsis": pos,
            "negative_non_sepsis": total - pos,
            "prevalence_percent": round(pos / total * 100, 2),
            "output_path": str(out_file),
        }
        print(f"[{split.upper()}] Saved {out_file.name}: {total:,} patients | Sepsis: {pos:,} ({stats[split]['prevalence_percent']}%)")

    meta_file = FUSION_DIR / "sepsis_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Metadata saved to {meta_file}")


if __name__ == "__main__":
    main()
