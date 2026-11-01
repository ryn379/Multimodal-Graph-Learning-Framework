
"""
eICU Multimodal Dataset Loaders
Provides Dataset classes for:
  - ICUSequenceDataset: (12, 92) tensors for the LSTM temporal encoder
  - ICUTabularDataset: (67,) tensors for the Tabular Autoencoder
  - ICUFusedDataset: merged temporal + tabular embeddings keyed by patient ID
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm


SEED = 42
MAX_BINS = 12

TEMPORAL_FEATURE_COLS: List[str] = [
    "lab_anion gap", "lab_bedside glucose", "lab_bicarbonate",
    "lab_bun", "lab_calcium", "lab_chloride", "lab_creatinine",
    "lab_glucose", "lab_hct", "lab_hgb", "lab_mch", "lab_mchc",
    "lab_mcv", "lab_mpv", "lab_platelets x 1000", "lab_potassium",
    "lab_rbc", "lab_rdw", "lab_sodium", "lab_wbc x 1000",
    "infusion_amiodarone (mg/min)",
    "infusion_dexmedetomidine (mcg/kg/hr)",
    "infusion_fentanyl (mcg/hr)",
    "infusion_heparin (units/hr)",
    "infusion_insulin (units/hr)",
    "infusion_midazolam (mg/hr)",
    "infusion_norepinephrine (mcg/min)",
    "infusion_norepinephrine (ml/hr)",
    "infusion_ns (ml/hr)",
    "infusion_propofol (mcg/kg/min)",
    "treatment_burns/trauma",
    "treatment_cardiovascular",
    "treatment_endocrine",
    "treatment_gastrointestinal",
    "treatment_general",
    "treatment_infectious diseases",
    "treatment_neurologic",
    "treatment_pulmonary",
    "treatment_renal",
    "treatment_surgery",
    "treatment_toxicology",
    "med_fentanyl",
    "med_furosemide",
    "med_lorazepam",
    "med_potassium chloride",
    "med_zofran",
]

_LAB_COLS = TEMPORAL_FEATURE_COLS[:20]

TABULAR_FEATURE_COLS: List[str] = [
    "age", "gender_Male", "BMI", "hospitaladmitoffset", "unitvisitnumber",
    "ethnicity_Caucasian", "ethnicity_African American",
    "ethnicity_Hispanic", "ethnicity_Asian", "ethnicity_Other",
    "acutephysiologyscore", "apachescore",
    "predictedicumortality", "predictediculos",
    "aids", "cirrhosis", "diabetes", "hepaticfailure",
    "leukemia", "lymphoma", "metastaticcancer", "immunosuppression",
    "verbal", "motor", "eyes", "pao2", "fio2", "creatinine",
    "diag_cardiovascular", "diag_pulmonary", "diag_neurologic",
    "diag_renal", "diag_gastrointestinal", "diag_endocrine",
    "diag_infectious diseases", "diag_hematology", "diag_burns/trauma",
    "diag_oncology", "diag_toxicology", "diag_surgery", "diag_general",
    "diag_transplant", "diag_obstetrics/gynecology",
    "diag_genitourinary", "diag_musculoskeletal", "diag_other_count",
    "admitdrug_aspirin", "admitdrug_lisinopril", "admitdrug_lasix",
    "admitdrug_omeprazole", "admitdrug_albuterol",
    "admitdrug_levothyroxine sodium", "admitdrug_simvastatin",
    "admitdrug_gabapentin", "admitdrug_multivitamin", "admitdrug_lantus",
    "admitdrug_metformin hcl", "admitdrug_metoprolol succinate",
    "admitdrug_atorvastatin calcium", "admitdrug_amlodipine besilate",
    "admitdrug_lipitor", "admitdrug_prednisone", "admitdrug_colace",
    "admitdrug_protonix", "admitdrug_plavix", "admitdrug_norco",
    "admitdrug_other_count",
]

TABULAR_CONT_COLS: List[str] = [
    "age", "BMI", "hospitaladmitoffset",
    "acutephysiologyscore", "apachescore",
    "predictedicumortality", "predictediculos",
    "verbal", "motor", "eyes",
    "pao2", "fio2", "creatinine",
]

TARGET_COL = "hospital_expired"


def get_hospital_splits(
    processed_root: str,
    seed: int = SEED,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split available hospitals 70/15/15 into train, validation, and test."""
    root = Path(processed_root) / "hospitals"

    all_dirs = sorted(
        [p for p in root.glob("hospital_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    )

    valid_dirs = [
        p for p in all_dirs
        if (p / "tabular" / "tabular_features.parquet").exists()
        and (p / "timeseries" / "timeseries_features.parquet").exists()
    ]

    if len(valid_dirs) < 3:
        raise ValueError(
            f"Need at least 3 hospitals with both parquet files; "
            f"found {len(valid_dirs)} under {root}."
        )

    train_dirs, temp_dirs = train_test_split(
        valid_dirs, test_size=0.30, random_state=seed
    )
    val_dirs, test_dirs = train_test_split(
        temp_dirs, test_size=0.50, random_state=seed
    )

    return train_dirs, val_dirs, test_dirs


class ICUSequenceDataset(Dataset):
    """
    Each patient sample contains:
      x:      float32 array, shape (12, 92)
      mask:   bool array, shape (12,)
      y_mort: mortality label, 0 or 1
      y_seps: (if include_sepsis=True) sepsis label, 0 or 1

    With return_pid=True, __getitem__ additionally returns patientunitstayid.
    The default return format remains (x, mask, y_mort) when include_sepsis=False.
    """

    def __init__(
        self,
        hospital_dirs: List[Path],
        feature_cols: List[str] = TEMPORAL_FEATURE_COLS,
        max_bins: int = MAX_BINS,
        return_pid: bool = False,
        include_sepsis: bool = False,
        use_cache: bool = True,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.lab_cols = [c for c in _LAB_COLS if c in self.feature_cols]
        self.max_bins = max_bins
        self.input_size = len(self.feature_cols) * 2
        self.return_pid = return_pid
        self.include_sepsis = include_sepsis
        self.use_cache = use_cache

        self.samples: List[Tuple] = []
        self.patient_ids: List[int] = []

        if self.use_cache:
            loaded_cached = self._load_cached(hospital_dirs)
            if not loaded_cached:
                self._load(hospital_dirs)
        else:
            self._load(hospital_dirs)

    def _load_cached(self, hospital_dirs: List[Path]) -> bool:
        """Fast-loads pre-aligned temporal sequences and dual labels from multitask_cache."""
        cache_dirs = []
        if hospital_dirs:
            cache_dirs.append(hospital_dirs[0].parents[1] / "multitask_cache")
        cache_dirs.append(Path("processed") / "multitask_cache")

        cache_dir = None
        for cd in cache_dirs:
            if cd.exists():
                cache_dir = cd
                break

        if cache_dir is None:
            return False

        for hdir in hospital_dirs:
            cache_file = cache_dir / f"{hdir.name}.npz"
            if not cache_file.exists():
                continue

            data = np.load(cache_file)
            x_seq = data["x_seq"]
            mask = data["mask"]
            y_mort = data["y_mort"]
            y_seps = data["y_seps"]
            pids = data["pids"]

            for i in range(len(pids)):
                if self.include_sepsis:
                    self.samples.append((x_seq[i], mask[i], float(y_mort[i]), float(y_seps[i])))
                else:
                    self.samples.append((x_seq[i], mask[i], float(y_mort[i])))
                self.patient_ids.append(int(pids[i]))

        return len(self.samples) > 0

    def _load(self, hospital_dirs: List[Path]) -> None:
        sepsis_pids = set()
        if self.include_sepsis:
            try:
                from src.data.extract_sepsis_labels import extract_sepsis_patient_ids
                if hospital_dirs:
                    sepsis_pids = extract_sepsis_patient_ids(hospital_dirs[0].parent)
            except Exception:
                pass

        for hdir in tqdm(
            hospital_dirs,
            desc="Loading ICU hospitals",
            unit="hospital"
            ):
            tab_path = hdir / "tabular" / "tabular_features.parquet"
            ts_path = hdir / "timeseries" / "timeseries_features.parquet"

            tab = pd.read_parquet(tab_path)
            ts = pd.read_parquet(ts_path)

            labels = (
                tab[["patientunitstayid", TARGET_COL]]
                .dropna(subset=[TARGET_COL])
                .drop_duplicates("patientunitstayid")
            )
            label_map = labels.set_index("patientunitstayid")[TARGET_COL]

            # Use only the first 48 hours.
            ts = ts[
                ts["hour_bin"].between(0, self.max_bins - 1)
            ].copy()

            ts = ts.reindex(
                columns=["patientunitstayid", "hour_bin"] + self.feature_cols
            )

            ts[self.feature_cols] = (
                ts[self.feature_cols]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )

            for pid, grp in ts.groupby("patientunitstayid", sort=False):
                if pid not in label_map.index:
                    continue

                y = float(label_map.loc[pid])
                if y not in (0.0, 1.0):
                    continue

                grp = (
                    grp.sort_values("hour_bin")
                    .drop_duplicates("hour_bin", keep="last")
                    .set_index("hour_bin")
                )

                valid_bins = grp.index[grp.index.notna()]
                if len(valid_bins) == 0:
                    continue

                last_bin = int(valid_bins.max())
                grp = grp.reindex(range(self.max_bins))

                # Capture missingness before imputation.
                observed = grp[self.feature_cols].notna()

                # Last observation carried forward for lab values.
                available_labs = [
                    c for c in self.lab_cols if c in grp.columns
                ]
                grp[available_labs] = grp[available_labs].ffill()

                # Remaining missing values become zero.
                values = (
                    grp[self.feature_cols]
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                )
                indicators = observed.to_numpy(dtype=np.float32)

                x = np.concatenate(
                    [values, indicators], axis=1
                ).astype(np.float32)

                # Valid bins form a continuous prefix.
                mask = np.zeros(self.max_bins, dtype=bool)
                mask[:last_bin + 1] = True

                y_seps = 1.0 if int(pid) in sepsis_pids else 0.0
                if self.include_sepsis:
                    self.samples.append((x, mask, y, y_seps))
                else:
                    self.samples.append((x, mask, y))
                self.patient_ids.append(int(pid))

        if not self.samples:
            raise ValueError(
                "No patient sequences were loaded. Check data paths and labels."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        if self.include_sepsis:
            x, mask, y_mort, y_seps = sample
            result = (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.bool),
                torch.tensor(y_mort, dtype=torch.float32),
                torch.tensor(y_seps, dtype=torch.float32),
            )
        else:
            x, mask, y_mort = sample
            result = (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.bool),
                torch.tensor(y_mort, dtype=torch.float32),
            )

        if self.return_pid:
            pid = torch.tensor(self.patient_ids[idx], dtype=torch.long)
            return (*result, pid)

        return result


class ICUTabularDataset(Dataset):
    """Returns tabular features and labels; patient IDs remain in samples."""

    def __init__(
        self,
        hospital_dirs: List[Path],
        scaler: Optional[dict] = None,
        feature_cols: List[str] = TABULAR_FEATURE_COLS,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.scaler = scaler
        self.samples: List[Tuple] = []
        self._load(hospital_dirs)

    def _load(self, hospital_dirs: List[Path]) -> None:
        frames = []

        for hdir in hospital_dirs:
            tab_path = hdir / "tabular" / "tabular_features.parquet"
            frames.append(pd.read_parquet(tab_path))

        if not frames:
            raise ValueError("No tabular parquet files were loaded.")

        all_df = pd.concat(frames, ignore_index=True)
        all_df = all_df.dropna(subset=[TARGET_COL])

        feat_df = all_df.reindex(
            columns=self.feature_cols
        ).fillna(0.0)
        feat_df = feat_df.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0)

        if self.scaler is not None:
            for col, (mu, sigma) in self.scaler.items():
                if col in feat_df.columns and sigma > 0:
                    feat_df[col] = (feat_df[col] - mu) / sigma

        labels = all_df[TARGET_COL].to_numpy(dtype=np.float32)
        pids = all_df["patientunitstayid"].to_numpy()

        for i in range(len(feat_df)):
            y = float(labels[i])
            if y not in (0.0, 1.0):
                continue

            x = feat_df.iloc[i].to_numpy(dtype=np.float32)
            self.samples.append((x, pids[i], y))

        if not self.samples:
            raise ValueError(
                "No tabular samples were loaded. Check data paths and labels."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, pid, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    @staticmethod
    def fit_scaler(hospital_dirs: List[Path]) -> dict:
        """Fit continuous-column scaling statistics on training hospitals only."""
        frames = []

        for hdir in hospital_dirs:
            tab_path = hdir / "tabular" / "tabular_features.parquet"
            frames.append(pd.read_parquet(tab_path))

        if not frames:
            raise ValueError("No tabular parquet files available to fit scaler.")

        all_df = pd.concat(frames, ignore_index=True)
        scaler = {}

        for col in TABULAR_CONT_COLS:
            if col in all_df.columns:
                vals = pd.to_numeric(
                    all_df[col], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan).dropna()

                if len(vals):
                    std = float(vals.std())
                    scaler[col] = (
                        float(vals.mean()),
                        std if np.isfinite(std) else 0.0,
                    )

        return scaler
