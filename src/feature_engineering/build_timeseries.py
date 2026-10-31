import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import HOSPITALS_DATA_PATH, TIMESERIES_MODALITY

PROCESSED_DATA_PATH = Path("data/processed")
BIN_SIZE_HOURS = 4  # Configurable bin size

def load_vocab():
    vocab_path = PROCESSED_DATA_PATH / 'feature_schema.json'
    if not vocab_path.exists():
        raise FileNotFoundError("Vocabulary file not found. Run collect_vocab.py first.")
    with open(vocab_path, 'r') as f:
        return json.load(f)

def load_stats():
    stats_path = PROCESSED_DATA_PATH / 'timeseries_stats.json'
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            return json.load(f)
    return {}

def aggregate_other_names(group, col_name, top_k_set):
    """Returns a list of names not in top_k_set."""
    others = [str(x).strip() for x in group[col_name] if str(x).strip().lower() not in top_k_set and pd.notna(x)]
    # Unique names only
    return list(set(others))

def build_hospital_timeseries(hospital_dir: Path, vocab: dict, stats: dict) -> pd.DataFrame:
    ts_dir = hospital_dir / TIMESERIES_MODALITY
    patient_file = hospital_dir / 'tabular' / 'patient.csv'
    
    if not patient_file.exists():
        return None
        
    patient_df = pd.read_csv(patient_file)
    # Get max time per patient based on unitdischargeoffset (in minutes)
    patient_max_time = patient_df.set_index('patientunitstayid')['unitdischargeoffset'].to_dict()
    
    all_dfs = []
    
    # --- LAB ---
    print('Processing lab...')
    lab_file = ts_dir / 'lab.csv'
    if lab_file.exists():
        lab_df = pd.read_csv(lab_file)
        lab_df['hour_bin'] = lab_df['labresultoffset'] // (60 * BIN_SIZE_HOURS)
        lab_df['labname_lower'] = lab_df['labname'].astype(str).str.lower().str.strip()
        lab_df['labresult'] = pd.to_numeric(lab_df['labresult'], errors='coerce')
        lab_df = lab_df.dropna(subset=['labresult'])
        
        top_labs = set(vocab.get('lab_name', []))
        
        # Top 20 labs: mean per bin
        top_lab_df = lab_df[lab_df['labname_lower'].isin(top_labs)]
        if not top_lab_df.empty:
            lab_pivot = top_lab_df.pivot_table(
                index=['patientunitstayid', 'hour_bin'],
                columns='labname_lower',
                values='labresult',
                aggfunc='mean'
            )
            # Normalize with global stats
            for c in lab_pivot.columns:
                if c in stats:
                    m = stats[c]['mean']
                    s = stats[c]['std']
                    lab_pivot[c] = (lab_pivot[c] - m) / s
            lab_pivot = lab_pivot.add_prefix('lab_')
            all_dfs.append(lab_pivot)
            
        # Other labs: collect names
        other_lab_df = lab_df[~lab_df['labname_lower'].isin(top_labs)]
        if not other_lab_df.empty:
            other_labs = other_lab_df.drop_duplicates(['patientunitstayid', 'hour_bin', 'labname']).groupby(
                ['patientunitstayid', 'hour_bin']
            )['labname'].agg(list).rename('lab_other_names').to_frame()
            all_dfs.append(other_labs)

    # --- INFUSION DRUG ---
    print('Processing infusion...')
    inf_file = ts_dir / 'infusionDrug.csv'
    if inf_file.exists():
        inf_df = pd.read_csv(inf_file)
        inf_df['hour_bin'] = inf_df['infusionoffset'] // (60 * BIN_SIZE_HOURS)
        inf_df['drugname_lower'] = inf_df['drugname'].astype(str).str.lower().str.strip()
        
        top_infs = set(vocab.get('infusion_drug', []))
        
        # Binary flag for top 20
        top_inf_df = inf_df[inf_df['drugname_lower'].isin(top_infs)].copy()
        if not top_inf_df.empty:
            top_inf_df['presence'] = 1
            inf_pivot = top_inf_df.pivot_table(
                index=['patientunitstayid', 'hour_bin'],
                columns='drugname_lower',
                values='presence',
                aggfunc='max',
                fill_value=0
            )
            inf_pivot = inf_pivot.add_prefix('infusion_')
            all_dfs.append(inf_pivot)
            
        # Other infusions
        other_inf_df = inf_df[~inf_df['drugname_lower'].isin(top_infs)]
        if not other_inf_df.empty:
            other_infs = other_inf_df.drop_duplicates(['patientunitstayid', 'hour_bin', 'drugname']).groupby(
                ['patientunitstayid', 'hour_bin']
            )['drugname'].agg(list).rename('infusion_other_names').to_frame()
            all_dfs.append(other_infs)

    # --- TREATMENT ---
    print('Processing treatment...')
    tr_file = ts_dir / 'treatment.csv'
    if tr_file.exists():
        tr_df = pd.read_csv(tr_file)
        tr_df['hour_bin'] = tr_df['treatmentoffset'] // (60 * BIN_SIZE_HOURS)
        tr_df['category'] = tr_df['treatmentstring'].astype(str).str.split('|').str[0].str.lower().str.strip()
        
        top_trs = set(vocab.get('treatment_category', []))
        
        top_tr_df = tr_df[tr_df['category'].isin(top_trs)]
        if not top_tr_df.empty:
            tr_pivot = pd.crosstab(
                [tr_df['patientunitstayid'], tr_df['hour_bin']], 
                top_tr_df['category']
            ).clip(upper=1).add_prefix('treatment_')
            all_dfs.append(tr_pivot)
            
        other_tr_df = tr_df[~tr_df['category'].isin(top_trs)]
        if not other_tr_df.empty:
            other_trs = other_tr_df.drop_duplicates(['patientunitstayid', 'hour_bin', 'treatmentstring']).groupby(
                ['patientunitstayid', 'hour_bin']
            )['treatmentstring'].agg(list).rename('treatment_other_names').to_frame()
            all_dfs.append(other_trs)

    # --- MEDICATION ---
    print('Processing medication...')
    med_file = ts_dir / 'medication.csv'
    if med_file.exists():
        med_df = pd.read_csv(med_file)
        med_df = med_df.dropna(subset=['drugstartoffset'])
        med_df['hour_bin'] = med_df['drugstartoffset'] // (60 * BIN_SIZE_HOURS)
        med_df['drugname_lower'] = med_df['drugname'].astype(str).str.lower().str.strip()
        
        top_meds = set(vocab.get('medication_name', []))
        
        top_med_df = med_df[med_df['drugname_lower'].isin(top_meds)].copy()
        if not top_med_df.empty:
            top_med_df['presence'] = 1
            med_pivot = top_med_df.pivot_table(
                index=['patientunitstayid', 'hour_bin'],
                columns='drugname_lower',
                values='presence',
                aggfunc='max',
                fill_value=0
            ).add_prefix('med_')
            all_dfs.append(med_pivot)
            
        other_med_df = med_df[~med_df['drugname_lower'].isin(top_meds)]
        if not other_med_df.empty:
            other_meds = other_med_df.drop_duplicates(['patientunitstayid', 'hour_bin', 'drugname']).groupby(
                ['patientunitstayid', 'hour_bin']
            )['drugname'].agg(list).rename('med_other_names').to_frame()
            all_dfs.append(other_meds)
            
    # Combine everything
    print('Combining dataframes...')
    if not all_dfs:
        return None
        
    combined = pd.concat(all_dfs, axis=1, join='outer').reset_index()
    
    # Ensure a continuous grid per patient from bin 0 to their max bin
    grid_dfs = []
    for pid, p_max in patient_max_time.items():
        max_b = int(p_max // (60 * BIN_SIZE_HOURS))
        if max_b < 0: max_b = 0
        grid = pd.DataFrame({
            'patientunitstayid': pid,
            'hour_bin': np.arange(max_b + 1)
        })
        grid_dfs.append(grid)
        
    full_grid = pd.concat(grid_dfs, ignore_index=True)
    print(f'Full grid shape: {full_grid.shape}')
    
    final_df = pd.merge(full_grid, combined, on=['patientunitstayid', 'hour_bin'], how='left')
    print(f'Final df shape before imputation: {final_df.shape}')
    
    # Impute Option A: Forward-fill, then back-fill per patient
    # Only for numeric columns! "other_names" lists should be filled with empty list
    list_cols = [c for c in final_df.columns if c.endswith('_other_names')]
    num_cols = [c for c in final_df.columns if c not in list_cols and c not in ['patientunitstayid', 'hour_bin']]
    
    print('Imputing missing values...')
    if num_cols:
        for c in num_cols:
            final_df[c] = final_df.groupby('patientunitstayid')[c].transform(lambda x: x.ffill().bfill())
        final_df[num_cols] = final_df[num_cols].fillna(0.0) # For patients with absolutely no measurements
        
    for c in list_cols:
        # Convert NaN to empty list
        final_df[c] = final_df[c].apply(lambda x: x if isinstance(x, list) else [])
        
    return final_df

def main():
    vocab = load_vocab()
    stats = load_stats()
    hospitals_dir = Path(HOSPITALS_DATA_PATH)
    hospital_dirs = [d for d in hospitals_dir.iterdir() if d.is_dir() and d.name.startswith("hospital_")]
    
    print(f"Building timeseries features for {len(hospital_dirs)} hospitals...")
    
    for hosp_dir in tqdm(hospital_dirs):
        df = build_hospital_timeseries(hosp_dir, vocab, stats)
        if df is not None and not df.empty:
            out_file = hosp_dir / TIMESERIES_MODALITY / 'timeseries_features.parquet'
            df.to_parquet(out_file, index=False)

if __name__ == "__main__":
    main()
