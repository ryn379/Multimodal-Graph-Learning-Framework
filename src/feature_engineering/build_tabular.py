import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import HOSPITALS_DATA_PATH, TABULAR_MODALITY

PROCESSED_DATA_PATH = Path("data/processed")

def load_vocab():
    vocab_path = PROCESSED_DATA_PATH / 'feature_schema.json'
    if not vocab_path.exists():
        raise FileNotFoundError("Vocabulary file not found. Run collect_vocab.py first.")
    with open(vocab_path, 'r') as f:
        return json.load(f)

def build_hospital_tabular(hospital_dir: Path, vocab: dict) -> pd.DataFrame:
    tabular_dir = hospital_dir / TABULAR_MODALITY
    
    # 1. Load patient
    patient_file = tabular_dir / 'patient.csv'
    if not patient_file.exists():
        return None
    patient_df = pd.read_csv(patient_file)
    
    # Process age, gender, ethnicity
    patient_df['age'] = pd.to_numeric(patient_df['age'].replace('> 89', 90), errors='coerce')
    patient_df['gender_Male'] = (patient_df['gender'] == 'Male').astype(float)
    
    # Simplistic ethnicity one-hot (assuming Caucasian, African American, Hispanic, Asian, Other)
    ethnicities = ['Caucasian', 'African American', 'Hispanic', 'Asian']
    for eth in ethnicities:
        patient_df[f'ethnicity_{eth}'] = (patient_df['ethnicity'] == eth).astype(float)
    patient_df['ethnicity_Other'] = (~patient_df['ethnicity'].isin(ethnicities)).astype(float)
    
    # BMI
    patient_df['admissionheight'] = pd.to_numeric(patient_df['admissionheight'], errors='coerce')
    patient_df['admissionweight'] = pd.to_numeric(patient_df['admissionweight'], errors='coerce')
    patient_df['BMI'] = patient_df['admissionweight'] / ((patient_df['admissionheight'] / 100) ** 2)
    
    # Outcomes and context
    patient_df['hospital_expired'] = (patient_df['hospitaldischargestatus'] == 'Expired').astype(float)
    patient_df['icu_expired'] = (patient_df['unitdischargestatus'] == 'Expired').astype(float)
    
    # Keep baseline cols
    cols_to_keep = ['patientunitstayid', 'age', 'gender_Male', 'BMI', 
                    'hospitaladmitoffset', 'unitvisitnumber',
                    'hospitaldischargeoffset', 'unitdischargeoffset',
                    'hospital_expired', 'icu_expired'] + \
                   [f'ethnicity_{eth}' for eth in ethnicities] + ['ethnicity_Other']
    
    df_out = patient_df[cols_to_keep].copy()
    
    # 2. APACHE Result
    apache_res = tabular_dir / 'apachePatientResult.csv'
    if apache_res.exists():
        res_df = pd.read_csv(apache_res)
        res_df = res_df[res_df['apacheversion'] == 'IV'] # take one version to avoid duplicates
        res_df = res_df[['patientunitstayid', 'acutephysiologyscore', 'apachescore', 
                         'predictedicumortality', 'predictediculos']].groupby('patientunitstayid').first().reset_index()
        df_out = df_out.merge(res_df, on='patientunitstayid', how='left')
        
    # 3. APACHE Pred Var
    apache_pred = tabular_dir / 'apachePredVar.csv'
    if apache_pred.exists():
        pred_df = pd.read_csv(apache_pred)
        pred_cols = ['patientunitstayid', 'aids', 'cirrhosis', 'diabetes', 'hepaticfailure', 
                     'leukemia', 'lymphoma', 'metastaticcancer', 'immunosuppression',
                     'verbal', 'motor', 'eyes', 'pao2', 'fio2', 'creatinine']
        pred_df = pred_df[[c for c in pred_cols if c in pred_df.columns]].groupby('patientunitstayid').first().reset_index()
        df_out = df_out.merge(pred_df, on='patientunitstayid', how='left')
        
    # 4. Diagnosis
    diag_file = tabular_dir / 'diagnosis.csv'
    if diag_file.exists():
        diag_df = pd.read_csv(diag_file)
        diag_df['category'] = diag_df['diagnosisstring'].str.split('|').str[0].str.lower().str.strip()
        top_diags = set(vocab.get('diagnosis_category', []))
        
        # Create dummy vars
        diag_dummies = pd.get_dummies(diag_df['category']).groupby(diag_df['patientunitstayid']).max().astype(float)
        for cat in top_diags:
            if cat in diag_dummies.columns:
                df_out[f'diag_{cat}'] = df_out['patientunitstayid'].map(diag_dummies[cat]).fillna(0.0)
            else:
                df_out[f'diag_{cat}'] = 0.0
                
        # "Other" count
        other_mask = ~diag_df['category'].isin(top_diags)
        other_counts = diag_df[other_mask].groupby('patientunitstayid').size()
        df_out['diag_other_count'] = df_out['patientunitstayid'].map(other_counts).fillna(0.0)

    # 5. Admission Drug
    drug_file = tabular_dir / 'admissionDrug.csv'
    if drug_file.exists():
        drug_df = pd.read_csv(drug_file)
        drug_df['drugname'] = drug_df['drugname'].astype(str).str.lower().str.strip()
        top_drugs = set(vocab.get('admission_drug_name', []))
        
        # Create dummy vars
        drug_dummies = pd.get_dummies(drug_df['drugname']).groupby(drug_df['patientunitstayid']).max().astype(float)
        for drug in top_drugs:
            if drug in drug_dummies.columns:
                df_out[f'admitdrug_{drug}'] = df_out['patientunitstayid'].map(drug_dummies[drug]).fillna(0.0)
            else:
                df_out[f'admitdrug_{drug}'] = 0.0
                
        # "Other" count
        other_mask = ~drug_df['drugname'].isin(top_drugs)
        other_counts = drug_df[other_mask].groupby('patientunitstayid').size()
        df_out['admitdrug_other_count'] = df_out['patientunitstayid'].map(other_counts).fillna(0.0)
        
    return df_out

def main():
    vocab = load_vocab()
    hospitals_dir = Path(HOSPITALS_DATA_PATH)
    hospital_dirs = [d for d in hospitals_dir.iterdir() if d.is_dir() and d.name.startswith("hospital_")]
    
    print(f"Building tabular features for {len(hospital_dirs)} hospitals...")
    
    # To collect global stats later if needed
    for hosp_dir in tqdm(hospital_dirs):
        df = build_hospital_tabular(hosp_dir, vocab)
        if df is not None:
            # Fill NAs in features (except targets which shouldn't have NAs, but just in case)
            df = df.fillna(0) # Simple imputation for missing features
            out_file = hosp_dir / TABULAR_MODALITY / 'tabular_features.parquet'
            df.to_parquet(out_file, index=False)

if __name__ == "__main__":
    main()
