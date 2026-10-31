import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
TOP_K = 20

def get_top_categories(file_path: Path, column: str, chunksize: int = 500_000, process_fn=None) -> list:
    print(f"Collecting top {TOP_K} for {column} in {file_path.name}...")
    value_counts = pd.Series(dtype=int)
    
    if not file_path.exists():
        print(f"File {file_path} not found.")
        return []

    for chunk in tqdm(pd.read_csv(file_path, usecols=[column], chunksize=chunksize, dtype=str)):
        s = chunk[column].dropna()
        if process_fn:
            s = s.apply(process_fn)
        # Convert to lowercase to avoid case issues
        s = s.str.lower().str.strip()
        counts = s.value_counts()
        value_counts = value_counts.add(counts, fill_value=0)
        
    top_k_items = value_counts.sort_values(ascending=False).head(TOP_K).index.tolist()
    return top_k_items

def main():
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    vocab = {}
    
    # 1. Lab names
    vocab['lab_name'] = get_top_categories(RAW_DATA_PATH / 'lab.csv', 'labname')
    
    # 2. Infusion drugs
    vocab['infusion_drug'] = get_top_categories(RAW_DATA_PATH / 'infusionDrug.csv', 'drugname')
    
    # 3. Treatments (top-level category)
    vocab['treatment_category'] = get_top_categories(
        RAW_DATA_PATH / 'treatment.csv', 
        'treatmentstring', 
        process_fn=lambda x: x.split('|')[0] if isinstance(x, str) else x
    )
    
    # 4. Diagnosis (top-level category)
    vocab['diagnosis_category'] = get_top_categories(
        RAW_DATA_PATH / 'diagnosis.csv', 
        'diagnosisstring', 
        process_fn=lambda x: x.split('|')[0] if isinstance(x, str) else x
    )
    
    # 5. Medication names
    vocab['medication_name'] = get_top_categories(RAW_DATA_PATH / 'medication.csv', 'drugname')
    
    # 6. Admission Drug names
    vocab['admission_drug_name'] = get_top_categories(RAW_DATA_PATH / 'admissionDrug.csv', 'drugname')
    
    # Save vocabulary
    vocab_file = PROCESSED_DATA_PATH / 'feature_schema.json'
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=2)
        
    print(f"Vocabulary saved to {vocab_file}")
    
    # Calculate global tabular stats (mean, std) for numeric columns
    print("Computing global stats for top labs...")
    lab_stats = {}
    if (RAW_DATA_PATH / 'lab.csv').exists():
        top_labs = set(vocab['lab_name'])
        sum_val = {lab: 0.0 for lab in top_labs}
        sum_sq = {lab: 0.0 for lab in top_labs}
        count_val = {lab: 0 for lab in top_labs}
        
        for chunk in tqdm(pd.read_csv(RAW_DATA_PATH / 'lab.csv', usecols=['labname', 'labresult'], chunksize=500_000)):
            chunk['labname'] = chunk['labname'].str.lower().str.strip()
            chunk = chunk[chunk['labname'].isin(top_labs)].dropna(subset=['labresult'])
            chunk['labresult'] = pd.to_numeric(chunk['labresult'], errors='coerce')
            chunk = chunk.dropna(subset=['labresult'])
            
            for lab in top_labs:
                lab_data = chunk[chunk['labname'] == lab]['labresult']
                if len(lab_data) > 0:
                    sum_val[lab] += lab_data.sum()
                    sum_sq[lab] += (lab_data ** 2).sum()
                    count_val[lab] += len(lab_data)
                    
        for lab in top_labs:
            if count_val[lab] > 0:
                mean = sum_val[lab] / count_val[lab]
                variance = (sum_sq[lab] / count_val[lab]) - (mean ** 2)
                std = variance ** 0.5 if variance > 0 else 1.0
                if std == 0: std = 1.0
                lab_stats[lab] = {'mean': mean, 'std': std}
                
    with open(PROCESSED_DATA_PATH / 'timeseries_stats.json', 'w') as f:
        json.dump(lab_stats, f, indent=2)
    print("Timeseries global stats saved.")

if __name__ == "__main__":
    main()
