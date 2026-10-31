RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
HOSPITALS_DATA_PATH = "data/processed/hospitals/"
MAPPING_DATA_PATH = "data/processed/mapping/"

PATIENT_ID = "patientunitstayid"
HOSPITAL_ID = "hospitalid"

# Number of rows to read per chunk when streaming large CSVs
DEFAULT_CHUNK_SIZE = 500_000

# ---------------------------------------------------------------------------
# Modality Configuration
# ---------------------------------------------------------------------------
TABULAR_MODALITY = "tabular"
TIMESERIES_MODALITY = "timeseries"

MODALITY_TABLES: dict[str, list[str]] = {
    TABULAR_MODALITY: [
        "hospital.csv",
        "patient.csv",
        "apachePatientResult.csv",
        "apachePredVar.csv",
        "admissionDrug.csv",
        "diagnosis.csv",
    ],
    TIMESERIES_MODALITY: [
        "lab.csv",
        "medication.csv",
        "infusionDrug.csv",
        "treatment.csv",
        "microLab.csv",
    ],
}