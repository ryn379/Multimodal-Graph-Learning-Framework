"""
partition_by_hospital.py
------------------------
Stream and partition all raw eICU datasets by hospitalid.

Strategy
--------
1. Load patient.csv (small, ~46 MB) fully into memory to build a dict:
       patientunitstayid -> hospitalid
2. Create one directory per hospital under HOSPITALS_DATA_PATH.
3. For each source CSV (except hospital.csv):
   - Read in chunks of DEFAULT_CHUNK_SIZE rows.
   - Map each chunk's patientunitstayid -> hospitalid.
   - Group by hospitalid and append each group to the matching hospital CSV.
4. medication.csv, medication_imputed.csv, medication_imputed1.csv are all
   streamed into a single merged medication.csv per hospital folder.
5. For hospitals that have no rows in a particular table (e.g. microLab),
   an empty CSV file with the correct headers is written so downstream code
   never raises FileNotFoundError.
6. After all files are written a row-count verification runs automatically.

Usage (run from project root)
------------------------------
    python -m src.partition_by_hospital
    python -m src.partition_by_hospital --chunk-size 200000
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make sure imports work when called as a module from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEFAULT_CHUNK_SIZE,
    HOSPITAL_ID,
    HOSPITALS_DATA_PATH,
    MAPPING_DATA_PATH,
    PATIENT_ID,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
# Files that will be merged into a SINGLE output CSV per hospital.
# The value is the common output file name.
MERGED_GROUPS: dict[str, list[str]] = {
    "medication.csv": [
        "medication.csv",
        "medication_imputed.csv",
        "medication_imputed1.csv",
    ],
}

# Datasets that should be handled individually (not merged).
# hospital.csv and patient.csv are treated separately, so exclude them here.
SKIP_FILES = {
    "hospital.csv",
    "patient.csv",
    "medication.csv",           # handled via MERGED_GROUPS (merged with imputed files)
    "medication_imputed.csv",   # handled via MERGED_GROUPS
    "medication_imputed1.csv",  # handled via MERGED_GROUPS
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hospital_dir(hospital_id: int, base: str = HOSPITALS_DATA_PATH) -> Path:
    """Return the Path object for a specific hospital directory."""
    return Path(base) / f"hospital_{hospital_id}"


def _csv_path(hospital_id: int, filename: str, base: str = HOSPITALS_DATA_PATH) -> Path:
    """Return the full path for a partitioned CSV file."""
    return _hospital_dir(hospital_id, base) / filename


def _ensure_dir(path: Path) -> None:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _write_or_append(df: pd.DataFrame, path: Path, *, header: bool) -> None:
    """Write a DataFrame to CSV, appending if already exists."""
    _ensure_dir(path.parent)   # create hospital dir on-the-fly if needed
    df.to_csv(path, mode="a" if path.exists() else "w", index=False, header=header)


# ---------------------------------------------------------------------------
# Step 1 – Build mapping
# ---------------------------------------------------------------------------

def build_patient_hospital_mapping(raw_path: str = RAW_DATA_PATH) -> dict[int, int]:
    """
    Read patient.csv and build a dict: patientunitstayid -> hospitalid.
    Also persists the mapping to MAPPING_DATA_PATH/patient_to_hospital.csv.
    """
    print("\n[1/6] Building patientunitstayid -> hospitalid mapping …")
    patient_csv = os.path.join(raw_path, "patient.csv")
    df = pd.read_csv(patient_csv, usecols=[PATIENT_ID, HOSPITAL_ID])

    # Persist mapping
    mapping_dir = Path(MAPPING_DATA_PATH)
    _ensure_dir(mapping_dir)
    mapping_out = mapping_dir / "patient_to_hospital.csv"
    df.to_csv(mapping_out, index=False)
    print(f"    Mapping saved -> {mapping_out}")
    print(f"    Total patient stays : {len(df):,}")
    print(f"    Unique hospitals    : {df[HOSPITAL_ID].nunique():,}")

    return dict(zip(df[PATIENT_ID].astype(int), df[HOSPITAL_ID].astype(int)))


# ---------------------------------------------------------------------------
# Step 2 – Create hospital directories
# ---------------------------------------------------------------------------

def setup_hospital_directories(
    hospital_ids: list[int],
    raw_path: str = RAW_DATA_PATH,
    base: str = HOSPITALS_DATA_PATH,
) -> None:
    """
    Create one directory per hospital and copy hospital metadata (hospital.csv row)
    into each folder.
    """
    print("\n[2/6] Creating hospital directories and writing metadata …")
    hosp_df = pd.read_csv(os.path.join(raw_path, "hospital.csv"))

    for hid in tqdm(hospital_ids, desc="   dirs", unit="hospital"):
        hdir = _hospital_dir(hid, base)
        _ensure_dir(hdir)

        # Write hospital metadata (single-row CSV)
        h_row = hosp_df[hosp_df[HOSPITAL_ID] == hid]
        h_row.to_csv(hdir / "hospital.csv", index=False)

    print(f"    Created {len(hospital_ids)} hospital directories under {base}")


# ---------------------------------------------------------------------------
# Step 3 – Partition patient.csv
# ---------------------------------------------------------------------------

def partition_patient(
    mapping: dict[int, int],
    raw_path: str = RAW_DATA_PATH,
    base: str = HOSPITALS_DATA_PATH,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Partition patient.csv by hospitalid (patient.csv already contains hospitalid)."""
    print("\n[3/6] Partitioning patient.csv …")
    patient_csv = os.path.join(raw_path, "patient.csv")
    total_rows = sum(1 for _ in open(patient_csv, "rb")) - 1  # minus header
    written: dict[int, bool] = {}  # tracks whether header was already written

    with tqdm(total=total_rows, desc="   patient.csv", unit="rows", unit_scale=True) as pbar:
        for chunk in pd.read_csv(patient_csv, chunksize=chunksize):
            for hid, group in chunk.groupby(HOSPITAL_ID):
                out = _csv_path(hid, "patient.csv", base)
                first_write = hid not in written
                _write_or_append(group, out, header=first_write)
                written[hid] = False
            pbar.update(len(chunk))

    print(f"    patient.csv partitioned into {len(written)} hospital folders")


# ---------------------------------------------------------------------------
# Step 4a – Generic single-file partitioning
# ---------------------------------------------------------------------------

def partition_single_file(
    filename: str,
    mapping: dict[int, int],
    hospital_ids: list[int],
    raw_path: str = RAW_DATA_PATH,
    base: str = HOSPITALS_DATA_PATH,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """
    Stream a single CSV in chunks, map patientunitstayid -> hospitalid,
    group and append to hospital folders. Retains original schema (no hospitalid col added).
    """
    src = os.path.join(raw_path, filename)
    if not os.path.exists(src):
        print(f"    [WARN] {filename} not found, skipping.")
        return

    # Read header to know columns
    header_cols = pd.read_csv(src, nrows=0).columns.tolist()
    # Track first-write per hospital to control header emission
    written: dict[int, bool] = {}

    total_rows = sum(1 for _ in open(src, "rb")) - 1

    with tqdm(total=total_rows, desc=f"   {filename}", unit="rows", unit_scale=True) as pbar:
        for chunk in pd.read_csv(src, chunksize=chunksize):
            # Map patientunitstayid -> hospitalid
            chunk["_hid"] = chunk[PATIENT_ID].map(mapping)

            for hid, group in chunk.groupby("_hid"):
                group = group.drop(columns=["_hid"])  # drop temp column
                out = _csv_path(int(hid), filename, base)
                first_write = hid not in written
                _write_or_append(group, out, header=first_write)
                written[hid] = False

            pbar.update(len(chunk))

    # Write empty CSVs (with header only) for hospitals that had 0 rows
    missing = set(hospital_ids) - set(written.keys())
    if missing:
        empty_df = pd.DataFrame(columns=header_cols)
        for hid in missing:
            out = _csv_path(hid, filename, base)
            _ensure_dir(out.parent)   # dir may not exist for very sparse datasets
            empty_df.to_csv(out, index=False)

    print(f"    {filename}: {len(written)} hospitals had data, {len(missing)} got empty stubs")


# ---------------------------------------------------------------------------
# Step 4b – Merged medication partitioning
# ---------------------------------------------------------------------------

def partition_merged_medication(
    output_filename: str,
    source_files: list[str],
    mapping: dict[int, int],
    hospital_ids: list[int],
    raw_path: str = RAW_DATA_PATH,
    base: str = HOSPITALS_DATA_PATH,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """
    Stream multiple source CSVs sequentially and append their data into a single
    output_filename per hospital folder. Original schema is retained (no hospitalid col added).
    """
    print(f"\n   Merging into '{output_filename}': {source_files}")

    header_cols: list[str] | None = None
    written: dict[int, bool] = {}

    for src_filename in source_files:
        src = os.path.join(raw_path, src_filename)
        if not os.path.exists(src):
            print(f"    [WARN] {src_filename} not found, skipping.")
            continue

        if header_cols is None:
            header_cols = pd.read_csv(src, nrows=0).columns.tolist()

        total_rows = sum(1 for _ in open(src, "rb")) - 1

        with tqdm(total=total_rows, desc=f"   {src_filename}", unit="rows", unit_scale=True) as pbar:
            for chunk in pd.read_csv(src, chunksize=chunksize):
                chunk["_hid"] = chunk[PATIENT_ID].map(mapping)

                for hid, group in chunk.groupby("_hid"):
                    group = group.drop(columns=["_hid"])
                    out = _csv_path(int(hid), output_filename, base)
                    first_write = hid not in written
                    _write_or_append(group, out, header=first_write)
                    written[hid] = False

                pbar.update(len(chunk))

    # Write empty stubs for hospitals with no medication rows
    missing = set(hospital_ids) - set(written.keys())
    if missing and header_cols:
        empty_df = pd.DataFrame(columns=header_cols)
        for hid in missing:
            out = _csv_path(hid, output_filename, base)
            _ensure_dir(out.parent)   # dir may not exist for very sparse datasets
            empty_df.to_csv(out, index=False)

    print(f"    '{output_filename}': {len(written)} hospitals had data, {len(missing)} got empty stubs")


# ---------------------------------------------------------------------------
# Step 5 – Verify row counts
# ---------------------------------------------------------------------------

def verify_integrity(
    hospital_ids: list[int],
    raw_path: str = RAW_DATA_PATH,
    base: str = HOSPITALS_DATA_PATH,
) -> bool:
    """
    Compare total rows in raw CSVs vs sum of rows across all hospital partitions.
    Returns True if all counts match (ignoring headers).
    """
    print("\n[6/6] Verifying row-count integrity …")

    # Build canonical list of files to verify (using output filenames)
    raw_files = sorted(
        f for f in os.listdir(raw_path)
        if f.endswith(".csv") and f not in {"hospital.csv", "medication_imputed.csv", "medication_imputed1.csv"}
    )

    all_ok = True
    print(f"\n{'File':<32} {'Raw rows':>12} {'Partitioned':>12} {'Match':>6}")
    print("-" * 65)

    for filename in raw_files:
        src = os.path.join(raw_path, filename)
        raw_count = sum(1 for _ in open(src, "rb")) - 1  # minus header

        # For medication, the partitioned file merges 3 sources
        # raw_count is just medication.csv; we need the combined count separately
        if filename == "medication.csv":
            # Build combined raw count across all three files
            combined = 0
            for med_file in ["medication.csv", "medication_imputed.csv", "medication_imputed1.csv"]:
                med_src = os.path.join(raw_path, med_file)
                if os.path.exists(med_src):
                    combined += sum(1 for _ in open(med_src, "rb")) - 1
            raw_count = combined

        part_count = 0
        for hid in hospital_ids:
            part_file = _csv_path(hid, filename, base)
            if part_file.exists():
                with open(part_file, "rb") as f:
                    rows = sum(1 for _ in f) - 1   # minus header
                    part_count += max(rows, 0)

        match = "OK" if raw_count == part_count else "!! MISMATCH"
        if raw_count != part_count:
            all_ok = False

        print(f"{filename:<32} {raw_count:>12,} {part_count:>12,} {match:>6}")

    if all_ok:
        print("\n[VERIFIED] All row counts match! Partitioning is complete and verified.")
    else:
        print("\n[FAILED]   Some row counts did NOT match. Review the mismatches above.")

    return all_ok


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_partition(chunksize: int = DEFAULT_CHUNK_SIZE) -> None:
    start_time = time.time()

    print("=" * 65)
    print("  eICU Dataset Partitioner — by Hospital")
    print("=" * 65)

    # ------------------------------------------------------------------ #
    # Step 1: Build mapping
    # ------------------------------------------------------------------ #
    mapping = build_patient_hospital_mapping(RAW_DATA_PATH)
    hospital_ids = sorted(set(mapping.values()))

    # ------------------------------------------------------------------ #
    # Step 2: Create directories + hospital metadata
    # ------------------------------------------------------------------ #
    setup_hospital_directories(hospital_ids, RAW_DATA_PATH, HOSPITALS_DATA_PATH)

    # ------------------------------------------------------------------ #
    # Step 3: Partition patient.csv
    # ------------------------------------------------------------------ #
    partition_patient(mapping, RAW_DATA_PATH, HOSPITALS_DATA_PATH, chunksize)

    # ------------------------------------------------------------------ #
    # Step 4: Partition all other files
    # ------------------------------------------------------------------ #
    print("\n[4/6] Partitioning individual datasets …")

    # Build flat list of raw CSV files to process individually
    all_raw = sorted(
        f for f in os.listdir(RAW_DATA_PATH)
        if f.endswith(".csv") and f not in SKIP_FILES
    )

    for filename in all_raw:
        partition_single_file(filename, mapping, hospital_ids, RAW_DATA_PATH, HOSPITALS_DATA_PATH, chunksize)

    # ------------------------------------------------------------------ #
    # Step 5: Merge medication files
    # ------------------------------------------------------------------ #
    print("\n[5/6] Merging medication datasets …")
    for output_name, sources in MERGED_GROUPS.items():
        partition_merged_medication(
            output_name,
            sources,
            mapping,
            hospital_ids,
            RAW_DATA_PATH,
            HOSPITALS_DATA_PATH,
            chunksize,
        )

    # ------------------------------------------------------------------ #
    # Step 6: Verify
    # ------------------------------------------------------------------ #
    verify_integrity(hospital_ids, RAW_DATA_PATH, HOSPITALS_DATA_PATH)

    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    print(f"\nTotal time: {m}m {s}s")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Partition eICU raw CSVs by hospital into data/processed/hospitals/."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per streaming chunk (default: {DEFAULT_CHUNK_SIZE:,})",
    )
    args = parser.parse_args()
    run_partition(chunksize=args.chunk_size)
