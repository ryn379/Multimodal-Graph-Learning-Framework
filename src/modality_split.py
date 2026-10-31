"""
modality_split.py
------------------
Divide partitioned eICU datasets within each hospital folder into two modalities:
1. tabular    - static demographics, hospital metadata, admission scores, baseline profile
2. timeseries - longitudinal lab tests, medications, infusions, treatments, microLab

Supported Actions
-----------------
- Partition (move / copy):
    python -m src.modality_split               # default: fast in-place move (< 1s)
    python -m src.modality_split --mode copy  # duplicate files into subfolders

- Verify:
    python -m src.modality_split --verify-only

- Revert:
    python -m src.modality_split --revert      # restore flat hospital structure
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make sure imports work when called as a module from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    HOSPITALS_DATA_PATH,
    MODALITY_TABLES,
    TABULAR_MODALITY,
    TIMESERIES_MODALITY,
)

# Reference row counts verified after hospital partitioning
EXPECTED_ROW_COUNTS: dict[str, int] = {
    "admissionDrug.csv": 874_920,
    "apachePatientResult.csv": 297_064,
    "apachePredVar.csv": 171_177,
    "diagnosis.csv": 2_710_672,
    "hospital.csv": 208,
    "infusionDrug.csv": 4_803_719,
    "lab.csv": 39_132_531,
    "medication.csv": 16_277_967,
    "microLab.csv": 16_996,
    "patient.csv": 200_859,
    "treatment.csv": 3_688_745,
}
TOTAL_EXPECTED_ROWS = sum(EXPECTED_ROW_COUNTS.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_hospital_dirs(base_path: Path) -> list[Path]:
    """Return sorted list of hospital_* directory paths."""
    if not base_path.exists():
        raise FileNotFoundError(f"Hospitals directory not found: {base_path}")
    dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("hospital_")]
    dirs.sort(key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else d.name)
    return dirs


def fast_count_csv_rows(path: Path) -> int:
    """
    Count data rows (excluding header) using fast binary buffer streaming.
    Handles files with or without trailing newlines accurately.
    """
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        first_line = f.readline()
        if not first_line:
            return 0  # completely empty file
        lines = 0
        last_byte = b""
        buf_size = 1024 * 1024
        while chunk := f.read(buf_size):
            lines += chunk.count(b"\n")
            last_byte = chunk[-1:]
        # If the file had content after the header and didn't end in newline
        if last_byte and last_byte != b"\n":
            lines += 1
        return lines


# ---------------------------------------------------------------------------
# Core Operations
# ---------------------------------------------------------------------------

def split_modalities(hospitals_dir: Path, mode: str = "move") -> None:
    """
    Divide datasets in each hospital directory into tabular/ and timeseries/.
    
    Parameters
    ----------
    hospitals_dir : Path
        Path to data/processed/hospitals/
    mode : str
        'move' (atomic in-place rename, instant) or 'copy' (duplicates files)
    """
    hosp_dirs = _get_hospital_dirs(hospitals_dir)
    print(f"\nDividing {len(hosp_dirs)} hospital directories into modalities (mode: {mode.upper()})...")

    transferred = 0
    skipped = 0

    for hosp_dir in tqdm(hosp_dirs, desc="Partitioning modalities", unit="hospital"):
        for modality, filenames in MODALITY_TABLES.items():
            modality_dir = hosp_dir / modality
            modality_dir.mkdir(parents=True, exist_ok=True)

            for filename in filenames:
                flat_file = hosp_dir / filename
                target_file = modality_dir / filename

                if flat_file.exists():
                    if mode == "move":
                        # If destination already exists (e.g. partial run), replace it
                        if target_file.exists():
                            target_file.unlink()
                        flat_file.rename(target_file)
                    elif mode == "copy":
                        shutil.copy2(flat_file, target_file)
                    transferred += 1
                elif target_file.exists():
                    # Already placed in modality subdirectory
                    skipped += 1
                else:
                    print(f"Warning: Missing file {filename} in {hosp_dir.name}")

    print(f"\nDone! Files transferred: {transferred}, Already in place: {skipped}")


def revert_modalities(hospitals_dir: Path) -> None:
    """
    Revert tabular/ and timeseries/ subdirectories back into a flat hospital directory.
    """
    hosp_dirs = _get_hospital_dirs(hospitals_dir)
    print(f"\nReverting modalities for {len(hosp_dirs)} hospitals to flat structure...")

    reverted = 0
    for hosp_dir in tqdm(hosp_dirs, desc="Reverting", unit="hospital"):
        for modality in [TABULAR_MODALITY, TIMESERIES_MODALITY]:
            modality_dir = hosp_dir / modality
            if not modality_dir.exists():
                continue

            for f in list(modality_dir.glob("*.csv")):
                target = hosp_dir / f.name
                if target.exists():
                    target.unlink()
                f.rename(target)
                reverted += 1

            # Clean up empty modality directory
            try:
                modality_dir.rmdir()
            except OSError:
                pass

    print(f"\nDone! Reverted {reverted} files to flat hospital directories.")


def verify_modalities(hospitals_dir: Path, check_rows: bool = True) -> bool:
    """
    Verify the modality partition structure and row counts across all hospitals.
    """
    hosp_dirs = _get_hospital_dirs(hospitals_dir)
    print(f"\n--- Verifying Modality Partition ({len(hosp_dirs)} hospitals) ---")

    all_ok = True
    missing_dirs = 0
    missing_files = 0
    counted_rows: dict[str, int] = {fname: 0 for fname in EXPECTED_ROW_COUNTS}

    print("Checking directory layout and files...")
    for hosp_dir in tqdm(hosp_dirs, desc="Checking structure", unit="hospital"):
        for modality, filenames in MODALITY_TABLES.items():
            modality_dir = hosp_dir / modality
            if not modality_dir.exists():
                missing_dirs += 1
                all_ok = False
                continue

            for filename in filenames:
                file_path = modality_dir / filename
                if not file_path.exists():
                    missing_files += 1
                    all_ok = False
                elif check_rows:
                    counted_rows[filename] += fast_count_csv_rows(file_path)

    if missing_dirs > 0:
        print(f"FAILED: {missing_dirs} modality directories missing!")
    if missing_files > 0:
        print(f"FAILED: {missing_files} table files missing!")

    if not check_rows:
        if all_ok:
            print("Structure verification PASSED (all directories & files present).")
        return all_ok

    print("\n--- Row Count Verification ---")
    row_count_ok = True
    header_str = f"{'Modality':<12} {'Dataset':<26} {'Expected Rows':>15} {'Partitioned Rows':>17} {'Match':>8}"
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))

    for modality, filenames in MODALITY_TABLES.items():
        for fname in filenames:
            exp = EXPECTED_ROW_COUNTS.get(fname, 0)
            actual = counted_rows.get(fname, 0)
            match_str = "OK" if exp == actual else "MISMATCH"
            if exp != actual:
                row_count_ok = False
                all_ok = False
            print(f"{modality:<12} {fname:<26} {exp:>15,d} {actual:>17,d} {match_str:>8}")

    print("-" * len(header_str))
    total_actual = sum(counted_rows.values())
    total_match = "ALL OK" if total_actual == TOTAL_EXPECTED_ROWS and row_count_ok else "MISMATCH"
    print(f"{'Total':<39} {TOTAL_EXPECTED_ROWS:>15,d} {total_actual:>17,d} {total_match:>8}")
    print("-" * len(header_str))

    if all_ok:
        print("\nAll datasets and row counts MATCH 100%! Modality split is fully verified.\n")
    else:
        print("\nVerification found discrepancies! Review the table above.\n")

    return all_ok


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Divide partitioned hospital datasets into tabular and timeseries modalities."
    )
    parser.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="move",
        help="Operation mode: 'move' (atomic in-place rename, instant, default) or 'copy' (duplicate files)",
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Revert tabular/ and timeseries/ subdirectories back to flat hospital structure",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run verification without modifying files",
    )
    parser.add_argument(
        "--skip-row-count",
        action="store_true",
        help="Skip full row-count checking during verification (faster)",
    )
    parser.add_argument(
        "--hospitals-dir",
        type=str,
        default=HOSPITALS_DATA_PATH,
        help=f"Path to hospitals directory (default: {HOSPITALS_DATA_PATH})",
    )

    args = parser.parse_args()
    hospitals_dir = Path(args.hospitals_dir)
    start_time = time.time()

    if args.revert:
        revert_modalities(hospitals_dir)
    elif args.verify_only:
        verify_modalities(hospitals_dir, check_rows=not args.skip_row_count)
    else:
        split_modalities(hospitals_dir, mode=args.mode)
        verify_modalities(hospitals_dir, check_rows=not args.skip_row_count)

    elapsed = time.time() - start_time
    print(f"Total time elapsed: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
