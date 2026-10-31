import argparse
import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--hospital-id", type=int, help="Build for a single hospital")
    parser.add_argument("--verify-only", action="store_true", help="Verify feature shapes")
    args = parser.parse_args()
    
    scripts_dir = Path(__file__).resolve().parent
    
    if args.verify_only:
        print("Verification logic here...")
        return
        
    # Step 1: Collect Vocab if needed
    vocab_path = Path("data/processed/feature_schema.json")
    if not vocab_path.exists():
        print("Running collect_vocab.py...")
        subprocess.run([sys.executable, str(scripts_dir / "collect_vocab.py")], check=True)
        
    # Step 2: Build Tabular
    print("Running build_tabular.py...")
    # Passing hospital ID logic could be added, but for now it builds all by default
    subprocess.run([sys.executable, str(scripts_dir / "build_tabular.py")], check=True)
    
    # Step 3: Build Timeseries
    print("Running build_timeseries.py...")
    subprocess.run([sys.executable, str(scripts_dir / "build_timeseries.py")], check=True)
    
    print("Feature engineering complete!")

if __name__ == "__main__":
    main()
