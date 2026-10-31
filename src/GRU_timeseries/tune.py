"""
tune.py — Hyperparameter Grid Search
======================================
Iterates over combinations of hyperparameters defined in TuningGrid,
trains a TemporalGRU for each configuration, and saves a ranked
results table to: models/tuning_results.json

Each run is logged with:
  - Hyperparameters
  - Best val mortality AUROC
  - Best val sepsis AUROC
  - Best val mean AUROC
  - Best val AUPRC (both tasks)
  - Approx training time

Usage:
    # Random search (recommended — exhaustive grid is too large for CPU)
    python -m src.GRU_timeseries.tune --num-trials 20 --epochs 5 --sample-hospitals 15

    # Grid search over a small subset
    python -m src.GRU_timeseries.tune --mode grid --epochs 5 --sample-hospitals 15
"""

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.GRU_timeseries.config import DataConfig, ModelConfig, TrainConfig, TuningGrid
from src.GRU_timeseries.train import run_training


# ---------------------------------------------------------------------------
# Helper: generate trial configs
# ---------------------------------------------------------------------------

def generate_grid_trials(grid: TuningGrid) -> List[Dict]:
    """
    Enumerate all combinations in the grid (full Cartesian product).
    WARNING: Can be very large. Use generate_random_trials() for CPU.
    """
    param_names = [
        "hidden_dim", "num_layers", "dropout", "bidirectional",
        "pooling", "learning_rate", "weight_decay", "batch_size",
    ]
    param_lists = [
        grid.hidden_dim, grid.num_layers, grid.dropout, grid.bidirectional,
        grid.pooling, grid.learning_rate, grid.weight_decay, grid.batch_size,
    ]
    trials = []
    for combo in itertools.product(*param_lists):
        trials.append(dict(zip(param_names, combo)))
    return trials


def generate_random_trials(grid: TuningGrid, num_trials: int, seed: int = 42) -> List[Dict]:
    """
    Sample random combinations from the grid — practical for CPU training.
    """
    rng = random.Random(seed)

    param_names = [
        "hidden_dim", "num_layers", "dropout", "bidirectional",
        "pooling", "learning_rate", "weight_decay", "batch_size",
    ]
    param_lists = [
        grid.hidden_dim, grid.num_layers, grid.dropout, grid.bidirectional,
        grid.pooling, grid.learning_rate, grid.weight_decay, grid.batch_size,
    ]

    trials = []
    for _ in range(num_trials):
        combo = {k: rng.choice(v) for k, v in zip(param_names, param_lists)}
        # Also sample loss weights
        combo["loss_weights"] = rng.choice(grid.loss_weights)
        trials.append(combo)

    return trials


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    trial: Dict,
    epochs: int,
    sample_hospitals: int,
    trial_idx: int,
    total_trials: int,
) -> Dict:
    """
    Run one hyperparameter configuration. Returns the trial result dict.
    """
    print(f"\n{'─'*60}")
    print(f"Trial {trial_idx+1}/{total_trials}")
    print(f"Config: {trial}")
    print(f"{'─'*60}")

    loss_weights = trial.get("loss_weights", (1.0, 1.0))

    data_cfg = DataConfig()
    model_cfg = ModelConfig(
        hidden_dim=trial["hidden_dim"],
        num_layers=trial["num_layers"],
        dropout=trial["dropout"],
        bidirectional=trial["bidirectional"],
        pooling=trial["pooling"],
    )
    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=trial["batch_size"],
        learning_rate=trial["learning_rate"],
        weight_decay=trial["weight_decay"],
        loss_w_mort=loss_weights[0],
        loss_w_sepsis=loss_weights[1],
        early_stopping_patience=3,  # tighter patience during tuning
    )

    t_start = time.time()
    try:
        best_metrics = run_training(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            sample_hospitals=sample_hospitals,
        )
        result = {
            **trial,
            "status": "ok",
            "val_mort_auroc":  best_metrics.get("val_mort_auroc",  0.0),
            "val_seps_auroc":  best_metrics.get("val_seps_auroc",  0.0),
            "val_mean_auroc":  best_metrics.get("val_mean_auroc",  0.0),
            "val_mort_auprc":  best_metrics.get("val_mort_auprc",  0.0),
            "val_seps_auprc":  best_metrics.get("val_seps_auprc",  0.0),
            "elapsed_s":       round(time.time() - t_start, 1),
        }
    except Exception as e:
        result = {**trial, "status": "error", "error": str(e), "elapsed_s": 0}
        print(f"  [WARN]  Trial failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tuning(
    mode: str          = "random",
    num_trials: int    = 20,
    epochs: int        = 10,
    sample_hospitals: int = None,
    output_path: Path  = None,
):
    """
    Run a hyperparameter search and save results.

    Args:
        mode:             'random' or 'grid'
        num_trials:       number of random trials (ignored for grid)
        epochs:           max epochs per trial (use fewer than full training)
        sample_hospitals: number of hospitals to use per trial
        output_path:      where to save results JSON
    """
    grid = TuningGrid()

    if mode == "grid":
        trials = generate_grid_trials(grid)
        print(f"Grid search: {len(trials)} total combinations")
    else:
        trials = generate_random_trials(grid, num_trials=num_trials)
        print(f"Random search: {num_trials} trials")

    results = []
    for i, trial in enumerate(trials):
        result = run_trial(trial, epochs, sample_hospitals, i, len(trials))
        results.append(result)

        # Save incrementally after each trial (in case of crash)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Sort by mean AUROC descending
    ok_results = [r for r in results if r.get("status") == "ok"]
    ok_results.sort(key=lambda r: r["val_mean_auroc"], reverse=True)

    print("\n" + "=" * 60)
    print("TUNING COMPLETE — Top 5 Configurations")
    print("=" * 60)
    for i, r in enumerate(ok_results[:5]):
        print(
            f"  #{i+1}  mean_auroc={r['val_mean_auroc']:.3f}  "
            f"mort={r['val_mort_auroc']:.3f}  seps={r['val_seps_auroc']:.3f}  "
            f"hidden={r['hidden_dim']}  layers={r['num_layers']}  "
            f"pool={r['pooling']}  lr={r['learning_rate']}"
        )

    if output_path:
        # Overwrite with sorted results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[OK] Results saved to {output_path}")

    return ok_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for TemporalGRU.")
    parser.add_argument("--mode",             type=str, default="random", choices=["random", "grid"])
    parser.add_argument("--num-trials",       type=int, default=20)
    parser.add_argument("--epochs",           type=int, default=10)
    parser.add_argument("--sample-hospitals", type=int, default=None)
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent.parent.parent / "models" / "tuning_results.json"
    run_tuning(
        mode=args.mode,
        num_trials=args.num_trials,
        epochs=args.epochs,
        sample_hospitals=args.sample_hospitals,
        output_path=output_path,
    )
