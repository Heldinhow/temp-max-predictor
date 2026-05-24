#!/usr/bin/env python3
"""
Training script for the temperature prediction model.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from src.data_loader import load_weather_data, prepare_features, get_feature_columns, split_temporal
from src.model import TempPredictor


def find_best_params(X_train, y_train, X_val, y_val):
    """Grid search over key hyperparameters."""
    param_grid = [
        {'learning_rate': 0.03, 'max_depth': 4, 'num_leaves': 15, 'min_child_samples': 20},
        {'learning_rate': 0.03, 'max_depth': 6, 'num_leaves': 31, 'min_child_samples': 20},
        {'learning_rate': 0.05, 'max_depth': 4, 'num_leaves': 15, 'min_child_samples': 20},
        {'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31, 'min_child_samples': 20},
        {'learning_rate': 0.05, 'max_depth': 8, 'num_leaves': 63, 'min_child_samples': 20},
        {'learning_rate': 0.08, 'max_depth': 4, 'num_leaves': 15, 'min_child_samples': 20},
        {'learning_rate': 0.08, 'max_depth': 6, 'num_leaves': 31, 'min_child_samples': 15},
        {'learning_rate': 0.10, 'max_depth': 4, 'num_leaves': 15, 'min_child_samples': 20},
    ]

    best_mae = float('inf')
    best_params = param_grid[0]

    for params in param_grid:
        p = TempPredictor(**params)
        metrics = p.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
        val_mae = metrics.get('val', {}).get('mae', float('inf'))
        n_rounds = p.model.current_iteration() if p.model else 0
        prefix = '✓' if val_mae < best_mae else ' '
        print(f"  {prefix} lr={params['learning_rate']} depth={params['max_depth']} "
              f"leaves={params['num_leaves']} min_child={params['min_child_samples']} "
              f"→ val_MAE={val_mae:.3f} ({n_rounds} rounds)")
        if val_mae < best_mae:
            best_mae = val_mae
            best_params = params

    return best_params


def main():
    parser = argparse.ArgumentParser(description='Train temperature prediction model')
    parser.add_argument('--data', type=str, required=True, help='Path to daily CSV data')
    parser.add_argument('--target', type=str, default='temp_max_real', help='Target column name')
    parser.add_argument('--cutoff', type=str, default='2024-01-01', help='Train/val split date')
    parser.add_argument('--output', type=str, default='models/model.txt', help='Output model path')
    parser.add_argument('--config', type=str, default='models/config.json', help='Output config path')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter search')
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = load_weather_data(args.data)
    df = prepare_features(df)

    # Drop rows with NaN in target
    df = df.dropna(subset=[args.target])

    features = get_feature_columns()

    available = [f for f in features if f in df.columns]
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
    print(f"Using {len(available)} features: {available}")

    train_df, val_df = split_temporal(df, args.cutoff)
    print(f"Train: {len(train_df)} days, Val: {len(val_df)} days")

    X_train = train_df[available].copy()
    y_train = train_df[args.target].copy()
    X_val = val_df[available].copy()
    y_val = val_df[args.target].copy()

    for col in available:
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)

    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)

    if args.tune:
        print("\n--- Hyperparameter Search ---")
        best = find_best_params(X_train, y_train, X_val, y_val)
        print(f"\nBest params: {best}")
    else:
        best = {}

    print("\nTraining final model...")
    predictor = TempPredictor(**best)
    metrics = predictor.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)

    print("\n=== Training Results ===")
    for split, m in metrics.items():
        print(f"  {split.upper()}: MAE={m['mae']:.2f}°C, RMSE={m['rmse']:.2f}°C")

    print("\n=== Feature Importance ===")
    importance = predictor.feature_importance()
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.1f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(str(output_path))
    print(f"\nModel saved to {args.output}")

    config_path = Path(args.config)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        'features': available,
        'target': args.target,
        'cutoff': args.cutoff,
        'params': best,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'metrics': {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in metrics.items()}
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to {args.config}")


if __name__ == '__main__':
    main()