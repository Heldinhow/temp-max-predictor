#!/usr/bin/env python3
"""
Training script for the temperature prediction model.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from data_loader import load_weather_data, prepare_features, get_feature_columns, split_temporal
from model import TempPredictor


def main():
    parser = argparse.ArgumentParser(description='Train temperature prediction model')
    parser.add_argument('--data', type=str, required=True, help='Path to daily CSV data')
    parser.add_argument('--target', type=str, default='temp_max_real', help='Target column name')
    parser.add_argument('--cutoff', type=str, default='2024-01-01', help='Train/val split date')
    parser.add_argument('--output', type=str, default='models/model.txt', help='Output model path')
    parser.add_argument('--config', type=str, default='models/config.json', help='Output config path')
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = load_weather_data(args.data)
    df = prepare_features(df)

    features = get_feature_columns()

    # Check for available columns
    available = [f for f in features if f in df.columns]
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
    print(f"Using {len(available)} features")

    # Split
    train_df, val_df = split_temporal(df, args.cutoff)
    print(f"Train: {len(train_df)} days, Val: {len(val_df)} days")

    # Prepare data
    X_train = train_df[available].copy()
    y_train = train_df[args.target].copy()
    X_val = val_df[available].copy()
    y_val = val_df[args.target].copy()

    # Fill NaN with column median from train
    for col in available:
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)

    # Convert to float64 to avoid LightGBM sparse issues
    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)

    # Train
    print("Training model...")
    predictor = TempPredictor()
    metrics = predictor.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50)

    print("\n=== Training Results ===")
    for split, m in metrics.items():
        print(f"  {split.upper()}: MAE={m['mae']:.2f}°C, RMSE={m['rmse']:.2f}°C")

    # Feature importance
    print("\n=== Feature Importance ===")
    importance = predictor.feature_importance()
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.1f}")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(str(output_path))
    print(f"\nModel saved to {args.output}")

    # Save config
    config_path = Path(args.config)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        'features': available,
        'target': args.target,
        'cutoff': args.cutoff,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'metrics': {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in metrics.items()}
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to {args.config}")


if __name__ == '__main__':
    main()