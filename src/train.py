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
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--target', type=str, default='temp_max_real', help='Target column name')
    parser.add_argument('--cutoff', type=str, default='2024-01-01', help='Train/val split date')
    parser.add_argument('--output', type=str, default='models/model.txt', help='Output model path')
    parser.add_argument('--config', type=str, default='models/config.json', help='Output config path')
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = load_weather_data(args.data)

    print(f"Preparing features...")
    df = prepare_features(df)

    features = get_feature_columns()

    # Check for missing features
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"WARNING: Missing features in data: {missing}")
        features = [f for f in features if f in df.columns]

    print(f"Using features: {features}")

    # Split
    train_df, val_df = split_temporal(df, args.cutoff)
    print(f"Train: {len(train_df)} days, Val: {len(val_df)} days")

    # Prepare data
    X_train = train_df[features]
    y_train = train_df[args.target]
    X_val = val_df[features]
    y_val = val_df[args.target]

    # Train
    print("Training model...")
    predictor = TempPredictor()
    metrics = predictor.fit(X_train, y_train, X_val, y_val)

    print("\n=== Training Results ===")
    for split, m in metrics.items():
        print(f"{split.upper()}: MAE={m['mae']:.2f}°C, RMSE={m['rmse']:.2f}°C")

    # Feature importance
    print("\n=== Feature Importance ===")
    importance = predictor.feature_importance()
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.1f}")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictor.model.save_model(str(output_path))
    print(f"\nModel saved to {args.output}")

    # Save config
    config_path = Path(args.config)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        'features': features,
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