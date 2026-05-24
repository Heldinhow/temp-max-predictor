#!/usr/bin/env python3
"""
Evaluation script - compare predictions with actual values.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_weather_data, prepare_features, get_feature_columns
from model import TempPredictor
from analog_days import AnalogDaysFinder


def evaluate(
    data_path: str,
    model_path: str = 'models/model.txt',
    config_path: str = 'models/config.json',
    cutoff: str = '2024-01-01'
):
    """
    Evaluate model on validation set.
    """
    print(f"Loading data from {data_path}...")
    df = load_weather_data(data_path)
    df = prepare_features(df)

    # Split
    train_df = df[df['date'] < cutoff].copy()
    val_df = df[df['date'] >= cutoff].copy()

    print(f"Train: {len(train_df)} days, Val: {len(val_df)} days")

    # Features
    features = get_feature_columns()
    features = [f for f in features if f in df.columns]
    target = 'temp_max_real'

    # Train LightGBM
    print("\nTraining LightGBM...")
    predictor = TempPredictor()
    metrics = predictor.fit(
        train_df[features], train_df[target],
        val_df[features], val_df[target]
    )

    # Predictions
    lgb_preds = predictor.predict(val_df[features])
    lgb_mae = mean_absolute_error(val_df[target], lgb_preds)
    lgb_rmse = mean_squared_error(val_df[target], lgb_preds, squared=False)

    print(f"\n=== LightGBM Results ===")
    print(f"MAE:  {lgb_mae:.2f}°C")
    print(f"RMSE: {lgb_rmse:.2f}°C")

    # Analog Days
    print("\n=== Analog Days Results ===")
    analog_finder = AnalogDaysFinder(n_neighbors=20, month_window=1)
    analog_finder.fit(train_df, features)

    analog_preds = []
    for _, row in val_df.iterrows():
        conditions = row[features].to_dict()
        conditions['date'] = row['date']
        pred = analog_finder.predict_max_temp(conditions, target_col=target)
        analog_preds.append(pred['prediction'])

    analog_mae = mean_absolute_error(val_df[target], analog_preds)
    analog_rmse = mean_squared_error(val_df[target], analog_preds, squared=False)

    print(f"MAE:  {analog_mae:.2f}°C")
    print(f"RMSE: {analog_rmse:.2f}°C")

    # Ensemble
    ensemble_preds = 0.6 * np.array(lgb_preds) + 0.4 * np.array(analog_preds)
    ensemble_mae = mean_absolute_error(val_df[target], ensemble_preds)
    ensemble_rmse = mean_squared_error(val_df[target], ensemble_preds, squared=False)

    print(f"\n=== Ensemble (0.6*LGB + 0.4*Analog) ===")
    print(f"MAE:  {ensemble_mae:.2f}°C")
    print(f"RMSE: {ensemble_rmse:.2f}°C")

    # Per-threshold accuracy
    print(f"\n=== Threshold Accuracy ===")
    for threshold in [20, 21, 22, 23, 24, 25, 26, 27, 28]:
        actual_binary = (val_df[target] >= threshold).astype(int)
        pred_binary_lgb = (np.array(lgb_preds) >= threshold).astype(int)
        pred_binary_analog = (np.array(analog_preds) >= threshold).astype(int)
        pred_binary_ensemble = (ensemble_preds >= threshold).astype(int)

        acc_lgb = (pred_binary_lgb == actual_binary).mean() * 100
        acc_analog = (pred_binary_analog == actual_binary).mean() * 100
        acc_ensemble = (pred_binary_ensemble == actual_binary).mean() * 100

        print(f"≥{threshold}°C: LGB={acc_lgb:.0f}%  Analog={acc_analog:.0f}%  Ensemble={acc_ensemble:.0f}%")

    # Save results
    results = {
        'lgb': {'mae': lgb_mae, 'rmse': lgb_rmse},
        'analog': {'mae': analog_mae, 'rmse': analog_rmse},
        'ensemble': {'mae': ensemble_mae, 'rmse': ensemble_rmse}
    }

    import json
    with open('models/evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to models/evaluation.json")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate temperature predictions')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--model', type=str, default='models/model.txt')
    parser.add_argument('--config', type=str, default='models/config.json')
    parser.add_argument('--cutoff', type=str, default='2024-01-01')
    args = parser.parse_args()

    evaluate(args.data, args.model, args.config, args.cutoff)


if __name__ == '__main__':
    main()