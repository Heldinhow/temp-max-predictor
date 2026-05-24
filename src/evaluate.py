#!/usr/bin/env python3
"""
Evaluation script - evaluate model accuracy on validation set.
"""

import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import load_weather_data, prepare_features, get_feature_columns, split_temporal
from src.model import TempPredictor


def evaluate(data_path: str, cutoff: str = '2024-01-01'):
    print(f"Loading data from {data_path}...")
    df = load_weather_data(data_path)
    df = prepare_features(df)

    features = get_feature_columns()
    available = [f for f in features if f in df.columns]
    target = 'temp_max_real'

    train_df, val_df = split_temporal(df, cutoff)
    print(f"Train: {len(train_df)} days, Val: {len(val_df)} days")

    for col in available:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        val_df[col] = val_df[col].fillna(median_val)

    X_train = train_df[available].astype(np.float64)
    y_train = train_df[target]
    X_val = val_df[available].astype(np.float64)
    y_val = val_df[target]

    print("\nTraining LightGBM...")
    predictor = TempPredictor()
    predictor.fit(X_train, y_train, X_val, y_val)

    preds = predictor.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f"\n=== Validation Results ===")
    print(f"MAE:  {mae:.2f}°C")
    print(f"RMSE: {rmse:.2f}°C")

    print(f"\n=== Feature Importance ===")
    for _, row in predictor.feature_importance().iterrows():
        print(f"  {row['feature']}: {row['importance']:.1f}")

    print(f"\n=== Threshold Accuracy ===")
    for threshold in [20, 21, 22, 23, 24, 25, 26, 27, 28]:
        actual_binary = (y_val >= threshold).astype(int)
        pred_binary = (preds >= threshold).astype(int)
        acc = (pred_binary == actual_binary).mean() * 100
        pred_prob = predictor.predict_probability(X_val, threshold)
        brier = np.mean((pred_prob - actual_binary) ** 2)
        print(f"≥{threshold}°C: Acc={acc:.0f}%  Brier={brier:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--cutoff', type=str, default='2024-01-01')
    args = parser.parse_args()
    evaluate(args.data, args.cutoff)


if __name__ == '__main__':
    main()