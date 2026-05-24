#!/usr/bin/env python3
"""
Prediction script using analog days + LightGBM hybrid approach.
"""

import argparse
import json
import sys
from datetime import datetime

import pandas as pd
import numpy as np

from data_loader import load_weather_data, prepare_features, get_feature_columns
from model import TempPredictor
from analog_days import AnalogDaysFinder


def load_model_and_config(model_path: str, config_path: str):
    """Load trained model and config."""
    import lightgbm as lgb

    model = lgb.Booster(model_file=model_path)
    with open(config_path) as f:
        config = json.load(f)

    return model, config


def predict_from_csv(
    csv_path: str,
    model_path: str = 'models/model.txt',
    config_path: str = 'models/config.json',
    today: str = None
):
    """
    Make predictions from a CSV file (typically today's conditions).

    Args:
        csv_path: Path to today's weather data CSV
        model_path: Path to trained model
        config_path: Path to config JSON
        today: Date string for today (YYYY-MM-DD). If None, uses current date.
    """
    if today is None:
        today = datetime.now().strftime('%Y-%m-%d')

    # Load today's data
    df_today = load_weather_data(csv_path)
    df_today = prepare_features(df_today)

    # Load model
    model_obj, config = load_model_and_config(model_path, config_path)
    features = config['features']

    # Ensure all features exist
    for f in features:
        if f not in df_today.columns:
            raise ValueError(f"Missing feature: {f}")

    X_today = df_today[features]

    # LightGBM prediction
    pred_lgb = model_obj.predict(X_today)[0]

    # Analog days prediction
    df_hist = load_weather_data(csv_path)  # Use same file as history (filter out today)
    df_hist = prepare_features(df_hist)
    df_hist = df_hist[df_hist['date'] < today]

    analog_finder = AnalogDaysFinder(n_neighbors=20, month_window=1)
    analog_finder.fit(df_hist, features)

    today_conditions = df_today.iloc[0].to_dict()
    today_conditions['date'] = today
    pred_analog = analog_finder.predict_max_temp(today_conditions, target_col=config['target'])

    # Ensemble: weighted average
    pred_ensemble = 0.6 * pred_lgb + 0.4 * pred_analog['prediction']

    print(f"\n=== Predictions for {today} ===")
    print(f"LightGBM:    {pred_lgb:.1f}°C")
    print(f"Analog Days: {pred_analog['prediction']:.1f}°C (from {pred_analog['n_analogs']} analogs)")
    print(f"ENSEMBLE:    {pred_ensemble:.1f}°C")

    # Probabilities for Polymarket thresholds
    thresholds = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    print(f"\n=== Polymarket Probabilities ===")
    for t in thresholds:
        # Approximate probability using analog distribution
        analogs_above = sum(1 for d in pred_analog['analog_days'] if d.get(config['target'], 0) >= t)
        prob = analogs_above / len(pred_analog['analog_days'])
        pct = prob * 100
        if pct >= 50:
            print(f"  P(max ≥ {t}°C): {pct:.0f}% ✅")
        else:
            print(f"  P(max ≥ {t}°C): {pct:.0f}%")

    return {
        'date': today,
        'lgb': pred_lgb,
        'analog': pred_analog['prediction'],
        'ensemble': pred_ensemble,
        'analog_stats': pred_analog
    }


def main():
    parser = argparse.ArgumentParser(description='Predict max temperature')
    parser.add_argument('--data', type=str, required=True, help='Path to today\'s CSV data')
    parser.add_argument('--model', type=str, default='models/model.txt', help='Model path')
    parser.add_argument('--config', type=str, default='models/config.json', help='Config path')
    parser.add_argument('--today', type=str, default=None, help='Date (YYYY-MM-DD)')
    args = parser.parse_args()

    predict_from_csv(args.data, args.model, args.config, args.today)


if __name__ == '__main__':
    main()