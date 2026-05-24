#!/usr/bin/env python3
"""
Prediction script - predict today's max temperature using LightGBM + Analog Days.
"""

import argparse
from datetime import datetime

import pandas as pd
import numpy as np

from data_loader import load_weather_data, prepare_features, get_feature_columns, split_temporal
from model import TempPredictor
from analog_days import AnalogDaysFinder


def predict_today(
    data_path: str,
    cutoff: str = '2024-01-01',
    today: str = None,
    model_path: str = 'models/model.txt'
):
    if today is None:
        today = datetime.now().strftime('%Y-%m-%d')

    print(f"Loading data from {data_path}...")
    df = load_weather_data(data_path)
    df = prepare_features(df)

    features = get_feature_columns()
    available = [f for f in features if f in df.columns]
    target = 'temp_max_real'

    # Split
    train_df, val_df = split_temporal(df, cutoff)

    # Fill NaN
    for col in available:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        val_df[col] = val_df[col].fillna(median_val)

    # Prepare arrays
    for col in available:
        train_df[col] = train_df[col].astype(np.float64)
        val_df[col] = val_df[col].astype(np.float64)

    # Train
    print("Training LightGBM...")
    predictor = TempPredictor()
    predictor.fit(train_df[available], train_df[target], val_df[available], val_df[target])

    # Get most recent day
    df_sorted = df.sort_values('date').reset_index(drop=True)
    today_row = df_sorted[df_sorted['date'] == pd.to_datetime(today)]

    if len(today_row) == 0:
        print(f"No data for {today} — using most recent day:")
        today_row = df_sorted.iloc[[-1]]
        today = str(today_row['date'].values[0])[:10]

    print(f"\nUsing data from: {today}")

    for col in available:
        today_row[col] = today_row[col].fillna(train_df[col].median()).astype(np.float64)

    # Analog days
    analog_finder = AnalogDaysFinder(n_neighbors=20, month_window=1)
    analog_finder.fit(train_df, available)

    today_conditions = {}
    for f in available:
        today_conditions[f] = today_row[f].values[0]
    today_conditions['date'] = pd.to_datetime(today)
    analog_pred = analog_finder.predict_max_temp(today_conditions, target_col=target)

    # Predictions
    X_today = today_row[available].values.astype(np.float64)
    lgb_pred = predictor.predict(pd.DataFrame(X_today, columns=available))[0]

    ensemble = 0.6 * lgb_pred + 0.4 * analog_pred['prediction']

    print(f"\n{'='*50}")
    print(f"📅 Prediction for {today}")
    print(f"{'='*50}")
    print(f"  LightGBM:    {lgb_pred:.1f}°C")
    print(f"  Analog Days: {analog_pred['prediction']:.1f}°C")
    print(f"  ─────────────────────")
    print(f"  ENSEMBLE:    {ensemble:.1f}°C  (σ={analog_pred['std']})")
    actual = today_row[target].values[0]
    if not pd.isna(actual):
        print(f"  Actual max:  {actual:.1f}°C  (error: {ensemble - actual:+.1f}°C)")

    # Polymarket probabilities
    print(f"\n📊 Polymarket Probabilities P(max ≥ X°C)")
    print(f"{'='*50}")

    for threshold in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        prob_lgb = predictor.predict_probability(pd.DataFrame(X_today, columns=available), threshold)[0]
        analogs_above = sum(1 for d in analog_pred['analog_days'] if d.get(target, 0) >= threshold)
        prob_analog = analogs_above / len(analog_pred['analog_days'])
        prob = 0.6 * prob_lgb + 0.4 * prob_analog
        pct = prob * 100
        bar = '█' * int(pct / 5)
        m = '🟢' if pct >= 95 else '🟡' if pct >= 70 else '🟠' if pct >= 30 else '🔴'
        print(f"  {m} ≥{threshold}°C: {pct:5.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser(description='Predict max temperature')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--cutoff', type=str, default='2024-01-01')
    parser.add_argument('--today', type=str, default=None)
    parser.add_argument('--model', type=str, default='models/model.txt')
    args = parser.parse_args()
    predict_today(args.data, args.cutoff, args.today, args.model)


if __name__ == '__main__':
    main()