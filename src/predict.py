#!/usr/bin/env python3
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

from src.data_loader import load_weather_data, prepare_features, get_feature_columns, split_temporal
from src.model import TempPredictor
from src.analog_days import AnalogDaysFinder


def find_ensemble_weight(train_df, val_df, features, target, analog_finder):
    """Find optimal ensemble weight on validation set."""
    lgb = TempPredictor()
    lgb.fit(train_df[features], train_df[target],
            val_df[features], val_df[target], early_stopping_rounds=50)

    lgb_preds = lgb.predict(val_df[features])
    analog_preds = []
    for _, row in val_df.iterrows():
        cond = {f: row[f] for f in features}
        cond['date'] = row['date']
        ap = analog_finder.predict_max_temp(cond, target_col=target)
        analog_preds.append(ap['prediction'])
    analog_preds = np.array(analog_preds)
    actuals = val_df[target].values

    best_weight = 0.5
    best_mae = float('inf')
    for w in np.arange(0, 1.05, 0.05):
        ensemble = w * lgb_preds + (1 - w) * analog_preds
        mae = np.mean(np.abs(ensemble - actuals))
        if mae < best_mae:
            best_mae = mae
            best_weight = w

    print(f"  Optimal ensemble weight: LGB={best_weight:.2f}, "
          f"Analog={1-best_weight:.2f} (val_MAE={best_mae:.2f}°C)")
    return best_weight, lgb, lgb_preds


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

    train_df, val_df = split_temporal(df, cutoff)

    for col in available:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        val_df[col] = val_df[col].fillna(median_val)

    for col in available:
        train_df[col] = train_df[col].astype(np.float64)
        val_df[col] = val_df[col].astype(np.float64)

    analog_finder = AnalogDaysFinder(n_neighbors=20, month_window=1)
    analog_finder.fit(train_df, available)

    print("\nFinding optimal ensemble weight...")
    lgb_weight, predictor, _ = find_ensemble_weight(
        train_df, val_df, available, target, analog_finder)
    analog_weight = 1 - lgb_weight

    df_sorted = df.sort_values('date').reset_index(drop=True)
    today_row = df_sorted[df_sorted['date'] == pd.to_datetime(today)]

    if len(today_row) == 0:
        print(f"No data for {today} — using most recent day:")
        today_row = df_sorted.iloc[[-1]]
        today = str(today_row['date'].values[0])[:10]

    print(f"\nUsing data from: {today}")

    for col in available:
        today_row[col] = today_row[col].fillna(train_df[col].median()).astype(np.float64)

    today_conditions = {f: today_row[f].values[0] for f in available}
    today_conditions['date'] = pd.to_datetime(today)
    analog_pred = analog_finder.predict_max_temp(today_conditions, target_col=target)

    X_today = today_row[available].values.astype(np.float64)
    lgb_pred = predictor.predict(pd.DataFrame(X_today, columns=available))[0]

    ensemble = lgb_weight * lgb_pred + analog_weight * analog_pred['prediction']

    observed_max = max(
        float(today_row['temp_06h'].values[0]) if 'temp_06h' in today_row else -999,
        float(today_row['temp_09h'].values[0]) if 'temp_09h' in today_row else -999,
        float(today_row['temp_morning_mean'].values[0]) if 'temp_morning_mean' in today_row else -999,
    )
    if observed_max > 0 and ensemble < observed_max:
        print(f"  ⚠ Previsão corrigida: {ensemble:.1f}°C → {observed_max:.1f}°C "
              f"(mínimo = máxima já observada hoje)")
        ensemble = round(observed_max, 1)

    print(f"\n{'='*50}")
    print(f"  Previsão para {today}")
    print(f"{'='*50}")
    print(f"\n  Condições matinais:")
    for f in ['temp_06h', 'temp_09h', 'humidity_06h', 'pressure_06h',
              'temp_morning_mean', 'wind_speed', 'cloud_cover']:
        if f in today_row and pd.notna(today_row[f].values[0]):
            print(f"    {f}: {today_row[f].values[0]:.1f}")
    print(f"\n  LightGBM ({lgb_weight:.0%}):      {lgb_pred:.1f}°C")
    print(f"  Analog Days ({analog_weight:.0%}):  {analog_pred['prediction']:.1f}°C")
    print(f"  ─────────────────────")
    print(f"  ENSEMBLE:          {ensemble:.1f}°C  (σ={analog_pred['std']})")
    actual = today_row[target].values[0]
    if not pd.isna(actual):
        print(f"  Real max:          {actual:.1f}°C  (erro: {ensemble - actual:+.1f}°C)")

    print(f"\n  Probabilidades P(max ≥ X°C)")
    for threshold in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        prob_lgb = predictor.predict_probability(
            pd.DataFrame(X_today, columns=available), threshold)[0]
        analogs_above = sum(1 for d in analog_pred['analog_days']
                            if d.get(target, 0) >= threshold)
        prob_analog = analogs_above / len(analog_pred['analog_days'])
        prob = lgb_weight * prob_lgb + analog_weight * prob_analog
        print(f"    ≥{threshold}°C: {prob * 100:.1f}%")


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
