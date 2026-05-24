#!/usr/bin/env python3
"""
Process raw rp5.lv weather data into daily format for the prediction model.

Input:  Raw CSV from rp5.lv (SBGR Guarulhos airport, hourly data, 2012-2026)
Output: Daily aggregated CSV with features expected by the model
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
import numpy as np


def parse_rp5_csv(filepath: str) -> pd.DataFrame:
    """Parse the raw rp5.lv CSV using Python's csv module (handles HTML inside fields)."""
    with open(filepath, 'r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter=';')
        records = []
        for row in reader:
            if row and isinstance(row[0], str) and row[0].startswith('#'):
                continue
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            cleaned = [cell.strip().strip('"').strip() for cell in row]
            while len(cleaned) < 14:
                cleaned.append('')
            cleaned = cleaned[:14]
            records.append(cleaned)

    if not records:
        raise ValueError("No data rows found in CSV")

    headers = [h.strip().strip('"').strip() for h in records[0]]
    df = pd.DataFrame(records[1:], columns=headers)
    return df


def clean_numeric(value: str) -> float:
    if not value or not isinstance(value, str) or value.strip() == '':
        return np.nan
    value = value.strip().replace(',', '.')
    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_datetime(date_str: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(date_str.strip(), format='%d.%m.%Y %H:%M')
    except Exception:
        return pd.NaT


WIND_DIR_MAP = {
    'norte': 0, 'nordeste': 45, 'leste': 90, 'sudeste': 135,
    'sul': 180, 'sudoeste': 225, 'oeste': 270, 'noroeste': 315,
    'n': 0, 'ne': 45, 'e': 90, 'se': 135, 's': 180, 'so': 225, 'o': 270, 'no': 315,
}


def parse_wind_dir(dd_str: str) -> float:
    if not dd_str or not isinstance(dd_str, str):
        return np.nan
    dd_lower = dd_str.lower()
    for key, val in WIND_DIR_MAP.items():
        if key in dd_lower:
            return val
    return np.nan


def parse_cloud_cover(cc_str: str) -> float:
    if not cc_str or not isinstance(cc_str, str):
        return np.nan
    cc_lower = cc_str.lower().strip()
    try:
        val = float(cc_lower)
        if 0 <= val <= 8:
            return val
    except ValueError:
        pass
    if '%' in cc_lower:
        match = re.search(r'(\d+)%', cc_lower)
        if match:
            return float(match.group(1)) / 100 * 8
    if 'quase limpo' in cc_lower or '10-30%' in cc_lower:
        return 1.0
    elif 'pouco nublado' in cc_lower or '40-50%' in cc_lower:
        return 2.0
    elif 'parcialmente nublado' in cc_lower or '60-90%' in cc_lower:
        return 4.0
    elif 'nublado' in cc_lower or '100%' in cc_lower:
        return 8.0
    return np.nan


def process_to_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly rp5 data into daily records."""
    df = df_raw.copy()
    df['datetime'] = df['Hora local em Guarulhos (aeroporto)'].apply(parse_datetime)
    df = df.dropna(subset=['datetime'])
    df['date'] = df['datetime'].dt.normalize()
    df['hour'] = df['datetime'].dt.hour

    # Parse numeric
    for col in ['T', 'P0', 'P', 'U', 'Ff', 'ff10', 'VV', 'Td']:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    if 'DD' in df.columns:
        df['wind_dir_deg'] = df['DD'].apply(parse_wind_dir)
    else:
        df['wind_dir_deg'] = np.nan

    if 'c' in df.columns:
        df['cloud_cover'] = df['c'].apply(parse_cloud_cover)
    else:
        df['cloud_cover'] = np.nan

    # === Build daily records directly via groupby ===
    morning = df[(df['hour'] >= 3) & (df['hour'] <= 12)].copy()

    # === Daily target (max temp from all hours) ===
    day_target = df.groupby('date')['T'].max().reset_index()
    day_target.columns = ['date', 'temp_max_real']

    day_min = df.groupby('date')['T'].min().reset_index()
    day_min.columns = ['date', 'temp_min_real']

    day_mean = df.groupby('date')['T'].mean().reset_index()
    day_mean.columns = ['date', 'temp_mean_real']

    # === Morning features via groupby aggregations ===
    # 06h values
    t06_df = morning[morning['hour'] == 6][['date', 'T']].copy()
    t06_df = t06_df.groupby('date')['T'].first().reset_index()
    t06_df.columns = ['date', 'temp_06h']

    u06_df = morning[morning['hour'] == 6][['date', 'U']].copy()
    u06_df = u06_df.groupby('date')['U'].first().reset_index()
    u06_df.columns = ['date', 'humidity_06h']

    p06_df = morning[morning['hour'] == 6][['date', 'P']].copy()
    p06_df = p06_df.groupby('date')['P'].first().reset_index()
    p06_df.columns = ['date', 'pressure_06h']

    # Morning mean stats (3h-12h)
    morn_agg = morning.groupby('date').agg({
        'T': ['mean', 'std', 'min'],
        'U': 'mean',
        'P': 'mean',
        'Ff': ['mean', 'max'],
        'cloud_cover': 'mean',
        'VV': 'mean',
        'wind_dir_deg': 'mean'
    })
    morn_agg.columns = ['_'.join(col).strip() for col in morn_agg.columns.values]
    morn_agg = morn_agg.reset_index()

    # 09h temperature
    t09_df = morning[morning['hour'] == 9][['date', 'T']].copy()
    t09_df = t09_df.groupby('date')['T'].first().reset_index()
    t09_df.columns = ['date', 'temp_09h']

    # === Merge all ===
    daily = day_target.copy()
    for frame in [day_min, day_mean, t06_df, t09_df, u06_df, p06_df, morn_agg]:
        daily = daily.merge(frame, on='date', how='left')

    # Add temp_range
    daily['temp_range'] = daily['temp_max_real'] - daily['temp_min_real']

    # Rename columns to match model
    col_map = {
        'T_mean': 'temp_morning_mean',
        'T_std': 'temp_morning_std',
        'T_min': 'temp_morning_min',
        'U_mean': 'humidity_morning_mean',
        'P_mean': 'pressure_morning_mean',
        'Ff_mean': 'wind_speed',
        'Ff_max': 'wind_gust_max',
        'cloud_cover_mean': 'cloud_cover',
        'VV_mean': 'visibility',
        'wind_dir_deg_mean': 'wind_dir',
    }
    daily = daily.rename(columns=col_map)

    # Fill missing 06h temp with 09h or morning mean
    daily['temp_06h'] = daily['temp_06h'].fillna(daily['temp_09h'])
    daily['temp_06h'] = daily['temp_06h'].fillna(daily['temp_morning_mean'])

    # Sort by date
    daily = daily.sort_values('date').reset_index(drop=True)

    # Keep useful columns
    keep_cols = [
        'date', 'temp_06h', 'temp_09h', 'humidity_06h', 'pressure_06h',
        'temp_morning_mean', 'temp_morning_std', 'temp_morning_min',
        'humidity_morning_mean', 'pressure_morning_mean',
        'wind_speed', 'wind_gust_max', 'wind_dir',
        'cloud_cover', 'visibility',
        'temp_max_real', 'temp_min_real', 'temp_mean_real', 'temp_range'
    ]
    final_cols = [c for c in keep_cols if c in daily.columns]
    daily = daily[final_cols]

    return daily


def main():
    parser = argparse.ArgumentParser(description='Process raw rp5.lv CSV to daily format')
    parser.add_argument('input', type=str, help='Path to raw rp5.lv CSV file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path(f"/tmp/{input_path.stem}_daily.csv")

    print(f"Reading {args.input}...")
    df_raw = parse_rp5_csv(args.input)
    print(f"Raw rows: {len(df_raw)}")
    print(f"Columns: {df_raw.columns.tolist()}")

    print("\nProcessing to daily (may take ~1 min)...")
    df_daily = process_to_daily(df_raw)
    print(f"Daily rows: {len(df_daily)}")
    print(f"Date range: {df_daily['date'].min()} → {df_daily['date'].max()}")

    print(f"\nSaving to {output_path}...")
    df_daily.to_csv(output_path, index=False)
    print("Done!")

    print("\n=== Last 5 days ===")
    print(df_daily.tail(5).to_string())

    print("\n=== Summary stats ===")
    num_cols = df_daily.select_dtypes(include=[np.number]).columns
    print(df_daily[num_cols].describe().round(2).to_string())

    missing = df_daily.isnull().sum()
    if missing.sum() > 0:
        print(f"\n=== Missing values ===")
        print(missing[missing > 0])


if __name__ == '__main__':
    main()