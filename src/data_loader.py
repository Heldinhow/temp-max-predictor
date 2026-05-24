#!/usr/bin/env python3
"""
Data loading and preparation for rp5.lv processed daily dataset.
"""

import pandas as pd
import numpy as np


def load_weather_data(csv_path: str) -> pd.DataFrame:
    """Load processed daily weather data from CSV."""
    df = pd.read_csv(csv_path, parse_dates=['date'])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all features needed for model training/prediction.
    Returns a dataframe with both raw and feature columns.
    """
    df = df.copy()

    df = df.sort_values('date').reset_index(drop=True)

    # Cyclical date features
    df['mes'] = df['date'].dt.month
    df['dia_do_ano'] = df['date'].dt.dayofyear
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_do_ano_sin'] = np.sin(2 * np.pi * df['dia_do_ano'] / 365)
    df['dia_do_ano_cos'] = np.cos(2 * np.pi * df['dia_do_ano'] / 365)

    # Lag features (yesterday's values)
    df['temp_max_yesterday'] = df['temp_max_real'].shift(1)
    df['temp_range_yesterday'] = df['temp_range'].shift(1)
    df['temp_mean_yesterday'] = df['temp_mean_real'].shift(1)
    df['temp_max_3day_avg'] = df['temp_max_real'].shift(1).rolling(3, min_periods=1).mean()
    df['temp_range_3day_avg'] = df['temp_range'].shift(1).rolling(3, min_periods=1).mean()

    return df


def get_feature_columns() -> list:
    """Return the list of feature column names for model input."""
    return [
        'temp_06h',
        'temp_09h',
        'humidity_06h',
        'pressure_06h',
        'temp_morning_mean',
        'temp_morning_std',
        'temp_morning_min',
        'humidity_morning_mean',
        'pressure_morning_mean',
        'wind_speed',
        'wind_gust_max',
        'wind_dir',
        'cloud_cover',
        'visibility',
        'temp_max_yesterday',
        'temp_range_yesterday',
        'temp_mean_yesterday',
        'temp_max_3day_avg',
        'temp_range_3day_avg',
        'mes_sin',
        'mes_cos',
        'dia_do_ano_sin',
        'dia_do_ano_cos',
    ]


def split_temporal(df: pd.DataFrame, cutoff_date: str) -> tuple:
    """Split data temporally (train on past, validate on future)."""
    train = df[df['date'] < cutoff_date].copy()
    val = df[df['date'] >= cutoff_date].copy()
    return train, val