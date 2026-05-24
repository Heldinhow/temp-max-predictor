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

    # Cyclical date features
    df['mes'] = df['date'].dt.month
    df['dia_do_ano'] = df['date'].dt.dayofyear
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_do_ano_sin'] = np.sin(2 * np.pi * df['dia_do_ano'] / 365)
    df['dia_do_ano_cos'] = np.cos(2 * np.pi * df['dia_do_ano'] / 365)

    return df


def get_feature_columns() -> list:
    """Return the list of feature column names for model input."""
    return [
        'temp_06h',
        'humidity_06h',
        'pressure_06h',
        'temp_morning_mean',
        'temp_morning_std',
        'humidity_morning_mean',
        'pressure_morning_mean',
        'wind_speed',
        'cloud_cover',
        'visibility',
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