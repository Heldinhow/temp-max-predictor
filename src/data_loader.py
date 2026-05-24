#!/usr/bin/env python3
"""
Data loading and preparation utilities.
"""

import pandas as pd
from pathlib import Path


def load_weather_data(csv_path: str, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load weather data from CSV.

    Args:
        csv_path: Path to the CSV file
        parse_dates: Whether to parse date columns

    Returns:
        DataFrame with weather data
    """
    df = pd.read_csv(csv_path)

    if parse_dates and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for the model.

    Args:
        df: DataFrame with raw weather data

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Time-based features
    df['mes'] = df['date'].dt.month
    df['dia_do_ano'] = df['date'].dt.dayofyear
    df['dia_da_semana'] = df['date'].dt.dayofweek

    # Cyclical encoding for month and day of year
    df['mes_sin'] = df['mes'].apply(lambda x: __import__('math').sin(2 * __import__('math').pi * x / 12))
    df['mes_cos'] = df['mes'].apply(lambda x: __import__('math').cos(2 * __import__('math').pi * x / 12))
    df['dia_do_ano_sin'] = df['dia_do_ano'].apply(lambda x: __import__('math').sin(2 * __import__('math').pi * x / 365))
    df['dia_do_ano_cos'] = df['dia_do_ano'].apply(lambda x: __import__('math').cos(2 * __import__('math').pi * x / 365))

    return df


def get_feature_columns() -> list:
    """
    Return the list of feature columns used for training/prediction.
    """
    return [
        'temp_06h',
        'humidity_06h',
        'pressure_06h',
        'wind_speed',
        'wind_dir',
        'mes_sin',
        'mes_cos',
        'dia_do_ano_sin',
        'dia_do_ano_cos',
    ]


def split_temporal(df: pd.DataFrame, cutoff_date: str) -> tuple:
    """
    Split data temporally (train on past, validate on future).

    Args:
        df: DataFrame with date column
        cutoff_date: Split date (e.g., '2024-01-01')

    Returns:
        (train_df, val_df)
    """
    train = df[df['date'] < cutoff_date].copy()
    val = df[df['date'] >= cutoff_date].copy()
    return train, val