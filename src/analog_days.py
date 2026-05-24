#!/usr/bin/env python3
"""
Analog days finder - finds historical days with similar initial conditions.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class AnalogDaysFinder:
    def __init__(self, n_neighbors: int = 20, month_window: int = 1):
        self.n_neighbors = n_neighbors
        self.month_window = month_window
        self.scaler = StandardScaler()
        self.df = None
        self.features = None

    def fit(self, df: pd.DataFrame, features: list):
        self.df = df.copy()
        self.features = features
        # Fill NaN for scaling
        df_feat = df[features].fillna(df[features].median())
        self.scaler.fit(df_feat)

    def find_analogs(self, current_conditions: dict, top_k: int = None) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Filter by month window
        current_month = pd.Timestamp(current_conditions.get('date', pd.Timestamp.now())).month
        month_min = ((current_month - self.month_window - 1) % 12) + 1
        month_max = ((current_month + self.month_window - 1) % 12) + 1

        if month_min <= month_max:
            filtered_df = self.df[(self.df['mes'] >= month_min) & (self.df['mes'] <= month_max)]
        else:
            filtered_df = self.df[(self.df['mes'] >= month_min) | (self.df['mes'] <= month_max)]

        if len(filtered_df) < self.n_neighbors:
            filtered_df = self.df

        # Build feature vector
        feat_vec = np.array([[current_conditions.get(f, 0) for f in self.features]])
        feat_vec = np.nan_to_num(feat_vec, nan=0)

        scaled_current = self.scaler.transform(feat_vec)
        scaled_hist = self.scaler.transform(filtered_df[self.features].fillna(0))

        k = min(self.n_neighbors, len(filtered_df))
        nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn_model.fit(scaled_hist)
        distances, indices = nn_model.kneighbors(scaled_current)

        analog_days = filtered_df.iloc[indices[0]].copy()
        analog_days['distance'] = distances[0]
        analog_days['weight'] = 1 / (distances[0] + 1e-6)
        analog_days = analog_days.sort_values('weight', ascending=False)

        if top_k:
            analog_days = analog_days.head(top_k)

        return analog_days

    def predict_max_temp(self, current_conditions: dict, target_col: str = 'temp_max_real') -> dict:
        analogs = self.find_analogs(current_conditions, top_k=self.n_neighbors)

        if target_col not in analogs.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        weights = analogs['weight'].values
        temps = analogs[target_col].values

        weighted_avg = np.sum(weights * temps) / np.sum(weights)

        return {
            'prediction': round(weighted_avg, 1),
            'mean': round(analogs[target_col].mean(), 1),
            'std': round(analogs[target_col].std(), 1),
            'min': round(analogs[target_col].min(), 1),
            'max': round(analogs[target_col].max(), 1),
            'n_analogs': len(analogs),
            'analog_days': analogs[['date', target_col, 'distance', 'weight']].to_dict('records')
        }