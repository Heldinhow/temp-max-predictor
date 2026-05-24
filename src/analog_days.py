#!/usr/bin/env python3
"""
Analog days finder - finds historical days with similar initial conditions.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class AnalogDaysFinder:
    """
    Find historical days with similar initial conditions (morning temperature,
    humidity, pressure, etc.) and use their outcomes to predict today's max temp.
    """

    def __init__(self, n_neighbors: int = 20, month_window: int = 1):
        """
        Args:
            n_neighbors: Number of analog days to consider
            month_window: How many months around current month to search
        """
        self.n_neighbors = n_neighbors
        self.month_window = month_window
        self.scaler = StandardScaler()
        self.df = None
        self.features = None

    def fit(self, df: pd.DataFrame, features: list):
        """
        Fit the analog finder on historical data.

        Args:
            df: DataFrame with historical weather data
            features: List of feature column names
        """
        self.df = df.copy()
        self.features = features

        # Fit scaler on all data
        self.scaler.fit(df[features])

    def find_analogs(self, current_conditions: dict, top_k: int = None) -> pd.DataFrame:
        """
        Find analog days for current conditions.

        Args:
            current_conditions: Dict with feature values (e.g., {'temp_06h': 18.5, ...})
            top_k: If set, return only top K most similar days

        Returns:
            DataFrame of analog days sorted by similarity
        """
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

        if len(filtered_df) == 0:
            filtered_df = self.df  # Fall back to all data

        # Build current feature vector
        current_vec = np.array([[current_conditions.get(f, 0) for f in self.features]])

        # Scale
        current_scaled = self.scaler.transform(current_vec)
        historical_scaled = self.scaler.transform(filtered_df[self.features])

        # Find nearest neighbors using distance weighting
        nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(filtered_df)), 
                               weights='distance')
        nn.fit(historical_scaled)

        distances, indices = nn.kneighbors(current_scaled)

        # Get analog days
        analog_days = filtered_df.iloc[indices[0]].copy()
        analog_days['distance'] = distances[0]
        analog_days['weight'] = 1 / (distances[0] + 1e-6)  # Avoid division by zero

        # Sort by weight (most similar first)
        analog_days = analog_days.sort_values('weight', ascending=False)

        if top_k:
            analog_days = analog_days.head(top_k)

        return analog_days

    def predict_max_temp(self, current_conditions: dict, target_col: str = 'temp_max_real') -> dict:
        """
        Predict max temperature using weighted average of analog days.

        Args:
            current_conditions: Dict with current feature values
            target_col: Target column name in historical data

        Returns:
            Dict with prediction and statistics
        """
        analogs = self.find_analogs(current_conditions, top_k=self.n_neighbors)

        if target_col not in analogs.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Weighted average
        weights = analogs['weight'].values
        temps = analogs[target_col].values

        weighted_avg = np.sum(weights * temps) / np.sum(weights)

        # Also calculate simple statistics
        result = {
            'prediction': round(weighted_avg, 1),
            'mean': round(analogs[target_col].mean(), 1),
            'std': round(analogs[target_col].std(), 1),
            'min': round(analogs[target_col].min(), 1),
            'max': round(analogs[target_col].max(), 1),
            'n_analogs': len(analogs),
            'analog_days': analogs[['date', target_col, 'distance', 'weight']].to_dict('records')
        }

        return result