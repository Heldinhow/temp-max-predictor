#!/usr/bin/env python3
"""
LightGBM model for temperature prediction.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Optional


class TempPredictor:
    """
    LightGBM-based temperature predictor with optional quantile regression
    for probabilistic predictions.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        random_state: int = 42
    ):
        """
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            num_leaves: Number of leaves per tree
            min_child_samples: Minimum samples in leaf
            random_state: Random seed
        """
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'random_state': random_state,
            'verbose': -1,
            'n_jobs': -1
        }

        self.model = None
        self.feature_names = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50
    ) -> dict:
        """
        Train the LightGBM model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            early_stopping_rounds: Rounds for early stopping

        Returns:
            Training metrics dict
        """
        self.feature_names = X_train.columns.tolist()

        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')

        callbacks = [lgb.early_stopping(early_stopping_rounds)]

        self.model = lgb.train(
            {k: v for k, v in self.params.items() if k != 'n_estimators'},
            train_data,
            num_boost_round=self.params['n_estimators'],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        # Evaluate
        metrics = {}
        for name, ds in zip(valid_names, valid_sets):
            preds = self.model.predict(ds.data)
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            metrics[name] = {
                'mae': mean_absolute_error(ds.label, preds),
                'rmse': mean_squared_error(ds.label, preds, squared=False)
            }

        return metrics

    def predict(self, X: pd.DataFrame, quantile: Optional[float] = None) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on
            quantile: If set, predict this quantile instead of mean

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if quantile is not None:
            return self.model.predict(X, quantile=quantile)

        return self.model.predict(X)

    def predict_probability(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        """
        Predict probability that max temp >= threshold.

        Uses quantile regression: P(y >= threshold) ≈
        (1/3) * [I(q<0.167) + I(q<0.5) + I(q<0.833)]

        Args:
            X: Features to predict on
            threshold: Temperature threshold

        Returns:
            Probability array
        """
        q17 = self.predict(X, quantile=0.17)
        q50 = self.predict(X, quantile=0.50)
        q83 = self.predict(X, quantile=0.83)

        probs = ((q17 >= threshold).astype(float) +
                 (q50 >= threshold).astype(float) +
                 (q83 >= threshold).astype(float)) / 3

        return probs

    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.

        Returns:
            DataFrame with features and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        })

        return importance.sort_values('importance', ascending=False)