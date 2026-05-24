#!/usr/bin/env python3
"""
LightGBM model for temperature prediction.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.metrics import mean_absolute_error, mean_squared_error


class TempPredictor:
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        random_state: int = 42
    ):
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'random_state': random_state,
            'verbose': -1,
            'n_jobs': -1,
            'force_row_wise': True,
        }
        self.model = None
        self.feature_names = None
        self.evals_result = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50
    ) -> dict:
        self.feature_names = list(X_train.columns)
        X_tr = X_train.values.astype(np.float64)
        y_tr = y_train.values.astype(np.float64)

        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names, free_raw_data=False)

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            X_v = X_val.values.astype(np.float64)
            y_v = y_val.values.astype(np.float64)
            val_data = lgb.Dataset(X_v, label=y_v, feature_name=self.feature_names, reference=train_data, free_raw_data=False)
            valid_sets.append(val_data)
            valid_names.append('val')

        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

        base_params = {k: v for k, v in self.params.items() if k != 'n_estimators'}
        self.model = lgb.train(
            base_params,
            train_data,
            num_boost_round=self.params['n_estimators'],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        metrics = {}
        for name, ds in zip(valid_names, valid_sets):
            preds = self.model.predict(ds.data)
            metrics[name] = {
                'mae': mean_absolute_error(ds.label, preds),
                'rmse': np.sqrt(mean_squared_error(ds.label, preds))
            }

        return metrics

    def predict(self, X: pd.DataFrame, quantile: Optional[float] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_arr = X.values.astype(np.float64)
        if quantile is not None:
            return self.model.predict(X_arr, quantile=quantile)
        return self.model.predict(X_arr)

    def predict_probability(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        q17 = self.predict(X, quantile=0.17)
        q50 = self.predict(X, quantile=0.50)
        q83 = self.predict(X, quantile=0.83)
        probs = ((q17 >= threshold).astype(float) +
                 (q50 >= threshold).astype(float) +
                 (q83 >= threshold).astype(float)) / 3
        return probs

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained.")
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

    def save(self, path: str):
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str) -> 'TempPredictor':
        instance = cls()
        instance.model = lgb.Booster(model_file=path)
        instance.feature_names = instance.model.feature_name()
        return instance