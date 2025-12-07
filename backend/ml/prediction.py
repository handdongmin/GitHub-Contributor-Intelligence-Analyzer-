# backend/ml/prediction.py
"""
Lightweight productivity prediction using linear regression.
"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class ActivityStats:
    commits: int
    issues: int
    additions: int
    deletions: int
    active_days: int
    prs: int = 0

    def to_features(self) -> List[float]:
        churn = self.additions + self.deletions
        velocity = churn / max(self.active_days, 1)
        return [
            float(self.commits),
            float(self.issues),
            float(self.prs),
            float(self.additions),
            float(self.deletions),
            float(churn),
            float(velocity),
        ]


class ProductivityRegressor:
    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, X: Iterable[ActivityStats], y: Iterable[float]) -> None:
        matrix = np.array([row.to_features() for row in X], dtype=float)
        target = np.array(list(y), dtype=float)
        self.model.fit(matrix, target)

    def predict(self, X: Iterable[ActivityStats]) -> List[float]:
        matrix = np.array([row.to_features() for row in X], dtype=float)
        return list(self.model.predict(matrix))

    def evaluate(self, X, y, test_size=0.2, random_state=42):
        matrix = np.array([row.to_features() for row in X], dtype=float)
        target = np.array(list(y), dtype=float)

        n_samples = len(matrix)

        # 데이터가 너무 적으면 split 불가 → full train
        if n_samples == 0:
            return 0.0, None
        if n_samples < 2:
            self.model.fit(matrix, target)
            preds = self.model.predict(matrix)
            mae = mean_absolute_error(target, preds)
            return mae, None
        if n_samples < 4:
            self.model.fit(matrix, target)
            preds = self.model.predict(matrix)
            mae = mean_absolute_error(target, preds)
            r2 = r2_score(target, preds)
            return mae, r2

        X_train, X_test, y_train, y_test = train_test_split(
            matrix, target, test_size=test_size, random_state=random_state
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        return mae, r2


    def feature_importance(self) -> List[Tuple[str, float]]:
        feature_names = [
            "commits",
            "issues",
            "prs",
            "additions",
            "deletions",
            "churn",
            "velocity",
        ]
        if not hasattr(self.model, "coef_"):
            return []
        return list(zip(feature_names, self.model.coef_.tolist()))
