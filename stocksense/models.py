"""Baselines and the gradient-boosting forecaster (point + 80% interval)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from .features import FEATURES


class SeasonalNaive:
    """Same weekday two weeks ago (freshest same-weekday value known at a 14-day horizon)."""
    name = "Seasonal naive"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lag_14"].to_numpy()


class SeasonalAverage:
    """Mean of the last four same-weekday values."""
    name = "Seasonal average (4 wk)"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[["lag_14", "lag_21", "lag_28", "lag_35"]].mean(axis=1).to_numpy()


class GBMForecaster:
    """Poisson-loss point model plus 10th/90th percentile models for an 80% interval."""
    name = "Gradient boosting"

    def __init__(self, seed=0):
        kw = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=24, random_state=seed,
                  categorical_features=[FEATURES.index("sku_code")])
        self.point = HGB(loss="poisson", **kw)
        self.lo = HGB(loss="quantile", quantile=0.1, **kw)
        self.hi = HGB(loss="quantile", quantile=0.9, **kw)

    def fit(self, X, y):
        X = X[FEATURES]
        self.point.fit(X, y)
        self.lo.fit(X, y)
        self.hi.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.point.predict(X[FEATURES]), 0, None)

    def predict_interval(self, X):
        p = self.predict(X)
        lo = np.minimum(np.clip(self.lo.predict(X[FEATURES]), 0, None), p)
        hi = np.maximum(self.hi.predict(X[FEATURES]), p)
        return p, lo, hi
