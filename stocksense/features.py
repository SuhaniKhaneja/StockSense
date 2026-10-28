"""Leakage-safe features: every history-based feature uses data at least HORIZON days old."""
import numpy as np
import pandas as pd

HORIZON = 14
LAGS = (14, 21, 28, 35)
FEATURES = ["lag_14", "lag_21", "lag_28", "lag_35", "roll_7", "roll_28", "dow", "month",
            "doy_sin", "doy_cos", "promo", "price_ratio", "sku_code"]


def extend_future(df, horizon=HORIZON):
    """Append `horizon` future rows per SKU (units unknown, no promo, median price)."""
    last = df["date"].max()
    fut = pd.date_range(last + pd.Timedelta(days=1), periods=horizon)
    parts = []
    for sku, g in df.groupby("sku"):
        parts.append(pd.DataFrame({"date": fut, "sku": sku, "category": g["category"].iloc[0],
                                   "units": np.nan, "price": g["price"].median(), "promo": 0}))
    return pd.concat([df, *parts], ignore_index=True)


def build_features(df, horizon=HORIZON):
    df = df.sort_values(["sku", "date"]).reset_index(drop=True).copy()
    g = df.groupby("sku")["units"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(horizon)
    df["roll_7"] = shifted.groupby(df["sku"]).transform(lambda s: s.rolling(7).mean())
    df["roll_28"] = shifted.groupby(df["sku"]).transform(lambda s: s.rolling(28).mean())
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    med = df.groupby("sku")["price"].transform("median")
    df["price_ratio"] = df["price"] / med
    df["sku_code"] = df["sku"].astype("category").cat.codes
    return df
