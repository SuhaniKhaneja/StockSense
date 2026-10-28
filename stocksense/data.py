"""Synthetic retail sales generator (documented, seeded, reproducible) and CSV loader."""
import numpy as np
import pandas as pd

CATEGORIES = ["Apparel", "Home", "Beauty", "Electronics", "Grocery"]


def generate_sales(n_skus=30, start="2023-01-01", days=1095, seed=42):
    """Daily unit sales per SKU: trend + weekly + yearly seasonality + promo lift + count noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    dow = dates.dayofweek.values
    frames = []
    for i in range(n_skus):
        base = rng.uniform(5, 60)
        trend = rng.uniform(-0.0004, 0.0010)
        weekend = rng.uniform(0.1, 0.5)
        yearly = rng.uniform(0.05, 0.4)
        phase = rng.uniform(0, 2 * np.pi)
        price = rng.uniform(5, 120)
        starts = (rng.random(days) < 0.03).astype(int)
        promo = pd.Series(starts).rolling(3, min_periods=1).max().values.astype(int)
        mu = (base * (1 + trend * t)
              * (1 + weekend * np.where(dow >= 5, 1.0, -0.4))
              * (1 + yearly * np.sin(2 * np.pi * t / 365.25 + phase))
              * (1 + 0.35 * promo))
        units = rng.negative_binomial(10, 10 / (10 + mu))
        frames.append(pd.DataFrame({
            "date": dates, "sku": f"SKU-{i + 1:03d}", "category": CATEGORIES[i % len(CATEGORIES)],
            "units": units, "price": np.round(price * (1 - 0.2 * promo), 2), "promo": promo}))
    return pd.concat(frames, ignore_index=True)


def load_sales(path):
    """Load a real dataset. Needs columns: date, sku, units (price, promo, category optional)."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in ("date", "sku", "units") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if "price" not in df.columns:
        df["price"] = 1.0
    if "promo" not in df.columns:
        df["promo"] = 0
    if "category" not in df.columns:
        df["category"] = "All"
    if (df["units"] < 0).any():
        raise ValueError("units must be non-negative")
    return df.sort_values(["sku", "date"]).reset_index(drop=True)
