"""Reorder-point policy driven by the forecast and its uncertainty."""
import math
from statistics import NormalDist
import numpy as np
import pandas as pd

Z80 = NormalDist().inv_cdf(0.9)  # the 10-90 interval spans 2*Z80 standard deviations


def recommend(forecast, stock, lead_time=7, review=7, service=0.95):
    """forecast: date, sku, pred, lo, hi (daily). stock: sku, on_hand, on_order."""
    z = NormalDist().inv_cdf(service)
    out = []
    for sku, g in forecast.sort_values("date").groupby("sku"):
        sigma = ((g["hi"] - g["lo"]) / (2 * Z80)).to_numpy()
        demand_lt = float(g["pred"].head(lead_time).sum())
        safety = z * math.sqrt(float((sigma[:lead_time] ** 2).sum()))
        rop = demand_lt + safety
        h = lead_time + review
        order_up_to = float(g["pred"].head(h).sum()) + z * math.sqrt(float((sigma[:h] ** 2).sum()))
        s = stock[stock["sku"] == sku].iloc[0]
        position = s["on_hand"] + s["on_order"]
        out.append({"sku": sku, "on_hand": int(s["on_hand"]), "on_order": int(s["on_order"]),
                    "demand_lead_time": round(demand_lt, 1), "safety_stock": round(safety, 1),
                    "reorder_point": round(rop, 1),
                    "days_of_cover": round(s["on_hand"] / max(g["pred"].head(lead_time).mean(), 1e-9), 1),
                    "order_qty": max(0, math.ceil(order_up_to - position)),
                    "action": "Reorder now" if position <= rop else "OK"})
    return pd.DataFrame(out).sort_values(["action", "days_of_cover"], ascending=[False, True])


def simulate_stock(skus, forecast, seed=7):
    """Demo-only stock levels (labeled simulated in the UI)."""
    rng = np.random.default_rng(seed)
    daily = forecast.groupby("sku")["pred"].mean()
    return pd.DataFrame({"sku": skus,
                         "on_hand": [int(daily[s] * rng.uniform(2, 20)) for s in skus],
                         "on_order": [int(daily[s] * rng.choice([0, 0, 7])) for s in skus]})
