"""Empirical safety stock from each product's own recent weekly forecast errors.

Same idea as the inventory backtest: the weekly TSB forecast plus a quantile of that
product's past forecast errors, floored at zero. Here it is extended to k = 1..4 weeks
of cover, using errors of cumulative k-week demand.

Usage:  python -m stocksense.safety .\\data\\sales.csv
Writes: artifacts/safety_empirical.csv  (sku, weeks, service, demand, safety)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .weekly_model import make_weekly

SERVICES = (0.90, 0.95, 0.97, 0.99)
MAX_WEEKS = 4
WINDOW = 26        # most recent forecast origins used for the error distribution
MIN_HISTORY = 8    # weeks of history needed before the first usable origin


def tsb_levels(y, alpha=0.2, beta=0.1):
    """TSB level after each week (causal: level[i] uses y[:i+1] only)."""
    level = np.zeros(len(y))
    p, z = 0.5, None
    for i, v in enumerate(y):
        if v > 0:
            z = v if z is None else z + alpha * (v - z)
            p += beta * (1 - p)
        else:
            p *= (1 - beta)
        level[i] = p * z if z is not None else 0.0
    return level


def compute_table(df, services=SERVICES, max_weeks=MAX_WEEKS, window=WINDOW):
    """One row per (sku, weeks of cover, service level):
    demand = expected demand over those weeks from now, safety = error quantile."""
    w = make_weekly(df)
    rows = []
    for sku, g in w.groupby("sku"):
        y = g.sort_values("t")["units"].to_numpy(dtype=float)
        level = tsb_levels(y)
        n = len(y)
        for k in range(1, max_weeks + 1):
            origins = list(range(MIN_HISTORY, n - k))[-window:]
            err = np.array([y[o + 1:o + k + 1].sum() - k * level[o] for o in origins])
            for s in services:
                q = max(float(np.quantile(err, s)), 0.0) if len(err) else 0.0
                rows.append({"sku": sku, "weeks": k, "service": s,
                             "demand": float(k * level[-1]), "safety": q})
    return pd.DataFrame(rows)


def recommend_empirical(table, stock, lead_time=7, review=7, service=0.95):
    """Same output columns as inventory.recommend. Lead time and review are in days and
    are rounded up to whole weeks (max 4). Reorder point = demand over the lead time +
    safety; order-up-to level = demand over lead time + review + safety."""
    k_lt = min(MAX_WEEKS, max(1, math.ceil(lead_time / 7)))
    k_h = min(MAX_WEEKS, max(1, math.ceil((lead_time + review) / 7)))
    t = table[np.isclose(table["service"], service)]
    lt = t[t["weeks"] == k_lt].set_index("sku")
    hz = t[t["weeks"] == k_h].set_index("sku")
    out = []
    for _, s in stock.iterrows():
        sku = s["sku"]
        if sku not in lt.index or sku not in hz.index:
            continue
        demand_lt, safety = float(lt.loc[sku, "demand"]), float(lt.loc[sku, "safety"])
        rop = demand_lt + safety
        order_up_to = float(hz.loc[sku, "demand"]) + float(hz.loc[sku, "safety"])
        position = s["on_hand"] + s["on_order"]
        daily = max(demand_lt / (7 * k_lt), 1e-9)
        out.append({"sku": sku, "on_hand": int(s["on_hand"]), "on_order": int(s["on_order"]),
                    "demand_lead_time": round(demand_lt, 1), "safety_stock": round(safety, 1),
                    "reorder_point": round(rop, 1),
                    "days_of_cover": round(s["on_hand"] / daily, 1),
                    "order_qty": max(0, math.ceil(order_up_to - position)),
                    "action": "Reorder now" if position <= rop else "OK"})
    return pd.DataFrame(out).sort_values(["action", "days_of_cover"], ascending=[False, True])


def main(path):
    df = pd.read_csv(path, parse_dates=["date"])
    table = compute_table(df)
    Path("artifacts").mkdir(exist_ok=True)
    table.round(4).to_csv("artifacts/safety_empirical.csv", index=False)
    print(table[table["weeks"] == 1].groupby("service")["safety"].describe().round(1).to_string())
    print(f"\nWrote artifacts/safety_empirical.csv ({len(table)} rows, {table['sku'].nunique()} products)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv")
