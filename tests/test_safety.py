import numpy as np
import pandas as pd
import pytest

from stocksense.safety import MAX_WEEKS, SERVICES, compute_table, recommend_empirical, tsb_levels

DATES = pd.date_range("2024-01-01", periods=300)


def _sales(n_skus=8, seed=0):
    rng = np.random.default_rng(seed)
    return pd.concat([
        pd.DataFrame({"date": DATES, "sku": f"s{i}",
                      "units": rng.poisson(4, len(DATES)) * (rng.random(len(DATES)) < 0.4)
                      * rng.choice([1, 1, 5], len(DATES)).astype(float)})
        for i in range(n_skus)], ignore_index=True)


def test_tsb_levels_are_causal():
    y = np.array([0, 3, 0, 0, 5, 2, 0, 4.0])
    y2 = y.copy()
    y2[-1] = 999
    assert np.allclose(tsb_levels(y)[:-1], tsb_levels(y2)[:-1])


def test_table_shape_and_monotonic_in_service():
    t = compute_table(_sales())
    assert len(t) == 8 * MAX_WEEKS * len(SERVICES)
    assert (t["safety"] >= 0).all() and (t["demand"] >= 0).all()
    for _, g in t.groupby(["sku", "weeks"]):
        assert g.sort_values("service")["safety"].is_monotonic_increasing
    one, two = t[t["weeks"] == 1].set_index(["sku", "service"]), t[t["weeks"] == 2].set_index(["sku", "service"])
    assert np.allclose(two["demand"], 2 * one["demand"])


def _toy_table():
    rows = []
    for sku, d1, s1, d2, s2 in (("a", 10.0, 5.0, 20.0, 8.0), ("b", 10.0, 5.0, 20.0, 8.0)):
        rows += [{"sku": sku, "weeks": 1, "service": 0.95, "demand": d1, "safety": s1},
                 {"sku": sku, "weeks": 2, "service": 0.95, "demand": d2, "safety": s2}]
    return pd.DataFrame(rows)


def test_recommend_empirical_logic():
    stock = pd.DataFrame({"sku": ["a", "b"], "on_hand": [10, 100], "on_order": [0, 0]})
    rec = recommend_empirical(_toy_table(), stock, lead_time=7, review=7, service=0.95).set_index("sku")
    assert rec.loc["a", "reorder_point"] == 15.0            # 10 demand + 5 safety (1 week)
    assert rec.loc["a", "action"] == "Reorder now"
    assert rec.loc["a", "order_qty"] == 18                   # order-up-to 20 + 8 = 28, minus 10
    assert rec.loc["b", "action"] == "OK" and rec.loc["b", "order_qty"] == 0
    assert list(rec.columns) == ["on_hand", "on_order", "demand_lead_time", "safety_stock",
                                 "reorder_point", "days_of_cover", "order_qty", "action"]


def test_more_service_never_orders_less():
    t = compute_table(_sales())
    stock = pd.DataFrame({"sku": sorted(t["sku"].unique()), "on_hand": 5, "on_order": 0})
    q = {s: recommend_empirical(t, stock, 7, 7, s).set_index("sku")["order_qty"] for s in SERVICES}
    for lo, hi in zip(SERVICES[:-1], SERVICES[1:]):
        assert (q[hi] >= q[lo]).all()
