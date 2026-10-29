import numpy as np
import pandas as pd
import pytest

from stocksense.inventory_backtest import add_forecasts, simulate, summarize, tsb_forecasts
from stocksense.segments import classify_skus, wape_by_segment
from stocksense.weekly import to_weekly
from stocksense.weekly_model import bootstrap_reduction, make_weekly, run, tsb_rate

DATES = pd.date_range("2024-01-01", periods=300)


def panel(series):
    return pd.concat([pd.DataFrame({"date": DATES, "sku": k, "units": np.asarray(v, dtype=float)})
                      for k, v in series.items()], ignore_index=True)


def test_segments_classify_known_patterns():
    n = len(DATES)
    idx = np.arange(n)
    every3 = idx % 3 == 0
    smooth = np.full(n, 10.0)
    erratic = np.where(idx % 2 == 0, 1.0, 100.0)
    intermittent = np.where(every3, 10.0, 0.0)
    lumpy = np.where(every3, np.where((idx // 3) % 2 == 0, 1.0, 100.0), 0.0)
    segs = classify_skus(panel({"s": smooth, "e": erratic, "i": intermittent, "l": lumpy}))
    assert dict(zip(segs["sku"], segs["segment"])) == {
        "s": "smooth", "e": "erratic", "i": "intermittent", "l": "lumpy"}


def test_classify_respects_until():
    series = np.r_[np.full(150, 10.0), np.where(np.arange(150) % 2 == 0, 1.0, 100.0)]
    df = panel({"a": series})
    assert classify_skus(df, until=DATES[149])["segment"].iloc[0] == "smooth"
    assert classify_skus(df)["segment"].iloc[0] == "erratic"


def test_wape_by_segment_is_pooled():
    preds = pd.DataFrame({"model": "m", "sku": ["a", "a", "b", "b"],
                          "y": [10.0, 10.0, 5.0, 5.0], "p": [8.0, 12.0, 5.0, 0.0]})
    segs = pd.DataFrame({"sku": ["a", "b"], "segment": ["smooth", "lumpy"]})
    t = wape_by_segment(preds, segs)
    assert t.loc["smooth", "m"] == pytest.approx(4 / 20)
    assert t.loc["lumpy", "m"] == pytest.approx(5 / 10)
    assert t.loc["ALL", "m"] == pytest.approx(9 / 30)
    assert t.loc["ALL", "n_skus"] == 2


def test_to_weekly_preserves_totals():
    d = pd.DataFrame({"model": "m", "fold": 0, "date": pd.date_range("2024-03-01", periods=14),
                      "sku": "a", "y": np.arange(14.0), "p": np.ones(14)})
    w = to_weekly(d)
    assert len(w) == 2 and w["y"].sum() == d["y"].sum() and w["p"].sum() == 14
    assert sorted(w["y"]) == [21.0, 70.0]


def test_make_weekly_anchors_at_end_and_uses_only_old_weeks():
    n = 100  # 14 full weeks plus 2 leftover days, which must be dropped
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "sku": "a",
                       "units": np.arange(n, dtype=float)})
    w = make_weekly(df)
    assert w["week_id"].nunique() == n // 7
    wk = w.set_index("week_id")["units"]
    assert wk[0] == df["units"].tail(7).sum()
    row = w[w["week_id"] == 0].iloc[0]
    assert row["lag_2"] == wk[2] and row["lag_2"] != wk[1]
    assert row["roll_4"] == pytest.approx(wk[[2, 3, 4, 5]].mean())


def test_tsb_rate_basic():
    assert tsb_rate([0, 0, 0]) == 0.0
    assert tsb_rate([7.0] * 50) == pytest.approx(7.0, rel=1e-6)


def test_run_produces_all_models_on_same_rows():
    rng = np.random.default_rng(0)
    df = panel({f"s{i}": rng.poisson(4, len(DATES)) * (rng.random(len(DATES)) < 0.5)
                for i in range(6)})
    preds = run(df)
    assert set(preds["model"]) == {"Weekly naive", "Weekly avg (4 wk)", "TSB", "Weekly GBM"}
    counts = preds.groupby("model").size()
    assert counts.nunique() == 1 and counts.iloc[0] == 6 * 8
    assert preds["p"].notna().all() and (preds["p"] >= 0).all()


def test_bootstrap_reduction_sign():
    rng = np.random.default_rng(1)
    skus = [f"s{i}" for i in range(30)]
    y = rng.poisson(10, 30 * 8).astype(float)
    rows = [pd.DataFrame({"model": m, "sku": np.repeat(skus, 8), "y": y,
                          "p": y + scale * rng.normal(0, 3, y.size)})
            for m, scale in (("good", 0.5), ("bad", 2.0))]
    point, lo, hi = bootstrap_reduction(pd.concat(rows), "good", "bad", n=200)
    assert point > 0 and lo > 0 and lo <= point <= hi


def test_tsb_forecasts_are_causal():
    y = np.array([0, 3, 0, 0, 5, 2, 0, 4.0])
    y2 = y.copy()
    y2[-1] = 999  # changing the last week must not change any forecast
    a, b = tsb_forecasts(y), tsb_forecasts(y2)
    assert np.isnan(a[:2]).all()
    assert np.allclose(a, b, equal_nan=True)


def test_inventory_fill_rate_and_excess_rise_with_target():
    rng = np.random.default_rng(2)
    n = len(DATES)
    df = panel({f"s{i}": rng.poisson(5, n) * (rng.random(n) < 0.4) * rng.choice([1, 1, 6], n)
                for i in range(20)})
    table = summarize(simulate(add_forecasts(make_weekly(df))))
    for _, g in table.groupby("method"):
        g = g.sort_values("target")
        assert g["fill_rate"].is_monotonic_increasing
        assert g["excess_ratio"].is_monotonic_increasing
        assert g["fill_rate"].between(0, 1).all()
