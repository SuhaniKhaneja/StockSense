import pandas as pd
import pytest
from stocksense.data import generate_sales, load_sales
from stocksense.features import build_features, extend_future, HORIZON, FEATURES
from stocksense.metrics import wape, coverage
from stocksense.inventory import recommend


@pytest.fixture(scope="module")
def df():
    return generate_sales(n_skus=3, days=200)


def test_generator_is_reproducible_and_valid(df):
    assert df.equals(generate_sales(n_skus=3, days=200))
    assert (df["units"] >= 0).all() and df["sku"].nunique() == 3


def test_features_do_not_leak_future(df):
    """Altering units from `cutoff` onward must not change any feature before cutoff+HORIZON."""
    base = build_features(df)
    changed = df.copy()
    cutoff = df["date"].iloc[120]
    changed.loc[changed["date"] >= cutoff, "units"] += 999
    new = build_features(changed)
    early = base["date"] < cutoff + pd.Timedelta(days=HORIZON)
    pd.testing.assert_frame_equal(base.loc[early, FEATURES], new.loc[early, FEATURES])


def test_future_rows_have_complete_features(df):
    f = build_features(extend_future(df))
    fut = f[f["units"].isna()]
    assert len(fut) == 3 * HORIZON and fut[FEATURES].notna().all().all()


def test_metrics():
    assert wape([10, 10], [8, 12]) == pytest.approx(0.2)
    assert coverage([1, 5], [0, 6], [2, 8]) == 0.5


def test_load_sales_validates(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-01"], "sku": ["A"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_sales(p)


def test_inventory_flags_low_stock():
    fc = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=14), "sku": "A",
                       "pred": 10.0, "lo": 7.0, "hi": 13.0})
    low = recommend(fc, pd.DataFrame({"sku": ["A"], "on_hand": [20], "on_order": [0]}))
    high = recommend(fc, pd.DataFrame({"sku": ["A"], "on_hand": [500], "on_order": [0]}))
    assert low.iloc[0]["action"] == "Reorder now" and low.iloc[0]["order_qty"] > 0
    assert high.iloc[0]["action"] == "OK" and high.iloc[0]["order_qty"] == 0
