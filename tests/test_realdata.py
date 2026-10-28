import pandas as pd
from stocksense.realdata import build_panel, read_raw
from stocksense.pipeline import run


def _raw(days=30):
    rows = []
    for d in pd.date_range("2024-01-01", periods=days):
        for sku, q in (("10001", 5), ("10002", 2)):
            rows.append(("INV" + d.strftime("%m%d") + sku, sku, q, d, 2.5))
    rows += [("C99999", "10001", -50, pd.Timestamp("2024-01-05"), 2.5),   # cancellation
             ("INV0105X", "POST", 1, pd.Timestamp("2024-01-05"), 9.0),    # non-product code
             ("INV0106X", "10003", 4, pd.Timestamp("2024-01-06"), 1.0)]   # too few active days
    return pd.DataFrame(rows, columns=["invoice", "sku", "qty", "ts", "price"])


def test_panel_filters_and_fills_zero_days():
    p = build_panel(_raw(), top=10, min_days=10)
    assert set(p["sku"]) == {"10001", "10002"}            # POST, cancellation, sparse SKU removed
    assert p.groupby("sku").size().nunique() == 1          # dense spine, same days per SKU
    assert (p[p["sku"] == "10001"]["units"] == 5).all()    # cancellation did not reduce sales


def test_read_raw_checks_columns(tmp_path):
    f = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(f, index=False)
    try:
        read_raw(f)
        assert False
    except ValueError as e:
        assert "missing" in str(e)


def test_pipeline_runs_on_real_format_file(tmp_path):
    raw = _raw(days=420).rename(columns={"invoice": "Invoice", "sku": "StockCode", "qty": "Quantity",
                                         "ts": "InvoiceDate", "price": "Price"})
    f = tmp_path / "retail.csv"
    raw.to_csv(f, index=False)
    panel = build_panel(read_raw(f), top=10, min_days=10)
    panel.to_csv(tmp_path / "sales.csv", index=False)
    summary, boot = run(tmp_path / "sales.csv", out=tmp_path / "art", data_path=tmp_path / "copy.csv")
    assert (tmp_path / "art" / "forecast.csv").exists() and "Seasonal naive" in boot
