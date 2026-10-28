"""Real-data adapter for the UCI Online Retail II transactions, with a SQL layer (SQLite).
Usage: python -m stocksense.realdata path\to\online_retail_II.csv --top 150
Writes data/sales.csv in the schema the pipeline expects."""
import argparse
import sqlite3
from pathlib import Path
import pandas as pd

RENAME = {"Invoice": "invoice", "InvoiceNo": "invoice", "StockCode": "sku", "Quantity": "qty",
          "InvoiceDate": "ts", "Price": "price", "UnitPrice": "price"}
NEEDED = ["invoice", "sku", "qty", "ts", "price"]

# Cancellations (invoice starts with C), returns and non-product codes (POST, M, BANK CHARGES...)
# are excluded; the top-N products by volume with enough active days are kept.
PANEL_SQL = """
WITH sales AS (
  SELECT date(ts) AS date, sku, SUM(qty) AS units, AVG(price) AS price
  FROM raw
  WHERE invoice NOT LIKE 'C%' AND qty > 0 AND price > 0
    AND sku GLOB '[0-9][0-9][0-9][0-9][0-9]*'
  GROUP BY date(ts), sku
),
ranked AS (
  SELECT sku, COUNT(*) AS active_days,
         RANK() OVER (ORDER BY SUM(units) DESC) AS rnk
  FROM sales GROUP BY sku
)
SELECT s.date, s.sku, s.units, s.price
FROM sales s JOIN ranked r ON r.sku = s.sku
WHERE r.rnk <= :top AND r.active_days >= :min_days
ORDER BY s.sku, s.date
"""


def read_raw(path):
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):  # needs openpyxl; both sheets are combined
        raw = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
    else:
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = pd.read_csv(path, encoding="latin-1")
    raw = raw.rename(columns=RENAME)
    missing = [c for c in NEEDED if c not in raw.columns]
    if missing:
        raise ValueError(f"Expected Online Retail II columns; missing {missing}. Found: {list(raw.columns)}")
    return raw[NEEDED]


def build_panel(raw, top=150, min_days=200):
    """Raw transactions -> dense daily panel (date, sku, category, units, price, promo)."""
    raw = raw.copy()
    raw["invoice"] = raw["invoice"].astype(str)
    raw["sku"] = raw["sku"].astype(str)
    raw["ts"] = pd.to_datetime(raw["ts"])
    con = sqlite3.connect(":memory:")
    raw.to_sql("raw", con, index=False)
    panel = pd.read_sql(PANEL_SQL, con, params={"top": top, "min_days": min_days})
    con.close()
    if panel.empty:
        raise ValueError("No products passed the filters; lower --min-days or check the file.")
    panel["date"] = pd.to_datetime(panel["date"])
    days = pd.date_range(raw["ts"].min().normalize(), raw["ts"].max().normalize())
    idx = pd.MultiIndex.from_product([sorted(panel["sku"].unique()), days], names=["sku", "date"])
    p = panel.set_index(["sku", "date"]).reindex(idx)
    p["units"] = p["units"].fillna(0).astype(int)
    p["price"] = p.groupby(level="sku")["price"].transform(lambda s: s.ffill().bfill())
    p = p.reset_index()
    p["category"], p["promo"] = "All", 0
    return p[["date", "sku", "category", "units", "price", "promo"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--min-days", type=int, default=200)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "sales.csv"))
    a = ap.parse_args()
    df = build_panel(read_raw(a.path), a.top, a.min_days)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{df['sku'].nunique()} products x {df['date'].nunique()} days -> {a.out}")
