import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stocksense.inventory import recommend, simulate_stock
from stocksense.safety import recommend_empirical

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
SAFETY = ART / "safety_empirical.csv"
SALES = ROOT / "data" / "sales.csv"
if not SALES.exists():  # deployed copy: data/ is not in git
    SALES = ART / "sales_recent.csv"
INK, TEAL, AMBER, MIST = "#14302f", "#1f7a6d", "#d98e04", "#e8efec"
HIST = "#7f9a96"  # readable on both light and dark themes

st.set_page_config(page_title="StockSense", page_icon="\U0001F4E6", layout="wide")
st.markdown(f"""<style>
html, body, [class*="css"] {{ font-family: 'Source Sans 3', 'Segoe UI', sans-serif; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stMetric"] {{ background: {MIST}; padding: 14px 18px; border-radius: 6px; }}
[data-testid="stMetric"] * {{ color: {INK} !important; }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    if not (ART / "forecast.csv").exists():
        return None
    return (pd.read_csv(ART / "forecast.csv", parse_dates=["date"]),
            pd.read_csv(SALES, parse_dates=["date"]),
            pd.read_csv(ART / "backtest_summary.csv", index_col=0),
            json.load(open(ART / "meta.json")))


data = load()
st.title("StockSense")
st.caption("Forecast demand 14 days ahead and see which products to reorder.")
if data is None:
    st.error("No forecasts found. Run `python -m stocksense.pipeline` first, then reload this page.")
    st.stop()
fc, hist, summary, meta = data

with st.sidebar:
    st.header("Ordering policy")
    lead = st.slider("Supplier lead time (days)", 1, 14, 7)
    review = st.slider("Days of stock to order for", 1, 14, 7)
    service = st.select_slider("Service level", [0.90, 0.95, 0.97, 0.99], value=0.95)
    INTERVAL = "Model interval (normal approximation)"
    EMPIRICAL = "Empirical (recent forecast errors)"
    method = st.radio("Safety stock method", [EMPIRICAL, INTERVAL] if SAFETY.exists() else [INTERVAL])
    if method == EMPIRICAL:
        st.caption("Safety stock is a quantile of each product's own recent weekly forecast errors. "
                   "The backtest found this better suited to lumpy demand than a normal approximation. "
                   "Cover is rounded up to whole weeks (max 4); the backtest tested the one-week policy.")
    st.info("Data: UCI Online Retail II, daily sales for 136 products. "
            "Stock levels are simulated for this demo.")

stock = simulate_stock(sorted(fc["sku"].unique()), fc)
if method == EMPIRICAL:
    rec = recommend_empirical(pd.read_csv(SAFETY), stock, lead, review, service)
else:
    rec = recommend(fc, stock, lead, review, service)
tab1, tab2, tab3 = st.tabs(["Reorder list", "Forecast", "Model accuracy"])

with tab1:
    n = int((rec["action"] == "Reorder now").sum())
    a, b, c = st.columns(3)
    a.metric("Products to reorder now", n)
    b.metric("Units to order", f"{int(rec['order_qty'].sum()):,}")
    c.metric("Lowest cover (days)", f"{rec['days_of_cover'].min():.1f}")
    st.dataframe(rec, hide_index=True, width="stretch")

with tab2:
    sku = st.selectbox("Product", sorted(fc["sku"].unique()))
    h = hist[hist["sku"] == sku].tail(60)
    f = fc[fc["sku"] == sku]
    fig = go.Figure()
    fig.add_scatter(x=h["date"], y=h["units"], name="Actual sales", line=dict(color=HIST, width=2))
    fig.add_scatter(x=f["date"], y=f["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=f["date"], y=f["lo"], fill="tonexty", fillcolor="rgba(217,142,4,0.25)",
                    line=dict(width=0), name="80% range")
    fig.add_scatter(x=f["date"], y=f["pred"], name="Forecast", line=dict(color=AMBER, width=3))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Units per day",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

with tab3:
    st.subheader("Daily forecasting")
    st.write("Rolling-origin backtest: 4 folds of 14 days, trained only on data known at each forecast date. "
             "Lower WAPE is better.")
    s = summary.reset_index()
    bar = go.Figure(go.Bar(x=s["wape"], y=s["model"], orientation="h", marker_color=TEAL))
    bar.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="WAPE")
    st.plotly_chart(bar, width="stretch")
    st.dataframe(s.round(4), hide_index=True, width="stretch")
    boot = ART / "bootstrap.json"
    if boot.exists():
        b = json.load(open(boot))["Seasonal naive"]
        st.write(f"Versus seasonal naive: {b['relative_wape_reduction']:.1%} lower WAPE "
                 f"(95% CI {b['ci95'][0]:.1%} to {b['ci95'][1]:.1%}, resampling {b['n_skus']} products).")
    st.caption("This is the daily grain, where error is dominated by when large orders land. "
               "Daily WAPE is high (about 0.89). See the weekly results below.")

    weekly = ART / "weekly_native.csv"
    if weekly.exists():
        st.subheader("Weekly planning grain")
        st.write("Weekly WAPE by demand segment, on the same test weeks. Lower is better. "
                 "A 4-week average and TSB (an intermittent-demand method) are statistically tied, "
                 "and both clearly beat a gradient-boosting model trained on weekly totals.")
        st.dataframe(pd.read_csv(weekly).round(3), hide_index=True, width="stretch")

    inv = ART / "inventory_backtest.csv"
    if inv.exists():
        st.subheader("Inventory backtest")
        st.write("Weekly order-up-to policy with safety stock from each method's own recent errors. "
                 "Fill rate is the share of demand served; excess stock is leftover stock as a share of demand.")
        st.dataframe(pd.read_csv(inv).round(3), hide_index=True, width="stretch")
        st.caption("A naive forecast needs roughly 10 to 13 points of demand more excess stock than the "
                   "4-week average for the same fill rate (approximate). TSB and the 4-week average sit on "
                   "essentially the same trade-off. No method reaches its target service level.")
