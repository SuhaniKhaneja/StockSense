import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from stocksense.inventory import recommend, simulate_stock

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
SALES = ROOT / "data" / "sales.csv"
if not SALES.exists():  # deployed copy: data/ is not in git
    SALES = ART / "sales_recent.csv"
INK, TEAL, AMBER, MIST = "#14302f", "#1f7a6d", "#d98e04", "#e8efec"

st.set_page_config(page_title="StockSense", page_icon="ðŸ“¦", layout="wide")
st.markdown(f"""<style>
html, body, [class*="css"] {{ font-family: 'Source Sans 3', 'Segoe UI', sans-serif; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
[data-testid="stMetric"] {{ background: {MIST}; padding: 14px 18px; border-radius: 6px; }}
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
    st.info(f"Data: {meta['data_source']}. Stock levels are simulated for this demo.")

stock = simulate_stock(sorted(fc["sku"].unique()), fc)
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
    fig.add_scatter(x=h["date"], y=h["units"], name="Actual sales", line=dict(color=INK, width=2))
    fig.add_scatter(x=f["date"], y=f["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=f["date"], y=f["lo"], fill="tonexty", fillcolor="rgba(217,142,4,0.25)",
                    line=dict(width=0), name="80% range")
    fig.add_scatter(x=f["date"], y=f["pred"], name="Forecast", line=dict(color=AMBER, width=3))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Units per day",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

with tab3:
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
