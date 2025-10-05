
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Portfolio Dashboard — 3-Stock Plan", layout="wide")
st.title("AI Portfolio Dashboard — 3-Stock Plan")

# -- Load data
@st.cache_data
def load_positions(path="positions.csv"):
    df = pd.read_csv(path)
    for c in ["buy_price","buy_limit","stop","expected_return_6m","qty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_indicators(df):
    df = df.copy()
    df["50dma"] = df["Close"].rolling(50).mean()
    df["200dma"] = df["Close"].rolling(200).mean()
    # ATR(14)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["52wk_high"] = df["Close"].rolling(252).max()
    return df

@st.cache_data
def load_history(tickers, period_days):
    hist = {}
    for t in tickers:
        data = yf.download(t, period=f"{period_days}d", interval="1d", progress=False, auto_adjust=True)
        if not data.empty:
            hist[t] = compute_indicators(data)
    return hist

pos = load_positions()
tickers = pos["ticker"].tolist()
hist = load_history(tickers, 365)

# --- Sidebar controls
with st.sidebar:
    st.header("Controls")
    lookback = st.slider("Lookback (days)", 120, 730, 365)
    show_atr = st.checkbox("Show ATR(14)", value=True)
    show_ma = st.checkbox("Show 50/200-DMA", value=True)
    st.caption("Edit the table and click Save to persist.")

# --- Summary
def summarize(df):
    out = df.copy()
    out["notional"] = out["qty"] * out["buy_price"]
    out["risk_to_stop"] = (out["buy_price"] - out["stop"]).clip(lower=0) * out["qty"]
    out["exp_gain_$"] = out["notional"] * out["expected_return_6m"]
    out["exp_value_6m"] = out["notional"] + out["exp_gain_$"]
    totals = {
        "budget": out["notional"].sum(),
        "risk_total": out["risk_to_stop"].sum(),
        "expected_gain_total": out["exp_gain_$"].sum(),
        "expected_value_6m": out["exp_value_6m"].sum(),
        "weighted_expected_return": (out["exp_gain_$"].sum() / max(out["notional"].sum(),1e-9))
    }
    return out, totals

st.subheader("Holdings")
edited = st.data_editor(pos, num_rows="dynamic", use_container_width=True, key="positions_editor")
summ, totals = summarize(edited)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Budget (sum of buys)", f"${totals['budget']:.2f}")
c2.metric("Total Risk to Stops", f"${totals['risk_total']:.2f}")
c3.metric("Expected Gain (6m)", f"${totals['expected_gain_total']:.2f}")
c4.metric("Weighted Exp. Return (6m)", f"{totals['weighted_expected_return']*100:.1f}%")

st.divider()

# --- Charts
for t in tickers:
    st.subheader(t)
    left, right = st.columns([2,1])
    if t not in hist:
        right.warning("No price history")
        continue
    df = hist[t].dropna().reset_index()
    df = df.tail(lookback)
    fig = px.line(df, x="Date", y="Close", title=f"{t} — Price")
    if show_ma:
        fig.add_scatter(x=df["Date"], y=df["50dma"], mode="lines", name="50-DMA")
        fig.add_scatter(x=df["Date"], y=df["200dma"], mode="lines", name="200-DMA")
    left.plotly_chart(fig, use_container_width=True)
    if show_atr:
        fig2 = px.line(df, x="Date", y="atr14", title=f"{t} — ATR(14)")
        left.plotly_chart(fig2, use_container_width=True)

    latest = df.iloc[-1]
    price = latest["Close"]; dma50 = latest["50dma"]; dma200 = latest["200dma"]
    atr = latest["atr14"]; high52 = latest["52wk_high"]
    dist_high = (price/high52 - 1)*100 if high52>0 else np.nan
    dist_50 = (price/dma50 - 1)*100 if dma50>0 else np.nan
    dist_200 = (price/dma200 - 1)*100 if dma200>0 else np.nan

    row = edited[edited["ticker"]==t]
    buy = float(row["buy_price"].iloc[0]) if not row.empty else float(price)
    stop = float(row["stop"].iloc[0]) if not row.empty else float(price*0.85)
    qty  = float(row["qty"].iloc[0]) if not row.empty else 0.0
    risk = max(buy-stop,0)*qty
    exp_r = float(row["expected_return_6m"].iloc[0]) if not row.empty else 0.0
    exp_gain = buy*qty*exp_r

    right.write(f"**Price**: ${price:.2f}")
    right.write(f"**50‑DMA / 200‑DMA**: ${dma50:.2f} / ${dma200:.2f}")
    right.write(f"**Dist to 52‑wk high**: {dist_high:.2f}%")
    right.write(f"**vs 50/200‑DMA**: {dist_50:.2f}% / {dist_200:.2f}%")
    right.write(f"**ATR(14)**: ${atr:.2f}")
    right.write("---")
    right.write(f"**Plan** — buy: ${buy:.2f} | stop: ${stop:.2f} | qty: {qty}")
    right.write(f"**Risk to stop**: ${risk:.2f} | **Expected gain (6m)**: ${exp_gain:.2f}")
    flags = []
    if price < dma200: flags.append("⚠ Below 200-DMA")
    if price < dma50: flags.append("⚠ Below 50-DMA")
    if not flags: flags = ["✅ Trend intact"]
    right.write("**Status:** " + "  •  ".join(flags))

st.divider()
st.subheader("Save changes")
if st.button("Save positions.csv"):
    edited.to_csv("positions.csv", index=False)
    st.success("Saved positions.csv")

# --- Watchlist with conditional triggers
st.divider()
st.subheader("Watchlist — Conditional Triggers")
try:
    wl = pd.read_csv("watchlist.csv")
except Exception:
    wl = pd.DataFrame(columns=["ticker","status","trigger_type","trigger_rule","buy_limit_rule","initial_stop_rule","add_on_rule","notes"])

st.dataframe(wl, use_container_width=True)

st.caption("Data source: Yahoo Finance (via yfinance). Research only; not investment advice.")
