import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from strategy import backtest, clean_prices, rolling_six_year, stats

st.set_page_config(page_title="Monthly Index Rule", page_icon="📈", layout="wide")
st.title("Does the monthly index rule beat buy and hold?")
st.caption("A transparent backtest of the rule in the screenshot. Past results do not predict future returns.")

with st.sidebar:
    st.header("Data and assumptions")
    source_choice = st.selectbox("Price data", ["Bundled SPY snapshot (fast)", "Yahoo Finance (live)"],
                                 help="The bundled file loads without waiting for an external price service.")
    ticker = st.text_input("Yahoo ticker", "SPY").strip().upper() if source_choice == "Yahoo Finance (live)" else "SPY"
    uploaded = st.file_uploader("Or upload dividend-adjusted daily prices (Date, Close CSV)", type="csv")
    start = st.date_input("Start date", value=pd.Timestamp("2004-01-02"), min_value=pd.Timestamp("1900-01-01"))
    end = st.date_input("End date (inclusive)", value=pd.Timestamp.today().date())
    comparison_label = st.radio("At month-end, compare price with", ["First close of that month", "Original purchase price"],
                                help="The post is ambiguous. The first choice tests whether the month lost value; "
                                     "the second can hold for years as long as the position remains above its entry.")
    comparison = "month_start" if comparison_label == "First close of that month" else "entry"
    stop_enabled = st.checkbox("Apply a close-based stop", value=False)
    stop = st.slider("Stop below entry (%)", 7, 12, 10) / 100 if stop_enabled else None
    fee = st.number_input("Cost per trade (basis points)", 0.0, 100.0, 0.0, 0.5)


@st.cache_data(show_spinner=False)
def bundled_prices():
    path = Path(__file__).with_name("spy_adjusted_snapshot.csv")
    if not path.exists():
        raise FileNotFoundError("Missing spy_adjusted_snapshot.csv. Add this file alongside app.py.")
    return clean_prices(pd.read_csv(path))


@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_prices(symbol, start_date, end_date):
    import yfinance as yf

    data = yf.download(symbol, start=str(start_date),
                       end=str(pd.Timestamp(end_date) + pd.Timedelta(days=1)),
                       auto_adjust=True, actions=False, progress=False, threads=False,
                       timeout=8)
    if data.empty:
        raise ValueError("Yahoo returned no prices, possibly due to rate limiting. Select the bundled SPY data.")
    series = data["Close"]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return clean_prices(series.rename("Close").reset_index().rename(columns={"index": "Date"}))


try:
    if end <= start:
        raise ValueError("End date must be after start date.")
    if uploaded is not None:
        prices = clean_prices(pd.read_csv(io.BytesIO(uploaded.getvalue())))
        prices = prices.loc[str(start):str(end)]
        if len(prices) < 10:
            raise ValueError("The selected dates contain fewer than ten valid daily rows.")
        label = "Uploaded data"
    else:
        if source_choice == "Bundled SPY snapshot (fast)":
            prices = bundled_prices().loc[str(start):str(end)]
            label = "SPY bundled adjusted-close snapshot"
        else:
            if not ticker:
                raise ValueError("Enter a Yahoo ticker or upload a CSV.")
            prices = yahoo_prices(ticker, start, end)
            label = f"{ticker} via Yahoo Finance"
        if len(prices) < 10:
            raise ValueError("The selected dates contain fewer than ten available daily prices.")
    curves, trades = backtest(prices, stop_pct=stop, fee_bps=fee, comparison=comparison)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.info("**Rules used:** Buy at the first available close. At each month-end close, sell if the price is below "
        f"the **{comparison_label.lower()}**. After a sale, re-enter only if a later month's first-day close "
        "exceeds the preceding month's final close. Signals execute at the **next trading day's adjusted "
        "close**. A trade execution day cannot trigger another trade. An optional stop checks daily closes "
        "against the position's entry price and exits at the next close. "
        "Cash earns 0%. Prices include reinvested dividends when Yahoo adjusted data or an adjusted CSV is used.")
st.caption(f"{label}: {len(curves):,} sessions, {curves.index[0]:%b %d, %Y}–{curves.index[-1]:%b %d, %Y}. "
           "If the selected dates begin mid-month, that first day is treated as the initial entry.")
if "snapshot" in label:
    st.warning("The bundled SPY snapshot starts in January 2004 and ends on August 10, 2026. "
               f"Your selected backtest ends on {curves.index[-1]:%B %d, %Y}; later market activity is excluded. "
               "Source: github.com/manisahni/marketdata (daily_adjclose.csv).")

a, b = stats(curves["Strategy"]), stats(curves["Buy & hold"])
cols = st.columns(4)
cols[0].metric("Strategy total", f"{a['Total return']:.1%}", f"{a['Total return'] - b['Total return']:+.1%} vs hold")
cols[1].metric("Strategy CAGR", f"{a['CAGR']:.1%}", f"{a['CAGR'] - b['CAGR']:+.1%} vs hold")
cols[2].metric("Max drawdown", f"{a['Max drawdown']:.1%}", f"{a['Max drawdown'] - b['Max drawdown']:+.1%} vs hold")
cols[3].metric("Time invested", f"{curves['Invested'].mean():.1%}", f"{len(trades)} trades")

chart = go.Figure()
for name in ("Strategy", "Buy & hold"):
    chart.add_trace(go.Scatter(x=curves.index, y=curves[name], name=name, mode="lines"))
chart.update_layout(title="Growth of $1", yaxis_title="Portfolio value", hovermode="x unified",
                    legend_orientation="h", height=480)
st.plotly_chart(chart, use_container_width=True)

annual = curves[["Strategy", "Buy & hold"]].resample("YE").last().pct_change()
# For the first (possibly partial) year, measure from the selected starting close.
annual.iloc[0] = curves[["Strategy", "Buy & hold"]].loc[:annual.index[0]].iloc[-1] / curves[["Strategy", "Buy & hold"]].iloc[0] - 1
annual["Excess"] = annual["Strategy"] - annual["Buy & hold"]
annual.index = annual.index.year
annual.index.name = "Year"

st.subheader("Calendar year returns")
st.dataframe(annual.style.format("{:+.1%}"), use_container_width=True)

rolling = rolling_six_year(curves)
st.subheader("Every rolling six-year window")
if rolling.empty:
    st.write("Select at least six years of data to test the post's six-year claim.")
else:
    wins = (rolling.Excess > 0).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Windows beating hold", f"{wins}/{len(rolling)}")
    c2.metric("Worst six-year excess", f"{rolling.Excess.min():+.1%}")
    c3.metric("Best six-year excess", f"{rolling.Excess.max():+.1%}")
    st.caption("Windows begin at the first trading day of each month. Excess is the difference in compounded "
               "six-year percentage returns, in percentage points. Overlapping windows are not independent.")
    st.dataframe(rolling.style.format({"Start": "{:%Y-%m-%d}", "End": "{:%Y-%m-%d}",
                                       "Strategy": "{:+.1%}", "Buy & hold": "{:+.1%}",
                                       "Excess": "{:+.1%}"}), use_container_width=True)
    st.download_button("Download six-year windows", rolling.to_csv(index=False).encode(),
                       "six_year_windows.csv", "text/csv")

with st.expander("Trades and methodology"):
    st.dataframe(trades, use_container_width=True)
    st.download_button("Download trades", trades.to_csv(index=False).encode(), "trades.csv", "text/csv")
    st.write("Costs apply at each strategy purchase and sale; buy and hold has no modeled transaction cost. "
             "The optional stop is based on daily closes, not an intraday stop order. Actual fills, gaps, "
             "spreads, cash interest, taxes, and tracking differences are not modeled. Both positions are "
             "marked to the same dividend-adjusted price series; upload prices must also be adjusted. "
             "The first data point's purchase cost reduces initial strategy equity. Results depend on the "
             "chosen interpretation of the vague re-entry wording.")
