import io
from urllib.request import Request, urlopen

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from strategy import backtest, clean_prices, rolling_six_year, stats

st.set_page_config(page_title="Monthly Index Rule", page_icon="📈", layout="wide")
st.title("Does the monthly index rule beat buy and hold?")
st.caption("A transparent backtest of the rule in the screenshot. Past results do not predict future returns.")

with st.sidebar:
    st.header("Data and assumptions")
    ticker = st.text_input("Ticker (Yahoo; SPY has a backup source)", "SPY").strip().upper()
    uploaded = st.file_uploader("Or upload dividend-adjusted daily prices (Date, Close CSV)", type="csv")
    start = st.date_input("Start date", value=pd.Timestamp("1993-01-29"), min_value=pd.Timestamp("1900-01-01"))
    end = st.date_input("End date (inclusive)", value=pd.Timestamp.today().date())
    comparison_label = st.radio("At month-end, compare price with", ["First close of that month", "Original purchase price"],
                                help="The post is ambiguous. The first choice tests whether the month lost value; "
                                     "the second can hold for years as long as the position remains above its entry.")
    comparison = "month_start" if comparison_label == "First close of that month" else "entry"
    stop_enabled = st.checkbox("Apply a close-based stop", value=False)
    stop = st.slider("Stop below entry (%)", 7, 12, 10) / 100 if stop_enabled else None
    fee = st.number_input("Cost per trade (basis points)", 0.0, 100.0, 0.0, 0.5)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch(symbol, start_date, end_date):
    yahoo_error = None
    try:
        data = yf.download(symbol, start=str(start_date),
                           end=str(pd.Timestamp(end_date) + pd.Timedelta(days=1)),
                           auto_adjust=True, actions=False, progress=False, threads=False)
        if not data.empty:
            series = data["Close"]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            prices = clean_prices(series.rename("Close").reset_index().rename(columns={"index": "Date"}))
            return prices, "Yahoo Finance"
        yahoo_error = "Yahoo returned no rows (often a rate limit)."
    except Exception as exc:
        yahoo_error = f"Yahoo error: {type(exc).__name__}: {exc}"

    if symbol == "SPY":
        try:
            url = "https://raw.githubusercontent.com/manisahni/marketdata/main/daily_adjclose.csv"
            request = Request(url, headers={"User-Agent": "Mozilla/5.0 monthly-backtest"})
            with urlopen(request, timeout=20) as response:
                snapshot = pd.read_csv(io.BytesIO(response.read()), usecols=["Date", "SPY"])
            snapshot = snapshot.rename(columns={"SPY": "Close"})
            prices = clean_prices(snapshot).loc[str(start_date):str(end_date)]
            if len(prices) >= 10:
                return prices, "Published SPY adjusted-close snapshot"
            raise ValueError("Backup snapshot has fewer than ten rows in the selected dates.")
        except Exception as exc:
            raise ValueError(f"{yahoo_error} SPY backup failed: {type(exc).__name__}: {exc}. "
                             "Upload an adjusted-price CSV as another option.") from exc
    raise ValueError(f"{yahoo_error} The backup covers SPY only. "
                     "Try again later or upload an adjusted-price CSV.")


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
        if not ticker:
            raise ValueError("Enter a ticker or upload a CSV.")
        prices, source = fetch(ticker, start, end)
        label = f"{ticker} via {source}"
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
    st.warning("Yahoo prices were unavailable. The backup SPY snapshot starts in January 2004 and currently "
               f"ends on {curves.index[-1]:%B %d, %Y}; results do not include later market activity. "
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
