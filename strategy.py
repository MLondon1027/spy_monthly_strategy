"""Monthly SPY timing strategy with decisions made only from known closing prices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Expect Date, Close; Close must be a dividend-adjusted total-return price."""
    frame = raw.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    if "Date" not in frame or "Close" not in frame:
        raise ValueError("Data must have Date and Close columns (dividend-adjusted Close).")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"])
    frame = frame[frame.Close > 0].drop_duplicates("Date", keep="last").sort_values("Date")
    if len(frame) < 10:
        raise ValueError("At least ten valid daily price rows are required.")
    return frame.set_index("Date")[["Close"]]


def backtest(prices: pd.DataFrame, stop_pct: float | None = None,
             fee_bps: float = 0.0, comparison: str = "month_start"):
    """Long/cash strategy, close signals, next trading close fills.

    The first observation is bought at its close. At the final close of each
    calendar month, sell if the adjusted close is below the chosen reference
    price (that month's first close or the original position entry). While in
    cash, a first-trading-day close above the previous month's
    last close queues a buy for the next trading close. An optional close-based
    stop queues a sell for the next close. Monthly exits take precedence over
    re-entry; no new signal may be created on the day a trade executes.
    """
    close = prices["Close"].astype(float)
    if stop_pct is not None and not 0 < stop_pct < 1:
        raise ValueError("Stop must be between 0 and 1.")
    if fee_bps < 0:
        raise ValueError("Fee cannot be negative.")
    if comparison not in ("month_start", "entry"):
        raise ValueError("Comparison must be month_start or entry.")
    fee = fee_bps / 10000
    dates = close.index
    months = dates.to_period("M")
    first_days = set(dates[~months.duplicated(keep="first")])
    last_days = set(dates[~months.duplicated(keep="last")])
    prior_month_end = {}
    for i, day in enumerate(dates):
        if i and day in first_days:
            prior_month_end[day] = float(close.iloc[i - 1])

    cash, shares = 1.0, 0.0
    entry_price = None
    month_start_price = None
    pending = None
    events = []
    equity = []
    exposure = []
    benchmark = close / close.iloc[0]

    for i, (date, price_value) in enumerate(close.items()):
        price = float(price_value)
        executed_today = False
        if i == 0:
            shares = cash * (1 - fee) / price
            cash = 0.0
            entry_price = price
            executed_today = True
            events.append({"Date": date, "Action": "BUY", "Reason": "Initial entry", "Price": price})
        elif pending is not None:
            if pending[0] == "SELL" and shares:
                cash = shares * price * (1 - fee)
                shares = 0.0
                entry_price = None
            elif pending[0] == "BUY" and cash:
                shares = cash * (1 - fee) / price
                cash = 0.0
                entry_price = price
            events.append({"Date": date, "Action": pending[0], "Reason": pending[1], "Price": price})
            executed_today = True
            pending = None

        if date in first_days:
            month_start_price = price

        equity.append(cash + shares * price)
        exposure.append(bool(shares))

        if i == len(dates) - 1:
            break
        if executed_today:
            continue
        if shares:
            if stop_pct is not None and price <= entry_price * (1 - stop_pct):
                pending = ("SELL", f"Close below {stop_pct:.0%} entry stop")
            elif date in last_days and price < (month_start_price if comparison == "month_start" else entry_price):
                reason = "Month-end below month start" if comparison == "month_start" else "Month-end below entry"
                pending = ("SELL", reason)
        elif date in first_days and date in prior_month_end and price > prior_month_end[date]:
            pending = ("BUY", "First-day close above prior month-end")

    result = pd.DataFrame({"Strategy": equity, "Buy & hold": benchmark.values,
                           "Invested": exposure}, index=dates)
    return result, pd.DataFrame(events)


def stats(curve: pd.Series) -> dict:
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    dd = curve / curve.cummax() - 1
    return {"Total return": curve.iloc[-1] - 1,
            "CAGR": curve.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
            "Max drawdown": dd.min()}


def rolling_six_year(curves: pd.DataFrame) -> pd.DataFrame:
    """Monthly starting dates, each ending on first observation >= six years later."""
    dates = curves.index
    starts = dates[~dates.to_period("M").duplicated(keep="first")]
    records = []
    for start in starts:
        target = start + pd.DateOffset(years=6)
        loc = dates.searchsorted(target)
        if loc >= len(dates):
            continue
        end = dates[loc]
        strat = curves.loc[end, "Strategy"] / curves.loc[start, "Strategy"] - 1
        hold = curves.loc[end, "Buy & hold"] / curves.loc[start, "Buy & hold"] - 1
        records.append({"Start": start, "End": end, "Strategy": strat,
                        "Buy & hold": hold, "Excess": strat - hold})
    return pd.DataFrame(records)
