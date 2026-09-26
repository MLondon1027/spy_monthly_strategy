# Monthly index strategy backtest

Backtests the Reddit screenshot's rule against SPY buy and hold. The screenshot's
re-entry and stop wording is imprecise; the app displays the exact assumptions.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

By default Yahoo Finance supplies dividend-adjusted daily prices. You can upload
a CSV with `Date,Close` columns if data download fails. `Close` must already be
dividend adjusted for a fair SPY total-return comparison. Yahoo data needs an
internet connection. On Streamlit Community Cloud, set the main file to `app.py`.

Signals at a closing price are filled at the following trading close, except
the unconditional initial purchase. Cash earns zero. This is a research tool,
not a claim of out-of-sample performance or investment advice.
