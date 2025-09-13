# ath_drawdown_app.py
# Shows stocks X% below their all-time high, with Market Cap filters
# and "Since ATH (days)" so you can limit to recent ATHs.

import os
import time
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from typing import List

# ---------- Config / helpers ----------

def get_api_key() -> str:
    return st.secrets.get("POLYGON_API_KEY") or os.getenv("POLYGON_API_KEY")

API_KEY = get_api_key()

POLY_TICKERS_URL = "https://api.polygon.io/v3/reference/tickers"

def fmt_cap(x: float) -> str:
    if pd.isna(x):
        return ""
    if x >= 1e12:
        return f"{x/1e12:.2f}T"
    if x >= 1e9:
        return f"{x/1e9:.2f}B"
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    return f"{x:,.0f}"

def fmt_price(x: float) -> str:
    return "" if pd.isna(x) else f"{x:,.2f}"

@st.cache_data(show_spinner=False, ttl=60*60*12)  # cache 12h
def fetch_polygon_universe(include_otc: bool, pages: int = 3) -> pd.DataFrame:
    """
    Pull active US stock tickers (type=CS) and market cap from Polygon across a few pages.
    Returns DataFrame with columns: ticker, name, primary_exchange, market_cap
    """
    if not API_KEY:
        raise RuntimeError("Missing POLYGON_API_KEY. Set in Secrets or env var.")

    out = []
    params = {
        "market": "stocks",
        "active": "true",
        "type": "CS",
        "limit": 1000,
        "sort": "market_cap",
        "order": "desc",
        "apiKey": API_KEY,
    }
    next_cursor = None
    for i in range(pages):
        if next_cursor:
            params["cursor"] = next_cursor
        r = requests.get(POLY_TICKERS_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", []) or []
        for row in results:
            out.append({
                "ticker": row.get("ticker"),
                "name": row.get("name"),
                "primary_exchange": row.get("primary_exchange"),
                "market_cap": row.get("market_cap"),
            })
        # polygon sometimes returns "next_url"/"next"/"next_cursor"
        next_cursor = data.get("next_url") or data.get("next") or data.get("next_cursor")
        if not next_cursor:
            break
        time.sleep(0.3)

    df = pd.DataFrame(out).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    if not include_otc:
        df = df[~df["primary_exchange"].fillna("").str.upper().str.contains("OTC")]
    return df

# ---------- ATH metrics (Yahoo Finance) ----------

import yfinance as yf

@st.cache_data(show_spinner=False, ttl=60*60*6)  # cache 6h
def get_ath_metrics(symbols: List[str]) -> pd.DataFrame:
    """
    Download max history for symbols (batched) and compute:
      - last close
      - ATH close
      - drawdown % from ATH
      - ATH date
      - Since ATH (days)
    Returns DataFrame index=ticker with columns:
      [last, ath, dd_pct, ath_date, since_ath_days]
    """
    if not symbols:
        return pd.DataFrame(columns=["last", "ath", "dd_pct", "ath_date", "since_ath_days"])

    rows = []
    batch_size = 40
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i+batch_size]
        try:
            df = yf.download(
                tickers=chunk,
                period="max",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception:
            continue

        # Single symbol shape vs multi-index columns
        if len(chunk) == 1:
            sym = chunk[0]
            sub = df.copy()
            if "Close" not in sub.columns:
                continue
            close = sub["Close"].dropna()
            if close.empty:
                continue
            last = float(close.iloc[-1])
            ath = float(close.max())
            ath_idx = close.idxmax()
            # days since ATH uses last available close date (also handles weekends/holidays)
            last_idx = close.index[-1]
            since_days = int((pd.Timestamp(last_idx).tz_localize(None) -
                              pd.Timestamp(ath_idx).tz_localize(None)).days)
            dd_pct = (ath - last) / ath * 100.0 if ath > 0 else np.nan
            rows.append((sym, last, ath, dd_pct, pd.to_datetime(ath_idx).date(), since_days))
        else:
            for sym in chunk:
                try:
                    sub = df[sym].copy()
                    close = sub["Close"].dropna()
                    if close.empty:
                        continue
                    last = float(close.iloc[-1])
                    ath = float(close.max())
                    ath_idx = close.idxmax()
                    last_idx = close.index[-1]
                    since_days = int((pd.Timestamp(last_idx).tz_localize(None) -
                                      pd.Timestamp(ath_idx).tz_localize(None)).days)
                    dd_pct = (ath - last) / ath * 100.0 if ath > 0 else np.nan
                    rows.append((sym, last, ath, dd_pct, pd.to_datetime(ath_idx).date(), since_days))
                except Exception:
                    continue

    out = pd.DataFrame(
        rows,
        columns=["ticker", "last", "ath", "dd_pct", "ath_date", "since_ath_days"]
    ).set_index("ticker")
    return out

# ---------- UI ----------

st.set_page_config(page_title="ATH Drawdown Screener", page_icon="📉", layout="wide")
st.title("📉 ATH Drawdown Screener")
st.caption("Universe & Market Caps from Polygon • ATH math from Yahoo Finance history")

with st.sidebar:
    st.subheader("Filters")

    include_otc = st.toggle("Include OTC", value=False)

    # Market cap range (USD, billions for UX)
    st.caption("Market Cap (USD)")
    min_cap_b = st.number_input("Min (Billions)", min_value=0.0, value=0.0, step=0.5, format="%.2f")
    max_cap_b = st.number_input("Max (Billions, 0 = no max)", min_value=0.0, value=0.0, step=0.5, format="%.2f")

    # Drawdown threshold
    dd_min = st.slider("≥ % below ATH", min_value=0, max_value=95, value=50, step=5)

    # NEW: since ATH days filter
    max_since_days = st.number_input(
        "Max days since ATH (0 = no limit)",
        min_value=0, value=1825, step=30, help="Limit to names whose ATH occurred within N days."
    )

    # limit/scaling
    max_symbols = st.slider("Max symbols to scan", 50, 2500, 600, step=50)
    pages = st.slider("Polygon pages to fetch", 1, 10, 3)

    run_btn = st.button("Run Screener", type="primary")

if not API_KEY:
    st.error("Missing POLYGON_API_KEY. Set env var or Streamlit secret.")
    st.stop()

# ---------- Run ----------

if run_btn:
    with st.spinner("Fetching universe & market caps from Polygon…"):
        uni = fetch_polygon_universe(include_otc=include_otc, pages=pages)
        if uni.empty:
            st.warning("No tickers from Polygon (check filters).")
            st.stop()

        # Apply market cap range early
        min_cap = min_cap_b * 1e9
        max_cap = max_cap_b * 1e9 if max_cap_b > 0 else None
        if min_cap > 0:
            uni = uni[uni["market_cap"].fillna(0) >= min_cap]
        if max_cap:
            uni = uni[uni["market_cap"].fillna(0) <= max_cap]

        # Top N by cap to control workload
        uni = uni.sort_values("market_cap", ascending=False).head(max_symbols).reset_index(drop=True)
        tickers = uni["ticker"].tolist()

    if not tickers:
        st.warning("No tickers after applying market cap / OTC filters.")
        st.stop()

    with st.spinner(f"Computing ATH metrics for {len(tickers)} symbols…"):
        ath_df = get_ath_metrics(tickers)

    if ath_df.empty:
        st.warning("No price history available for selected symbols.")
        st.stop()

    merged = (
        ath_df.reset_index()
              .merge(uni[["ticker", "name", "market_cap"]], on="ticker", how="left")
    )

    # Filter by drawdown threshold
    merged = merged[merged["dd_pct"] >= dd_min]

    # NEW: filter by since ATH days if user set a limit
    if max_since_days > 0:
        merged = merged[merged["since_ath_days"] <= int(max_since_days)]

    # Sort by biggest drawdown (farthest below ATH)
    merged = merged.sort_values("dd_pct", ascending=False)

    if merged.empty:
        st.info("No matches for the chosen filters (drawdown/market-cap/since-ATH).")
        st.stop()

    # Present
    show = merged.copy()
    show.rename(columns={
        "ticker": "Symbol",
        "name": "Company",
        "last": "Last",
        "ath": "ATH",
        "dd_pct": "Below ATH (%)",
        "market_cap": "Market Cap",
        "ath_date": "ATH Date",
        "since_ath_days": "Since ATH (days)",
    }, inplace=True)

    show["Last"] = show["Last"].map(fmt_price)
    show["ATH"] = show["ATH"].map(fmt_price)
    show["Market Cap"] = show["Market Cap"].map(fmt_cap)
    show["Below ATH (%)"] = show["Below ATH (%)"].map(lambda x: f"{x:,.1f}")
    # ATH Date already date; ensure string for display
    show["ATH Date"] = pd.to_datetime(show["ATH Date"]).dt.strftime("%Y-%m-%d")

    st.subheader(f"Results — {len(show):,} matches")
    st.dataframe(
        show[["Symbol", "Company", "Market Cap", "Last", "ATH", "Below ATH (%)", "ATH Date", "Since ATH (days)"]],
        use_container_width=True,
        height=600
    )

    # CSV
    csv = merged.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv,
        file_name=f"ath_drawdown_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%MUTC')}.csv",
        mime="text/csv"
    )
else:
    st.info("Set your filters (drawdown, market cap, **Max days since ATH**) and click **Run Screener**.")
