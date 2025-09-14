# ath_drawdown_app.py  (FAST BATCH VERSION)

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import yfinance as yf

# Optional holiday calendar; app works without it
try:
    import pandas_market_calendars as mcal
    _HAS_PMC = True
except Exception:
    _HAS_PMC = False

NY_TZ = pytz.timezone("America/New_York")


# ---------- Time helpers ----------
def most_recent_trading_day(now_et: Optional[datetime] = None) -> pd.Timestamp:
    if now_et is None:
        now_et = datetime.now(NY_TZ)

    # Use yesterday if before ~4:05pm ET
    cutoff = now_et.replace(hour=16, minute=5, second=0, microsecond=0)
    ref = now_et if now_et >= cutoff else now_et - timedelta(days=1)

    if _HAS_PMC:
        start = (ref - timedelta(days=14)).date()
        end = ref.date()
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=start, end_date=end)
        if not sched.empty:
            last_sess = pd.Timestamp(sched.index[-1]).tz_localize("UTC").tz_convert(NY_TZ)
            return pd.Timestamp(last_sess.date())

    # Weekend fallback
    d = ref.date()
    while d.weekday() >= 5:  # Sat/Sun
        d = d - timedelta(days=1)
    return pd.Timestamp(d)


# ---------- Keys ----------
def _get_polygon_key() -> str:
    key = (
        os.getenv("POLYGON_API_KEY")
        or st.secrets.get("POLYGON_API_KEY")
        or st.secrets.get("polygon_api_key")
    )
    if not key:
        st.error("Missing POLYGON_API_KEY. Set env var or Streamlit secret.")
        st.stop()
    return key


# ---------- Polygon universe ----------
def _get_page(url: str, params: Dict[str, str]) -> Dict:
    for attempt in range(5):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(1.0 + attempt * 0.5)  # gentle backoff
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}

def _walk_reference_tickers(market: str, pages: int, api_key: str, active: bool = True) -> List[Dict]:
    base = "https://api.polygon.io/v3/reference/tickers"
    params = {"market": market, "active": str(active).lower(), "limit": "1000", "order": "asc", "apiKey": api_key}
    out, next_url, page = [], base, 0
    while next_url and page < pages:
        data = _get_page(next_url, params if next_url == base else {})
        out.extend(data.get("results", []))
        next_url = data.get("next_url")
        if next_url and "apiKey=" not in next_url:
            next_url = f"{next_url}&apiKey={api_key}"
        page += 1
    return out

@st.cache_data(show_spinner=False, ttl=60 * 15)
def fetch_polygon_universe(include_otc: bool, pages: int) -> pd.DataFrame:
    api_key = _get_polygon_key()
    tickers = _walk_reference_tickers("stocks", pages, api_key, active=True)
    if include_otc:
        tickers += _walk_reference_tickers("otc", pages, api_key, active=True)

    if not tickers:
        return pd.DataFrame(columns=["ticker", "name", "market_cap", "type", "primary_exchange"])

    df = pd.DataFrame(tickers)
    for col in ["ticker", "name", "market_cap", "type", "primary_exchange"]:
        if col not in df.columns:
            df[col] = np.nan

    # Keep common stock/related types; adjust as you like
    df = df[df["type"].isin(["CS", "ADR", "CEF", "ETP", "ETN", "REIT", "UCI", "UNIT"])]
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df[["ticker", "name", "market_cap", "type", "primary_exchange"]]


# ---------- Batch yfinance ----------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def batch_download_closecalc(symbols: List[str], effective_close: pd.Timestamp) -> Dict[str, pd.Series]:
    """
    Download daily data for many symbols at once and return a dict:
       { "AAPL": pd.Series(closecalc), "MSFT": pd.Series(closecalc), ... }
    closecalc chooses Adj Close when available, otherwise Close.
    """
    if not symbols:
        return {}

    # yfinance end is exclusive; add 1 day to include effective close
    end_exclusive = pd.Timestamp(effective_close) + pd.Timedelta(days=1)

    # yfinance can accept a list or space-separated string
    sym_str = " ".join(symbols)

    # group_by='ticker' -> MultiIndex columns with top level tickers
    data = yf.download(
        tickers=sym_str,
        start="1990-01-01",
        end=end_exclusive,
        interval="1d",
        auto_adjust=False,
        threads=True,
        group_by="ticker",
        actions=False,
    )

    out: Dict[str, pd.Series] = {}

    # If only one symbol, yfinance returns a single-frame with columns not MultiIndex
    if not isinstance(data.columns, pd.MultiIndex):
        # Single symbol path
        df = data.rename(columns=str.title)
        if "Adj Close" in df.columns and df["Adj Close"].notna().any():
            out[symbols[0]] = df["Adj Close"].dropna()
        elif "Close" in df.columns:
            out[symbols[0]] = df["Close"].dropna()
        return out

    # Multi symbol path: top level is ticker
    top = data.columns.levels[0]

    for sym in symbols:
        if sym not in top:
            # fallback: try single download for this one
            try:
                one = yf.Ticker(sym).history(
                    interval="1d",
                    start="1990-01-01",
                    end=end_exclusive,
                    auto_adjust=False,
                    actions=False,
                )
                one = one.rename(columns=str.title)
                if one.empty:
                    continue
                if "Adj Close" in one.columns and one["Adj Close"].notna().any():
                    out[sym] = one["Adj Close"].dropna()
                elif "Close" in one.columns:
                    out[sym] = one["Close"].dropna()
            except Exception:
                continue
            continue

        sub = data[sym].rename(columns=str.title)
        # choose Adj Close if present
        if "Adj Close" in sub.columns and sub["Adj Close"].notna().any():
            out[sym] = sub["Adj Close"].dropna()
        elif "Close" in sub.columns:
            out[sym] = sub["Close"].dropna()

    return out


def compute_ath_from_closes(close_map: Dict[str, pd.Series], effective_close: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for sym, s in close_map.items():
        if s is None or s.empty:
            continue
        last = float(s.iloc[-1])
        ath_price = float(s.max())
        if ath_price <= 0 or np.isnan(ath_price):
            continue
        ath_idx = s.idxmax()
        ath_date = pd.Timestamp(ath_idx).tz_localize(None) if hasattr(ath_idx, "tzinfo") and ath_idx.tzinfo else pd.to_datetime(ath_idx)
        pct_below = 100.0 * (ath_price - last) / ath_price
        days_since = int((effective_close - ath_date.normalize()).days)
        rows.append(
            {
                "Symbol": sym,
                "Last": last,
                "ATH": ath_price,
                "% Below ATH": pct_below,
                "ATH Date": pd.to_datetime(ath_date.date()),
                "Days Since ATH": days_since,
            }
        )
    return pd.DataFrame(rows)


# ---------- UI ----------
st.set_page_config(page_title="ATH Drawdown Screener", page_icon="📉", layout="wide")
st.title("📉 ATH Drawdown Screener")
st.caption("Universe & Market Caps from Polygon • ATH math from Yahoo Finance history")

with st.sidebar:
    st.header("Filters")
    include_otc = st.toggle("Include OTC", value=False)

    st.subheader("Market Cap (USD)")
    min_cap_bil = st.number_input("Min (Billions)", min_value=0.0, value=0.0, step=0.5)
    max_cap_bil = st.number_input("Max (Billions, 0 = no max)", min_value=0.0, value=0.0, step=0.5)

    pct_below_min = st.slider("≥ % below ATH", min_value=0, max_value=95, value=10, step=1)
    max_days_since_ath = st.number_input("Max days since ATH (0 = no limit)", min_value=0, value=1825, step=5)

    max_symbols = st.slider("Max symbols to scan", min_value=50, max_value=2000, value=300, step=50)
    pages_to_fetch = st.slider("Polygon pages to fetch", min_value=1, max_value=10, value=1, step=1)

    run_btn = st.button("Run Screener", type="primary")

eff_close = most_recent_trading_day()
st.write(f"**Most recent completed NYSE session:** {eff_close.date()} (auto-selected)")

def run_screen():
    with st.status("Fetching universe from Polygon…", expanded=False) as status:
        try:
            uni = fetch_polygon_universe(include_otc=include_otc, pages=pages_to_fetch)
        except requests.HTTPError:
            st.error("Polygon API error. If rate-limited, lower 'pages to fetch', uncheck OTC, or wait ~60s.")
            st.stop()

        if uni.empty:
            st.warning("No tickers returned from Polygon.")
            st.stop()

        # Filter by market cap early
        uni["market_cap"] = pd.to_numeric(uni["market_cap"], errors="coerce")
        if min_cap_bil > 0:
            uni = uni[uni["market_cap"] >= (min_cap_bil * 1e9)]
        if max_cap_bil > 0:
            uni = uni[uni["market_cap"] <= (max_cap_bil * 1e9)]

        uni = uni.dropna(subset=["ticker"])
        uni = uni[uni["ticker"].str.len().between(1, 6)]
        if uni.empty:
            st.warning("No symbols matched your filters.")
            st.stop()

        # Cap the workload
        uni = uni.sort_values("market_cap", ascending=False).head(max_symbols).reset_index(drop=True)
        tickers = uni["ticker"].tolist()

        status.update(label=f"Downloading history for {len(tickers)} symbols (batch)…", state="running")

    # Batch download closes
    close_map = batch_download_closecalc(tickers, eff_close)

    # Compute ATH metrics from batched closes
    df = compute_ath_from_closes(close_map, eff_close)
    if df.empty:
        st.warning("No symbols produced historical data. Try increasing 'Max symbols to scan' or relaxing filters.")
        return

    # attach name + market cap
    df = df.merge(
        uni[["ticker", "name", "market_cap"]].rename(columns={"ticker": "Symbol", "name": "Name"}),
        on="Symbol",
        how="left",
    )
    df["Market Cap (B)"] = df["market_cap"] / 1e9
    df.drop(columns=["market_cap"], inplace=True)

    # Filters
    df = df[df["% Below ATH"] >= float(pct_below_min)]
    if max_days_since_ath > 0:
        df = df[df["Days Since ATH"] <= int(max_days_since_ath)]

    if df.empty:
        st.warning("No symbols matched your filters. Lower % below ATH or expand 'days since ATH'.")
        return

    df = df.sort_values(by=["% Below ATH", "Market Cap (B)"], ascending=[False, False]).reset_index(drop=True)

    st.subheader("Results")
    st.write(f"Matched **{len(df):,}** symbols.")
    st.dataframe(
        df[["Symbol", "Name", "Market Cap (B)", "Last", "ATH", "% Below ATH", "ATH Date", "Days Since ATH"]],
        use_container_width=True,
        hide_index=True,
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"ath_drawdown_{eff_close.date()}.csv",
        mime="text/csv",
    )

if run_btn:
    run_screen()
else:
    st.info("Adjust filters in the sidebar and click **Run Screener**.")
