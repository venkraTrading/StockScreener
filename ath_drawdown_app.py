import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import pytz

# Optional: better exchange holiday handling (falls back gracefully if missing)
try:
    import pandas_market_calendars as mcal
    _HAS_PMC = True
except Exception:
    _HAS_PMC = False

NY_TZ = pytz.timezone("America/New_York")


# -------------------------------
# Helpers: time/calendar
# -------------------------------
def most_recent_trading_day(now_et: Optional[datetime] = None) -> pd.Timestamp:
    """
    Return the most recent NYSE trading day as a pandas Timestamp (date only).
    - If it's a weekend/holiday, returns the last open session.
    - If it's a weekday but before ~4:05pm ET, still use the previous session.
    """
    if now_et is None:
        now_et = datetime.now(NY_TZ)

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

    # Fallback: previous weekday
    d = ref.date()
    while d.weekday() >= 5:  # Sat/Sun
        d = d - timedelta(days=1)
    return pd.Timestamp(d)


# -------------------------------
# Secrets / API key
# -------------------------------
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


# -------------------------------
# Polygon: universe fetch
# -------------------------------
def _get_page(url: str, params: Dict[str, str]) -> Dict:
    """
    Robust GET with simple retries and 429 handling.
    """
    for attempt in range(5):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(1.0 + attempt * 0.5)
            continue
        resp.raise_for_status()
        return resp.json()
    # if we get here, last response probably was 429
    resp.raise_for_status()
    return {}  # not reached

def _walk_reference_tickers(
    market: str,
    pages: int,
    api_key: str,
    active: bool = True,
) -> List[Dict]:
    """
    Walk v3/reference/tickers using 'next_url'.
    - market: "stocks" or "otc"
    - pages: max number of pages to fetch
    """
    base = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": market,
        "active": str(active).lower(),
        "limit": "1000",
        "order": "asc",
        "apiKey": api_key,
    }

    out = []
    next_url = base
    page = 0
    while next_url and page < pages:
        data = _get_page(next_url, params if next_url == base else {})
        results = data.get("results", [])
        out.extend(results)
        next_url = data.get("next_url")
        page += 1
        # After the first request via 'next_url', do not pass params (Polygon expects just next_url+apiKey)
        if next_url:
            # ensure apiKey is appended
            if "apiKey=" not in next_url:
                next_url = f"{next_url}&apiKey={api_key}"
    return out


@st.cache_data(show_spinner=False, ttl=60 * 15)
def fetch_polygon_universe(
    include_otc: bool,
    pages: int,
) -> pd.DataFrame:
    """
    Fetch US stocks (and optionally OTC) tickers with market caps from Polygon.
    Returns a DataFrame with columns: ticker, name, market_cap, type, primary_exchange
    """
    api_key = _get_polygon_key()
    tickers = _walk_reference_tickers("stocks", pages, api_key, active=True)
    if include_otc:
        tickers += _walk_reference_tickers("otc", pages, api_key, active=True)

    if not tickers:
        return pd.DataFrame(columns=["ticker", "name", "market_cap", "type", "primary_exchange"])

    df = pd.DataFrame(tickers)
    # Normalize expected columns
    for col in ["ticker", "name", "market_cap", "type", "primary_exchange"]:
        if col not in df.columns:
            df[col] = np.nan

    # Keep common stock first; you can adjust if you want ETFs etc
    df = df[df["type"].isin(["CS", "ADR", "CEF", "ETP", "ETN", "REIT", "UCI", "UNIT"])]
    # Deduplicate
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df[["ticker", "name", "market_cap", "type", "primary_exchange"]]


# -------------------------------
# yfinance: load daily to most-recent close
# -------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_prices_for_ath(symbol: str) -> pd.DataFrame:
    """
    Fetch daily history up to the most recent *completed* NYSE session.
    yfinance's 'end' is exclusive, so add +1 day to include that session.
    """
    effective_close = most_recent_trading_day()                    # date
    end_exclusive = effective_close + pd.Timedelta(days=1)        # yfinance end is exclusive

    hist = yf.Ticker(symbol).history(
        interval="1d",
        start="1990-01-01",
        end=end_exclusive,
        auto_adjust=False,
        actions=False,
        prepost=False,
    )

    if hist is None or hist.empty:
        return pd.DataFrame()

    # Normalize columns and ensure we have a clean 'Close' to use
    hist = hist.rename(columns=str.title)
    # Prefer adjusted close if available
    if "Adj Close" in hist.columns and hist["Adj Close"].notna().any():
        hist["CloseCalc"] = hist["Adj Close"]
    else:
        hist["CloseCalc"] = hist["Close"]

    hist = hist.dropna(subset=["CloseCalc"])
    return hist


def compute_ath_metrics(symbol: str) -> Optional[Dict]:
    """
    Compute ATH metrics for a single symbol from yfinance daily data.
    Returns:
      {
        "Symbol", "Last", "ATH", "Pct Below ATH", "ATH Date", "Days Since ATH"
      } or None if data missing.
    """
    hist = load_prices_for_ath(symbol)
    if hist is None or hist.empty:
        return None

    # Last close
    last_close = float(hist["CloseCalc"].iloc[-1])

    # ATH on adjusted series
    ath_price = float(hist["CloseCalc"].max())
    ath_idx = hist["CloseCalc"].idxmax()
    ath_date = pd.Timestamp(ath_idx).tz_localize(None) if pd.api.types.is_datetime64_any_dtype(pd.Index([ath_idx])) else pd.to_datetime(ath_idx)

    if ath_price <= 0 or np.isnan(ath_price):
        return None

    pct_below = 100.0 * (ath_price - last_close) / ath_price
    last_close_day = most_recent_trading_day()
    days_since_ath = int((last_close_day - ath_date.normalize()).days)

    return {
        "Symbol": symbol,
        "Last": last_close,
        "ATH": ath_price,
        "ATH Date": pd.to_datetime(ath_date.date()),
        "% Below ATH": pct_below,
        "Days Since ATH": days_since_ath,
    }


# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="ATH Drawdown Screener", page_icon="📉", layout="wide")

st.title("📉 ATH Drawdown Screener")
st.caption("Universe & Market Caps from Polygon • ATH math from Yahoo Finance history")

with st.sidebar:
    st.header("Filters")
    include_otc = st.toggle("Include OTC", value=False, help="Also scan OTC universe from Polygon (slower).")

    st.subheader("Market Cap (USD)")
    min_cap_bil = st.number_input("Min (Billions)", min_value=0.0, value=0.0, step=0.5, help="Minimum market cap (billions).")
    max_cap_bil = st.number_input("Max (Billions, 0 = no max)", min_value=0.0, value=0.0, step=0.5, help="0 means no upper cap filter.")

    pct_below_min = st.slider("≥ % below ATH", min_value=0, max_value=95, value=10, step=1)

    max_days_since_ath = st.number_input("Max days since ATH (0 = no limit)", min_value=0, value=1825, step=5,
                                         help="Only include symbols whose ATH happened within this many days. 0 = no limit.")

    max_symbols = st.slider("Max symbols to scan", min_value=50, max_value=2000, value=400, step=50,
                            help="Caps how many tickers we compute. Increase if you can wait longer.")

    pages_to_fetch = st.slider("Polygon pages to fetch", min_value=1, max_value=10, value=3, step=1,
                               help="Each page ~1000 tickers per market. More pages = bigger universe + slower.")

    run_btn = st.button("Run Screener", type="primary")

# Show “effective close date”
eff_close = most_recent_trading_day()
st.write(f"**Most recent completed NYSE session:** {eff_close.date()} (auto-selected)")

def run_screen():
    with st.status("Fetching universe from Polygon…", expanded=False) as status:
        try:
            uni = fetch_polygon_universe(include_otc=include_otc, pages=pages_to_fetch)
        except requests.HTTPError as e:
            st.error("Polygon API error. If rate limited, lower 'pages to fetch', uncheck OTC, or wait a minute.")
            st.stop()

        if uni.empty:
            st.warning("No tickers returned from Polygon.")
            st.stop()

        # Market cap (billions) filter
        uni["market_cap"] = pd.to_numeric(uni["market_cap"], errors="coerce")
        if min_cap_bil > 0:
            uni = uni[uni["market_cap"] >= (min_cap_bil * 1e9)]
        if max_cap_bil > 0:
            uni = uni[uni["market_cap"] <= (max_cap_bil * 1e9)]

        # Basic sanity: keep symbols that are alnum-ish and not test tickers
        uni = uni[uni["ticker"].str.len().between(1, 6)]
        uni = uni.dropna(subset=["ticker"]).reset_index(drop=True)

        if uni.empty:
            st.warning("No symbols matched your market-cap filters.")
            st.stop()

        status.update(label=f"Scanning {min(max_symbols, len(uni))} symbols… (yfinance)", state="running")

    # Reduce to max_symbols deterministically
    uni = uni.sort_values(by="market_cap", ascending=False).head(max_symbols).reset_index(drop=True)

    rows: List[Dict] = []
    prog = st.progress(0, text="Computing ATH metrics…")

    for i, sym in enumerate(uni["ticker"].tolist(), start=1):
        m = compute_ath_metrics(sym)
        if m:
            # Attach name + cap billions
            m["Name"] = uni.loc[uni["ticker"] == sym, "name"].values[0]
            cap = uni.loc[uni["ticker"] == sym, "market_cap"].values[0]
            m["Market Cap (B)"] = (cap / 1e9) if pd.notna(cap) else np.nan
            rows.append(m)
        prog.progress(i / len(uni), text=f"{i}/{len(uni)} symbols")

    if not rows:
        st.info("No symbols produced historical data. Try increasing 'Max symbols to scan' or relaxing filters.")
        return

    df = pd.DataFrame(rows)

    # Filters: pct below and days since ATH
    df = df[df["% Below ATH"] >= float(pct_below_min)]
    if max_days_since_ath > 0:
        df = df[df["Days Since ATH"] <= int(max_days_since_ath)]

    if df.empty:
        st.warning("No symbols matched your filters. Try lowering % below ATH, expanding days, or raising max scan.")
        return

    df = df.sort_values(by=["% Below ATH", "Market Cap (B)"], ascending=[False, False]).reset_index(drop=True)

    # Pretty formats
    fmts = {
        "Last": "{:.2f}",
        "ATH": "{:.2f}",
        "% Below ATH": "{:.2f}",
        "Market Cap (B)": "{:.2f}",
    }

    # Display
    st.subheader("Results")
    st.write(f"Matched **{len(df):,}** symbols.")
    st.dataframe(
        df[["Symbol", "Name", "Market Cap (B)", "Last", "ATH", "% Below ATH", "ATH Date", "Days Since ATH"]],
        use_container_width=True,
        hide_index=True,
    )

    # CSV download
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
