import os
import time
import math
from datetime import datetime, timedelta, timezone

import requests
import yfinance as yf
import pandas as pd
import streamlit as st

# ---------- Config ----------
APP_TITLE = "📈 ATH Drawdown Screener"
POLY_KEY = os.getenv("POLYGON_API_KEY") or st.secrets.get("POLYGON_API_KEY", "")

# ---------- Helpers ----------

def human_billions(x):
    if pd.isna(x):
        return None
    return round(float(x) / 1e9, 2)

def nice_error_from_response(resp: requests.Response) -> str:
    try:
        data = resp.json()
        # Polygon typically returns: {"status":"ERROR","error": "...", "message": "..."}
        return data.get("error") or data.get("message") or resp.text
    except Exception:
        return resp.text

@st.cache_data(show_spinner=False, ttl=300)
def fetch_polygon_universe(include_otc: bool, pages: int = 2):
    """
    Fetch a list of active US stock tickers from Polygon.
    We use v3/reference/tickers with cursor pagination.
    We do NOT use 'sort=market_cap' to avoid API errors on some accounts.
    Returns a DataFrame with columns: ticker, name, market_cap, primary_exchange, locale
    """
    if not POLY_KEY:
        raise RuntimeError("Missing POLYGON_API_KEY. Set env var or Streamlit secret.")

    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "limit": 1000,           # max allowed by Polygon
        "order": "desc",         # we'll sort in pandas later if we want
    }

    out = []
    cursor = None
    fetched_pages = 0

    while fetched_pages < pages:
        q = params.copy()
        if cursor:
            q["cursor"] = cursor
        q["apiKey"] = POLY_KEY

        resp = requests.get(url, params=q, timeout=30)
        if not resp.ok:
            # Surface a friendly API message
            raise requests.HTTPError(nice_error_from_response(resp), response=resp)

        data = resp.json()
        results = data.get("results", []) or []
        for r in results:
            # common fields in v3/reference/tickers
            out.append({
                "ticker": r.get("ticker"),
                "name": r.get("name"),
                "market_cap": r.get("market_cap"),
                "primary_exchange": r.get("primary_exchange"),
                "locale": r.get("locale"),
            })

        cursor = data.get("next_url") or data.get("next_url")  # prefer next_url
        if not cursor:
            break

        fetched_pages += 1
        # Be gentle with Polygon
        time.sleep(0.15)

    df = pd.DataFrame(out)
    # drop non-US if present
    if not df.empty and "locale" in df.columns:
        df = df[df["locale"].fillna("").str.upper().eq("US")]

    if not include_otc and "primary_exchange" in df.columns:
        # Filter out anything that looks OTC
        df = df[~df["primary_exchange"].fillna("").str.upper().str.contains("OTC")]

    # Some rows may not have market_cap; standardize type
    if "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")

    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history_yf(ticker: str, lookback_years: int = 15):
    """
    Grab long history from yfinance (15y by default).
    """
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(365.25 * lookback_years))
        y = yf.Ticker(ticker)
        # yfinance needs naive timestamps; it will handle tz internally
        hist = y.history(start=start.date().isoformat(), end=end.date().isoformat(), auto_adjust=True)
        if hist is None or hist.empty:
            return pd.DataFrame()
        hist = hist.rename(columns=str.lower).reset_index()
        hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
        return hist
    except Exception:
        return pd.DataFrame()

def compute_ath_metrics(hist: pd.DataFrame):
    """
    From daily adjusted history, compute:
      - last close
      - ATH and date
      - drawdown %
      - days since ATH
    Returns dict or None
    """
    if hist is None or hist.empty or "close" not in hist.columns:
        return None

    # Ensure numeric
    close = pd.to_numeric(hist["close"], errors="coerce")
    hist = hist.assign(close=close).dropna(subset=["close"])
    if hist.empty:
        return None

    last_close = float(hist["close"].iloc[-1])
    ath_val = float(hist["close"].max())
    # first date the ATH occurred
    ath_idx = hist["close"].idxmax()
    ath_date = hist.loc[ath_idx, "date"]
    if pd.isna(ath_val) or ath_val <= 0:
        return None

    dd_pct = (ath_val - last_close) / ath_val * 100.0
    days_since_ath = (hist["date"].iloc[-1] - ath_date).days
    return {
        "last_close": last_close,
        "ath": ath_val,
        "ath_date": ath_date.date().isoformat() if not pd.isna(ath_date) else None,
        "drawdown_pct": dd_pct,
        "days_since_ath": int(days_since_ath) if pd.notna(days_since_ath) else None,
    }

# ---------- UI ----------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Universe & Market Caps from Polygon • ATH math from Yahoo Finance history")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    include_otc = st.toggle("Include OTC", value=False)

    st.subheader("Market Cap (USD)")
    min_mc_bil = st.number_input("Min (Billions)", min_value=0.0, value=0.0, step=0.5)
    max_mc_bil = st.number_input("Max (Billions, 0 = no max)", min_value=0.0, value=0.0, step=0.5)

    st.subheader("≥ % below ATH")
    min_pct_below = st.slider("", min_value=0, max_value=95, value=50, step=1)

    st.subheader("Max days since ATH (0 = no limit)")
    max_days_since_ath = st.number_input("", min_value=0, value=1825, step=5, help="Limit recency of ATH (e.g., 365 = within 1 year).")

    st.subheader("Max symbols to scan")
    max_symbols = st.slider("", min_value=50, max_value=2000, value=600, step=50)

    st.subheader("Polygon pages to fetch")
    pages_to_fetch = st.slider("", min_value=1, max_value=10, value=3, step=1,
                               help="Each page ~1000 tickers (subject to your plan). Fewer pages reduces API usage.")

    run_btn = st.button("Run Screener", type="primary")

# Guard: key must be present
if not POLY_KEY:
    st.error("Missing POLYGON_API_KEY. Set env var or Streamlit secret.")
    st.stop()

if run_btn:
    with st.spinner("Fetching universe from Polygon…"):
        try:
            uni = fetch_polygon_universe(include_otc=include_otc, pages=pages_to_fetch)
        except requests.HTTPError as e:
            msg = str(e)
            st.error(f"Polygon API error: {msg}")
            st.info("Tips: lower 'pages to fetch', disable OTC, or wait a minute to cool down rate limits.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to fetch from Polygon: {e}")
            st.stop()

    if uni.empty:
        st.warning("Polygon returned an empty universe. Try adjusting pages/OTC.")
        st.stop()

    # Basic market cap filter (billions -> absolute)
    if "market_cap" in uni.columns:
        uni["market_cap_bil"] = uni["market_cap"].apply(human_billions)
        if min_mc_bil > 0:
            uni = uni[uni["market_cap_bil"].fillna(0) >= min_mc_bil]
        if max_mc_bil > 0:
            uni = uni[uni["market_cap_bil"].fillna(0) <= max_mc_bil]

    if uni.empty:
        st.warning("No symbols after market-cap filter.")
        st.stop()

    # Cap how many we scan with yfinance
    uni = uni.head(int(max_symbols))

    st.write(f"Scanning **{len(uni)}** symbols… (yfinance)")

    rows = []
    progress = st.progress(0)
    for i, row in uni.reset_index(drop=True).iterrows():
        t = row["ticker"]
        # yfinance uses Yahoo tickers; most Polygon US tickers match directly
        hist = fetch_history_yf(t)
        if not hist.empty:
            m = compute_ath_metrics(hist)
            if m:
                pct_below = round(m["drawdown_pct"], 2)
                days_ = m["days_since_ath"] if m["days_since_ath"] is not None else 10**9
                if pct_below >= min_pct_below and (max_days_since_ath == 0 or days_ <= max_days_since_ath):
                    rows.append({
                        "Ticker": t,
                        "Name": row.get("name"),
                        "Market Cap (B)": row.get("market_cap_bil"),
                        "Last Close": round(m["last_close"], 2),
                        "ATH": round(m["ath"], 2),
                        "% Below ATH": pct_below,
                        "ATH Date": m["ath_date"],
                        "Days Since ATH": m["days_since_ath"],
                        "Finviz": f"https://finviz.com/quote.ashx?t={t}",
                    })
        if (i + 1) % 10 == 0 or i == len(uni) - 1:
            progress.progress((i + 1) / len(uni))

    progress.empty()

    if not rows:
        st.warning("No symbols matched your filters.")
        st.stop()

    df = pd.DataFrame(rows)
    df = df.sort_values("% Below ATH", ascending=False, na_position="last")

    # Turn Finviz into links
    def linkify(val, label):
        if pd.isna(val):
            return ""
        return f'<a href="{val}" target="_blank">{label}</a>'

    df_display = df.copy()
    df_display["Ticker"] = df_display.apply(
        lambda r: f'<a href="https://finviz.com/quote.ashx?t={r["Ticker"]}" target="_blank">{r["Ticker"]}</a>', axis=1
    )
    df_display["Finviz"] = df_display["Finviz"].apply(lambda u: linkify(u, "Finviz"))

    # Order columns
    cols = ["Ticker", "Name", "Market Cap (B)", "Last Close", "ATH", "% Below ATH",
            "ATH Date", "Days Since ATH", "Finviz"]
    df_display = df_display[cols]

    st.subheader(f"Results — {len(df_display)} matches")
    st.caption("Click a ticker or the Finviz link to open details in a new tab.")
    st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.info("Set your filters, then press **Run Screener**.")
