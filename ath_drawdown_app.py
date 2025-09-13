# streamlit run ath_drawdown_app.py
import os
import time
import math
import pytz
import numpy as np
import pandas as pd
import datetime as dt
import requests
import streamlit as st
from typing import List, Dict

ET = pytz.timezone("America/New_York")
BASE = "https://api.polygon.io"

# ─────────────────────────────────────────────────────────────────────────────
# Config & helpers
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("POLYGON_API_KEY") or st.secrets.get("POLYGON_API_KEY", "")
if not API_KEY:
    st.error("Missing POLYGON_API_KEY. Set env var or Streamlit secret.")
    st.stop()

def poly_get(path: str, params: Dict=None, retries=3, backoff=0.9):
    """GET with light retry/backoff + 429 handling."""
    params = dict(params or {})
    params["apiKey"] = API_KEY
    url = f"{BASE}{path}"
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff * (i+1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff * (i+1))
    raise last_err if last_err else RuntimeError("Polygon GET failed")

def today_et() -> dt.date:
    return dt.datetime.now(ET).date()

def prev_business_day(d: dt.date) -> dt.date:
    wd = d.weekday()
    if wd == 0: return d - dt.timedelta(days=3)  # Mon -> Fri
    if wd == 6: return d - dt.timedelta(days=2)  # Sun -> Fri
    if wd == 5: return d - dt.timedelta(days=1)  # Sat -> Fri
    return d - dt.timedelta(days=1)

def finviz(sym: str) -> str:
    return f"https://finviz.com/quote.ashx?t={sym.upper()}"

# ─────────────────────────────────────────────────────────────────────────────
# Universe from grouped bars (one API call for the market)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=600)
def get_grouped_bars(date_str: str, locale="us", market="stocks"):
    """Return grouped daily bars (dict list). Falls back to prev biz day if empty."""
    j = poly_get(f"/v2/aggs/grouped/locale/{locale}/market/{market}/{date_str}")
    results = j.get("results", []) or []
    if not results:
        # fallback once to the previous business day
        d = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        d2 = prev_business_day(d)
        j2 = poly_get(f"/v2/aggs/grouped/locale/{locale}/market/{market}/{d2.strftime('%Y-%m-%d')}")
        results = j2.get("results", []) or []
        return results, d2.strftime("%Y-%m-%d")
    return results, date_str

def clean_symbol(sym: str) -> bool:
    """Filter out obvious test/warrants/units if you want (very light filter)."""
    # Exclude extremely odd tickers commonly not common shares
    bad_chars = ["/", "^", " "]
    if any(ch in sym for ch in bad_chars): return False
    if sym.endswith(("W", "WS", "U", "RT")):  # warrants/units/rights
        return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# ATH & current price
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=24*3600)
def get_symbol_ath(symbol: str, start="1990-01-01") -> dict:
    """Fetch daily history and compute ATH (max high) & date."""
    end = today_et().strftime("%Y-%m-%d")
    j = poly_get(f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
                 params={"adjusted":"true","sort":"asc","limit":50000})
    res = j.get("results", [])
    if not res:
        return {"ath": None, "ath_date": None}
    df = pd.DataFrame(res)
    if "h" not in df or df.empty:
        return {"ath": None, "ath_date": None}
    idx = int(df["h"].idxmax())
    ath = float(df.loc[idx, "h"])
    ts = pd.to_datetime(df.loc[idx, "t"], unit="ms", utc=True).tz_convert(ET).date()
    return {"ath": ath, "ath_date": ts}

@st.cache_data(show_spinner=False, ttl=20)
def get_snapshot_prices(symbols: List[str]) -> dict:
    """Batch snapshot for last trade prices. Returns dict {symbol: price}."""
    out = {}
    BATCH = 50
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i+BATCH]
        try:
            j = poly_get("/v2/snapshot/locale/us/markets/stocks/tickers",
                         params={"tickers": ",".join(batch)})
            for t in j.get("tickers", []):
                sym = t.get("ticker")
                p = None
                if t.get("lastTrade") and t["lastTrade"].get("p"):
                    p = float(t["lastTrade"]["p"])
                elif t.get("day") and t["day"].get("c"):
                    p = float(t["day"]["c"])
                if sym and p:
                    out[sym] = p
        except Exception:
            # continue gracefully
            pass
    return out

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ATH Distance Screener", page_icon="📉", layout="wide")
st.title("📉 Distance from All-Time High (ATH) — Screener")

with st.sidebar:
    st.subheader("Universe")
    default_date = today_et().strftime("%Y-%m-%d")
    date_str = st.text_input("Universe from grouped bars date (YYYY-MM-DD)", value=default_date)
    min_price = st.number_input("Min Price ($)", value=5.0, step=0.5)
    top_by_volume = st.slider("Universe size (Top by Volume from that day)", 50, 1000, 300, step=50)
    include_watch = st.text_area("Extra symbols (comma-separated)", "AAPL,NVDA,TSLA,AMD,MSFT,AMZN")

    st.subheader("Price Source")
    use_live = st.checkbox("Use live snapshot price (more API calls)", value=False)

    st.subheader("Filter")
    # Interpret as “at least X% below ATH”
    thr = st.slider("Minimum % below ATH", min_value=10, max_value=95, value=50, step=5)

    st.subheader("Options")
    add_finviz = st.checkbox("Add Finviz links", value=True)
    auto = st.checkbox("Auto refresh (20s)", value=False)

if auto:
    st.experimental_rerun()

# Load grouped bars
with st.status("Building universe from grouped bars…", expanded=False):
    try:
        grp, used_date = get_grouped_bars(date_str)
    except Exception as e:
        st.error(f"Failed to load grouped bars: {e}")
        st.stop()

    if not grp:
        st.warning("No grouped bars found for the date. Try previous business day.")
        st.stop()

    # Build universe DF
    gdf = pd.DataFrame(grp)
    # Keep common shares-ish
    gdf = gdf[gdf["T"].apply(clean_symbol)]
    # Filter by price
    gdf = gdf[gdf["c"] >= float(min_price)]
    # Sort by volume, pick top N
    gdf = gdf.sort_values("v", ascending=False).head(int(top_by_volume))
    symbols = gdf["T"].tolist()

    if include_watch.strip():
        extra = [s.strip().upper() for s in include_watch.split(",") if s.strip()]
        symbols = sorted(set(symbols + extra))

st.caption(f"Universe date used: **{used_date}** • Universe size: **{len(symbols)}**")

# Get prices
price_map = {}
if use_live:
    with st.status("Fetching snapshot prices…", expanded=False):
        price_map = get_snapshot_prices(symbols)
        # Fallback: if not found in snapshot, use that grouped close (if symbol present)
        for _, r in gdf.iterrows():
            sym = r["T"]
            if sym not in price_map:
                price_map[sym] = float(r["c"])
else:
    # Use grouped closes for all in universe
    price_map = {r["T"]: float(r["c"]) for _, r in gdf.iterrows()}

# Compute ATHs (cached per-symbol)
rows = []
skipped = 0
with st.status("Computing ATH & drawdowns…", expanded=False):
    for sym in symbols:
        try:
            ath_info = get_symbol_ath(sym)
            ath = ath_info["ath"]
            ath_date = ath_info["ath_date"]
            last = price_map.get(sym)
            if not ath or not last or ath <= 0 or last <= 0:
                continue
            dd = 100.0 * (ath - last) / ath  # % below ATH
            if dd >= thr:
                rows.append({
                    "Symbol": sym,
                    "Last": last,
                    "ATH": ath,
                    "% Below ATH": dd,
                    "ATH Date": ath_date,
                    "Since ATH (days)": (today_et() - ath_date).days if ath_date else None,
                    "Finviz": finviz(sym) if add_finviz else "",
                })
        except Exception:
            skipped += 1
            continue

if not rows:
    st.info("No symbols matched your drawdown threshold. Try lowering the threshold, increasing the universe size, or enabling live prices.")
    st.stop()

df = pd.DataFrame(rows)
df = df.sort_values("% Below ATH", ascending=False).reset_index(drop=True)

# Format nicely
df_fmt = df.copy()
df_fmt["Last"] = df_fmt["Last"].map(lambda x: f"{x:,.2f}")
df_fmt["ATH"]  = df_fmt["ATH"].map(lambda x: f"{x:,.2f}")
df_fmt["% Below ATH"] = df_fmt["% Below ATH"].map(lambda x: f"{x:,.2f}%")
if add_finviz:
    df_fmt["Finviz"] = df_fmt["Finviz"].map(lambda u: f'<a href="{u}" target="_blank">Open</a>')

st.subheader(f"Stocks ≥ {thr}% below their All-Time High")
st.write(
    df_fmt[["Symbol","Last","ATH","% Below ATH","ATH Date","Since ATH (days)"] + (["Finviz"] if add_finviz else [])]
      .to_html(escape=False, index=False),
    unsafe_allow_html=True
)

# CSV download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name=f"ath_drawdown_{thr}pct_{used_date}.csv", mime="text/csv")

if skipped:
    st.caption(f"Skipped {skipped} symbol(s) due to missing/invalid data.")
st.caption("Tip: Enable snapshot prices for ‘as-of-now’ moves (uses more API calls).")
