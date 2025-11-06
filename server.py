# server.py
import os, re, math, threading, crud
import datetime as dt
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import func

from model import connect_to_db, db, Stock, PredictionLog, Ticker
from kmeans import KMeans

# ---------------- App / CORS ----------------
FLASK_APP_SECRET_KEY = os.environ.get("FLASK_APP_SECRET_KEY", "dev-secret")

# --- startup background population toggle ---
STARTUP_POPULATE = (os.getenv("STARTUP_POPULATE", "1").lower() in ("1", "true", "yes"))
STARTUP_FETCH_NAMES = (os.getenv("STARTUP_FETCH_NAMES", "0").lower() in ("1", "true", "yes"))
STARTUP_MULTITHREAD = (os.getenv("STARTUP_MULTITHREAD", "1").lower() in ("1", "true", "yes"))
STARTUP_MAX_WORKERS = int(os.getenv("STARTUP_MAX_WORKERS", "4"))

FMP_API_KEY = os.getenv("FMP_API_KEY")

# prevent double-start (e.g., dev reloader, multi workers)
_startup_once = threading.Event()

app = Flask(__name__)
app.secret_key = FLASK_APP_SECRET_KEY

FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
CORS(app, resources={
    r"/api/*": {
        "origins": FRONTEND_ORIGINS,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False,   # set True only if you use cookies/auth from the browser
    }
})

connect_to_db(app, "stocks")

# ---------------- Config ----------------
LOOKBACK_DAYS = 60
VOLATILITY_WINDOW = 20
MOMENTUM_WINDOW = 20
DEFAULT_RANGE = "6mo"
CANDIDATE_LIMIT = 5

# ---------------- Utilities ----------------
def parse_range_to_start(range_str: str) -> dt.date:
    today = dt.date.today()
    key = (range_str or "").strip().lower()
    if key == "1mo":  return today - dt.timedelta(days=31)
    if key == "3mo":  return today - dt.timedelta(days=93)
    if key == "6mo":  return today - dt.timedelta(days=186)
    if key == "1y":   return today - dt.timedelta(days=372)
    return today - dt.timedelta(days=186)

def fetch_and_upsert_yfinance_stock(ticker: str, start_date: dt.date, end_date: Optional[dt.date] = None) -> int:
    """Download daily bars via yfinance and upsert into Stock (your schema)."""
    ticker = str(ticker).upper().strip()
    yf_end = end_date.isoformat() if end_date else None

    hist = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=yf_end,
        interval="1d",
        auto_adjust=True,   # your crud.py uses auto_adjust=True
        progress=False
    )
    if hist is None or hist.empty:
        return 0

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = ["_".join([c for c in tup if c]).strip() for tup in hist.columns.values]

    count = 0
    ses = db.session
    for idx, row in hist.iterrows():
        # idx is a pandas Timestamp
        dt_val = idx.to_pydatetime()  # Stock.date is DateTime
        rid = f"{ticker}-{idx.strftime('%Y%m%d')}"

        exists = ses.query(Stock).filter(Stock.record_id == rid).first()
        if exists:
            # Optional: update prices if you want to keep values fresh
            continue

        # Convert to native types
        open_price  = float(row["Open"])   if pd.notna(row.get("Open"))   else None
        high_price  = float(row["High"])   if pd.notna(row.get("High"))   else None
        low_price   = float(row["Low"])    if pd.notna(row.get("Low"))    else None
        close_price = float(row["Close"])  if pd.notna(row.get("Close"))  else None
        volume      = int(row["Volume"])   if pd.notna(row.get("Volume")) else None

        ses.add(Stock(
            record_id       = rid,
            ticker          = ticker,
            date            = dt_val,
            open_price      = open_price,
            high_price      = high_price,
            low_price       = low_price,
            close_price     = close_price,
            adj_close_price = None,  # yfinance auto_adjust=True already adjusts Close
            volume          = volume
        ))
        count += 1

    ses.commit()
    return count

def load_prices_df(ticker: str, start_date: dt.date, end_date: Optional[dt.date] = None) -> pd.DataFrame:
    """Query the stocks table; if coverage is insufficient, backfill via yfinance and retry."""
    ticker = ticker.upper().strip()
    ses = db.session

    q = ses.query(Stock).filter(
        Stock.ticker == ticker,
        func.date(Stock.date) >= start_date  # Stock.date is DateTime -> cast to date
    )
    if end_date:
        q = q.filter(func.date(Stock.date) <= end_date)

    rows = q.order_by(Stock.date.asc()).all()
    if not rows or len(rows) < LOOKBACK_DAYS:
        fetch_and_upsert_yfinance_stock(ticker, start_date, end_date)
        rows = q.order_by(Stock.date.asc()).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "date": r.date,
        "open_price": r.open_price,
        "high_price": r.high_price,
        "low_price": r.low_price,
        "close_price": r.close_price,
        "adj_close_price": r.adj_close_price,
        "volume": r.volume
    } for r in rows])

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date  # downstream uses date objects
    return df

def feature_vector_from_close(close_series: pd.Series) -> Optional[np.ndarray]:
    if len(close_series) < max(VOLATILITY_WINDOW, MOMENTUM_WINDOW):
        return None
    returns = close_series.pct_change()
    vol = returns.rolling(VOLATILITY_WINDOW).std()
    mom = returns.rolling(MOMENTUM_WINDOW).mean()
    vol_last = vol.iloc[-1]
    mom_last = mom.iloc[-1]
    if pd.isna(vol_last) or pd.isna(mom_last) or not np.isfinite(vol_last) or not np.isfinite(mom_last):
        return None
    return np.array([float(vol_last), float(mom_last)], dtype=float)

def normalize_features(X: np.ndarray) -> np.ndarray:
    Xn = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        cmin, cmax = np.nanmin(col), np.nanmax(col)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
            Xn[:, j] = 0.0
        else:
            Xn[:, j] = (col - cmin) / (cmax - cmin)
    return Xn

def score_stock(feature_row: np.ndarray) -> float:
    vol_norm, mom_norm = feature_row[0], feature_row[1]
    return float(max(0.0, min(1.0, (1.0 - vol_norm) * mom_norm)))

def choose_decision(score: float) -> str:
    if score >= 0.70: return "BUY"
    if score >= 0.55: return "CONSIDER"
    return "HOLD"

def describe_regime(X: np.ndarray) -> str:
    if X.size == 0:
        return "Insufficient data"
    vol_med = np.median(X[:, 0])
    mom_med = np.median(X[:, 1])
    if mom_med > 0 and vol_med < 0.5:  return "Positive momentum; moderate volatility"
    if mom_med > 0 and vol_med >= 0.5: return "Positive momentum; high volatility"
    if mom_med <= 0 and vol_med < 0.5: return "Weak momentum; stable volatility"
    return "Weak momentum; choppy volatility"

# ---------------- API ----------------
@app.get("/api/history")
def api_history():
    """
    Query: ticker=XXX&range=6mo
    Returns: [{date, close, volume}]
    """
    ticker = (request.args.get("ticker") or "").upper().strip()
    r = request.args.get("range") or DEFAULT_RANGE
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400

    start = parse_range_to_start(r)
    df = load_prices_df(ticker, start)
    if df.empty:
        return jsonify([]), 200

    out = [{
        "date": d.strftime("%Y-%m-%d"),
        "close": float(c) if c is not None else None,
        "volume": int(v) if v is not None and not math.isnan(v) else 0
    } for d, c, v in zip(df["date"], df["close_price"], df["volume"]) if c is not None]

    return jsonify(out), 200

@app.post("/api/predict")
def api_predict():
    """
    Body (flexible):
      {
        tickers: string[] | "AAPL, MSFT" | "AAPL MSFT",
        k?: number,
        startDate?: "YYYY-MM-DD",
        endDate?: "YYYY-MM-DD",
        range?: "3mo" | "3 months" | "6mo" | "1y" ...
      }
    """
    data = request.get_json(force=True, silent=True) or {}

    # Parse tickers flexibly
    raw_tickers = data.get("tickers")
    if isinstance(raw_tickers, list):
        parsed = [str(t) for t in raw_tickers]
    elif isinstance(raw_tickers, str):
        parsed = [p for p in re.split(r"[,\s]+", raw_tickers) if p]
    else:
        parsed = []

    def _ok(t: str) -> bool:
        return all(ch.isalnum() or ch in ".-" for ch in t)

    tickers: List[str] = [t.upper().strip() for t in parsed if _ok(t.strip())]
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    k: int = int(data.get("k") or 3)

    # Date / range handling
    startDate = data.get("startDate")
    endDate   = data.get("endDate")
    range_in  = (data.get("range") or "").strip().lower()

    start = None
    end: Optional[dt.date] = None
    try:
        if startDate: start = dt.datetime.strptime(startDate, "%Y-%m-%d").date()
        if endDate:   end   = dt.datetime.strptime(endDate,   "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    if start is None:
        alias = {
            "1 month": "1mo", "1mo": "1mo", "1 m": "1mo",
            "3 months": "3mo", "3mo": "3mo", "3 m": "3mo",
            "6 months": "6mo", "6mo": "6mo", "6 m": "6mo",
            "1 year": "1y", "1y": "1y", "12 months": "1y"
        }
        start = parse_range_to_start(alias.get(range_in, range_in) or DEFAULT_RANGE)

    feats: List[np.ndarray] = []
    names: List[str] = []
    raw_feats: Dict[str, Dict[str, float]] = {}
    skipped: Dict[str, str] = {}

    for t in tickers:
        df = load_prices_df(t, start, end)
        if df.empty or df["close_price"].isna().all():
            skipped[t] = "no data"
            continue

        f = feature_vector_from_close(df["close_price"])
        if f is None:
            # Retry with more history
            df2 = load_prices_df(t, start - dt.timedelta(days=200), end)
            if df2.empty or df2["close_price"].isna().all():
                skipped[t] = f"insufficient window (len={len(df)})"
                continue
            f = feature_vector_from_close(df2["close_price"])
            if f is None:
                skipped[t] = f"insufficient window after retry (len={len(df2)})"
                continue

        feats.append(f)
        names.append(t)
        raw_feats[t] = {"volatility20": float(f[0]), "momentum20": float(f[1])}

    if not feats:
        return jsonify({
            "best": None,
            "alternatives": [],
            "regime": "Insufficient data across requested tickers",
            "debug": {
                "skipped": skipped,
                "parsedTickers": names,
                "start": start.isoformat(),
                "end": (end and end.isoformat()),
            }
        }), 200

    X  = np.vstack(feats)
    Xn = normalize_features(X)

    k = max(1, min(k, len(names)))
    km = KMeans(n_clusters=k, random_state=42, max_iter=300)
    km.fit(X)
    labels = km.labels_

    scores = [score_stock(row) for row in Xn]
    order  = np.argsort(scores)[::-1]
    best_idx = int(order[0])
    alt_idxs = [int(i) for i in order[1:1 + min(4, len(order) - 1)]]

    best = {
        "ticker": names[best_idx],
        "score": round(float(scores[best_idx]), 4),
        "cluster": int(labels[best_idx])
    }
    alts = [{
        "ticker": names[i],
        "score": round(float(scores[i]), 4),
        "cluster": int(labels[i])
    } for i in alt_idxs]

    regime = describe_regime(Xn)

    decision = choose_decision(best["score"])
    details = {
        "scores":  {names[i]: float(scores[i]) for i in range(len(names))},
        "features": raw_feats,
        "k": k,
        "labels": {names[i]: int(labels[i]) for i in range(len(names))},
        "regime": regime
    }
    db.session.add(PredictionLog(
        ticker=best["ticker"],
        score=float(best["score"]),
        decision=decision,
        details=details
    ))
    db.session.commit()

    return jsonify({
        "best": best,
        "alternatives": alts,
        "regime": regime
    }), 200

@app.get("/api/recent-predictions")
def api_recent_predictions():
    rows = db.session.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(20).all()
    out = [{
        "timestamp": r.timestamp.isoformat(),
        "ticker": r.ticker,
        "score": float(r.score),
        "decision": r.decision
    } for r in rows]
    return jsonify(out), 200


# --- add to your imports ---
from sqlalchemy import func, case
from flask import request, jsonify
from model import Ticker, db  # you already import connect_to_db, db, etc.

@app.get("/api/tickers/suggest")
def api_ticker_suggest():
    """
    GET /api/tickers?q=tes&limit=12&require_in_stocks=true
    Returns: [{ "symbol": "TSLA", "name": "Tesla, Inc." }, ...]
    """
    q = (request.args.get("q") or "").strip()
    limit = request.args.get("limit", "12")
    require_in_stocks_raw = (request.args.get("require_in_stocks") or "true").lower()

    require_in_stocks = require_in_stocks_raw not in ("false", "0", "no")

    try:
        suggestions = crud.suggest_tickers(
            q,
            limit=limit,
            require_in_stocks=require_in_stocks,
            include_inactive=False,
        )
        print(suggestions)
        return jsonify(suggestions), 200
    except Exception as e:
        return jsonify({"error": repr(e)}), 500

@app.get("/api/status")
def api_status():
    """
    Returns background population status + live DB counts.
    Useful to display readiness / progress in the frontend.
    """
    try:
        status = crud.get_population_status()
    except Exception as e:
        status = {"error": repr(e)}
    status["startup_triggered"] = _startup_once.is_set()
    return jsonify(status), 200

_startup_once = threading.Event()

@app.before_request
def _kickoff_population_once():
    if not STARTUP_POPULATE:
        return
    if _startup_once.is_set():
        return

    crud.kickoff_background_population(
        verify_in_stocks=True,        # ✅ only add tickers that we truly have prices for
        use_batched_prices=True,      # if you implemented a batched yf loader
        batch_size=64,
    )
    _startup_once.set()

if __name__ == "__main__":
    print("Connected to the db!")
    app.run(debug=True, port=5000)
