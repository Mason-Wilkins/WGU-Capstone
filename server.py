"""Flask API server for stock data, feature engineering, clustering, and scoring."""

from __future__ import annotations

import os
import re
import statistics
import threading
import datetime as dt
from functools import wraps
from typing import Dict, List, Optional

import crud
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import func

from kmeans import KMeans
from model import PredictionLog, Stock, Ticker, connect_to_db, db

# ---------------------------------------------------------------------------
# Flask setup & runtime configuration
# ---------------------------------------------------------------------------

FLASK_APP_SECRET_KEY = os.environ.get("FLASK_APP_SECRET_KEY", "dev-secret")
API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN", "").strip()
STARTUP_POPULATE = os.getenv("STARTUP_POPULATE", "1").lower() in {"1", "true", "yes"}

LOOKBACK_DAYS = 60
DEFAULT_RANGE = "6mo"
MAX_RANGE_DAYS = 365 * 3
DEFAULT_METRICS_WINDOW = int(os.getenv("ACCURACY_WINDOW_DAYS", "45") or 45)
DEFAULT_METRICS_HORIZON = int(os.getenv("ACCURACY_HORIZON_DAYS", "5") or 5)

app = Flask(__name__)
app.secret_key = FLASK_APP_SECRET_KEY
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": False,
        }
    },
)

connect_to_db(app, "stocks")
_startup_once = threading.Event()

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _get_request_api_key() -> Optional[str]:
    """Return API key supplied via header or query parameter."""

    hdr = request.headers.get("X-API-Key") or request.headers.get("X-Api-Key")
    if hdr:
        return hdr.strip()
    arg = request.args.get("api_key") if request else None
    return arg.strip() if arg else None


def api_key_required(func):
    """Decorator that validates the API key when one is configured."""

    @wraps(func)
    def _wrapped(*args, **kwargs):
        if API_AUTH_TOKEN:
            provided = _get_request_api_key()
            if provided != API_AUTH_TOKEN:
                return jsonify({"error": "Unauthorized: valid API key required."}), 401
        return func(*args, **kwargs)

    return _wrapped

# ---------------------------------------------------------------------------
# Data + feature helpers
# ---------------------------------------------------------------------------

def parse_range_to_start(rng: str) -> dt.date:
    """Map human inputs like '45d', '6 months', '2y' into an anchor start date."""

    today = dt.date.today()
    raw = (rng or DEFAULT_RANGE).strip().lower()

    aliases = {
        "1 month": "1mo",
        "1 m": "1mo",
        "3 months": "3mo",
        "6 months": "6mo",
        "1 year": "1y",
        "1 yr": "1y",
        "12 months": "12mo",
    }
    raw = aliases.get(raw, raw)

    match = re.match(r"^(\d+)\s*(d|day|days|m|mo|mon|month|months|y|yr|yrs|year|years)?$", raw)
    if match:
        qty = int(match.group(1))
        unit = match.group(2) or "d"
        if unit in {"d", "day", "days"}:
            days = qty
        elif unit in {"m", "mo", "mon", "month", "months"}:
            days = qty * 30
        else:
            days = qty * 365
        days = min(MAX_RANGE_DAYS, max(1, days))
        return today - dt.timedelta(days=days)

    # fallback to default 6 months when parsing fails
    return today - dt.timedelta(days=180)


def safe_float(value, default=None):
    """Coerce to float, returning a default when conversion fails."""

    try:
        v = float(value)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def fetch_and_upsert_yfinance_stock(
    ticker: str, start_date: dt.date, end_date: Optional[dt.date] = None
) -> int:
    """Download daily bars via yfinance and upsert them into `Stock`."""

    ticker = str(ticker).upper().strip()
    yf_end = end_date.isoformat() if end_date else None

    hist = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=yf_end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if hist is None or hist.empty:
        return 0

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = ["_".join([c for c in tup if c]).strip() for tup in hist.columns.values]

    inserted = 0
    session = db.session
    for idx, row in hist.iterrows():
        dt_val = idx.to_pydatetime()
        record_id = f"{ticker}-{idx.strftime('%Y%m%d')}"

        if session.query(Stock).filter(Stock.record_id == record_id).first():
            continue

        session.add(
            Stock(
                record_id=record_id,
                ticker=ticker,
                date=dt_val,
                open_price=float(row["Open"]) if pd.notna(row.get("Open")) else None,
                high_price=float(row["High"]) if pd.notna(row.get("High")) else None,
                low_price=float(row["Low"]) if pd.notna(row.get("Low")) else None,
                close_price=float(row["Close"]) if pd.notna(row.get("Close")) else None,
                adj_close_price=None,
                volume=int(row["Volume"]) if pd.notna(row.get("Volume")) else None,
            )
        )
        inserted += 1

    session.commit()
    return inserted


def load_prices_df(
    ticker: str, start_date: dt.date, end_date: Optional[dt.date] = None
) -> pd.DataFrame:
    """Load price rows from the DB (backfilling from yfinance when needed)."""

    ticker = ticker.upper().strip()
    session = db.session

    query = session.query(Stock).filter(
        Stock.ticker == ticker,
        func.date(Stock.date) >= start_date,
    )
    if end_date:
        query = query.filter(func.date(Stock.date) <= end_date)

    rows = query.order_by(Stock.date.asc()).all()

    def _usable_count(rs):
        return sum(1 for r in rs if r.close_price is not None)

    if not rows or _usable_count(rows) < LOOKBACK_DAYS:
        fetch_and_upsert_yfinance_stock(ticker, start_date, end_date)
        rows = query.order_by(Stock.date.asc()).all()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "open_price": r.open_price,
                "high_price": r.high_price,
                "low_price": r.low_price,
                "close_price": r.close_price,
                "adj_close_price": r.adj_close_price,
                "volume": r.volume,
            }
            for r in rows
        ]
    ).sort_values("date")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.reset_index(drop=True)


def rolling_mean(arr: np.ndarray, window: int) -> Optional[float]:
    """Return the mean of the last `window` entries, guarding against NaNs."""

    if len(arr) < window:
        return None
    sample = arr[-window:]
    mask = np.isfinite(sample)
    if mask.sum() < window:
        return None
    return float(sample.mean())


def rolling_std(arr: np.ndarray, window: int) -> Optional[float]:
    """Return the stddev of the last `window` entries, guarding against NaNs."""

    if len(arr) < window:
        return None
    sample = arr[-window:]
    mask = np.isfinite(sample)
    if mask.sum() < window:
        return None
    return float(sample.std(ddof=0))


def compute_features(close_series, momentum_w: int, vol_w: int) -> Optional[np.ndarray]:
    """Return [volatility, momentum] features for the given close series."""

    closes = np.asarray(close_series, dtype=float)
    if closes.size < max(momentum_w, vol_w) + 1:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(closes) / closes[:-1]
    mom = rolling_mean(rets, momentum_w)
    vol = rolling_std(rets, vol_w)
    if mom is None or vol is None or not np.isfinite(mom) or not np.isfinite(vol):
        return None
    return np.array([vol, mom], dtype=float)


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Min-max normalize each feature column to the [0, 1] interval."""

    Xn = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        cmin, cmax = np.nanmin(col), np.nanmax(col)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
            Xn[:, j] = 0.0
        else:
            Xn[:, j] = (col - cmin) / (cmax - cmin)
    return Xn


def score_stock_row(row_norm: np.ndarray, risk: str) -> float:
    """Convert normalized [volatility, momentum] into a scalar score."""

    vol_norm, mom_norm = float(row_norm[0]), float(row_norm[1])

    risk = (risk or "balanced").lower()
    if risk == "conservative":
        w_mom, w_vol = 0.25, 0.75
    elif risk == "aggressive":
        w_mom, w_vol = 0.75, 0.25
    else:
        w_mom, w_vol = 0.5, 0.5

    mom_util = 1 / (1 + np.exp(-mom_norm))
    vol_util = 1 / (1 + np.exp(vol_norm))
    return float(w_mom * mom_util + w_vol * vol_util)


def choose_decision(score: float) -> str:
    """Map a score into BUY/CONSIDER/HOLD buckets."""

    if score >= 0.70:
        return "BUY"
    if score >= 0.55:
        return "CONSIDER"
    return "HOLD"


def describe_regime(X_norm: np.ndarray) -> str:
    """Return a simple text description of the feature regime."""

    vol_bar, mom_bar = X_norm[:, 0].mean(), X_norm[:, 1].mean()
    vol_str = (
        "Low volatility"
        if vol_bar < -0.3
        else "Moderate volatility"
        if vol_bar < 0.4
        else "High volatility"
    )
    mom_str = (
        "positive momentum"
        if mom_bar > 0.2
        else "mixed momentum"
        if mom_bar > -0.2
        else "negative momentum"
    )
    return f"{vol_str}, {mom_str}"

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _lookup_close_on_or_after(ticker: str, ts: dt.datetime) -> Optional[float]:
    """Return the first close_price recorded on or after the timestamp."""

    row = (
        db.session.query(Stock.close_price)
        .filter(Stock.ticker == ticker, Stock.close_price.isnot(None), Stock.date >= ts)
        .order_by(Stock.date.asc())
        .first()
    )
    return float(row[0]) if row and row[0] is not None else None


def evaluate_prediction_accuracy(
    window_days: int, horizon_days: int, max_rows: int = 200
) -> Dict[str, object]:
    """Compare historical predictions to realized future returns."""

    window_days = max(1, int(window_days))
    horizon_days = max(1, int(horizon_days))
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=window_days)

    preds = (
        db.session.query(PredictionLog)
        .filter(PredictionLog.timestamp >= cutoff)
        .order_by(PredictionLog.timestamp.desc())
        .limit(max_rows)
        .all()
    )

    evaluated: List[float] = []
    hits = buy_hits = buy_total = hold_hits = hold_total = 0

    for pred in preds:
        entry_price = _lookup_close_on_or_after(pred.ticker, pred.timestamp)
        if entry_price in (None, 0):
            continue
        future_ts = pred.timestamp + dt.timedelta(days=horizon_days)
        future_price = _lookup_close_on_or_after(pred.ticker, future_ts)
        if future_price is None:
            continue

        ret = (future_price - entry_price) / entry_price
        decision = (pred.decision or "").upper()
        positive_call = decision in {"BUY", "CONSIDER"}
        is_hit = (ret > 0) if positive_call else (ret <= 0)
        hits += int(is_hit)

        if decision in {"BUY", "CONSIDER"}:
            buy_total += 1
            buy_hits += int(ret > 0)
        elif decision == "HOLD":
            hold_total += 1
            hold_hits += int(ret <= 0)

        evaluated.append(ret * 100)

    count = len(evaluated)
    if not count:
        return {
            "window_days": window_days,
            "horizon_days": horizon_days,
            "evaluated": 0,
            "hit_rate": None,
            "avg_return_pct": None,
            "median_return_pct": None,
            "buy_precision": None,
            "hold_precision": None,
        }

    return {
        "window_days": window_days,
        "horizon_days": horizon_days,
        "evaluated": count,
        "hit_rate": round(hits / count, 3),
        "avg_return_pct": round(statistics.mean(evaluated), 2),
        "median_return_pct": round(statistics.median(evaluated), 2),
        "buy_precision": round(buy_hits / buy_total, 3) if buy_total else None,
        "hold_precision": round(hold_hits / hold_total, 3) if hold_total else None,
    }

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/history")
def api_history():
    """Return clean (date, close, volume) rows for charting."""

    ticker = (request.args.get("ticker") or "").upper().strip()
    rng = request.args.get("range") or DEFAULT_RANGE
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400

    start = parse_range_to_start(rng)
    df = load_prices_df(ticker, start)
    if df.empty:
        return jsonify([]), 200

    out = []
    for d, close, volume in zip(df["date"], df["close_price"], df["volume"]):
        if pd.isna(close):
            continue
        out.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "close": float(close),
                "volume": int(volume) if pd.notna(volume) else 0,
            }
        )
    return jsonify(out), 200


@app.post("/api/predict")
@api_key_required
def api_predict():
    """Score tickers based on engineered features and a custom K-Means model."""

    data = request.get_json(silent=True) or {}

    raw_tickers = data.get("tickers", [])
    if isinstance(raw_tickers, str):
        raw_tickers = [t for t in re.split(r"[,\s]+", raw_tickers) if t]
    tickers = [str(t).upper().strip() for t in raw_tickers if t]

    raw_ex = data.get("excludeTickers", [])
    if isinstance(raw_ex, str):
        raw_ex = [t for t in re.split(r"[,\s]+", raw_ex) if t]
    exclude = {str(t).upper().strip() for t in raw_ex if t}

    def _ok(sym: str) -> bool:
        return all(ch.isalnum() or ch in ".-" for ch in sym)

    tickers = [t for t in tickers if _ok(t) and t not in exclude]
    if not tickers:
        return jsonify({"error": "No valid tickers provided."}), 400

    start_raw = data.get("startDate")
    end_raw = data.get("endDate")
    rng = (data.get("range") or DEFAULT_RANGE).strip().lower()

    start = end = None
    try:
        if start_raw:
            start = dt.datetime.strptime(start_raw, "%Y-%m-%d").date()
        if end_raw:
            end = dt.datetime.strptime(end_raw, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format; use YYYY-MM-DD."}), 400
    if start is None:
        start = parse_range_to_start(rng)

    k = int(safe_float(data.get("k"), 3) or 3)
    momentum_w = int(safe_float(data.get("momentumWindow"), 20) or 20)
    vol_w = int(safe_float(data.get("volWindow"), 20) or 20)
    min_avg_vol = float(safe_float(data.get("minAvgVolume"), 300_000) or 300_000)
    min_price = float(safe_float(data.get("minPrice"), 5) or 5)
    risk = (data.get("riskProfile") or "balanced").lower()
    alts = max(0, min(int(safe_float(data.get("alts"), 4) or 4), 5))

    feats: List[np.ndarray] = []
    names: List[str] = []
    raw_features: Dict[str, Dict[str, float]] = {}
    skipped: Dict[str, str] = {}

    for symbol in tickers:
        df = load_prices_df(symbol, start, end)
        if df.empty:
            skipped[symbol] = "no data"
            continue

        df = df.dropna(subset=["close_price"])
        if df.empty:
            skipped[symbol] = "no close data"
            continue

        try:
            avg_vol = float(df["volume"].tail(max(20, vol_w)).mean())
        except Exception:
            avg_vol = None
        med_price = safe_float(df["close_price"].median())
        if avg_vol is None or med_price is None:
            skipped[symbol] = "bad volume/price"
            continue
        if avg_vol < min_avg_vol:
            skipped[symbol] = f"avg volume {int(avg_vol):,} < {int(min_avg_vol):,}"
            continue
        if med_price < min_price:
            skipped[symbol] = f"median price {med_price:.2f} < {min_price:.2f}"
            continue

        feat = compute_features(df["close_price"].tolist(), momentum_w, vol_w)
        if feat is None:
            skipped[symbol] = f"insufficient window (need >= {max(momentum_w, vol_w) + 1} rows)"
            continue

        feats.append(feat)
        names.append(symbol)
        raw_features[symbol] = {"volatility": float(feat[0]), "momentum": float(feat[1])}

    if not feats:
        return jsonify(
            {
                "best": None,
                "alternatives": [],
                "regime": "Insufficient usable data across requested tickers.",
                "debug": {
                    "skipped": skipped,
                    "parsedTickers": names,
                    "start": start.isoformat(),
                    "end": (end and end.isoformat()),
                    "params": {
                        "momentumWindow": momentum_w,
                        "volWindow": vol_w,
                        "minAvgVolume": min_avg_vol,
                        "minPrice": min_price,
                        "riskProfile": risk,
                        "k": k,
                    },
                },
            }
        ), 200

    X = np.vstack(feats)
    X_norm = normalize_features(X)

    k = max(1, min(k, len(names)))
    km = KMeans(n_clusters=k, random_state=42, max_iter=300)
    km.fit(X)
    labels = km.labels_

    scores = [score_stock_row(row, risk) for row in X_norm]
    order = np.argsort(scores)[::-1]
    best_idx = int(order[0])
    alt_idxs = [int(i) for i in order[1 : 1 + min(alts, len(order) - 1)]]

    best = {
        "ticker": names[best_idx],
        "score": round(float(scores[best_idx]), 4),
        "cluster": int(labels[best_idx]),
    }
    alternatives = [
        {
            "ticker": names[i],
            "score": round(float(scores[i]), 4),
            "cluster": int(labels[i]),
        }
        for i in alt_idxs
    ]

    regime = describe_regime(X_norm)

    try:
        decision = choose_decision(best["score"])
        details = {
            "scores": {names[i]: float(scores[i]) for i in range(len(names))},
            "features": raw_features,
            "k": k,
            "labels": {names[i]: int(labels[i]) for i in range(len(names))},
            "regime": regime,
            "params": {
                "momentumWindow": momentum_w,
                "volWindow": vol_w,
                "minAvgVolume": min_avg_vol,
                "minPrice": min_price,
                "riskProfile": risk,
            },
        }
        db.session.add(
            PredictionLog(
                ticker=best["ticker"],
                score=float(best["score"]),
                decision=decision,
                details=details,
            )
        )
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify({"best": best, "alternatives": alternatives, "regime": regime}), 200


@app.get("/api/tickers/suggest")
def api_ticker_suggest():
    """Return ticker/name suggestions for the UI typeahead."""

    q = (request.args.get("q") or "").strip()
    limit = request.args.get("limit", "12")
    require_in_stocks_raw = (request.args.get("require_in_stocks") or "true").lower()
    require_in_stocks = require_in_stocks_raw not in {"false", "0", "no"}

    if not q:
        return jsonify([]), 200

    try:
        suggestions = crud.suggest_tickers(
            q,
            limit=limit,
            require_in_stocks=require_in_stocks,
            include_inactive=False,
        )
        return jsonify(suggestions), 200
    except Exception as exc:
        return jsonify({"error": repr(exc)}), 500


@app.get("/api/metrics/accuracy")
@api_key_required
def api_metrics_accuracy():
    """Expose rolling accuracy metrics for the dashboard KPI."""

    try:
        window = int(request.args.get("windowDays") or DEFAULT_METRICS_WINDOW)
        horizon = int(request.args.get("horizonDays") or DEFAULT_METRICS_HORIZON)
    except ValueError:
        return jsonify({"error": "windowDays and horizonDays must be integers"}), 400

    metrics = evaluate_prediction_accuracy(window, horizon)
    return jsonify(metrics), 200


@app.get("/api/recent-predictions")
def api_recent_predictions():
    """Return the most recent prediction log entries for monitoring."""

    rows = (
        db.session.query(PredictionLog)
        .order_by(PredictionLog.timestamp.desc())
        .limit(20)
        .all()
    )
    out = [
        {
            "timestamp": r.timestamp.isoformat(),
            "ticker": r.ticker,
            "score": float(r.score),
            "decision": r.decision,
        }
        for r in rows
    ]
    return jsonify(out), 200


@app.get("/api/status")
def api_status():
    """Return background population status + live DB counts."""

    try:
        status = crud.get_population_status()
    except Exception as exc:
        status = {"error": repr(exc)}
    status["startup_triggered"] = _startup_once.is_set()
    return jsonify(status), 200

# ---------------------------------------------------------------------------
# Startup hooks & entry point
# ---------------------------------------------------------------------------

@app.before_request
def _kickoff_population_once():
    """Kick off the background population job exactly once per process."""

    if not STARTUP_POPULATE:
        return
    if _startup_once.is_set():
        return

    crud.kickoff_background_population(
        verify_in_stocks=True,
        use_batched_prices=True,
        batch_size=64,
    )
    _startup_once.set()


if __name__ == "__main__":
    print("Connected to the db!")
    app.run(debug=True, port=5000)
