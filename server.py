from __future__ import annotations

import csv, json, os, re, statistics, threading 
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import datetime as dt
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

import crud
from kmeans import KMeans

# ---------------------------------------------------------------------------
# Flask setup & runtime configuration
# ---------------------------------------------------------------------------

FLASK_APP_SECRET_KEY = os.environ.get("FLASK_APP_SECRET_KEY", "dev-secret")
API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN", "").strip()
DEFAULT_RANGE = "6mo"
MAX_RANGE_DAYS = 365 * 3
DEFAULT_METRICS_WINDOW = int(os.getenv("ACCURACY_WINDOW_DAYS", "45") or 45)
DEFAULT_METRICS_HORIZON = int(os.getenv("ACCURACY_HORIZON_DAYS", "5") or 5)
RECENT_PREDICTIONS_FILE = Path("recent_predictions.csv")
PREDICTION_FIELDS = ["timestamp", "ticker", "score", "decision", "details"]
_SYMBOL_ROWS: List[Dict[str, str]] | None = None
_SAMPLE_CACHE: Dict[Any, Any] = {}

app = Flask(__name__)
app.secret_key = FLASK_APP_SECRET_KEY
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ["http://localhost:5000", "http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            # include X-API-Key since the UI sends it and allow cookies via credentials: "include"
            "allow_headers": ["Content-Type", "Authorization", "X-API-Key"],
            "supports_credentials": True,
        }
    },
)

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


def _ensure_predictions_csv() -> None:
    if RECENT_PREDICTIONS_FILE.exists():
        return
    RECENT_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RECENT_PREDICTIONS_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()


def _normalize_ts(ts: Optional[dt.datetime]) -> Optional[dt.datetime]:
    """Return a tz-naive UTC datetime for consistent comparisons."""

    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(dt.timezone.utc).replace(tzinfo=None)


def append_recent_prediction(record: Dict[str, Any]) -> None:
    """Append a prediction record to the CSV log."""

    _ensure_predictions_csv()
    row = {
        "timestamp": record.get("timestamp", ""),
        "ticker": record.get("ticker", ""),
        "score": record.get("score", ""),
        "decision": record.get("decision", ""),
        "details": json.dumps(record.get("details") or {}),
    }
    with RECENT_PREDICTIONS_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writerow(row)


def load_recent_prediction_rows(
    limit: Optional[int] = None, cutoff: Optional[dt.datetime] = None
) -> List[Dict[str, Any]]:
    """Load predictions from CSV, optionally filtering by time and limit."""

    cutoff = _normalize_ts(cutoff)
    if not RECENT_PREDICTIONS_FILE.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with RECENT_PREDICTIONS_FILE.open() as f:
        reader = csv.DictReader(f)
        for raw in reader:
            ts_raw = raw.get("timestamp")
            try:
                ts = dt.datetime.fromisoformat(ts_raw) if ts_raw else None
            except Exception:
                ts = None
            ts = _normalize_ts(ts)
            if ts is None:
                continue
            if cutoff and ts < cutoff:
                continue
            try:
                details = json.loads(raw.get("details") or "{}")
            except Exception:
                details = {}
            rows.append(
                {
                    "timestamp": ts,
                    "ticker": (raw.get("ticker") or "").upper(),
                    "score": crud.safe_float(raw.get("score"), 0.0) or 0.0,
                    "decision": raw.get("decision") or "",
                    "details": details,
                }
            )
    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_symbol_directory() -> List[Dict[str, str]]:
    """Load cached symbol/name rows from the Kaggle metadata CSV."""

    global _SYMBOL_ROWS
    if _SYMBOL_ROWS is not None:
        return _SYMBOL_ROWS

    path = Path("data/symbols_valid_meta.csv")
    if not path.exists():
        _SYMBOL_ROWS = []
        return _SYMBOL_ROWS

    rows: List[Dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for raw in reader:
            symbol = (raw.get("Symbol") or "").strip().upper()
            name = (raw.get("Security Name") or "").strip()
            if not symbol:
                continue
            rows.append({"symbol": symbol, "name": name})

    _SYMBOL_ROWS = rows
    return rows


def _load_sample_tickers(limit: int = 25) -> List[str]:
    path = Path("tickers.csv")
    allow = Path("tickers_allowlist.txt")
    allow_syms = set(
        line.strip().upper()
        for line in (allow.read_text().splitlines() if allow.exists() else [])
        if line.strip()
    )

    def ok(sym: str) -> bool:
        return sym.isalpha() and not sym.endswith(("W", "R", "U"))

    if path.exists():
        out: List[str] = []
        for line in path.read_text().splitlines():
            sym = line.strip().upper()
            if not sym:
                continue
            if ok(sym) or sym in allow_syms:
                out.append(sym)
            if len(out) >= limit:
                break
        if out:
            return out

    return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"][:limit]


def _compute_sample_clusters(limit: int = 25) -> Dict[str, Any]:
    tickers = _load_sample_tickers(limit)
    start = dt.date.today() - dt.timedelta(days=365)

    result = crud.run_prediction_pipeline(
        tickers=tickers,
        start=start,
        end=None,
        k=min(3, max(2, len(tickers))),
        momentum_w=20,
        vol_w=20,
        min_avg_vol=200_000,
        min_price=2,
        risk="balanced",
        alts=0,
    )

    feats = result.get("raw_features") or {}
    labels = result.get("labels") or {}
    points = []
    for sym, vals in feats.items():
        if sym not in labels:
            continue
        try:
            vol = float(vals.get("volatility"))
            mom = float(vals.get("momentum"))
        except Exception:
            continue
        points.append(
            {
                "ticker": sym,
                "vol": vol,
                "mom": mom,
                "cluster": int(labels[sym]),
            }
        )

    # Compute centroids from points
    centroids = []
    by_cluster: Dict[int, List[Dict[str, float]]] = {}
    for p in points:
        by_cluster.setdefault(p["cluster"], []).append(p)
    for cl, arr in sorted(by_cluster.items()):
        vol_avg = sum(a["vol"] for a in arr) / len(arr)
        mom_avg = sum(a["mom"] for a in arr) / len(arr)
        centroids.append({"cluster": cl, "vol": vol_avg, "mom": mom_avg})

    return {
        "points": points,
        "centroids": centroids,
        "k": result.get("k"),
        "silhouette": result.get("silhouette"),
        "tickers": tickers,
    }


def _get_sample_clusters(limit: int = 25) -> Dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    cache = _SAMPLE_CACHE.get(limit) if isinstance(_SAMPLE_CACHE, dict) else None
    if cache and cache.get("expires") and cache["expires"] > now:
        return cache["payload"]

    payload = _compute_sample_clusters(limit)
    _SAMPLE_CACHE[limit] = {
        "expires": now + dt.timedelta(hours=6),
        "payload": payload,
    }
    return payload


def load_prices_df(
    ticker: str,
    start_date: dt.date,
    end_date: Optional[dt.date] = None
) -> pd.DataFrame:
    """Compatibility wrapper for legacy callers."""

    return crud.fetch_prices_from_yf(ticker, start_date, end_date)

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _lookup_close_on_or_after(ticker: str, ts: dt.datetime) -> Optional[float]:
    """Return the first close_price recorded on or after the timestamp."""

    if ts.tzinfo is None:
        target = ts.replace(tzinfo=dt.timezone.utc)
    else:
        target = ts.astimezone(dt.timezone.utc)

    start_date = target.date() - dt.timedelta(days=5)
    end_date = target.date() + dt.timedelta(days=10)
    df = crud.fetch_prices_from_yf(ticker, start_date, end_date)
    if df.empty:
        return None

    df = df.sort_values("date")
    for d, close in zip(df["date"], df["close_price"]):
        if pd.isna(close):
            continue
        row_ts = dt.datetime.combine(d, dt.time.min, dt.timezone.utc)
        if row_ts >= target:
            return float(close)
    return None


def evaluate_prediction_accuracy(
    window_days: int, horizon_days: int, max_rows: int = 200
) -> Dict[str, object]:
    """Compare historical predictions to realized future returns."""

    window_days = max(1, int(window_days))
    horizon_days = max(1, int(horizon_days))
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)

    preds = load_recent_prediction_rows(limit=max_rows, cutoff=cutoff)

    evaluated: List[float] = []
    hits = buy_hits = buy_total = hold_hits = hold_total = 0

    for pred in preds:
        entry_price = _lookup_close_on_or_after(pred["ticker"], pred["timestamp"])
        if entry_price in (None, 0):
            continue
        future_ts = pred["timestamp"] + dt.timedelta(days=horizon_days)
        future_price = _lookup_close_on_or_after(pred["ticker"], future_ts)
        if future_price is None:
            continue

        ret = (future_price - entry_price) / entry_price
        decision = (pred["decision"] or "").upper()
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
    # df = load_prices_df(ticker, start)
    df = crud.fetch_prices_from_yf(ticker, start)
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

    k = int(crud.safe_float(data.get("k"), 3) or 3)
    momentum_w = int(crud.safe_float(data.get("momentumWindow"), 20) or 20)
    vol_w = int(crud.safe_float(data.get("volWindow"), 20) or 20)
    min_avg_vol = float(crud.safe_float(data.get("minAvgVolume"), 300_000) or 300_000)
    min_price = float(crud.safe_float(data.get("minPrice"), 5) or 5)
    risk = (data.get("riskProfile") or "balanced").lower()
    alts = max(0, min(int(crud.safe_float(data.get("alts"), 4) or 4), 5))

    result = crud.run_prediction_pipeline(
        tickers,
        start=start,
        end=end,
        k=k,
        momentum_w=momentum_w,
        vol_w=vol_w,
        min_avg_vol=min_avg_vol,
        min_price=min_price,
        risk=risk,
        alts=alts,
    )

    if not result["best"]:
        return jsonify(
            {
                "best": None,
                "alternatives": [],
                "regime": result["regime"],
                "debug": {
                    "skipped": result["skipped"],
                    "parsedTickers": tickers,
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

    best = {
        "ticker": result["best"]["ticker"],
        "score": result["best"]["score"],
        "cluster": result["best"]["cluster"],
    }
    alternatives = [
        {"ticker": alt["ticker"], "score": alt["score"], "cluster": alt["cluster"]}
        for alt in result["alternatives"]
    ]

    details = {
        "scores": result["scores"],
        "features": result["raw_features"],
        "k": result["k"],
        "labels": result["labels"],
        "regime": result["regime"],
        "thresholds": result["thresholds"],
        "params": {
            "momentumWindow": momentum_w,
            "volWindow": vol_w,
            "minAvgVolume": min_avg_vol,
            "minPrice": min_price,
            "riskProfile": risk,
        },
    }
    try:
        append_recent_prediction(
            {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "ticker": best["ticker"],
                "score": float(result["best"]["score"]),
                "decision": result["best"]["decision"],
                "details": details,
            }
        )
    except Exception:
        pass

    return (
        jsonify(
            {
                "best": best,
                "alternatives": alternatives,
                "regime": result["regime"],
                "silhouette": result.get("silhouette"),
                "raw_features": result.get("raw_features", {}),
                "labels": result.get("labels", {}),
                "scores": result.get("scores", {}),
                "thresholds": result.get("thresholds", {}),
                "k": result.get("k"),
                "skipped": result.get("skipped", {}),
            }
        ),
        200,
    )


@app.get("/api/tickers/suggest")
def api_ticker_suggest():
    """Return ticker/name suggestions for the UI typeahead."""

    q = (request.args.get("q") or "").strip()
    limit_raw = request.args.get("limit", "12")
    if not q:
        return jsonify([]), 200

    try:
        limit = max(1, min(int(limit_raw), 50))
    except ValueError:
        limit = 12

    rows = load_symbol_directory()
    if not rows:
        return jsonify([]), 200

    q_upper = q.upper()
    q_lower = q.lower()

    matches: List[Dict[str, str]] = []
    for row in rows:
        symbol = row["symbol"]
        name = row["name"]
        prefix = symbol.startswith(q_upper)
        contains = q_lower in name.lower() if name else False
        if prefix or contains:
            matches.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "matchRank": 0 if prefix else 1,
                }
            )

    matches.sort(key=lambda r: (r["matchRank"], r["symbol"]))
    out = [{"symbol": m["symbol"], "name": m["name"]} for m in matches[:limit]]
    return jsonify(out), 200


@app.get("/api/metrics/accuracy")
@api_key_required
def api_metrics_accuracy():
    """Expose rolling accuracy metrics for the dashboard KPI."""

    try:
        window = int(request.args.get("windowDays") or DEFAULT_METRICS_WINDOW)
        horizon = int(request.args.get("horizonDays") or DEFAULT_METRICS_HORIZON)
    except ValueError:
        return jsonify({"error": "windowDays and horizonDays must be integers"}), 400

    try:
        metrics = evaluate_prediction_accuracy(window, horizon)
    except Exception as exc:
        return jsonify({"error": f"failed to compute metrics: {exc}"}), 500
    return jsonify(metrics), 200


@app.get("/api/clusters/sample")
def api_clusters_sample():
    """Return a reference clustering cloud to visualize K-Means even before predictions."""

    try:
        limit = int(request.args.get("limit") or 25)
        limit = max(5, min(limit, 60))
    except ValueError:
        limit = 25

    try:
        payload = _get_sample_clusters(limit)
    except Exception as exc:
        return jsonify({"error": f"failed to build sample clusters: {exc}"}), 500

    return jsonify(payload), 200


@app.get("/api/recent-predictions")
def api_recent_predictions():
    """Return the most recent prediction log entries for monitoring."""

    rows = load_recent_prediction_rows(limit=20)
    out = []
    for r in rows:
        out.append(
            {
                "timestamp": r["timestamp"].isoformat(),
                "ticker": r["ticker"],
                "score": float(r["score"]),
                "decision": r["decision"],
            }
        )
    return jsonify(out), 200


@app.get("/api/status")
def api_status():
    """Return a lightweight status payload without DB dependencies."""

    recent = len(load_recent_prediction_rows(limit=5))
    return jsonify({"status": "ok", "recentPredictions": recent}), 200


def kickoff_population_job() -> bool:
    """Trigger the Kaggle/price population background task."""

    try:
        crud.kickoff_background_population()
        return True
    except Exception as exc:
        app.logger.warning("kickoff_background_population failed: %s", exc)
        return False


@app.post("/api/population/kickoff")
@api_key_required
def api_kickoff_population():
    """Endpoint to manually start the population worker."""

    started = kickoff_population_job()
    if not started:
        return jsonify({"started": False, "error": "failed to start"}), 500
    return jsonify({"started": True}), 202

_thread_started = False
_thread_lock = threading.Lock()

def background_worker():
    while True:
        kickoff_population_job()
        return False

@app.before_request
def _start_population_daemon():
    """Ensure the Kaggle population job starts once per process."""

    # kickoff_population_job()
    global _thread_started
    if not _thread_started:
        with _thread_lock:
            if not _thread_started:
                t = threading.Thread(target=background_worker, daemon=True)
                t.start()
                _thread_started = True

# ---------------------------------------------------------------------------
# Startup hooks & entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # kickoff_population_job()
    app.run(debug=True, port=5000)
