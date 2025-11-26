# --- add near top of crud.py (or keep your existing ones) ---
import os
import requests
import datetime as dt
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from kmeans import KMeans

def ensure_tickers():
    """Create a tickers.csv from NASDAQ list if not present."""
    if not os.path.exists("tickers.csv"):
        url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        nasdaq = pd.read_csv(url, sep="|")
        tickers = [str(t).strip() for t in nasdaq["Symbol"].tolist() if str(t).strip() != "nan"]
        with open("tickers.csv", "w") as f:
            for t in tickers:
                t = str(t).strip()
                if t == 'nan':
                    continue
                if len(t) > 5:
                    break
                f.write(t + "\n")

# ======= status helpers (reuse if you already have them) =======
_population_status = {
    "started": False,
    "finished": False,
    "error": None,
    "steps": [],
    "started_at": None,
    "finished_at": None,
}

def _now_utc_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def _mark(msg: str):
    _population_status["steps"].append(msg)
    # keep steps list bounded
    if len(_population_status["steps"]) > 200:
        _population_status["steps"] = _population_status["steps"][-200:]


def safe_float(value, default=None):
    """Coerce to float, returning a default when conversion fails."""

    try:
        v = float(value)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def fetch_prices_from_yf(
    ticker: str, start_date: dt.date, end_date: Optional[dt.date] = None
) -> pd.DataFrame:
    """Fetch OHLCV rows for a symbol directly from yfinance."""

    ticker = (ticker or "").upper().strip()
    if not ticker:
        return pd.DataFrame()

    yf_df = yf.download(
        ticker,
        start=start_date,
        end=end_date or dt.date.today(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if yf_df is None or yf_df.empty:
        return pd.DataFrame()

    if isinstance(yf_df.columns, pd.MultiIndex):
        try:
            yf_df = yf_df.xs(ticker, axis=1, level=1)
        except (KeyError, IndexError):
            yf_df = yf_df.droplevel(1, axis=1)

    yf_df = yf_df.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Adj Close": "adj_close_price",
            "Volume": "volume",
        }
    )
    yf_df["date"] = pd.to_datetime(yf_df["date"]).dt.date
    return yf_df

# ======= Kaggle fetch & populate =======
def fetch_latest_kaggle_dataset(output_dir="./data") -> str:
    """
    Downloads and extracts the latest stock-market-dataset from Kaggle
    into the current relative directory (default ./data).

    No Kaggle CLI or authentication required.
    Returns the path to symbols_valid_meta.csv if found.
    """
    os.makedirs(output_dir, exist_ok=True)
    kaggle_zip_url = (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "jacksoncrow/stock-market-dataset"
    )
    zip_path = os.path.join(output_dir, "stock-market-dataset.zip")

    print("📦 Downloading dataset from Kaggle...")
    resp = requests.get(kaggle_zip_url, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed: {resp.status_code} — Kaggle may require login.")
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Downloaded to {zip_path}")

    print("📂 Extracting ZIP...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Filter out unwanted directories
        members = [
            m for m in zip_ref.namelist()
            if not (m.startswith("etfs/") or m.startswith("stocks/"))
        ]
        zip_ref.extractall(output_dir, members)
    os.remove(zip_path)

    # Find relevant CSV
    candidates = [
        os.path.join(output_dir, "symbols_valid_meta.csv"),
        os.path.join(output_dir, "symbols.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"✅ Found symbols CSV: {p}")
            return p
    raise FileNotFoundError("symbols_valid_meta.csv not found after extraction.")


def rolling_mean(arr: np.ndarray, window: int) -> Optional[float]:
    if len(arr) < window:
        return None
    sample = arr[-window:]
    mask = np.isfinite(sample)
    if mask.sum() < window:
        return None
    return float(sample.mean())


def rolling_std(arr: np.ndarray, window: int) -> Optional[float]:
    if len(arr) < window:
        return None
    sample = arr[-window:]
    mask = np.isfinite(sample)
    if mask.sum() < window:
        return None
    return float(sample.std(ddof=0))


def compute_features(close_series, momentum_w: int, vol_w: int) -> Optional[np.ndarray]:
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
    Xn = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        cmin, cmax = np.nanmin(col), np.nanmax(col)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
            Xn[:, j] = 0.0
        else:
            Xn[:, j] = (col - cmin) / (cmax - cmin)
    return Xn


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Lightweight silhouette coefficient (no sklearn dependency)."""

    if X.ndim != 2 or X.shape[0] < 3:
        return None
    uniq = np.unique(labels)
    if uniq.size < 2:
        return None

    # Pairwise distances
    diff = X[:, None, :] - X[None, :, :]
    dists = np.sqrt((diff * diff).sum(axis=2))

    out = []
    for i in range(X.shape[0]):
        same = labels == labels[i]
        other = ~same

        a_vals = dists[i, same]
        a_vals = a_vals[a_vals > 0]
        a = a_vals.mean() if a_vals.size else 0.0

        b = None
        for lbl in uniq:
            if lbl == labels[i]:
                continue
            mask = labels == lbl
            if mask.sum() == 0:
                continue
            mean_dist = dists[i, mask].mean()
            b = mean_dist if b is None else min(b, mean_dist)
        if b is None or max(a, b) == 0:
            s = 0.0
        else:
            s = (b - a) / max(a, b)
        out.append(s)

    return float(np.mean(out)) if out else None


def score_stock_row(raw_row: np.ndarray, risk: str) -> float:
    """
    Convert raw (volatility, momentum) features into a smoother 0–1 confidence score.

    Previously we normalized per-request (min/max across the submitted tickers), which
    collapses to coarse values like 0.0, 0.5, or 1.0 when the list is short. Here we
    instead map the absolute feature magnitudes through soft squashing functions so a
    single ticker can still land anywhere in [0, 1].
    """

    vol = float(raw_row[0])
    mom = float(raw_row[1])

    # Map momentum (average daily return) into [0, 1]; ~1% avg daily return ≈ 0.88
    MOM_SCALE = 0.01
    mom_util = 0.5 + 0.5 * np.tanh(mom / MOM_SCALE)

    # Map volatility (stdev of daily returns) into [0, 1]; ~2.5% stdev is neutral.
    VOL_CENTER = 0.025
    VOL_SCALE = 0.02
    vol_util = 1.0 - (0.5 + 0.5 * np.tanh((vol - VOL_CENTER) / VOL_SCALE))

    mom_util = float(np.clip(mom_util, 0.0, 1.0))
    vol_util = float(np.clip(vol_util, 0.0, 1.0))

    risk = (risk or "balanced").lower()
    if risk == "conservative":
        w_mom, w_vol = 0.25, 0.75
    elif risk == "aggressive":
        w_mom, w_vol = 0.75, 0.25
    else:
        w_mom, w_vol = 0.5, 0.5

    return float(w_mom * mom_util + w_vol * vol_util)


def compute_decision_thresholds(scores: Sequence[float]) -> tuple[float, float]:
    base_buy, base_consider = 0.65, 0.45
    if len(scores) < 3:
        return base_buy, base_consider

    buy_cut = float(np.percentile(scores, 75))
    consider_cut = float(np.percentile(scores, 35))
    buy_thr = max(base_buy, buy_cut)
    consider_thr = min(buy_thr - 0.05, max(base_consider, consider_cut))
    return buy_thr, consider_thr


def choose_decision(score: float, buy_thr: float, consider_thr: float) -> str:
    if score >= buy_thr:
        return "BUY"
    if score >= consider_thr:
        return "CONSIDER"
    return "HOLD"


def describe_regime(X_norm: np.ndarray) -> str:
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


# ======= Startup: integrate Kaggle + your existing price population =======
def kickoff_background_population() -> bool:
    """
    Non-blocking startup population to 
    populate tickers from NASDAQ and
    fetch Kaggle dataset (symbols_valid_meta.csv)
    """

    _population_status.update({
        "started": True,
        "finished": False,
        "error": None,
        "steps": [],
        "started_at": _now_utc_iso(),
        "finished_at": None,
    })
    try:
        # 1) Ensure tickers.csv
        _mark("Ensuring tickers.csv from NASDAQ")
        ensure_tickers()
        # 2) Kaggle download
        _mark("Kaggle: download latest stock-market-dataset")
        fetch_latest_kaggle_dataset(output_dir="./data")

        _population_status["finished"] = True
        _population_status["finished_at"] = _now_utc_iso()
        _mark("done")
    except Exception as e:
        _population_status["error"] = repr(e)
        _population_status["finished"] = True
        _population_status["finished_at"] = _now_utc_iso()

    return True


def run_prediction_pipeline(
    tickers: List[str],
    *,
    start: dt.date,
    end: Optional[dt.date],
    k: int,
    momentum_w: int,
    vol_w: int,
    min_avg_vol: float,
    min_price: float,
    risk: str,
    alts: int,
) -> Dict[str, Any]:
    """Load candles, build features, cluster, and score tickers."""

    feats: List[np.ndarray] = []
    names: List[str] = []
    raw_features: Dict[str, Dict[str, float]] = {}
    skipped: Dict[str, str] = {}

    for symbol in tickers:
        df = fetch_prices_from_yf(symbol, start, end)
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
        return {
            "best": None,
            "alternatives": [],
            "regime": "Insufficient usable data across requested tickers.",
            "raw_features": raw_features,
            "scores": {},
            "labels": {},
            "k": 0,
            "thresholds": {"buy": None, "consider": None},
            "skipped": skipped,
        }

    X = np.vstack(feats)
    X_norm = normalize_features(X)

    k = max(1, min(k, len(names)))
    km = KMeans(n_clusters=k, random_state=42, max_iter=300)
    km.fit(X)
    labels = km.labels_

    scores = [score_stock_row(row, risk) for row in X]
    buy_thr, consider_thr = compute_decision_thresholds(scores)
    order = np.argsort(scores)[::-1]
    best_idx = int(order[0])
    alt_idxs = [int(i) for i in order[1 : 1 + min(alts, len(order) - 1)]]

    regime = describe_regime(X_norm)
    labels_map = {names[i]: int(labels[i]) for i in range(len(names))}
    scores_map = {names[i]: float(scores[i]) for i in range(len(names))}
    sil = silhouette_score(X_norm, labels)

    best = {
        "ticker": names[best_idx],
        "score": round(float(scores[best_idx]), 4),
        "cluster": int(labels[best_idx]),
        "decision": choose_decision(float(scores[best_idx]), buy_thr, consider_thr),
    }
    alternatives = [
        {
            "ticker": names[i],
            "score": round(float(scores[i]), 4),
            "cluster": int(labels[i]),
            "decision": choose_decision(float(scores[i]), buy_thr, consider_thr),
        }
        for i in alt_idxs
    ]

    return {
        "best": best,
        "alternatives": alternatives,
        "regime": regime,
        "raw_features": raw_features,
        "scores": scores_map,
        "labels": labels_map,
        "k": k,
        "thresholds": {"buy": buy_thr, "consider": consider_thr},
        "skipped": skipped,
        "silhouette": sil,
    }
