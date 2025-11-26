#!/usr/bin/env python3
"""
Offline validation + sample output generator.

What it does:
- Pulls features for a ticker set (from tickers.csv, with a safe fallback list).
- Fits the in-repo KMeans model and computes an internal validation metric
  (Silhouette Coefficient) plus inertia.
- Saves validation results to reports/model_validation.json.
- Runs a sample prediction pipeline call and saves the output to
  reports/sample_prediction.json to show the model's shape of results.

References:
- Rousseeuw, 1987 (Silhouette): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- MacQueen, 1967 (K-Means): https://en.wikipedia.org/wiki/K-means_clustering
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from crud import compute_features, fetch_prices_from_yf
from kmeans import KMeans
ARTIFACT_ROOT = Path("reports")
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------

def load_tickers(limit: int = 120) -> List[str]:
    """
    Read tickers.csv if present and pick a reasonable subset.
    Filters out symbols that look like units/warrants unless explicitly allowed.
    """
    ticker_file = Path("tickers.csv")
    allow_file = Path("tickers_allowlist.txt")

    manual_allow = {
        line.strip().upper()
        for line in (allow_file.read_text().splitlines() if allow_file.exists() else [])
        if line.strip()
    }

    def is_equity(sym: str) -> bool:
        return sym.isalpha() and not sym.endswith(("W", "R", "U"))

    if ticker_file.exists():
        tickers: List[str] = []
        for line in ticker_file.read_text().splitlines():
            sym = line.strip().upper()
            if not sym:
                continue
            if is_equity(sym) or sym in manual_allow:
                tickers.append(sym)
            if len(tickers) >= limit:
                break
    else:
        tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA"]

    return tickers


def build_feature_matrix(
    tickers: List[str],
    start: dt.date,
    momentum_w: int = 20,
    vol_w: int = 20,
) -> Tuple[np.ndarray, List[str]]:
    """Fetch OHLCV data and build (volatility, momentum) feature vectors."""
    feats: List[np.ndarray] = []
    names: List[str] = []
    for sym in tickers:
        df = fetch_prices_from_yf(sym, start)
        if df.empty or df["close_price"].isna().all():
            continue
        feat = compute_features(df["close_price"].to_list(), momentum_w, vol_w)
        if feat is None:
            continue
        feats.append(feat)
        names.append(sym)
    if not feats:
        return np.empty((0, 2)), []
    return np.vstack(feats), names


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute average Silhouette Coefficient (range [-1, 1]).
    Manual implementation to avoid extra dependencies.
    References: scikit-learn silhouette guide
    (https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient),
    Wikipedia summary (https://en.wikipedia.org/wiki/Silhouette_(clustering)).
    """
    n = X.shape[0]
    if n < 3 or len(np.unique(labels)) < 2:
        return 0.0

    # Precompute all pairwise distances
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    sil_values = []
    for i in range(n):
        same = labels == labels[i]
        other = ~same

        # a: mean intra-cluster distance
        a = dists[i, same]
        a = a[a > 0]  # drop self
        a = a.mean() if a.size else 0.0

        # b: lowest mean distance to any other cluster
        b = np.inf
        for lbl in np.unique(labels[other]):
            mask = labels == lbl
            mean_dist = dists[i, mask].mean()
            b = min(b, mean_dist)

        s = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
        sil_values.append(s)
    return float(np.mean(sil_values))


def validate_kmeans(X: np.ndarray, names: List[str], k: int = 3) -> Dict[str, object]:
    """Fit KMeans and compute internal validation metrics."""
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    sil = silhouette_score(X, km.labels_)

    return {
        "k": k,
        "n_samples": int(X.shape[0]),
        "silhouette": sil,
        "inertia": km.inertia,
        "cluster_sizes": {
            int(lbl): int((km.labels_ == lbl).sum()) for lbl in np.unique(km.labels_)
        },
        "centroids": km.cluster_centers_.tolist(),
        "tickers": names,
    }


def run_sample_prediction(tickers: List[str]) -> Dict[str, object]:
    """
    Run a tiny example prediction to show shape of outputs.
    Uses the same horizon/risk defaults as the dashboard.
    """
    # Lazily import to avoid pulling Flask on import
    import crud

    today = dt.date.today()
    start = today - dt.timedelta(days=365)
    payload = crud.run_prediction_pipeline(
        tickers=tickers[:6],
        start=start,
        end=None,
        k=min(3, max(1, len(tickers[:6]))),
        momentum_w=20,
        vol_w=20,
        min_avg_vol=300_000,
        min_price=5,
        risk="balanced",
        alts=3,
    )
    return {
        "requested_tickers": tickers[:6],
        "result": payload,
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def main() -> None:
    # Try to build with enough tickers to ensure at least 3 usable after filtering.
    # If the initial pull yields fewer than 3 points, widen the pool.
    tickers = load_tickers(limit=200)
    start = dt.date.today() - dt.timedelta(days=365)

    X, names = build_feature_matrix(tickers, start)
    if X.shape[0] < 3:
        # Second attempt with a larger pool and lower min requirements via a broader ticker list
        tickers = load_tickers(limit=400)
        X, names = build_feature_matrix(tickers, start)
    if X.shape[0] < 3:
        raise SystemExit("No usable features (need at least 3 tickers); check yfinance access or ticker list.")

    validation = validate_kmeans(X, names, k=min(3, max(2, len(names) // 10)))
    validation["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()

    sample = run_sample_prediction(names)

    save_json(ARTIFACT_ROOT / "model_validation.json", validation)
    save_json(ARTIFACT_ROOT / "sample_prediction.json", sample)

    print("Validation metric (silhouette): {:.3f}".format(validation["silhouette"]))
    print(f"Used {validation['n_samples']} tickers; artifacts saved under {ARTIFACT_ROOT}/")


if __name__ == "__main__":
    main()
