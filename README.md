# Stock Recommender

Full-stack demo that pulls equity prices from yfinance, engineers momentum/volatility features, clusters tickers with a custom K-Means implementation, and surfaces BUY/CONSIDER/HOLD guidance plus recent prediction logs in a React dashboard.

## What’s here
- `server.py` — Flask API (`/api/predict`, `/api/history`, `/api/metrics/accuracy`, `/api/tickers/suggest`)
- `crud.py` — price fetchers (yfinance), feature engineering, Kaggle metadata helpers
- `kmeans.py` — lightweight K-Means used by the pipeline
- `src/` — React app (Create React App + Recharts)
- `recent_predictions.csv` — append-only log for accuracy metrics (created on first POST to `/api/predict`)
- No database or SQLAlchemy dependencies; everything is in-memory with yfinance plus CSV logs.

## Prerequisites
- Python 3.10+
- Node.js 18+ (with npm)
- Git

Ubuntu quick install:
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip nodejs npm git
```

## Local setup
1) Backend
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

   export FLASK_APP_SECRET_KEY="dev-secret"
   export API_AUTH_TOKEN="replace-with-strong-token"
   # optional tuning
   export ACCURACY_WINDOW_DAYS=45
   export ACCURACY_HORIZON_DAYS=5

   FLASK_ENV=development python server.py  # http://localhost:5000
   ```

2) Frontend
   ```bash
   npm ci
   cat > .env <<'EOF'
   REACT_APP_API_BASE=http://localhost:5000
   REACT_APP_API_KEY=replace-with-strong-token
   EOF
   # Run in a separate terminal while the backend is running:
   npm start  # http://localhost:3000
   ```

## Build and serve the frontend
- Development: `npm start` (CRA dev server on http://localhost:3000, proxying to your API)
- Production build: `npm run build` (outputs `./build`)
- Ensure the frontend `.env` and backend `API_AUTH_TOKEN` match before running or building.

## Model validation & testing
- **Cluster quality (Silhouette Coefficient)**: run `python3 generate_artifacts.py` to compute an internal validation metric for the KMeans clustering (silhouette in [-1, 1]; higher means tighter, better-separated clusters). The script saves `reports/model_validation.json` with the silhouette, inertia, and cluster sizes, plus `reports/sample_prediction.json` showing a real prediction payload for a handful of tickers.
- **Temporal hit-rate backtest**: use the `/api/metrics/accuracy` endpoint (or `evaluate_prediction_accuracy` in `server.py`) with a window (e.g., 45 days) and horizon (e.g., 5 days). It compares past BUY/CONSIDER/HOLD calls to realized forward returns and reports hit rates for positive (BUY/CONSIDER) vs HOLD decisions.
- **Manual spot checks**: invoke `/api/predict` with a short ticker list (e.g., 5–10 names) and verify scores/decisions are plausible given recent price action; combine with `/api/history` plots in the UI for context.

## Notes
- yfinance provides prices via YFinance's api; no database is required.

## Troubleshooting
- yfinance errors: retry or reduce ticker lists; ensure internet access.
- CORS errors: update `FRONTEND_ORIGINS` in `server.py` to include your UI host.
- `API key required`: backend `API_AUTH_TOKEN` must match frontend `REACT_APP_API_KEY` and be passed via `X-API-Key`.
