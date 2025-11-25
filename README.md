# Stock Recommender (WGU Capstone)

Full-stack demo that pulls equity prices from yfinance, engineers momentum/volatility features, clusters tickers with a custom K-Means implementation, and surfaces BUY/CONSIDER/HOLD guidance plus recent prediction logs in a React dashboard.

## What’s here
- `server.py` — Flask API (`/api/predict`, `/api/history`, `/api/metrics/accuracy`, `/api/tickers/suggest`)
- `crud.py` — price fetchers (yfinance), feature engineering, Kaggle metadata helpers
- `kmeans.py` — lightweight NumPy K-Means used by the pipeline
- `src/` — React app (Create React App + Recharts)
- `recent_predictions.csv` — append-only log for accuracy metrics (created on first POST to `/api/predict`)

## Prerequisites
- Python 3.10+
- Node.js 18+ (with npm)
- Git
- Optional: Kaggle CLI for bulk metadata refresh

## Quick start (local)
1) Clone and install backend deps  
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Set backend env vars (replace with strong values)  
```bash
export FLASK_APP_SECRET_KEY="dev-secret"
export API_AUTH_TOKEN="replace-with-strong-token"
# Optional tuning
export ACCURACY_WINDOW_DAYS=45
export ACCURACY_HORIZON_DAYS=5
```

3) Run the API  
```bash
FLASK_ENV=development python server.py
# listens on http://localhost:5000
```

4) Install frontend deps and run UI  
```bash
npm ci
cat > .env <<'EOF'
REACT_APP_API_BASE=http://localhost:5000
REACT_APP_API_KEY=replace-with-strong-token
EOF
npm start
# visit http://localhost:3000
```

## Common API calls
- History: `GET /api/history?ticker=MSFT&range=6mo`
- Predict (requires API key):  
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: replace-with-strong-token" \
  -d '{"tickers":["AAPL","MSFT","NVDA"],"range":"1y","riskProfile":"balanced"}'
```
- Recent predictions: `GET /api/recent-predictions`
- Accuracy (requires API key): `GET /api/metrics/accuracy?windowDays=45&horizonDays=5`

## Using it in a notebook
```python
import datetime as dt
from crud import run_prediction_pipeline

tickers = ["AAPL", "MSFT", "NVDA", "AMZN"]
start = dt.date.today() - dt.timedelta(days=365)

result = run_prediction_pipeline(
    tickers=tickers,
    start=start,
    end=None,
    k=3,
    momentum_w=20,
    vol_w=20,
    min_avg_vol=300_000,
    min_price=5,
    risk="balanced",
    alts=4,
)
print(result["best"], result["alternatives"])
```
yfinance provides prices on the fly; no database is required for notebook experiments.

## Minimal single-box deployment (for a few evaluators)
- Build frontend: `npm ci && npm run build`
- Run backend with Gunicorn:  
  ```bash
  source env/bin/activate
  API_AUTH_TOKEN=... FLASK_APP_SECRET_KEY=... gunicorn -b 0.0.0.0:5000 server:app
  ```
- Serve `build/` with nginx and proxy `/api/` to `http://127.0.0.1:5000`. Ensure CORS in `server.py` includes your host if not using the default localhost origins.

## Troubleshooting
- yfinance errors: retry or reduce ticker lists; ensure internet access.
- CORS errors: update `FRONTEND_ORIGINS` in `server.py` to include your UI host.
- `API key required`: backend `API_AUTH_TOKEN` must match frontend `REACT_APP_API_KEY` and be passed via `X-API-Key`.
