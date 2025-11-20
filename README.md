# WGU Capstone – Stock Recommender

Full-stack data product that ingests equity prices, engineers technical factors, clusters tickers with a custom K-Means implementation, and surfaces BUY/CONSIDER/HOLD guidance plus model-accuracy telemetry via a React dashboard.

The repo hosts:

- `server.py`: Flask API (history, predictions, accuracy metrics, ticker suggestions)
- `crud.py`: Kaggle + yfinance ingestion helpers, ticker maintenance utilities
- `model.py`: SQLAlchemy models (`Stock`, `PredictionLog`, `Ticker`)
- `src/`: React dashboard built with Create React App + Recharts visualizations
- `kmeans.py`: Standalone implementation used by `/api/predict`

---

## 1. Prerequisites

| Tool | Version (tested) | Notes |
| --- | --- | --- |
| Python | 3.10+ | Required for the Flask API and data tooling |
| Node.js | 18+ (with npm) | Required for the React dashboard |
| PostgreSQL | 13+ | Default DSN is `postgresql:///stocks` |
| Kaggle CLI (optional) | latest | Needed only if you run the Kaggle ingest |

---

## 2. Backend (Flask) Setup

1. **Create and activate a virtual environment**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

2. **Install dependencies from `requirements.txt`**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Create the local database (`stocks`)**
   ```bash
   # Using the createdb helper
   createdb stocks

   # or via psql if createdb isn't in PATH
   psql postgres -c "CREATE DATABASE stocks;"
   ```

4. **Configure environment variables**  
   Create `backend.env` (or export via your shell profile):
   ```bash
   export FLASK_APP_SECRET_KEY="dev-secret"
   export API_AUTH_TOKEN="replace-with-strong-token"
   export STARTUP_POPULATE=1           # optional background data load
   export ACCURACY_WINDOW_DAYS=45      # rolling window for evaluation endpoint
   export ACCURACY_HORIZON_DAYS=5      # forward return horizon
   ```
   The API refuses write endpoints unless `API_AUTH_TOKEN` is supplied with each request (`X-API-Key` header or `api_key` query param).

5. **Initialize tables**
   ```bash
   python - <<'PY'
   from server import app
   from model import connect_to_db
   connect_to_db(app, "stocks")
   PY
   ```

6. **Seed price data (two options)**
   - Minimal: pull recent history for tickers already present in the DB.
     ```bash
     python - <<'PY'
     from crud import populate_recent_stocks
     populate_recent_stocks(days_back=180)
     PY
     ```
   - Full: download the Kaggle dataset, populate tickers, then fetch prices.
     ```bash
     python - <<'PY'
     from crud import kickoff_background_population
     kickoff_background_population()
     PY
     ```

7. **Run the API** (startup will auto-populate the dataset if `STARTUP_POPULATE=1`)
   ```bash
   FLASK_ENV=development python server.py
   ```
   The server listens on `http://localhost:5000`. Endpoints:
   - `GET /api/history?ticker=MSFT&range=6mo`
   - `POST /api/predict` (requires JSON body + `X-API-Key`)
   - `GET /api/tickers/suggest?q=tes`
   - `GET /api/recent-predictions`
   - `GET /api/metrics/accuracy?windowDays=45&horizonDays=5` (requires API key)

> **Tip:** keep `/api/predict` running periodically to log predictions (`PredictionLog`) so the accuracy dashboard has material to evaluate.

---

## 3. Frontend (React) Setup

1. **Install dependencies (locks to `package-lock.json`)**
   ```bash
   # npm ci uses package-lock for deterministic installs
   npm ci

   # For local tweaking you can still use:
   # npm install
   ```

2. **Environment variables**  
   Create `./.env` in the repo root (CRA loads vars prefixed with `REACT_APP_`):
   ```bash
   REACT_APP_API_BASE=http://localhost:5000
   REACT_APP_API_KEY=replace-with-strong-token
   ```
   - `REACT_APP_API_KEY` must match the backend `API_AUTH_TOKEN`.
   - Adjust `REACT_APP_API_BASE` if the Flask server runs elsewhere.

3. **Run the dev server**
   ```bash
   npm start
   ```
   Visit `http://localhost:3000`. The dashboard supports:
   - Custom ticker inputs + suggestions
   - Range controls, risk profile, and feature-engineering knobs
   - Multiple visualizations (price line, volume bar, histogram, scatter, etc.)
   - KPI tiles covering recommended ticker, market regime, advisor text, and model accuracy

4. **Build for production** (optional)
   ```bash
   npm run build
   ```

---

## 4. Working With the Accuracy Endpoint

- The backend evaluates historical predictions by comparing each logged `PredictionLog` entry against future closes (`DEFAULT_METRICS_HORIZON` days ahead).
- View metrics via `GET /api/metrics/accuracy` (must include API key). Query params:
  - `windowDays` (default 45) – look-back period of predictions to assess
  - `horizonDays` (default 5) – forward return horizon per prediction
- The React dashboard automatically polls this endpoint on load and displays hit rate, average/median returns, and precision for BUY vs HOLD calls.

---

## 5. Security Notes

- All mutating or sensitive endpoints (`/api/predict`, `/api/metrics/accuracy`, `api/tickers/suggest`) require `X-API-Key`.
- Set `FLASK_APP_SECRET_KEY` and `API_AUTH_TOKEN` to strong, non-default values in production.
- CORS is restricted to `http://localhost:3000`/`127.0.0.1:3000` by default. Extend `FRONTEND_ORIGINS` in `server.py` when deploying elsewhere.
- Sensitive environment files (`backend.env`, `.env`) should **not** be committed.

---

## 6. Common Tasks

| Task | Command |
| --- | --- |
| Remove orphan tickers | `python - <<'PY'\nfrom crud import remove_empty_tickers\nprint(remove_empty_tickers(dry_run=False))\nPY` |
| Inspect population status | `curl http://localhost:5000/api/status` |
| View recent predictions | `curl http://localhost:5000/api/recent-predictions` |
| Refresh Kaggle tickers only | `python - <<'PY'\nfrom crud import populate_tickers_from_kaggle\nsummary = populate_tickers_from_kaggle('./data/symbols_valid_meta.csv')\nprint(summary)\nPY` |
| Generate reports/models artifacts | `python scripts/generate_artifacts.py` |

---

## 7. Troubleshooting

- **`psycopg2` missing headers**: Install PostgreSQL client libraries (`sudo apt install libpq-dev` or equivalent) before running `pip install`.
- **`API key required` errors**: Ensure both backend (`API_AUTH_TOKEN`) and frontend (`REACT_APP_API_KEY`) are set and the header is forwarded when calling endpoints manually.
- **Accuracy card shows “Not enough logged predictions”**: run `/api/predict` a few times so `PredictionLog` contains records inside the configured `windowDays`.
- **CORS errors from React**: double-check that Flask is running on the same origin you configured in `REACT_APP_API_BASE` and that the origin is listed in `FRONTEND_ORIGINS`.

---

Happy building! If you expand the feature set (additional models, auth providers, or monitoring hooks), update this README so future individuals can reproduce the environment quickly.
