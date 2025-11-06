# --- add near top of crud.py (or keep your existing ones) ---
import os, subprocess, threading, requests, datetime as dt, zipfile
import pandas as pd

from model import db, Stock, Ticker
from ctx import with_app_context
from sqlalchemy import or_, case, exists, func

def ensure_tickers():
    """Create a tickers.csv from NASDAQ list if not present."""
    if not os.path.exists("tickers.csv"):
        url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        nasdaq = pd.read_csv(url, sep="|")
        tickers = [str(t).strip() for t in nasdaq["Symbol"].tolist() if str(t).strip() != "nan"]
        with open("tickers.csv", "w") as f:
            for t in tickers:
                # keep it reasonable
                if len(t) <= 5:
                    f.write(t + "\n")

@with_app_context
def populate_recent_stocks(days_back: int = 365) -> dict[str, int]:
    """
    Incrementally backfill per ticker using yfinance (single-thread).
    For each Ticker in DB, fetch candles from (last_date+1) -> today.
    """
    _mark("populate_recent_stocks: start")
    session = db.session

    # Prefer Ticker table; if empty, fall back to distinct symbols from Stock
    tickers = [t.ticker for t in session.query(Ticker.ticker).all()]
    if not tickers:
        tickers = [t[0] for t in session.query(Stock.ticker).distinct().all()]

    if not tickers:
        _mark("populate_recent_stocks: no tickers found in DB")
        return {"tickers": 0, "inserted": 0}

    inserted_total = 0
    today = dt.datetime.utcnow()

    for sym in tickers:
        sym = (sym or "").strip().upper()
        if not sym:
            continue

        try:
            last_date = (
                session.query(func.max(Stock.date))
                .filter(Stock.ticker == sym)
                .scalar()
            )
            start = (last_date + dt.timedelta(days=1)) if last_date else (today - dt.timedelta(days=days_back))
            if start >= today:
                continue

            df = yf.download(sym, start=start, end=today, interval="1d", progress=False, threads=False, auto_adjust=True)
            if df is None or df.empty:
                continue

            new_rows = []
            for idx, row in df.iterrows():
                # idx may be pandas Timestamp (tz-aware sometimes), normalize to naive UTC
                ts = dt.datetime.utcfromtimestamp(idx.timestamp()) if hasattr(idx, "timestamp") else idx
                record_id = f"{sym}-{ts.strftime('%Y%m%d')}"
                # upsert guard
                exists = session.query(Stock).filter(Stock.record_id == record_id).first()
                if exists:
                    continue
                new_rows.append(Stock(
                    record_id=record_id,
                    ticker=sym,
                    date=ts,
                    open_price=float(row.get("Open", 0.0)) if row.get("Open", None) is not None else None,
                    high_price=float(row.get("High", 0.0)) if row.get("High", None) is not None else None,
                    low_price=float(row.get("Low", 0.0)) if row.get("Low", None) is not None else None,
                    close_price=float(row.get("Close", 0.0)) if row.get("Close", None) is not None else None,
                    adj_close_price=float(row.get("Adj Close", 0.0)) if row.get("Adj Close", None) is not None else None,
                    volume=int(row.get("Volume", 0)) if row.get("Volume", None) is not None else None,
                    created_at=dt.datetime.utcnow(),
                ))

            if new_rows:
                session.bulk_save_objects(new_rows)
                session.commit()
                inserted_total += len(new_rows)
                _mark(f"{sym}: inserted {len(new_rows)} rows")

        except Exception as e:
            session.rollback()
            _mark(f"{sym}: error {e}")

    _mark(f"populate_recent_stocks: done, inserted={inserted_total}")
    print("✅ Stock data population complete.")
    return {"tickers": len(tickers), "inserted": inserted_total}



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
    return dt.datetime.utcnow().isoformat()

def _mark(msg: str):
    _population_status["steps"].append(msg)
    # keep steps list bounded
    if len(_population_status["steps"]) > 200:
        _population_status["steps"] = _population_status["steps"][-200:]

@with_app_context
def get_population_status():
    """Add live counts safely."""
    try:
        stock_rows = db.session.query(Stock).count()
        stocks_distinct = db.session.query(Stock.ticker).distinct().count()
        tickers_rows = db.session.query(Ticker).count()
    except Exception as e:
        stock_rows = stocks_distinct = tickers_rows = 0
        _population_status["error"] = _population_status.get("error") or repr(e)

    return {
        **_population_status,
        "stock_rows": stock_rows,
        "stocks_distinct_tickers": stocks_distinct,
        "tickers_rows": tickers_rows,
    }

# ======= Kaggle fetch & populate =======

def _ensure_kaggle_auth_ready():
    """
    Best-effort check that kaggle CLI can run with a valid kaggle.json.
    Raises if 'kaggle' CLI not found; warns (marks) if config is missing.
    """
    try:
        # Will raise if binary missing
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise RuntimeError("Kaggle CLI not found. Install with `pip install kaggle` and place kaggle.json in ~/.kaggle/") from e

    # Check for kaggle.json
    kaggle_dir = os.environ.get("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
    cred = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(cred):
        _mark("WARNING: ~/.kaggle/kaggle.json not found — Kaggle download will fail.")

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

    print("📦 Downloading dataset from Kaggle (no auth)...")
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

@with_app_context
def populate_tickers_from_kaggle(csv_path: str, *, verify_in_stocks: bool = True, chunk_size: int = 1000) -> dict:
    """
    Populate `Ticker` table from Kaggle symbols CSV.
    - If verify_in_stocks=True, only insert symbols that already exist in `stocks` table.
    Returns a summary dict.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    _mark(f"Loading symbols from {csv_path}")
    df = pd.read_csv(csv_path)

    # Column normalization (dataset uses 'Symbol' and 'Security Name')
    # Fallbacks for robustness.
    sym_col = None
    for c in ["Symbol", "symbol", "Ticker", "ticker"]:
        if c in df.columns:
            sym_col = c; break
    if sym_col is None:
        raise ValueError(f"No symbol column found in {csv_path}")

    name_col = None
    for c in ["Security Name", "Name", "companyName", "company_name"]:
        if c in df.columns:
            name_col = c; break

    # Build symbol -> name
    raw = df[[sym_col] + ([name_col] if name_col else [])].dropna(subset=[sym_col])
    raw[sym_col] = raw[sym_col].astype(str).str.upper().str.strip()
    # Basic filter: avoid clearly invalid rows
    raw = raw[(raw[sym_col] != "") & (~raw[sym_col].str.contains(r"\s", regex=True))]

    symbols_all = raw[sym_col].unique().tolist()
    _mark(f"Kaggle symbols loaded: {len(symbols_all)}")

    # Restrict to symbols actually present in prices, if requested
    if verify_in_stocks:
        present_rows = db.session.query(Stock.ticker).filter(Stock.ticker.in_(symbols_all)).distinct().all()
        symbols = sorted({r[0].upper() for r in present_rows})
        _mark(f"Symbols verified present in stocks: {len(symbols)}")
    else:
        symbols = symbols_all

    # Already in Ticker?
    existing_rows = db.session.query(Ticker.ticker).filter(Ticker.ticker.in_(symbols)).all()
    existing = {r[0].upper() for r in existing_rows}
    to_insert = [s for s in symbols if s not in existing]

    inserted = 0
    batch = []

    def flush():
        nonlocal inserted, batch
        if not batch:
            return
        db.session.add_all(batch)
        db.session.commit()
        inserted += len(batch)
        batch = []

    # Optional name mapping
    name_map = {}
    if name_col:
        sub = raw[[sym_col, name_col]].drop_duplicates()
        name_map = {r[sym_col]: r[name_col] for _, r in sub.iterrows()}

    _mark(f"Inserting {len(to_insert)} new tickers")
    for sym in to_insert:
        batch.append(Ticker(
            ticker=sym,
            name=name_map.get(sym),
            active=True,
            created_at=dt.datetime.utcnow(),
        ))
        if len(batch) >= chunk_size:
            flush()
    flush()

    return {
        "kaggle_symbols_total": len(symbols_all),
        "considered": len(symbols),
        "already_present": len(existing),
        "inserted": inserted,
        "verify_in_stocks": verify_in_stocks,
    }

# ======= Startup: integrate Kaggle + your existing price population =======

@with_app_context
def kickoff_background_population(
    *,
    verify_in_stocks: bool = True,   # only add tickers that already have price rows
    enrich_names: bool = True,       # kept for API compatibility; Kaggle already includes names
    use_batched_prices: bool = True, # call your faster price loader if you have it
    batch_size: int = 64,            # batch size for your batched loader
):
    """
    Non-blocking startup population:
    1) Fetch Kaggle dataset (symbols_valid_meta.csv)
    2) Populate Ticker table (optionally verifying presence in Stocks)
    3) Populate prices (using your existing functions)
    """
    def _run():
        _population_status.update({
            "started": True,
            "finished": False,
            "error": None,
            "steps": [],
            "started_at": _now_utc_iso(),
            "finished_at": None,
        })
        try:
            # 1) Kaggle download
            _mark("Kaggle: download latest stock-market-dataset")
            csv_path = fetch_latest_kaggle_dataset(output_dir="./data")

            # 2) Populate tickers from Kaggle
            _mark(f"Kaggle: populate tickers (verify_in_stocks={verify_in_stocks})")
            summary_tickers = populate_tickers_from_kaggle(csv_path, verify_in_stocks=verify_in_stocks)
            _mark(f"tickers summary: {summary_tickers}")

            # 3) Populate/refresh prices using your existing code paths.
            #    If you have populate_stocks_yf_batched(...) use it; otherwise fallback to populate_recent_stocks().
            try:
                from crud import populate_stocks_yf_batched  # only if you implemented it
                has_batched = True
            except Exception:
                has_batched = False

            if use_batched_prices and has_batched:
                _mark(f"populate stocks (batched yf.download, batch_size={batch_size})")
                prices_summary = populate_stocks_yf_batched(batch_size=batch_size)
            else:
                _mark("populate stocks (incremental yfinance)")
                prices_summary = populate_recent_stocks()

            _mark(f"prices summary: {prices_summary}")

            _population_status["finished"] = True
            _population_status["finished_at"] = _now_utc_iso()
            _mark("done")
        except Exception as e:
            _population_status["error"] = repr(e)
            _population_status["finished"] = True
            _population_status["finished_at"] = _now_utc_iso()

    t = threading.Thread(target=_run, daemon=True, name="startup-populate")
    t.start()
    return True

def _coerce_limit(raw, default=12, lo=1, hi=50):
    try:
        n = int(raw)
    except Exception:
        n = default
    return max(lo, min(hi, n))

def _norm_q(q: str) -> str:
    return (q or "").strip()

def _like_pattern_contains(q: str) -> str:
    return f"%{q}%"

def _like_pattern_prefix(q: str) -> str:
    return f"{q}%"

def _base_ticker_query():
    return db.session.query(
        Ticker.ticker.label("symbol"),
        Ticker.name.label("name"),
    )

def suggest_tickers(
    q: str,
    *,
    limit: int = 12,
    require_in_stocks: bool = True,
    include_inactive: bool = False,
):
    """
    Return a list[ {symbol, name} ] for search suggestions.

    - q: user query (ticker or company fragment), case-insensitive
    - limit: 1..50 (default 12)
    - require_in_stocks: only tickers that appear at least once in `stocks`
    - include_inactive: include rows where Ticker.active == False
    """
    q = _norm_q(q)
    if not q:
        return []

    limit = _coerce_limit(limit)

    # case-insensitive patterns
    pat_prefix = _like_pattern_prefix(q)
    pat_contains = _like_pattern_contains(q)

    # Base WHERE: match by ticker prefix first OR name contains
    filters = [
        or_(
            Ticker.ticker.ilike(pat_prefix),         # prefix match on ticker
            (Ticker.name.ilike(pat_contains) if Ticker.name is not None else False),
        )
    ]

    if not include_inactive:
        filters.append(or_(Ticker.active.is_(True), Ticker.active.is_(None)))

    query = _base_ticker_query().filter(*filters)

    if require_in_stocks:
        # Only tickers that have at least one price row recorded
        subq_exists = db.session.query(Stock.ticker).filter(Stock.ticker == Ticker.ticker).exists()
        query = query.filter(subq_exists)

    # Order: prioritize prefix ticker matches, then alphabetically
    priority = case(
        (Ticker.ticker.ilike(pat_prefix), 0),
        else_=1,
    )
    query = query.order_by(priority.asc(), Ticker.ticker.asc()).limit(limit)

    rows = query.all()
    return [{"symbol": r.symbol, "name": r.name} for r in rows]