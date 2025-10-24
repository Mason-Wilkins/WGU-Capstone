import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans
from ctx import with_app_context
import datetime, os, subprocess, threading, math

def download_stocks_data(populate_db=False):
    if not os.path.exists("stocks.sql"):
        os.system('curl -L -o stocks.sql "https://drive.usercontent.google.com/download?export=download&id=1_d8Ivcjw5WsGcUyR8C9d6Uyi-OwPumN3&confirm=t"')
    elif populate_db:
        os.system('dropdb stocks')
        os.system('createdb stocks')
        subprocess.run(
            [
                "psql",
                "--set", "ON_ERROR_STOP=1",
                "--dbname", "postgresql:///stocks",
                "--file", "stocks.sql"
            ]
            ,check=True
        )
    else:
        print("stocks.sql already exists. Skipping download.")

def ensure_tickers():
    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    nasdaq = pd.read_csv(url, sep="|")
    tickers = nasdaq["Symbol"].tolist()
    tickers_csv = pd.read_csv("tickers.csv")
    if tickers_csv is None or tickers_csv.empty:
        with open("tickers.csv", "w") as f:
            for t in tickers:
                t = str(t).strip()
                if t == 'nan':
                    continue
                if len(t) > 5:
                    break
                f.write(t + "\n")

def chunk(lst, n):
    size = math.ceil(len(lst) / n)
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def fetch_and_store_batch(tickers_chunk):
    """
    Runs INSIDE a thread. Creates its own app context and DB session.
    """
    from model import app, db, Stock  # import inside to avoid circulars; adjust path

    # Each thread needs an app context for Flask-SQLAlchemy's scoped_session
    with app.app_context():
        session = db.create_scoped_session()  # thread-local session
        try:
            today = datetime.date.today().strftime("%Y-%m-%d")
            # Process each ticker in this chunk
            for ticker in tickers_chunk:
                t = yf.Ticker(ticker)

                # Find last date we have
                last = (
                    session.query(Stock)
                    .filter(Stock.ticker == ticker)
                    .order_by(Stock.date.desc())
                    .first()
                )
                start = last.date.strftime("%Y-%m-%d") if last else "2022-10-01"

                hist = t.history(start=start, end=today, interval="1d", auto_adjust=True)
                if hist is None or hist.empty:
                    continue

                # Build rows, commit in batches for speed
                to_insert = []
                for date, row in hist.iterrows():  # use itertuples() if you like
                    record_id = f"{ticker}-{date.strftime('%Y%m%d')}"
                    # optimistic insert; let UNIQUE(record_id) prevent dupes
                    s = Stock(
                        record_id=record_id,
                        ticker=ticker,
                        date=date.to_pydatetime(),
                        open_price=float(row["Open"]),
                        high_price=float(row["High"]),
                        low_price=float(row["Low"]),
                        close_price=float(row["Close"]),
                        volume=int(row["Volume"]),
                    )
                    to_insert.append(s)

                # Bulk-ish insert with conflict handling (Postgres example below)
                if to_insert:
                    session.add_all(to_insert)
                    # For Postgres, prefer "ON CONFLICT DO NOTHING" to avoid races:
                    # session.flush() with ORM upsert requires SQLAlchemy 2.0 + insert(Stock).on_conflict_do_nothing()
                    session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

def multithread_stock_population(max_workers=4):
    from model import app, db  # adjust path
    with app.app_context(): 
        ensure_tickers()
        db.create_all()
        tickers = pd.read_csv("tickers.csv").values.flatten().tolist()

    # Split work so threads don't step on each other
    chunks = list(chunk(tickers, max_workers))

    # Run threads; each will create its own app context/session
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_and_store_batch, ch) for ch in chunks]
        for f in as_completed(futures):
            f.result()  # will raise if thread failed

    print("✅ Multithreaded stock data population complete.")


@with_app_context
def multithread_stock_population():
    thread1 = threading.Thread(target=populate_recent_stocks)
    thread2 = threading.Thread(target=populate_recent_stocks)
    thread3 = threading.Thread(target=populate_recent_stocks)
    thread1.start()
    thread2.start()
    thread3.start()
    thread1.join()
    thread2.join()
    thread3.join()
    print("✅ Multithreaded stock data population complete.")

@with_app_context
def populate_all_stocks():
    ensure_tickers()
    db.create_all()
    tickers = pd.read_csv("tickers.csv").values.flatten().tolist()
    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        ticker_obj = yf.Ticker(ticker)
        ticker_in_db = (
            db.session.query(Stock)
            .filter(Stock.ticker == ticker)
            .order_by(Stock.date.desc())
            .first()
        )
        today = datetime.date.today().strftime("%Y-%m-%d")
        if ticker_in_db:
            historical_data = ticker_obj.history(start=ticker_in_db.date, end=today, interval="1d", auto_adjust=True)
        else:
            historical_data = ticker_obj.history(start="2022-10-01", end=today, interval="1d", auto_adjust=True)
        # historical_data = ticker_obj.history(period="1y", interval="1d", auto_adjust=True)
        # stock = yf.download(ticker, period="10d", interval="1d", auto_adjust=True)
        if historical_data is None or historical_data.empty:
            continue
        for date, row in historical_data.iterrows():
            # Convert from pandas datatypes to python datatypes for db storage
            record_id = f"{ticker}-{date.strftime('%Y%m%d')}"
            ticker_data_in_db = (
                db.session.query(Stock)
                .filter(Stock.record_id == record_id)
                .first()
            )
            if ticker_data_in_db:
                continue
            row = row.astype(object)
            s = Stock(
                record_id = f"{ticker}-{date.strftime('%Y%m%d')}",
                ticker = ticker,
                date = date.to_pydatetime(),
                open_price = row["Open"],
                high_price = row["High"],
                low_price = row["Low"],
                close_price = row["Close"],
                volume = int(row["Volume"])
            )
            db.session.add(s)
            db.session.commit()

    print("✅ Stock data population complete.")

@with_app_context
def populate_recent_stocks():
    tickers = pd.read_csv("tickers.csv").values.flatten().tolist()
    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        ticker_obj = yf.Ticker(ticker)
        ticker_in_db = (
            db.session.query(Stock)
            .filter(Stock.ticker == ticker)
            .order_by(Stock.date.desc())
            .first()
        )
        today = datetime.date.today().strftime("%Y-%m-%d")
        if ticker_in_db:
            historical_data = ticker_obj.history(start=ticker_in_db.date, end=today, interval="1d", auto_adjust=True)
        else:
            historical_data = ticker_obj.history(start="2022-10-01", end=today, interval="1d", auto_adjust=True)
        # historical_data = ticker_obj.history(period="1y", interval="1d", auto_adjust=True)
        # stock = yf.download(ticker, period="10d", interval="1d", auto_adjust=True)
        if historical_data is None or historical_data.empty:
            continue
        for date, row in historical_data.iterrows():
            record_id = f"{ticker}-{date.strftime('%Y%m%d')}"
            ticker_data_in_db = (
                db.session.query(Stock)
                .filter(Stock.record_id == record_id)
                .first()
            )
            if ticker_data_in_db:
                continue
            # Convert from pandas datatypes to python datatypes for db storage
            row = row.astype(object)
            db_record_id = db.session.query(Stock).filter(Stock.record_id == record_id).first()
            if db_record_id:
                continue
            s = Stock(
                record_id = f"{ticker}-{date.strftime('%Y%m%d')}",
                ticker = ticker,
                date = date.to_pydatetime(),
                open_price = row["Open"],
                high_price = row["High"],
                low_price = row["Low"],
                close_price = row["Close"],
                volume = int(row["Volume"])
            )
            db.session.add(s)
            db.session.commit()

    print("✅ Stock data population complete.")

def test_stock_data():
    dat = yf.Ticker("MSFT")
    dat.info
    ensure_tickers()
    tickers = pd.read_csv("tickers.csv")
    
    data = {}
    i = 0
    for ticker in tickers:
        # print(ticker)
        ticker_obj = yf.Ticker(ticker)
        historical_data = ticker_obj.history(period="10d", interval="1d", auto_adjust=True)
        stock = yf.download(ticker, period="10d", interval="1d", auto_adjust=True)
        if stock is None or stock.empty:
            continue
        data[ticker] = stock
        if i == 2:
            break
        i+=1

    return data

def fit_data(data):
    features = []
    names = []

    for t, df in data.items():
        df["Return"] = df["Close"].pct_change()

        avg_return = df["Return"].mean()
        volatility = df["Return"].std()

        # Skip if we can’t compute both numbers
        if np.isnan(avg_return) or np.isnan(volatility):
            print(f"Skipping {t} due to NaN values")
            continue

        features.append([avg_return, volatility])
        names.append(t)

    X = np.array(features)
    print("Cleaned feature matrix:\n", X)
    # Example: get data for first 5
    # data = yf.download(tickers[:5], period="1d")
    # print(data)
    model = KMeans(n_clusters=3, max_iter=100)
    model.fit(X)
    labels = model.predict(X)
    for stock, cluster in zip(names, labels):
        print(f"{stock}: Cluster {cluster}")

    # Compute score (return / volatility = Sharpe-like ratio)
    scores = [r/v if v > 0 else -np.inf for r, v in X]

    # Pick best stock
    best_index = np.argmax(scores)
    best_stock = names[best_index]

    # Show all scores
    print("\n=== Stock Scores ===")
    for name, score, label in zip(names, scores, labels):
        print(f"{name}: Score={score:.4f}, Cluster={label}")

    print(f"\n✅ Best stock to pick: {best_stock}")
    return best_stock
