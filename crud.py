import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import db, Stock, connect_to_db
from kmeans import KMeans
from ctx import with_app_context
import time

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

def normalize_stock(stock_df, tickers):
    """
    Ensure columns are MultiIndex with levels ['Ticker','Price'] like:
    Ticker  AAPL
    Price  Open  High  Low  Close  Adj Close  Volume
    """
    # If single ticker string, make a list for uniform handling
    if isinstance(tickers, str):
        tickers = [tickers]

    if isinstance(stock_df.columns, pd.MultiIndex):
        # yfinance usually returns level 0 = price fields, level 1 = tickers
        price_level_values = set(stock_df.columns.get_level_values(0))
        if {"Open","High","Low","Close","Adj Close","Volume"} & price_level_values:
            stock_df = stock_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
        # else: already in desired order; no-op
    else:
        # Single-ticker case: wrap columns into a MultiIndex
        stock_df.columns = pd.MultiIndex.from_product(
            [ [tickers[0]], list(stock_df.columns) ],
            names=["Ticker", "Price"]
        )

    # Index is already datetime; just set a friendly name
    stock_df.index.name = "Date"
    return stock_df

@with_app_context
def populate_stocks():
    ensure_tickers()
    tickers = pd.read_csv("tickers.csv").values.flatten().tolist()
    j = 0
    for i in range(len(tickers)-1):
        ticker = tickers[i][0]
        print(f"Fetching data for {ticker}")
        ticker_obj = yf.Ticker(ticker)
        ticker_in_db = (
            db.session.query(Stock)
            .filter(Stock.ticker == ticker)
            .order_by(Stock.date.desc())
            .first()
        )
        if ticker_in_db:
            historical_data = ticker_obj.history(start=ticker_in_db.date, interval="1d", auto_adjust=True)
        else:
            historical_data = ticker_obj.history(period="1y", interval="1d", auto_adjust=True)
        # historical_data = ticker_obj.history(period="1y", interval="1d", auto_adjust=True)
        # stock = yf.download(ticker, period="10d", interval="1d", auto_adjust=True)
        if historical_data is None or historical_data.empty:
            continue
        historical_data.infer_objects()
        print(historical_data.info())
        for date, row in historical_data.iterrows():
            print("Sleeping for one second")
            time.sleep(1)
            # Convert from pandas datatypes to python datatypes for db storage
            row = row.astype(object)
            record_id = f"{ticker}-{date.strftime('%Y%m%d')}"
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
        # print(normalize_stock(stock, ticker))
        print(historical_data.index)
        print(historical_data.columns)
        print(historical_data.shape)
        print(historical_data.info())
        if stock is None or stock.empty:
            continue
        data[ticker] = stock
        if i == 2:
            break
        i+=1

    # return data

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
