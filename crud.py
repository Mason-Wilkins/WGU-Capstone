import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import connect_to_db
from kmeans import KMeans
import werkzeug


def do_stuff():
    dat = yf.Ticker("MSFT")
    dat.info
    # print(dat.history())
    # print(dat.info.keys())
    # print(yf.download(tickers))

    # Load NASDAQ symbols
    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    nasdaq = pd.read_csv(url, sep="|")

    tickers = nasdaq["Symbol"].tolist()
    data = {}
    i = 0
    for ticker in tickers:
        # print(ticker)
        stock = yf.download(ticker, period="360d", interval="1d", auto_adjust=True)
        data[ticker] = stock
        if i == 10:
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

if __name__ == "__main__":
    from server import app
    connect_to_db(app)
    