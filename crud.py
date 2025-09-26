import yfinance as yf
import pandas as pd

dat = yf.Ticker("MSFT")
dat.info
# print(dat.history())
# print(dat.info.keys())
# print(yf.download(tickers))

# Load NASDAQ symbols
url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
nasdaq = pd.read_csv(url, sep="|")

tickers = nasdaq["Symbol"].tolist()
print(tickers)

# Example: get data for first 5
# data = yf.download(tickers[:5], period="1d")
# print(data)

if __name__ == "__main__":
    main()
    