"""Database models and DB helper for the application.

This module defines the SQLAlchemy models used by the app and exposes
`db` (a Flask-SQLAlchemy instance) and `connect_to_db` helper to wire
the models to a Flask application.

Models:
 - Stock: normalized storage of historical daily bars ingested from yfinance
 - StockPrice: legacy or alternative table for per-day prices (kept for compatibility)
 - PredictionLog: stores model predictions and decision metadata
 - Ticker: metadata about tickers
"""

from sqlalchemy.orm import DeclarativeBase
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

class Base(DeclarativeBase):
    """Base declarative class used by SQLAlchemy (when needed).

    Using a DeclarativeBase keeps compatibility with SQLAlchemy's
    declarative features while we use Flask-SQLAlchemy's `db` object
    for model definitions.
    """

    pass

# `db` is the Flask-SQLAlchemy facade used throughout the app. It
# provides `Model`, `Column`, session and helper methods that are
# convenient inside a Flask application context.
db = SQLAlchemy(model_class=Base)

class Stock(db.Model):
    """Normalized table for historical stock bars.

    Each row represents a single daily bar for a ticker. This table is
    the primary place the application writes and reads price history.
    """
    
    __tablename__ = "stocks"
    
    stock_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    record_id = db.Column(db.String, unique=True, nullable=False)
    ticker = db.Column(db.String, nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    open_price = db.Column(db.Float)
    high_price = db.Column(db.Float)
    low_price = db.Column(db.Float)
    close_price = db.Column(db.Float)
    adj_close_price = db.Column(db.Float)
    volume = db.Column(db.BigInteger)
    created_at = db.Column(db.DateTime, default=datetime.now(tz=None))
    
    def __repr__(self):
        """Show info for Stock class."""
        
        return f"""
        <Stock ID = {self.stock_id}
        Record ID = {self.record_id}
        Ticker = {self.ticker}
        Date = {self.date}
        Open Price = {self.open_price}
        High Price = {self.high_price}
        Low Price = {self.low_price}
        Close Price = {self.close_price}
        Adj Close Price = {self.adj_close_price}
        Volume = {self.volume}
        Created At = {self.created_at}
        >
        """

class StockPrice(db.Model):
    __tablename__ = "stock_prices"
    # Alternative / legacy schema for per-ticker daily prices. The
    # application uses `Stock` as the canonical table, but this model
    # may exist for compatibility with other tools or earlier versions.
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(16), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, index=True)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    adj_close = db.Column(db.Float, nullable=True)
    volume = db.Column(db.BigInteger)
    # use a python-side default timestamp (UTC)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # ensure we don't insert duplicate (ticker, date) rows
    __table_args__ = (db.UniqueConstraint("ticker", "date", name="uq_ticker_date"),)

class PredictionLog(db.Model):
    """Stores prediction results and metadata.

    Each row records a single prediction run for a ticker: the numeric
    score, the string decision (BUY/CONSIDER/HOLD), and an optional
    JSON blob that contains features, labels, and any debug info.
    """
    __tablename__ = "prediction_logs"
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    ticker = db.Column(db.String(16), nullable=False)
    score = db.Column(db.Float, nullable=False)
    decision = db.Column(db.String(16), nullable=False)  # e.g., BUY / CONSIDER / HOLD
    details = db.Column(db.JSON, nullable=True)          # store scoring, features, etc.

class Ticker(db.Model):
    """Metadata about known tickers (symbol, name, exchange, etc.).

    This table helps power the ticker suggestion endpoint and stores
    informational fields about each ticker. Fields are mostly nullable
    because different data vendors expose different subsets of info.
    """
    
    __tablename__ = "tickers"
    
    ticker_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    ticker = db.Column(db.String, unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(tz=None))
    market = db.Column(db.String)
    currency_name = db.Column(db.String)
    locale = db.Column(db.String)
    name = db.Column(db.String)
    primary_exchange = db.Column(db.String)
    type = db.Column(db.String)
    active = db.Column(db.Boolean)
    cik = db.Column(db.String)
    composite_figi = db.Column(db.String)
    share_class_figi = db.Column(db.String)
    last_updated_utc = db.Column(db.DateTime)
    
    def __repr__(self):
        """Show info for Ticker class."""
        
        return f"""
        <Ticker ID = {self.ticker_id}
        Ticker = {self.ticker}
        Created At = {self.created_at}
        >
        """

def connect_to_db(app, db_name):
    app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql:///{db_name}"
    app.config["SQLALCHEMY_ECHO"] = True
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Initialize the db instance with the app
    db.init_app(app)
    with app.app_context():
        db.create_all()
    print("Connected to the db!")


if __name__ == "__main__":
    from server import app
    connect_to_db(app, "stocks")