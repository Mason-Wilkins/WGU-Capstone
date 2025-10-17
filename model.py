from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import json
from datetime import datetime

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class Stock(db.Model):
    """Model for Stock class."""
    
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
    volume = db.Column(db.Integer)
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
    
class Ticker(db.Model):
    """Model for Ticker class."""
    
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
    # db.create_all()
    print("Connected to the db!")


if __name__ == "__main__":
    from server import app
    connect_to_db(app, "stocks")