from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import json
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """Model for User class."""
    
    __tablename__ = "users"
    # table name is users
    
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    fname = db.Column(db.String, nullable=False)
    lname = db.Column(db.String, nullable=False)
    email = db.Column(db.String, nullable=False)
    username = db.Column(db.String, nullable=False)
    password = db.Column(db.String, nullable=False)
    phone = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.now(tz=None))

    favorites = db.relationship('Favorite', back_populates='users')
    reviews = db.relationship('Review', back_populates='users')
    images = db.relationship('Images', back_populates='users')
    
    #many-to-one relationship  favorites table
    
    def __repr__(self):
        """Show info for User class."""
        
        return f"""
        <User ID = {self.user_id}
        First Name = {self.fname}
        Last Name = {self.lname}
        Email = {self.email}
        Username = {self.username}
        Password = {self.password}
        Phone = {self.phone}
        >
        """

     
class Review(db.Model):
    """Model foe Review class"""
    
    __tablename__ = "reviews"
    
    review_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    trail_id = db.Column(db.Integer, db.ForeignKey("trails.trail_id"), nullable=False)
    num_stars = db.Column(db.Integer)
    review_text = db.Column(db.String)
    created_at = db.Column(db.DateTime, default=datetime.utcnow())
    
    users = db.relationship("User", back_populates="reviews")
    trails = db.relationship("Trail", back_populates="reviews")
    
    
    def __repr__(self):
        """Show info aboue Review class"""
        
        return f"""review_id:{self.review_id},user_id: {self.user_id},trail_id:{self.trail_id},num_stars:{self.num_stars},review_text:{self.review_text}>"""
        

class Trail(db.Model):
    """Model for Property class."""
    
    __tablename__ = "trails"
    
    trail_id = db.Column(db.Integer, primary_key=True, nullable=False)
    name = db.Column(db.String, nullable=True)
    area_name = db.Column(db.String, nullable=True)
    city_name = db.Column(db.String, nullable=False)
    state_name = db.Column(db.String, nullable=False)
    country_name = db.Column(db.String, nullable=True)
    _geoloc = db.Column(db.String, nullable=False)
    popularity = db.Column(db.Float, nullable=True)
    length = db.Column(db.Float, nullable=False)
    elevation_gain = db.Column(db.Float, nullable=True)
    difficulty_rating = db.Column(db.Integer, nullable=False)
    route_type = db.Column(db.String, nullable=False)
    visitor_usage = db.Column(db.String, nullable=True)
    avg_rating = db.Column(db.Float, nullable=True)
    num_reviews = db.Column(db.Integer, nullable=True)
    features = db.Column(db.String, nullable=True)
    activities = db.Column(db.String, nullable=True)

    favorites = db.relationship("Favorite", back_populates="trails")
    reviews = db.relationship("Review", back_populates="trails")
    images = db.relationship("Images", back_populates="trails")

    
    def __repr__(self):
        """Show info about trail class."""
        
        return f"""
        Trail ID: {self.trail_id}
        Trail Name: {self.name}
        Area Name: {self.area_name}
        City Name: {self.city_name}
        State Name: {self.state_name}
        Country Name: {self.country_name}
        Geolocation: {self._geoloc}
        Popularity: {self.popularity}
        Trail Length: {self.length}
        Elevation gain: {self.elevation_gain}
        Difficulty Rating: {self.difficulty_rating}
        Route Type: {self.route_type}
        Visitor Usage: {self.visitor_usage}
        Average Rating: {self.avg_rating}
        Number of Reviews: {self.num_reviews}
        Features: {self.features}
        """
        
        
class Favorite(db.Model):
    
    __tablename__ = "favorites"
    
    favorite_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    trail_id = db.Column(db.Integer, db.ForeignKey('trails.trail_id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow())
    
    users = db.relationship("User", back_populates="favorites")
    trails = db.relationship("Trail", back_populates="favorites")
    
    
    
    def __repr__(self):
        """Returns info aboue Favorite class."""
        
        return f"""
            <Favorite ID: {self.favorite_id}
            User ID: {self.user_id}
            Trail ID: {self.trail_id}>
            """ 
            

class Images(db.Model):
    """Model for TrailImages class"""
    
    __tablename__ = "images"
    
    image_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    image_url = db.Column(db.String, nullable=False)
    trail_id = db.Column(db.Integer, db.ForeignKey("trails.trail_id"), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now(tz=None))
    
    users = db.relationship("User", back_populates="images")
    trails = db.relationship("Trail", back_populates="images")


def connect_to_db(flask_app, db_uri=None, echo=True):
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
    flask_app.config["SQLALCHEMY_ECHO"] = echo
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # db.app = flask_app
    db.init_app(flask_app)



if __name__ == "__main__":
    from server import app
    connect_to_db(app)