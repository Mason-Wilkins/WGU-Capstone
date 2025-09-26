import os
from flask import Flask
from model import connect_to_db, db


FLASK_APP_SECRET_KEY = os.environ['FLASK_APP_SECRET_KEY']

app = Flask(__name__)
app.secret_key = FLASK_APP_SECRET_KEY

# @app.route("/")
# def index():
#     """Render Home page"""
#     return render_template('index.html')