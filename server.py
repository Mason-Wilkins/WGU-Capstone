import os
from flask import Flask, jsonify
from model import connect_to_db, db, Stock
from flask_cors import CORS, cross_origin

FLASK_APP_SECRET_KEY = os.environ['FLASK_APP_SECRET_KEY']

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
app.secret_key = FLASK_APP_SECRET_KEY

connect_to_db(app, "stocks")

# ✅ Create tables once, WITH an app context
with app.app_context():
    db.create_all()

@app.route("/api/data", methods=['GET'])
def get_data():
    return crud.test_stock_data(db.session)

# def create_app():
#     app = Flask(__name__)
#     CORS(app, origins=["http://localhost:3000"])
#     app.secret_key = os.environ["FLASK_APP_SECRET_KEY"]

#     connect_to_db(app, "stocks")

#     # create tables once on startup
#     with app.app_context():
#         db.create_all()

#     # register routes
#     # from routes import api_bp
#     # app.register_blueprint(api_bp)

#     return app

# # exported app (so other modules/scripts can import it)
# app = create_app()

if __name__ == "__main__":
    import crud
    app.run(debug=True, port=5000)