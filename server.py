import os
from flask import Flask, jsonify
from model import connect_to_db, db
from flask_cors import CORS


FLASK_APP_SECRET_KEY = os.environ['FLASK_APP_SECRET_KEY']

app = Flask(__name__)
CORS(app)
app.secret_key = FLASK_APP_SECRET_KEY

@app.route("/api/data", methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from Flask!"})

if __name__ == "__main__":
    # connect_to_db(app)
    app.run(debug=True)