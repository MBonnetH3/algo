from flask import Flask
from flask import jsonify
from markupsafe import escape
import joblib
app = Flask(__name__)

algo = joblib.load("model.pkl")

@app.route("/")
def index():
 
    return 'Hello my app'
