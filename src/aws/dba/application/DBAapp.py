from flask import Flask

app = Flask(__name__)

@app.route('/predict_base_score')
def index():
    return '10'
