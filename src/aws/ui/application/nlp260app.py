from flask import Flask, render_template, request

import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sandbox.html')
def sandbox():
    return render_template('sandbox.html', title='Sandbox')

@app.route('/api', methods=['POST'])
def api():
    try:
        inputText = request.form['text']
    except KeyError:
        return 'Bad input'

    url = 'http://127.0.0.1:81/predict'
    res = requests.post(url, data={'text': inputText})
    try:
        res.raise_for_status()
    except:
        return 'Internal Error' # TODO: improve error handling/messaging

    return res.text
