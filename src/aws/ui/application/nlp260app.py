import json

from flask import Flask, render_template, request

import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sandbox')
def sandbox():
    return render_template('sandbox.html', title='Sandbox')

@app.route('/explore')
def explore():
    return render_template('explore.html', title='Explore')

#
# Rest API for CVSS prediction on text
#
@app.route('/getprediction', methods=['POST'])
def getPrediction():
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

#
# Rest API for CVE data
#
@app.route('/getcves', methods=['POST'])
def getCVES():
    try:
        inputText = request.form['text']
    except KeyError:
        return 'Bad input'

    url = 'http://127.0.0.1:81/cves'
    res = requests.post(url, data={'text': inputText})
    try:
        res.raise_for_status()
    except:
        return 'Internal Error' # TODO: improve error handling/messaging

    return res.text
