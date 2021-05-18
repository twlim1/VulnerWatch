from flask import Flask, render_template, request

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

    return inputText
