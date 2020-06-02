from flask import Flask, send_from_directory
import os

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World"


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'images/icon.ico')

            