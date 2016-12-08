import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

#-------- MODEL GOES HERE -----------#
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# load the model from disk
uber_PREDICTOR = pickle.load(open('uber.pkl', 'rb'))
lyft_PREDICTOR = pickle.load(open('lyft.pkl', 'rb'))

#-------- ROUTES GO HERE -----------#
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        Humidity = request.form['Humidity']
        Prec = request.form['Prec']
        Temp = request.form['Temp']
        hour = request.form['hour']
        month = request.form['month']
        day = request.form['day']
        return predict(pclass, sex, age, fare, sibsp)
    return render_template('index.html')

@app.route('/predict', methods=["GET"])
def predict(Humidity, Prec, Temp, hour, month, day):
    item = [Humidity, Prec, Temp, hour, month, day]
    uber = uber_PREDICTOR.predict(item)
    lyft = lyft_PREDICTOR.predict(item)
    results = {'Uber': uber, 'Lyft': lyft}
    return jsonify(results)

if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'
    app.run(HOST, PORT)