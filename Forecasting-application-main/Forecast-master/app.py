from flask import Flask, request, jsonify, render_template, redirect,session
from server.Helper_Functions import train_test_split, read_file
import pandas as pd
from server.Arima import Arima
from server.Simple_Exponential_Smoothing import SimpleExponentialSmoothing
from server.Holts_Winter import HoltWinter
from server.Moving_Average import MovingAverage
from server.route_helpers import helper_predict_arima, helper_predict_simple_exponential_smoothing, helper_predict_holt_winter,helper_predict_moving_average
import sqlite3
import pickle
import os
import json

app = Flask(__name__, template_folder='client/template', static_folder='client/static')
app.secret_key = 'BAC'  

LOGIN_TEMPLATE = 'login.html'
SQLITE_DB_FILE_PATH = 'Forecast-master/data/'
SQLITE_DB_FILE_NAME = 'Sqlite3.db'
DATE_FORMAT = "%d-%m-%Y"
PICKLE_FILE_PATH = 'Forecast-master/Artifacts/'

@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form:
        name = request.form['name']
        password = request.form['password']

        if not name or not password:
            error = 'Please enter both Name and Password'
            return render_template(LOGIN_TEMPLATE, error=error)

        print(SQLITE_DB_FILE_PATH + SQLITE_DB_FILE_NAME)
        connection = sqlite3.connect(SQLITE_DB_FILE_PATH + SQLITE_DB_FILE_NAME)
        cursor = connection.cursor()

        query = "SELECT name, password from users WHERE name = ? AND password = ? "
        cursor.execute(query, (name, password))
        # cursor.execute('SELECT  name, password FROM users WHERE name = %s AND password = %s', (name, password, ))
        results = cursor.fetchone()

        if results is None:
            error = 'Sorry, Incorrect Credentials'
            return render_template(LOGIN_TEMPLATE, error=error)
        else:
            session['loggedin'] = True
            session['username'] = name
            return redirect('/main')

    return render_template(LOGIN_TEMPLATE)

@app.route('/main', methods=["GET", "POST"])
def main_page():
    if not session.get('loggedin'):
        return redirect('/login')
    
    return render_template('main_page.html')

@app.route('/logout')
def logout():
   # remove the username from the session if it is there
   session.pop('loggedin', None)
   return redirect("/login")

@app.route('/main/predict_arima', methods=['GET', 'POST'])
def predict_arima():
    """
    Route for predicting with ARIMA model.
    """

    data = request.json
    specialty = data['specialty']
    procedure = data['procedure']
    frequency = data['frequency']
    response = helper_predict_arima(specialty, procedure, frequency)
    return jsonify(response)


@app.route('/main/predict_holt_winter', methods=['GET', 'POST'])
def predict_holt_winter():
    """
    Route for predicting with Holt-Winters model.
    """
    data = request.json
    specialty = data['specialty']
    procedure = data['procedure']
    frequency = data['frequency']
    response = helper_predict_holt_winter(specialty, procedure, frequency)
    return jsonify(response)


@app.route('/main/predict_simple_exponential_smoothing', methods=['GET', 'POST'])
def predict_simple_exponential_smoothing():
    """
    Route for predicting with Simple Exponential Smoothing model.
    """
    data = request.json
    specialty = data['specialty']
    procedure = data['procedure']
    frequency = data['frequency']
    response = helper_predict_simple_exponential_smoothing(specialty, procedure, frequency)
    return jsonify(response)


@app.route('/main/predict_moving_average', methods=['GET', 'POST'])
def predict_moving_average():
    """
    Route for predicting with Moving Average model.
    """
    data = request.json
    specialty = data['specialty']
    procedure = data['procedure']
    frequency = data['frequency']
    response = helper_predict_moving_average(specialty, procedure, frequency)    

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
