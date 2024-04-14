# import required libraries
import os
import json
import pickle
import pandas as pd
from server.Helper_Functions import read_file, train_test_split
from server.Arima import Arima
from server.Simple_Exponential_Smoothing import SimpleExponentialSmoothing
from server.Holts_Winter import HoltWinter
from server.Moving_Average import MovingAverage

# constants

LOGIN_TEMPLATE = 'login.html'
SQLITE_DB_FILE_PATH = 'Forecast-master/data/'
SQLITE_DB_FILE_NAME = 'Sqlite3.db'
DATE_FORMAT = "%d-%m-%Y"
PICKLE_FILE_PATH = 'Forecast-master/Artifacts/'

def helper_predict_arima(specialty, procedure, frequency):
    """
    Route for predicting with ARIMA model.
    """

    df = read_file(SQLITE_DB_FILE_PATH,SQLITE_DB_FILE_NAME)
    filtered_df = df[(df['speciality'] == specialty) & (df['procedure'] == procedure)]
    filtered_df = filtered_df[['date', 'no_of_procedures']]
    # Perform train-test split
    train, test = train_test_split(filtered_df, 0.2)
    # Create an instance of the Arima class
    arima = Arima(train, date_column='date', dep_var='no_of_procedures', freq=frequency)
    # Preprocess the data
    train = arima.pre_process()
    test_arima = Arima(test, date_column='date', dep_var='no_of_procedures', freq=frequency)
    test = test_arima.pre_process()
    filtered_df_arima = Arima(filtered_df, date_column='date', dep_var='no_of_procedures', freq=frequency)
    filtered_df = filtered_df_arima.pre_process()
    if not os.path.exists(PICKLE_FILE_PATH + 'arima' + '_' + frequency + '.pkl'):
        # Fit the ARIMA model
        model = arima.fit(train)
        # Predict the test values
        predicted = arima.predict(model, test)
        # evaluate the model
        metrics = arima.evaluate(test['no_of_procedures'], predicted)
        # write metrics dictionary as a json to Artifact folder
        with open(PICKLE_FILE_PATH + 'arima' + '_' + frequency + '.json', 'w') as f:
            json.dump(metrics, f)
        # Build arima model on entire dataset
        full_model = arima.fit(filtered_df)

        # write the model as pickle object to Artifact folder if pickle object is not available
        
        with open(PICKLE_FILE_PATH + 'arima' + '_' + frequency + '.pkl', 'wb') as f:
            pickle.dump(full_model, f)
    else:
        with open(PICKLE_FILE_PATH + 'arima' + '_' + frequency + '.pkl', 'rb') as f:
            full_model = pickle.load(f)
        with open(PICKLE_FILE_PATH + 'arima' + '_' + frequency + '.json', 'r') as f:
            metrics = json.load(f)
    
    #finding confidence interval
    confidence_intervals = arima.get_confidence_intervals(model=full_model, n_steps=len(filtered_df),alpha=0.05)
    actual_lower_ci = confidence_intervals['lower'].tolist()
    actual_upper_ci = confidence_intervals['upper'].tolist()
    forecast_steps = {'M': 1, 'W': 4, '2W': 2}
    # Predict the future values
    forecast = arima.forecast(full_model, forecast_steps[frequency])

    forecast_confidence_intervals = arima.get_confidence_intervals(model=full_model, n_steps=forecast_steps[frequency], alpha=0.05)
    forecast_lower_ci = forecast_confidence_intervals['lower'].tolist()
    forecast_upper_ci = forecast_confidence_intervals['upper'].tolist()

    lower_ci=actual_lower_ci +forecast_lower_ci
    upper_ci=actual_upper_ci+forecast_upper_ci
    # Prepare the response
    forecast_dates = pd.date_range(start=test['date'].max(), periods=forecast_steps[frequency], freq=frequency)
    forecast_dates = forecast_dates.strftime(DATE_FORMAT).tolist()
    dates = filtered_df['date'].dt.strftime(DATE_FORMAT).tolist() + forecast_dates
    actual = filtered_df['no_of_procedures'].tolist()
    # get fitted values from full model as list
    forecast_values = full_model.fittedvalues().tolist() + forecast.tolist()

    # convert forecast_values to int
    forecast_values = [int(i) for i in forecast_values]
    forecasted = [int(i) for i in forecast.tolist()]
    lower_ci = [int(i) for i in lower_ci]
    upper_ci = [int(i) for i in upper_ci]

    response = {
        'dates': dates,
        'actual': actual,
        'name': 'Arima',
        'forecast_values': forecast_values,
        'forecast_size': forecast_steps[frequency],
        'forecasted_dates':forecast_dates,
        'forecasted':forecasted,
        'lower_ci':lower_ci,
        'upper_ci':upper_ci

    }
    return response


def helper_predict_holt_winter(specialty, procedure, frequency):
    
    df = read_file(SQLITE_DB_FILE_PATH,SQLITE_DB_FILE_NAME)
    filtered_df = df[(df['speciality'] == specialty) & (df['procedure'] == procedure)]
    filtered_df = filtered_df[['date', 'no_of_procedures']]
    train, test = train_test_split(filtered_df, 0.2)

    # Create an instance of the HoltWinter class
    hw = HoltWinter(train, date_column='date', dep_var='no_of_procedures', freq=frequency)
    train = hw.pre_process()
    test_hw = HoltWinter(test, date_column='date', dep_var='no_of_procedures', freq=frequency)
    test = test_hw.pre_process()
    filtered_df_hm = HoltWinter(filtered_df, date_column='date', dep_var='no_of_procedures', freq=frequency)
    filtered_df = filtered_df_hm.pre_process()

    if not os.path.exists(PICKLE_FILE_PATH + 'holt_winter' + '_' + frequency + '.pkl'):
        # Perform grid search to find the optimal values of alpha, beta, and gamma
        alpha, beta, gamma = hw.grid_search(train)

        # Fit the model using the optimal values
        model = hw.fit(train, alpha, beta, gamma)

        # predict the future values
        forecast = hw.predict(model, test)

        # evaluate the model
        metrics = hw.evaluate(test['no_of_procedures'], forecast)

        # write metrics dictionary as a json to Artifact folder
        with open(PICKLE_FILE_PATH + 'holt_winter' + '_' + frequency + '.json', 'w') as f:
            json.dump(metrics, f)

        # Build holt winter model on entire dataset
        full_model = hw.fit(filtered_df, alpha, beta, gamma)

        with open(PICKLE_FILE_PATH + 'holt_winter' + '_' + frequency + '.pkl', 'wb') as f:
            pickle.dump(full_model, f)
    else:
        with open(PICKLE_FILE_PATH + 'holt_winter' + '_' + frequency + '.pkl', 'rb') as f:
            full_model = pickle.load(f)

        with open(PICKLE_FILE_PATH + 'holt_winter' + '_' + frequency + '.json', 'r') as f:
            metrics = json.load(f)

    confidence_intervals = hw.get_confidence_intervals(model=full_model, n_steps=len(filtered_df),alpha=0.05)
    actual_lower_ci = confidence_intervals['lower'].tolist()
    actual_upper_ci = confidence_intervals['upper'].tolist()
    forecast_steps = {'M': 1, 'W': 4, '2W': 2}
    # forecast for n_steps ahead
    forecast = hw.forecast(full_model, forecast_steps[frequency])
    forecast_confidence_intervals = hw.get_confidence_intervals(model=full_model, n_steps=forecast_steps[frequency], alpha=0.05)
    forecast_lower_ci = forecast_confidence_intervals['lower'].tolist()
    forecast_upper_ci = forecast_confidence_intervals['upper'].tolist()

    lower_ci=actual_lower_ci +forecast_lower_ci
    upper_ci=actual_upper_ci+forecast_upper_ci

    forecast_dates = pd.date_range(start=test['date'].max(), periods=forecast_steps[frequency], freq=frequency)
    forecast_dates = forecast_dates.strftime(DATE_FORMAT).tolist()
   
    # Prepare the response
    dates = filtered_df['date'].dt.strftime(DATE_FORMAT).tolist() + forecast_dates
    actual = filtered_df['no_of_procedures'].tolist()
    # get fitted values from full model as list
    forecast_values = full_model.fittedvalues.tolist() + forecast.tolist()
    
    # convert forecast_values to int
    forecast_values = [int(i) for i in forecast_values]
    forecasted = [int(i) for i in forecast.tolist()]
    lower_ci = [int(i) for i in lower_ci]
    upper_ci = [int(i) for i in upper_ci]
    
    response = {
        'dates': dates,
        'actual': actual,
        'name': 'Holts Winter',
        'forecast_values': forecast_values,
        'forecast_size': forecast_steps[frequency],
        'forecasted_dates':forecast_dates,
        'forecasted':forecasted,
        'lower_ci':lower_ci,
        'upper_ci':upper_ci

    }
    return response

def helper_predict_simple_exponential_smoothing(specialty, procedure, frequency):
    
    df = read_file(SQLITE_DB_FILE_PATH,SQLITE_DB_FILE_NAME)
    filtered_df = df[(df['speciality'] == specialty) & (df['procedure'] == procedure)]
    filtered_df = filtered_df[['date', 'no_of_procedures']]
    train, test = train_test_split(filtered_df, 0.2)
    # Create an instance of the SimpleExponentialSmoothing class
    ses = SimpleExponentialSmoothing(train, date_column='date', dep_var='no_of_procedures', freq=frequency)
    # preprocess the data
    train = ses.pre_process()
    test_ses = SimpleExponentialSmoothing(test, date_column='date', dep_var='no_of_procedures', freq=frequency)
    test = test_ses.pre_process()
    filtered_df_ses = SimpleExponentialSmoothing(filtered_df, date_column='date', dep_var='no_of_procedures', freq=frequency)
    filtered_df = filtered_df_ses.pre_process()
    if not os.path.exists(PICKLE_FILE_PATH + 'ses' + '_' + frequency + '.pkl'):
        model = ses.fit(train)
        print(model.summary())
        forecast = ses.predict(model, test)
        # evaluate the model
        metrics = ses.evaluate(test['no_of_procedures'], forecast)

        # write metrics dictionary as a json to Artifact folder
        with open(PICKLE_FILE_PATH + 'ses' + '_' + frequency + '.json', 'w') as f:
            json.dump(metrics, f)
        
        # Build ses model on entire dataset
        full_model = ses.fit(filtered_df)

    # write the model as pickle object to Artifact folder if pickle object is not available
        with open(PICKLE_FILE_PATH + 'ses' + '_' + frequency + '.pkl', 'wb') as f:
            pickle.dump(full_model, f)
    else:
        with open(PICKLE_FILE_PATH + 'ses' + '_' + frequency + '.pkl', 'rb') as f:
            full_model = pickle.load(f)

        with open(PICKLE_FILE_PATH + 'ses' + '_' + frequency + '.json', 'r') as f:
            metrics = json.load(f)

    confidence_intervals = ses.get_confidence_intervals(model=full_model, n_steps=len(filtered_df),alpha=0.05)
    actual_lower_ci = confidence_intervals['lower'].tolist()
    actual_upper_ci = confidence_intervals['upper'].tolist()
    forecast_steps = {'M': 1, 'W': 4, '2W': 2}
    # forecast for n_steps ahead
    forecast = ses.forecast(full_model, forecast_steps[frequency])
    forecast_confidence_intervals = ses.get_confidence_intervals(model=full_model, n_steps=forecast_steps[frequency], alpha=0.05)
    forecast_lower_ci = forecast_confidence_intervals['lower'].tolist()
    forecast_upper_ci = forecast_confidence_intervals['upper'].tolist()

    lower_ci=actual_lower_ci +forecast_lower_ci
    upper_ci=actual_upper_ci+forecast_upper_ci

    forecast_dates = pd.date_range(start=test['date'].max(), periods=forecast_steps[frequency], freq=frequency)
    forecast_dates = forecast_dates.strftime(DATE_FORMAT).tolist()
   
    # Prepare the response
    dates = filtered_df['date'].dt.strftime(DATE_FORMAT).tolist() + forecast_dates
    actual = filtered_df['no_of_procedures'].tolist()
    # get fitted values from full model as list
    forecast_values = full_model.fittedvalues.tolist() + forecast.tolist()
    
    # convert forecast_values to int
    forecast_values = [int(i) for i in forecast_values]
    forecasted = [int(i) for i in forecast.tolist()]
    lower_ci = [int(i) for i in lower_ci]
    upper_ci = [int(i) for i in upper_ci]
    
    response = {
        'dates': dates,
        'actual': actual,
        'name': 'Simple Exponential Smoothing',
        'forecast_values': forecast_values,
        'forecast_size': forecast_steps[frequency],
        'forecasted_dates':forecast_dates,
        'forecasted':forecasted,
        'lower_ci':lower_ci,
        'upper_ci':upper_ci

    }
    return response

def helper_predict_moving_average(specialty, procedure, frequency):
    df = read_file(SQLITE_DB_FILE_PATH,SQLITE_DB_FILE_NAME)
    filtered_df = df[(df['speciality'] == specialty) & (df['procedure'] == procedure)]
    filtered_df = filtered_df[['date', 'no_of_procedures']]
    # Perform train-test split
    train, test = train_test_split(filtered_df, 0.2)
    # Create an instance of the Arima class
    ma = MovingAverage(train, date_column='date', dep_var='no_of_procedures',window=None, freq=frequency)
    # Preprocess the data
    train = ma.pre_process()
    test_ma = MovingAverage(test, date_column='date', dep_var='no_of_procedures', window=None,freq=frequency)
    test = test_ma.pre_process()
    filtered_df_ma = MovingAverage(filtered_df, date_column='date', dep_var='no_of_procedures',window=None, freq=frequency)
    filtered_df = filtered_df_ma.pre_process()

    model = ma.fit(filtered_df, test)
    # Creating train and test set
    train, test = train_test_split(model, 0.2)

    forecast = ma.predict(test)

    metrics = ma.evaluate(test['no_of_procedures'], forecast)

    model = model.drop(columns=['moving_avg_forecast'])
    forecast_steps = {'M': 1, 'W': 4, '2W': 2}
    forecast = ma.forecast(model, forecast_steps[frequency])

    forecast_dates = pd.date_range(start=test['date'].max(), periods=forecast_steps[frequency], freq=frequency)
    forecast_dates = forecast_dates.strftime(DATE_FORMAT).tolist()
    train_dates = train['date'].dt.strftime(DATE_FORMAT).tolist()
    test_dates = test['date'].dt.strftime(DATE_FORMAT).tolist()
    # Prepare the response
    dates = train_dates + test_dates
    actual = filtered_df['no_of_procedures'].tolist()
    forecast_values = train['no_of_procedures'].tolist() + test['no_of_procedures'].tolist()+forecast.tolist()

    response = {
        'dates': dates,
        'actual': actual,
        'name': 'Moving Average',
        'forecast_values': forecast_values,
        'forecast_size': forecast_steps[frequency],
        'forecasted_dates':forecast_dates,
        'forecasted':forecast.tolist(),
        #'lower_ci':lower_ci,
        #'upper_ci':upper_ci

    }

    return response

if __name__ == "__main__":

    from Helper_Functions import read_file, train_test_split
    from Arima import Arima
    specialty  = 'Orthopedic'
    procedure = 'orknarth'
    frequency = 'W'
    response = helper_predict_arima(specialty, procedure, frequency)
    # convert dict response to a json
    response = json.dumps(response)

