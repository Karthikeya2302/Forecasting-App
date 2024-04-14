import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from itertools import product
from server.Forecast import Forecast
from server.Pre_Process import Preprocess
from server.Helper_Functions import get_frequency, set_frequency, find_metrics, train_test_split, read_file
"""
Holts Winter model for forecasting

"""

# create a class for Holt's Winter method inheriting from base class Forecast and Preprocess
class HoltWinter(Forecast, Preprocess):
    """
    A class for performing Holt-Winters forecasting on time series data.

    Parameters:
    df (DataFrame): The input DataFrame containing the time series data.
    date_column (str): The name of the column representing the dates.
    dep_var (str): The name of the dependent variable column.
    freq (str): The desired frequency for the time series data. Default is 'D' (daily).

    Methods:
    pre_process(): Preprocesses the data by filling missing dates, changing frequency if necessary.
                   Returns the preprocessed DataFrame.
    fit(df, alpha, beta, gamma): Fits the Holt-Winters model to the provided DataFrame using the given alpha, beta, and gamma values.
                                Returns the fitted model.
    grid_search(train): Performs a grid search to find the optimal values of alpha, beta, and gamma based on the train dataset.
                        Returns a tuple of the best parameters.
    predict(model, df): Predicts the values for the given DataFrame using the provided model.
                        Returns the predicted values.
    evaluate(actual, predicted): Evaluates the model using metrics based on the actual and predicted values.
                                Returns a dictionary of evaluation metrics.
    forecast(model, n_steps): Generates a forecast for n_steps ahead using the provided model.
                              Returns the forecasted values.
    """

    # initialize the class with the dataframe and the date column and the dependent variable
    def __init__(self, df, date_column, dep_var, freq='D'):
        """
        Initializes the HoltWinter class.

        Parameters:
        df : The input DataFrame containing the time series data.
        date_column : The name of the column representing the dates.
        dep_var : The name of the dependent variable column.
        freq : The desired frequency for the time series data. Default is 'D' (daily).
        """
        super().__init__(df, date_column, dep_var)
        self.freq = freq

    def pre_process(self):
        """
        Preprocesses the time series data by filling missing dates, changing frequency if necessary.

        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
        self.df = self.preprocessing(threshold=3, fill_method='linear')
        # change the frequency of the data based on user input
        if self.freq != get_frequency(self.df, self.date_column):
            self.df = set_frequency(self.df, self.date_column, get_frequency(self.df, self.date_column), self.freq)
        return self.df

    def fit(self, df, alpha, beta, gamma):
        """
        Fits the Holt-Winters model to the provided DataFrame using the given alpha, beta, and gamma values.

        Parameters:
        df : The DataFrame to fit the model on.
        alpha (float): The level smoothing parameter.
        beta (float): The trend smoothing parameter.
        gamma (float): The seasonal smoothing parameter.

        Returns:
        statsmodels.tsa.holtwinters.ExponentialSmoothing: The fitted Holt-Winters model.
        """
        # fit the model using Holt's Winter method
        model = ExponentialSmoothing(df[self.dep_var], trend='add', seasonal='add', seasonal_periods=4).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
        return model

    def grid_search(self, train):
        """
        Performs a grid search to find the optimal values of alpha, beta, and gamma based on the train dataset.

        Parameters:
        train : The DataFrame used for training the model.

        Returns:
        tuple: A tuple containing the best alpha, beta, and gamma values.
        """

        best_rmse = np.inf
        best_param = None

        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Iterate over all possible combinations of alpha, beta, and gamma
        for alpha, beta, gamma in product(alphas, betas, gammas):
            model = self.fit(train, alpha, beta, gamma)
            forecast = model.predict(start=train.index[0], end=train.index[-1])

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(train[self.dep_var][:len(forecast)], forecast))

            # Update best RMSE and best parameters if current combination is better
            if rmse < best_rmse:
                best_rmse = rmse
                best_param = (alpha, beta, gamma)

        return best_param


    def predict(self, model, df):
        """
        Predicts the values for the given DataFrame using the provided model.

        Parameters:
        model : The fitted Holt-Winters model.
        df : The DataFrame to predict on.

        Returns:
        pd.Series: The predicted values.
        """
        # predict the future values
        start_index = df.index[0]
        end_index = df.index[-1]
        forecast = model.predict(start=start_index, end=end_index)
        return forecast

    def evaluate(self, actual, predicted):
        """
        Evaluates the model using metrics based on the actual and predicted values.

        Parameters:
        actual: The actual values of the dependent variable.
        predicted: The predicted values of the dependent variable.

        Returns:
        dict: A dictionary of evaluation metrics.
        """
        # evaluate the model using the metrics
        metrics = find_metrics(actual, predicted)
        return metrics

    # forecast for n_steps ahead
    def forecast(self, model, n_steps):
        """
        Generates a forecast for n_steps ahead using the provided model.

        Parameters:
        model : The fitted Holt-Winters model.
        n_steps (int): The number of steps to forecast ahead.

        Returns:
        pd.Series: The forecasted values.
        """
        forecast = model.forecast(n_steps)
        return forecast

    def get_confidence_intervals(self, model, n_steps, alpha=0.05):
   
        # Get the forecasted values
        forecast = self.forecast(model, n_steps)

    # Calculate the standard deviation of the residuals
        std_residuals = model.resid.std()

    # Calculate the confidence intervals
        lower_bound = forecast - std_residuals * norm.ppf(1 - alpha / 2)
        upper_bound = forecast + std_residuals * norm.ppf(1 - alpha / 2)

    # Create a DataFrame with the confidence intervals
        conf_int = pd.DataFrame({'lower': lower_bound, 'upper': upper_bound})

        return conf_int

if __name__ == '__main__':
    df = read_file("Forecast-master/data/","Sqlite3.db")
    df = df[(df['speciality'] == 'Orthopedic') & (df['procedure'] == 'orknarth')]
    # select date and no_of_procedures columns
    df = df[['date', 'no_of_procedures']]
    train, test = train_test_split(df, 0.2)

    # create an object of the class
    hw = HoltWinter(train, date_column='date', dep_var='no_of_procedures', freq='M')
    train = hw.pre_process()
    test = hw.pre_process()

    # Perform grid search to find the optimal values of alpha, beta, and gamma
    alpha, beta, gamma = hw.grid_search(train)
    print("Best alpha:", alpha)
    print("Best beta:", beta)
    print("Best gamma:", gamma)

    # Fit the model using the optimal values
    model = hw.fit(train, alpha, beta, gamma)
    print(model.summary())

    # predict the future values
    forecast = hw.predict(model, test)

    # create a loop to show actual and forecast parallelly
    for i in range(len(forecast)):
        print('predicted=%f, expected=%f' % (forecast.iloc[i], test['no_of_procedures'].iloc[i]))

    # evaluate the model
    metrics = hw.evaluate(test['no_of_procedures'], forecast)
    print(metrics)

    # forecast for n_steps ahead
    forecast = hw.forecast(model, 20)
    print(forecast)
    
