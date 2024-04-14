# import the required libraries
import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from server.Forecast import Forecast
from server.Pre_Process import Preprocess
from server.Helper_Functions import get_frequency, set_frequency, find_metrics, train_test_split,read_file

# create a class for arima inheriting from base class Forecast and Preprocess

class Arima(Forecast,Preprocess):
    """
    ARIMA class for time series forecasting using the ARIMA model.

    Args:
    - df (DataFrame): Input data as a pandas DataFrame.
    - date_column (str): Name of the date column in the DataFrame.
    - dep_var (str): Name of the dependent variable column in the DataFrame.
    - freq (str, optional): Frequency of the time series data. Defaults to 'D'.

    Attributes:
    - df (DataFrame): Preprocessed DataFrame.
    - date_column (str): Name of the date column in the DataFrame.
    - dep_var (str): Name of the dependent variable column in the DataFrame.
    - freq (str): Frequency of the time series data.

    Methods:
    - pre_process(): Preprocesses the input DataFrame.
    - fit(df): Fits the ARIMA model to the training data.
    - predict(model, df): Makes predictions using the ARIMA model.
    - evaluate(actual, predicted): Evaluates the performance of the ARIMA model.
    - forecast(model, n_steps): Forecasts future values using the ARIMA model.
    """
    
    # initialize the class with the dataframe and the date column and the dependent variable
    def __init__(self,df,date_column,dep_var, freq='D'):
        """
        Initializes the Arima class.

        Args:
        - df (DataFrame): Input data as a pandas DataFrame.
        - date_column (str): Name of the date column in the DataFrame.
        - dep_var (str): Name of the dependent variable column in the DataFrame.
        - freq (str, optional): Frequency of the time series data. Defaults to 'D'.
        """

        super().__init__(df,date_column,dep_var)
        self.freq = freq
  
    def pre_process(self):
        """
        Preprocesses the input DataFrame.

        Returns:
        - df (DataFrame): Preprocessed DataFrame.
        """

        self.df = self.preprocessing(threshold=0.3, fill_method='linear')
        # change the frequency of the data based on user input
        if self.freq != get_frequency(self.df, self.date_column):
            self.df = set_frequency(self.df, self.date_column, get_frequency(self.df, self.date_column),self.freq)
        return self.df
    
    def fit(self, df):
        """
        Fits the ARIMA model to the training data.

        Args:
        - df (DataFrame): Training data as a pandas DataFrame.

        Returns:
        - model: Fitted ARIMA model.
        """

        # fit the model using auto_arima
        model = auto_arima(df[self.dep_var], trace=True, error_action='ignore', suppress_warnings=True, test='kpss',
                           information_criterion='aic', stepwise=True, seasonal=True, n_fits=50)
        return model
    
    def predict(self, model, df):
        """
        Makes predictions using the ARIMA model.

        Args:
        - model: Fitted ARIMA model.
        - df (DataFrame): Data for which predictions are made as a pandas DataFrame.

        Returns:
        - forecast: Predicted values.
        """

        # predict the future values
        forecast = model.predict(n_periods=len(df))
        return forecast
    
    def evaluate(self, actual, predicted):
        """
        Evaluates the performance of the ARIMA model.

        Args:
        - actual: Actual values.
        - predicted: Predicted values.

        Returns:
        - metrics (dict): Evaluation metrics.
        """

        # evaluate the model using the metrics
        metrics = find_metrics(actual, predicted)
        return metrics
    
    # forecast for n_steps ahead
    def forecast(self, model, n_steps):
        forecast = model.predict(n_periods=n_steps)
        return forecast

    def get_confidence_intervals(self, model, n_steps, alpha=0.05):

        """
        Calculates the confidence intervals for the forecasted values.
        Args:
        - model: Fitted ARIMA model.
        - n_steps (int): Number of steps ahead to forecast.
        - alpha (float, optional): Significance level. Defaults to 0.05.
        Returns:
        - conf_int (DataFrame): Confidence intervals for the forecasted values.
        """
    # get the confidence intervals
        conf_int = model.predict(n_periods=n_steps, return_conf_int=True, alpha=alpha)
        conf_int = pd.DataFrame(conf_int[1], columns=['lower', 'upper'])
        return conf_int


if __name__ == '__main__':

    from Forecast import Forecast
    from Pre_Process import Preprocess
    from Helper_Functions import get_frequency, set_frequency, find_metrics, train_test_split,read_file

    df = read_file("Forecast-master/data/","Sqlite3.db")
    df = df[(df['speciality'] == 'Orthopedic') & (df['procedure'] == 'orknarth')]
    # select date and no_of_procedures columns
    df = df[['date', 'no_of_procedures']]
    train, test = train_test_split(df, 0.2)
    # create an object of the class
    arima = Arima(train, date_column='date', dep_var='no_of_procedures', freq='W')
    # preprocess the data
    train = arima.pre_process()
    test = arima.pre_process()
    model = arima.fit(train)
    # predict the future values
    forecast = arima.predict(model, test)
    # create a loop to show actual and forecast parallely
    for i in range(len(forecast)):
        print('predicted=%f, expected=%f' % (forecast.iloc[i], test['no_of_procedures'].iloc[i]))

    # evaluate the model
    metrics = arima.evaluate(test['no_of_procedures'], forecast)
    print(metrics)

    # forecast for n_steps ahead
    forecast = arima.forecast(model, 10)
    print(forecast)
   

    pickle.dump(model, open('model.pkl', 'wb'))

    pickled_model = pickle.load(open('model.pkl', 'rb'))
    print(pickled_model.predict(20))

    



    

    
