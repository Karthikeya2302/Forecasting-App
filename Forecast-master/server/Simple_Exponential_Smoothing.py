import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from server.Forecast import Forecast
from server.Pre_Process import Preprocess
from server.Helper_Functions import get_frequency, set_frequency, find_metrics, train_test_split, read_file

class SimpleExponentialSmoothing(Forecast, Preprocess):
    """
    SimpleExponentialSmoothing class for time series forecasting using Simple Exponential Smoothing.

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
    - fit(df): Fits the Simple Exponential Smoothing model to the training data.
    - predict(model, df): Makes predictions using the Simple Exponential Smoothing model.
    - evaluate(actual, predicted): Evaluates the performance of the Simple Exponential Smoothing model.
    - forecast(model, n_steps): Forecasts future values using the Simple Exponential Smoothing model.
    - __evaluate_alpha(df, dep_var): Finds the optimal alpha value.
    - __simple_exponential_smoothing(series, alpha): Performs simple exponential smoothing on the series.
    """

    # initialize the class with the dataframe and the date column and the dependent variable
    def __init__(self, df, date_column, dep_var, freq='D'):
        """
        Initializes the SimpleExponentialSmoothing class.

        Args:
        - df (DataFrame): Input data as a pandas DataFrame.
        - date_column (str): Name of the date column in the DataFrame.
        - dep_var (str): Name of the dependent variable column in the DataFrame.
        - freq (str, optional): Frequency of the time series data. Defaults to 'D'.
        """

        super().__init__(df, date_column, dep_var)
        self.freq = freq

    def pre_process(self):
        """
        Preprocesses the input DataFrame.

        Returns:
        - df (DataFrame): Preprocessed DataFrame.
        """

        self.df = self.preprocessing(threshold=3, fill_method='linear')
        # change the frequency of the data based on user input
        if self.freq != get_frequency(self.df, self.date_column):
            self.df = set_frequency(self.df, self.date_column, get_frequency(self.df, self.date_column), self.freq)
        return self.df

    def fit(self, df):
        """
        Fits the Simple Exponential Smoothing model to the training data.

        Args:
        - df (DataFrame): Training data as a pandas DataFrame.

        Returns:
        - model: Fitted Simple Exponential Smoothing model.
        """

        # To obtain optimal alpha value
        alpha=self.__evaluate_alpha(df,self.dep_var)
        # fit the model using SimpleExpSmoothing
        model = SimpleExpSmoothing(df[self.dep_var])
        model = model.fit(smoothing_level=alpha)
        # Return the model after fitting
        return model

    def predict(self, model, df):
        """
        Makes predictions using the Simple Exponential Smoothing model.

        Args:
        - model: Fitted Simple Exponential Smoothing model.
        - df: Data for which predictions are made as a pandas DataFrame.

        Returns:
        - forecast (Series): Predicted values.
        """

        # predict the future values
        forecast = model.predict(start=df.index[0], end=df.index[-1])
        return forecast

    def evaluate(self, actual, predicted):
        """
        Evaluates the performance of the Simple Exponential Smoothing model.

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
        """
        Forecasts future values using the Simple Exponential Smoothing model.

        Args:
        - model: Fitted Simple Exponential Smoothing model.
        - n_steps (int): Number of steps to forecast ahead.

        Returns:
        - forecast (Series): Forecasted values.
        """
        forecast = model.forecast(steps=n_steps)
        return forecast
    
    # To find optimal alpha value
    def __evaluate_alpha(self,df,dep_var):
        """
        Finds the optimal alpha value for Simple Exponential Smoothing.

        Args:
        - df (DataFrame): Data as a pandas DataFrame.
        - dep_var (str): Name of the dependent variable column in the DataFrame.

        Returns:
        - optimal_alpha (float): Optimal alpha value.
        """
        
        series=df[dep_var]
        # To store the SSE for each alpha value
        errors = []  
        # Try alpha values from 0 to 1 with a step size of 0.01
        for alpha in np.linspace(0, 1, 101):  
            smoothed_series = self.__simple_exponential_smoothing(series, alpha)
            error = ((series - smoothed_series) ** 2).sum()
            errors.append(error)
         # Get the index of the smallest SSE    
        optimal_alpha_index = np.argmin(errors) 
        # Get the corresponding alpha value
        optimal_alpha = np.linspace(0, 1, 101)[optimal_alpha_index]  
        # Return optimal alpha value
        return optimal_alpha
    
    def __simple_exponential_smoothing(self,series, alpha):
        """
        Performs simple exponential smoothing on the series.

        Args:
        - series: Time series data.
        - alpha (float): Smoothing parameter.

        Returns:
        - smoothed_series (Series): Smoothed series.
        """
        
        smoothed_series = pd.Series(index=series.index)  # Initialize the smoothed series with the same index
        # Initialize the first smoothed value as the first observation
        smoothed_series.iloc[0] = series.iloc[0]
        for i in range(1, len(series)):
            smoothed_value = alpha * series.iloc[i] + (1 - alpha) * smoothed_series.iloc[i-1]
            smoothed_series.iloc[i] = smoothed_value
        # Return smoothed Series
        return smoothed_series
    
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

    ses = SimpleExponentialSmoothing(train, date_column='date', dep_var='no_of_procedures', freq='D')
    # preprocess the data
    train = ses.pre_process()
    test = ses.pre_process()
    model = ses.fit(train)
    print(model.summary())
    forecast = ses.predict(model, test)
    # create a loop to show actual and forecast parallely
    for i in range(len(forecast)):
        print('predicted=%f, expected=%f' % (forecast.iloc[i], test['no_of_procedures'].iloc[i]))

    # evaluate the model
    metrics = ses.evaluate(test['no_of_procedures'], forecast)
    print(metrics)

    # forecast for n_steps ahead
    forecast = ses.forecast(model, 10)
    print(forecast)
    print(test['no_of_procedures'].mean())
