import pandas as pd
import numpy as np
from server.Forecast import Forecast
from server.Pre_Process import Preprocess
from server.Helper_Functions import get_frequency, set_frequency, find_metrics, train_test_split, read_file

class MovingAverage(Forecast, Preprocess):
    """
    Moving Average forecasting model implementation.

    Args:
    - df (DataFrame): Input data as a pandas DataFrame.
    - date_column (str): Name of the date column in the DataFrame.
    - dep_var (str): Name of the dependent variable column in the DataFrame.
    - window (int): Window size for the moving average.
    - freq (str, optional): Frequency of the time series data. Defaults to 'D'.

    Methods:
    - pre_process(): Preprocesses the input data.
    - fit(): Fits the forecasting model to the data.
    - predict(): Predicts the future values using the fitted model.
    - evaluate(): Evaluates the performance of the forecasting model.
    - forecast(): Forecasts the future values for a given number of steps.
    """

    # initialize the class with the dataframe and the date column and the dependent variable
    def __init__(self, df, date_column, dep_var, window, freq='D'):
        super().__init__(df, date_column, dep_var)
        self.window = window
        self.freq = freq

    def pre_process(self):
        """
        Preprocesses the input data.

        Returns:
        - df (DataFrame): Preprocessed DataFrame.
        """

        self.df = self.preprocessing(threshold=3, fill_method='linear')
        # change the frequency of the data based on user input
        if self.freq != get_frequency(self.df, self.date_column):
            self.df = set_frequency(self.df, self.date_column, get_frequency(self.df, self.date_column), self.freq)
        return self.df

    def fit(self,df_preprocess,test):
        """
        Fits the forecasting model to the data.

        Args:
        - df_preprocess: Preprocessed DataFrame.
        - test: Test dataset.

        Returns:
        - model: Fitted forecasting model.
        """

        # To obtain window size
        window_size,df1=self.__find_window_size(df_preprocess,test)
        self.window=window_size
        model=self.__moving_average(df_preprocess,df1,window_size)

        return model

    def predict(self,df):
        """
        Predicts the future values.

        Args:
        - df: DataFrame to make predictions on.

        Returns:
        - predicted: Predicted values.
        """

        # Obtain predicted values 
        predicted=df['moving_avg_forecast']
        return predicted
        

    def evaluate(self, actual, predicted):
        """
        Evaluates the performance of the forecasting model.

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
    def forecast(self, df, n_steps):
        """
        Forecasts the future values for a specified number of steps.

        Args:
        - df: DataFrame to forecast on.
        - n_steps (int): Number of steps to forecast.

        Returns:
        - forecast: Forecasted values.
        """

        # Get the last index of the data
        last_index = df.index[-1]
        # Create the index for the forecasted values
        forecast_index = pd.RangeIndex(start=last_index + 1, stop=last_index + n_steps + 1)
        # Calculate the moving average forecast for future steps
        forecast = pd.Series([df[self.dep_var].rolling(window=self.window).mean().iloc[-1]] * n_steps,index=forecast_index)
        return forecast
    
    # To find optimal window size
    def __find_window_size(self,df,test):
        """
        Finds the optimal window size based on evaluation metrics.

        Args:
        - df: DataFrame to search for optimal window size.
        - test: Test dataset.

        Returns:
        - window_size (int): Optimal window size.
        - df1: DataFrame with moving average forecast column.
        """
        df1 = df.copy()
        y_test = test[['no_of_procedures']]
        rolling_window=dict()
        for i in range(1,13):
            df1['moving_avg_forecast'] = df[self.dep_var].rolling(i).mean()
            train, test = train_test_split(df1, 0.2)
            y_hat_avg = test.copy()
            abs_error = np.abs(y_test[self.dep_var]-y_hat_avg.moving_avg_forecast)
            actual = y_test[self.dep_var]
            mape = np.round(np.mean(abs_error/actual),4)
            rolling_window[i]=mape

        mape_keys=list(rolling_window.keys())
        mape_values=list(rolling_window.values())
        index=mape_values.index(min(mape_values[1:]))
        window_size=mape_keys[index]

        return window_size,df1


    def __moving_average(self,df,df1,window_size):
        """
        Calculates the moving average forecast.

        Args:
        - df (DataFrame): DataFrame to calculate moving average forecast on.
        - df1 (DataFrame): DataFrame to store the moving average forecast column.
        - window_size (int): Size of the moving average window.

        Returns:
        - df1 (DataFrame): DataFrame with the moving average forecast column.
        """
        df1['moving_avg_forecast'] = df[self.dep_var].rolling(window_size).mean()

        return df1

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
    df = read_file("Forecast-master/data/","Sqlite3.db")
    df = df[(df['speciality'] == 'Orthopedic') & (df['procedure'] == 'orknarth')]
    # select date and no_of_procedures columns
    df = df[['date', 'no_of_procedures']]
    
    # create an object of the class 
    ma = MovingAverage(df, date_column='date', dep_var='no_of_procedures', window=None, freq='D')
    pre_processed = ma.pre_process()

    train, test = train_test_split(pre_processed, 0.2)
    # preprocess the data
    model = ma.fit(pre_processed,test)
    #Creating train and test set 
    train,test= train_test_split(model, 0.2)

    forecast = ma.predict(test)

    #print(forecast)
    metrics = ma.evaluate(test['no_of_procedures'], forecast)
    print(metrics)
    model=model.drop(columns=['moving_avg_forecast'])
    forecast = ma.forecast(model, 10)
    print(forecast)


