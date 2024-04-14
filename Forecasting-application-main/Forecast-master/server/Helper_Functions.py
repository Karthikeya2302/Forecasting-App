import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import sqlite3

def read_file(file_path, file_name):
    """
    Reads a data from database and parses the specified date column as datetime.

    Returns:
        pandas.DataFrame: The DataFrame read from the CSV file.
    """

    # create a string of file path and file name
    _file = file_path + file_name

    try:
        # Making a connection between sqlite3 database and Python Program
        conn = sqlite3.connect(_file)
        # If sqlite3 makes a connection with python program then it will print "Connected to SQLite"
        # Otherwise it will show errors
        print("Connected to SQLite")
    except sqlite3.Error as error:
        print("Failed to connect with sqlite3 database", error)
    df = pd.read_sql_query("SELECT * FROM ts_data", conn)
    # Convert the 'date_column' to a specific date format
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

    conn.close()

    return df

def get_frequency(df, date_column):
    """
    Infers the frequency of the data based on the date column.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        date_column (str): The name of the column containing dates.

    Returns:
        str: The inferred frequency of the data.
    """

    # Get the frequency of the data
    frequency = pd.infer_freq(df[date_column])
    return frequency

def set_frequency(df,date_column, frequency,set_freq):
    """
    Sets the frequency of the data to the desired frequency.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        date_column (str): The name of the column containing dates.
        frequency (str): The current frequency of the data.
        set_freq (str): The desired frequency to set.

    Returns:
        pandas.DataFrame: The resampled DataFrame with the desired frequency.
    """

    # convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # set date column as index
    df.set_index(date_column, inplace=True)
    
    freq_dict={'D':1,'W':7,'M':30,'Y':365, 'W-SUN': 7,'2W':14,'2W-SUN':14,'A-DEC':365}

    if freq_dict[frequency]<freq_dict[set_freq]:
        resampled = df.resample(set_freq).sum()

    elif freq_dict[frequency]>freq_dict[set_freq]:
        resampled = df.resample(set_freq).mean()
    else:
        resampled = df

    # reset index
    resampled.reset_index(inplace=True)

    return resampled


def train_test_split(df, test_size = 0.2):
    """
    Splits the DataFrame into training and test sets based on the specified test size.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        test_size (float): The proportion of the data to include in the test set.

    Returns:
        tuple: A tuple containing the training and test DataFrames.
    """
    train_size = int(len(df) * (1-test_size))
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    return train, test


def find_metrics(actual,predicted):
    """
    Calculates evaluation metrics between the actual and predicted values.

    Args:
        actual (array-like): The actual values.
        predicted (array-like): The predicted values.

    Returns:
        dict: A dictionary containing the evaluation metrics.
            - r_2_score (float): The R-squared score.
            - mae (float): The mean absolute error (MAE).
            - rmse (float): The root mean squared error (RMSE).
            - mape (float): The mean absolute percentage error (MAPE).
    """
    metrics = {}
    metrics['r_2_score'] = r2_score(actual,predicted)
    metrics['mae'] = np.mean(np.abs(predicted - actual))
    metrics['rmse'] = np.sqrt(np.mean((predicted- actual) ** 2))
    metrics['mape'] = np.mean(np.abs((predicted - actual) / actual)) * 100
    return metrics

