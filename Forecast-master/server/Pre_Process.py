import pandas as pd
import numpy as np
from server.Helper_Functions import read_file

class Preprocess:
    """
    A class for preprocessing time series data.
    """

    # initialize the class
    def __init__(self, df, date_column, dep_var):
        """
        Initialize the Preprocess class.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        date_column (str): The name of the column representing the date.
        dep_var (str): The name of the column representing the dependent variable.
        """
        self.df = df
        self.date_column = date_column
        self.dep_var = dep_var

    # create a public method to preprocess the data
    def preprocessing(self, threshold=3, fill_method='linear'):
       """
        Preprocess the time series data by filling missing dates, filling missing values,
        and replacing outliers.

        Parameters:
        threshold (float): The threshold value for identifying outliers. Defaults to 3.
        fill_method (str): The method to use for filling missing values. Defaults to 'linear'.

        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
       
       # Fill missing dates
       self.df = self.__fill_missing_dates()
       # Fill missing values
       self.df = self.__fill_missing_values(method=fill_method)
       # Replace outliers
       self.df = self.__replace_outliers(threshold=threshold)
       # Return the DataFrame
       return self.df

    # create a private method to fill missing dates

    def __fill_missing_dates(self):
        """
        Fill missing dates in the DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with missing dates filled.
        """
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_column]):
            # Convert the date column to datetime type
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])

        # Sort the DataFrame by the date column
        self.df = self.df.sort_values(self.date_column)

        # Calculate the time differences between consecutive dates
        time_diff = self.df[self.date_column].diff()

        # Find the most common time difference
        frequency = time_diff.value_counts().idxmax().days

        # create a dictionary of frequency integer to frequency string
        frequency_dict = {1: 'D', 7: 'W', 30: 'M', 365: 'A-DEC', 31: 'M',14:'2W-SUN'}
        

        # Set the 'date' column as the index
        self.df.set_index(self.date_column, inplace=True)

        # Resample the data to the most common frequency
        full_data = self.df.resample(frequency_dict[frequency]).sum()

        # Reset the index of the DataFrame
        full_data.reset_index(inplace=True)

        # Return the DataFrame
        return full_data
    
    # create a private method to fill missing values

    def __fill_missing_values(self, method='linear'):
        """
        Fill missing values in the DataFrame.

        Parameters:
        method (str): The method to use for filling missing values. Defaults to 'linear'.

        Returns:
        pd.DataFrame: The DataFrame with missing values filled.
        """
        # Fill missing values using linear interpolation
        self.df[self.dep_var] = self.df[self.dep_var].interpolate(method=method)

        # Return the DataFrame
        return self.df
    
    # create a private method to replace outliers

    def __replace_outliers(self, threshold=3):
        """
        Replace outliers in the DataFrame.

        Parameters:
        threshold (float): The threshold value for identifying outliers. Defaults to 3.

        Returns:
        pd.DataFrame: The DataFrame with outliers replaced.
        """
        # Calculate the mean and standard deviation
        mean, std = self.df[self.dep_var].mean(), self.df[self.dep_var].std()

        # Compute the upper and lower threshold
        lower, upper = mean - threshold * std, mean + threshold * std

        # Replace outliers
        self.df[self.dep_var] = np.where(self.df[self.dep_var] > upper, upper, np.where(self.df[self.dep_var] < lower, lower, self.df[self.dep_var]))

        # Return the DataFrame
        return self.df
    

if __name__ == '__main__':
    # Read the data
    df = read_file("Forecast-master/data/","Sqlite3.db")
    df = df[(df['speciality'] == 'Orthopedic') & (df['procedure'] == 'orknarth')]
    # select date and no_of_procedures columns
    df = df[['date', 'no_of_procedures']]

    # Create an instance of the Preprocess class
    preprocess = Preprocess(df, 'date', 'no_of_procedures')

    # Preprocess the data
    df = preprocess.preprocessing()

    # Print the head of the DataFrame
    print(df.head())



    

    
