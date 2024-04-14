from abc import ABC, abstractmethod
 
class Forecast(ABC):
    """
    Abstract base class for time series forecasting models.

    Methods:
    - pre_process(): Preprocesses the input data.
    - fit(): Fits the forecasting model to the data.
    - evaluate(): Evaluates the performance of the forecasting model.
    """

    def __init__(self,df,date_column,dep_var):
        """
        Initializes the Forecast class.

        Args:
        - df (DataFrame): Input data as a pandas DataFrame.
        - date_column (str): Name of the date column in the DataFrame.
        - dep_var (str): Name of the dependent variable column in the DataFrame.
        """
        self.df=df
        self.date_column=date_column
        self.dep_var=dep_var

    @abstractmethod
    def pre_process(self):
        """
        Preprocesses the input data.

        Returns:
        - df (DataFrame): Preprocessed DataFrame.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Fits the forecasting model to the data.
        """
        pass
    
    @abstractmethod
    def evaluate(self):
        """
        Evaluates the performance of the forecasting model.

        Returns:
        - metrics (dict): Evaluation metrics.
        """
        pass