import logging
from zenml import step
import pandas as pd
from src.datacleaning import DataCleaning,DataDivideStrategy,DataPreproseeingStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"x_train"],
    Annotated[pd.DataFrame,"x_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],
    ]:

    """
    This step is used to clean the data and divide the data into train and test
    args:
        df: pandas dataframe
        returns:
            x_train: pandas dataframe
            x_test: pandas dataframe
            y_train: pandas series
            y_test: pandas series
            """
    try:
        preprocess_strategy=DataPreproseeingStrategy()
        data_cleaning = DataCleaning(df,preprocess_strategy)
        preprocessed_df=data_cleaning.handle_data() 

        dividestrategy = DataDivideStrategy()   
        data_cleaning = DataCleaning(preprocessed_df,dividestrategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")
        return x_train, x_test, y_train, y_test
    except Exception as e:  
        logging.error(f"Error in Data Cleaning{e}")
        raise e
