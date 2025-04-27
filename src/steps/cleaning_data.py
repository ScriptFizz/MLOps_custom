import logging
import numpy as np
import pandas as pd
from utils.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def clean_df(df: pd.DataFrame) -> Tuple[
	Annotated[pd.DataFrame, "X_train"],
	Annotated[pd.DataFrame, "X_test"],
	Annotated[pd.Series, "y_train"],
	Annotated[pd.Series, "y_test"],]:
	
	try:
		preprocess_strategy = DataPreprocessStrategy()
		data_cleaning = DataCleaning(df, preprocess_strategy)
		df_preprocessed = data_cleaning.handle_data()
		
		divide_strategy = DataDivideStrategy()
		data_cleaning = DataCleaning(df_preprocessed, divide_strategy)
		X_train, X_test, y_train, y_test = data_cleaning.handle_data()
		logger.info("Data Cleaning completed")
		return X_train, X_test, y_train, y_test
	except Exception as e:
		logger.error(f"Error in cleaning data: {e}")
		raise e

def preprocess_input(x: np.array) -> np.array:
	try:
		preprocess_strategy = DataPreprocessStrategy()
		data_cleaning = DataCleaning(x, preprocess_strategy)
		x_preprocessed = data_cleaning.handle_data()
		logger.info("Data preprocessing completed")
		return x_preprocessed
	except Exception as e:
		logger.error(f"Error in preprocessing data: {e}")
		raise e
