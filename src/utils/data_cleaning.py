import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataStrategy(ABC):
	"""
	Abstract class defining strategy for handling data
	"""
	@abstractmethod
	def handle_data(self, df: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, pd.Series]:
		pass

class DataPreprocessStrategy(DataStrategy):
	"""
	Class that preprocess the data
	"""
	def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
		try:
			df['date'] = pd.to_datetime(df['date'].apply(lambda x: x[:8]), format = "%Y%m%d")
			df['date'] = (dt.datetime.now() - df['date']).dt.days
			return df
		except Exception as e:
			logger.error(f"Preprocessing failed: {e}")
			raise e

class DataDivideStrategy(DataStrategy):
	"""
	Class that split the data
	"""
	def handle_data(self, df: pd.DataFrame, test_size: float, random_state: int) -> Union[pd.DataFrame, pd.Series]:
		try:
			train, test = train_test_split(df, test_size = test_size, random_state = random_state)
			return train, test
		except Exception as e:
			logger.error(f"Data splitting failed: {e}")
			raise e


class DataCleaning:
	"""
	Class that preprocess and divide the data
	"""
	def __init__(self, df: pd.DataFrame, strategy: DataStrategy):
		self.df = df
		self.strategy = strategy
	
	def handle_data(self, **kwargs) -> Union[pd.DataFrame, pd.Series]:
		try:
			return self.strategy.handle_data(self.df, **kwargs)
		except Exception as e:
			logger.error(f"Error in preprocessing data: {e}")
			raise e	


