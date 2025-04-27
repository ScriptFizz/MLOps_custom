import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IngestData:
	"""
	Ingesting data from data_path
	"""
	def __init__(self, data_path: str):
		self.data_path = data_path
	
	def get_data(self, **kwargs) -> pd.DataFrame:
		logger.info(f"Ingesting data from {self.data_path}")
		return pd.read_csv(self.data_path, **kwargs)

def ingest_df(data_path: str) -> pd.DataFrame:
	"""
	Ingesting the data from the data path:
	Args:
		data_path: path to the data
	Returns:
		pd.DataFrame: the ingested data
	"""
	try:
		ingest_data = IngestData(data_path)
		df = ingest_data.get_data()
		return df
	except Exception as e:
		logger.error("Error while ingesting data: {e}")
		raise e


