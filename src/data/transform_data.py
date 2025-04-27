import logging
import numpy as np
import pandas as pd
import argparse
from utils.methods import read_config
from utils.data_cleaning import DataCleaning, DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def transform_df(config_path: str):
	config = read_config(config_path)
	raw_data_path = config['raw_data_config']['raw_data_csv']
	transformed_data_path = config['raw_data_config']['transformed_data_csv']
	try:
		df = pd.read_csv(raw_data_path)
		preprocess_strategy = DataPreprocessStrategy()
		data_cleaning = DataCleaning(df, preprocess_strategy)
		df_transformed = data_cleaning.handle_data()
		df_transformed.to_csv(transformed_data_path)
		logger.info("Data Tranformation completed")
	except Exception as e:
		logger.error(f"Error in cleaning data: {e}")
		raise e


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="config.yaml")
    parsed_args = args.parse_args()
    transform_df(config_path=parsed_args.config)
