import pandas as pd
import numpy as np
import yaml
import argparse
import logging
from utils.methods import read_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data(data_path: str, features: list) -> pd.DataFrame:
	try:
		df = pd.read_csv(data_path, sep = ',')
		df = df[features]
		return df
	except Exception as e:
		logger.error(f"Error in loading external data: {e}")
		raise e

def load_raw_data(config_path: str):
	
	config: dict = read_config(config_path)
	external_data_path: str = config["external_data_config"]["external_data_csv"]
	raw_data_path: str = config["raw_data_config"]["raw_data_csv"]
	features: list = config["raw_data_config"]["model_var"]
	
	df = load_data(external_data_path, features)
	df = df[features]
	df.to_csv(raw_data_path, index = False)
	
if __name__ == "__main__":
	args = argparse.ArgumentParser()
	args.add_argument("--config", default = "params.yaml")
	parsed_args = args.parse_args()
	load_raw_data(parsed_args.config)
