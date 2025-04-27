import os
import argparse
import pandas as pd
from utils.methods import read_config
from utils.data_cleaning import DataDivideStrategy, DataCleaning
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def split_data(config_path: str):
	config = read_config(config_path)
	transformed_data_path = config["raw_data_config"]["transformed_data_csv"]
	test_data_path = config["processed_data_config"]["test_data_csv"] 
	train_data_path = config["processed_data_config"]["train_data_csv"]
	test_size = config["raw_data_config"]["train_test_split_ratio"]
	random_state = config["raw_data_config"]["random_state"]
	df_transformed = pd.read_csv(transformed_data_path)
	divide_strategy = DataDivideStrategy()
	data_splitting = DataCleaning(df_transformed, divide_strategy)
	train, test = data_splitting.handle_data(test_size = test_size, random_state = random_state)
	train.to_csv(train_data_path, sep = ',', index = False)
	test.to_csv(test_data_path, sep = ',', index = False)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_data(config_path=parsed_args.config)
