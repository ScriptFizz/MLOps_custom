import yaml
import pandas as pd
from typing import Union

def read_config(config_path: str) -> dict:

	with open(config_path) as yaml_file:
		config = yaml.safe_load(yaml_file)
	return config


def get_feats_target(df: pd.DataFrame, target: str) -> Union[pd.DataFrame, pd.Series]:
	X = df.drop(target, axis = 1)
	y = df[target]
	return X, y
