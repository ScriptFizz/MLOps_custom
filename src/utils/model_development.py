import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import RegressorMixin
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def select_model(
	model_name:str, **kwargs) -> RegressorMixin:
	try: 
		model = None
		match model_name:
			case "LinearRegression":
				model = LinearRegression(**kwargs)
			
			case "RandomForestRegressor":
				model = RandomForestRegressor(**kwargs)
			
			case "KNeighborsRegressor":
				model = KNeighborsRegressor(**kwargs)
		
			case default:
				raise ValueError(f"Model {model_name} not supported")
		
		return model
		
	except Exception as e:
		logger.error(f"Error in selecting model: {e}")
		raise e


def select_preprocessor(preprocess_config: dict) -> ColumnTransformer:
	preprocess_steps = []
	try:
		for key in preprocess_config.keys():
			if (key.startswith('imputer') and (preprocess_config[key] is not None)):
				name = preprocess_config[key]['name']
				feats = preprocessor_config[key]['feats']
				feats = feats if feats is not None else make_column_selector(dtype_include = [np.number, object])
				
				if name == 'SimpleImputer':
					preprocess_steps.append((SimpleImputer(strategy = 'median'), feats))
				else:
					logger.warning(f"Imputer {name} is not yet supported and will be ignored.")
						
			elif (key.startswith('scaler') and (preprocess_config[key] is not None)):
				name = preprocess_config[key]['name']
				feats = preprocess_config[key]['feats']
				feats = feats if feats is not None else make_column_selector(dtype_include = np.number)
				
				if name == 'StandardScaler':
					preprocess_steps.append((StandardScaler(), feats))
				elif name =='MinMaxScaler':
					preprocess_steps.append((MinMaxScaler(), feats))
				else:
					logger.warning(f"Scaler {name} is not yet supported and will be ignored.")
					
			elif (key.startswith('encoder') and (preprocess_config[key] is not None)):
				name = preprocess_config[key]['name']
				feats = preprocess_config[key]['feats']
				feats = feats if feats is not None else make_column_selector(dtype_include = object)
				
				if name == 'OneHotEncoder':
					preprocess_steps.append((OneHotEncoder(), feats))
				elif name == 'OrdinalEncoder':
					preprocess_steps.append((OrdinalEncoder(), feats))
				else:
					logger.warning(f"Encoder {name} is not yet supported and will be ignored.")
			
			elif (key.startswith('pca') and (preprocess_config[key] is not None)):
				name = preprocess_config[key]['name']
				
				if name == 'PCA':
					preprocess_steps.append((PCA(n_components = 0.95), make_column_selector(dtype_include = np.number)))
				else:
					logger.warning(f"PCA {name} is not yet supported and will be ignored.")
			
			else:
				logger.warning(f'Preprocessing step {key} is not yet supported and will be ignored')
			
		print(f'preprocess_steps: {preprocess_steps}')
		column_transformer = make_column_transformer(*preprocess_steps, remainder = 'passthrough', verbose_feature_names_out = False)
		return column_transformer
			
	except Exception as e:
		logger.error(f'Error in creating preprocessor: {e}')
		
"""
def select_preprocessor(preprocess_config: dict) -> ColumnTransformer:
	preprocess_steps = []
	try:
		for key in preprocess_config.keys():
			print(f'key: {key}')
			print(f'value: {preprocess_config[key]}')
			if (key.startswith('imputer') and (preprocess_config[key] is not None)):
				print('inside imputer')
				name = preprocess_config[key]['name']
				feats = preprocessor_config[key]['feats']
				feats = feats if feats is not None else make_column_selector(dtype_include = [np.number, object])
				match name:
					case 'SimpleImputer':
						preprocess_steps.append((SimpleImputer(strategy = 'median'), feats))
					case default:
						logger.warning(f"Imputer {name} is not yet supported and will be ignored.")
						
			elif (key.startswith('scaler') and (preprocess_config[key] is not None)):
				name = preprocess_config[key]['name']
				feats = preprocess_config[key]['feats']
				feats = feats if feats is not None else make_column_selector(dtype_include = np.number)
				
				match name:
					case 'StandardScaler':
						preprocess_steps.append((StandardScaler(), feats))
					case 'MinMaxScaler':
						preprocess_steps.append((MinMaxScaler(), feats))
					case default:
						logger.warning(f"Scaler {name} is not yet supported and will be ignored.")
					
			elif (key.startswith('encoder') and (preprocess_config[key] is not None)):
				name = preprocess_config[key]['name']
				feats = preprocess_config[key]['feats']
				feats = feats if feats is not None else make_column_selector(dtype_include = object)
				match name:
					case 'OneHotEncoder':
						preprocess_steps.append((OneHotEncoder(), feats))
					case 'OrdinalEncoder':
						preprocess_steps.append((OrdinalEncoder(), feats))
					case default:
						logger.warning(f"Encoder {name} is not yet supported and will be ignored.")
			
			elif (key.startswith('pca') and (preprocess_config[key] is not None)):
				name = preprocess_config[key]['name']
				match name:
					case 'PCA':
						preprocess_steps.append((PCA(n_components = 0.95), make_column_selector(dtype_include = np.number)))
					case default:
						logger.warning(f"PCA {name} is not yet supported and will be ignored.")
			
			else:
				logger.warning(f'Preprocessing step {key} is not yet supported and will be ignored')
			
		print(f'preprocess_steps: {preprocess_steps}')
		column_transformer = make_column_transformer(*preprocess_steps, remainder = 'passthrough', verbose_feature_names_out = False)
		return column_transformer
			
	except Exception as e:
		logger.error(f'Error in creating preprocessor: {e}')
"""
