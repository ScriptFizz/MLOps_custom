import logging
from sklearn.pipeline import Pipeline
import pandas as pd
from utils.model_development import select_model, select_preprocessor#LinearRegressionModel, RandomForestModel, KNNModel
from sklearn.base import RegressorMixin
import mlflow
import mlflow.pyfunc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RegressionModel(mlflow.pyfunc.PythonModel):
	
	def __init__(self, preprocessor_config: dict, model_name: str, **kwargs):
		preprocessor = select_preprocessor(preprocessor_config)
		model = select_model(model_name, **kwargs)
		self.pipeline = Pipeline([
			('preprocessor', preprocessor),
			('model', model)
			])
			
	def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
		self.pipeline.fit(X_train, y_train)
		#return self.pipeline
	
	def predict(self, X_test: pd.DataFrame):
		return self.pipeline.predict(X_test)
			
"""
def train_model(
	X_train: pd.DataFrame, 
	y_train: pd.DataFrame,
	model_name: str, **kwargs) -> RegressorMixin:
	try: 
		model = None
		match model_name:
			case "LinearRegression":
				model = LinearRegressionModel()
			
			case "RandomForestRegressor":
				model = RandomForestModel()
			
			case "KNeighborsRegressor":
				model = KNNModel()
		
			case default:
				raise ValueError(f"Model {model_name} not supported")
		
		trained_model = model.train(X_train, y_train, **kwargs)
		return trained_model
		
	except Exception as e:
		logger.error(f"Error in training model: {e}")
		raise e
"""
