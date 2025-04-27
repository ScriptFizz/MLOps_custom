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
	
	def predict(self, X_test: pd.DataFrame):
		return self.pipeline.predict(X_test)
