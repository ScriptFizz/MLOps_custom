import logging
import pandas as pd
from sklearn.base import RegressorMixin
from utils.evaluation import MSE, RMSE, R2
from typing import Tuple
from typing_extensions import Annotated
import mlflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate_model(model: RegressorMixin,
		X_test: pd.DataFrame,
		y_test: pd.DataFrame
		) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
	"""
	evaluate R2 score and RMSE of a model
	"""
	try:
		prediction = model.predict(X_test)
		
		mse_class = MSE()
		mse = mse_class.calculate_scores(y_test, prediction)
		#mlflow.log_metric("mse", mse)
		
		r2_class = R2()
		r2_score = r2_class.calculate_scores(y_test, prediction)
		#mlflow.log_metric("r2_score", r2_score)
		
		rmse_class = RMSE()
		rmse = rmse_class.calculate_scores(y_test, prediction)
		#mlflow.log_metric("rmse", rmse)
		
		return r2_score, rmse
	
	except Exception as e:
		logger.error(f"Error in evaluating model: {e}")
		raise e
