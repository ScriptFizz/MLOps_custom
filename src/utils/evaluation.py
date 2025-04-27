import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Evaluation(ABC):
	"""
	"""
	@abstractmethod
	def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
		pass

class MSE(Evaluation):
	"""
	Evaluation strategy with MSE
	"""
	def calculate_scores(self,  y_true: np.ndarray, y_pred: np.ndarray):
		try:
			logger.info("Calculating MSE...")
			mse = mean_squared_error(y_true, y_pred)
			logger.info(f"MSE: {mse}")
			return mse
		except Exception as e:
			logger.error(f"Error in calculating MSE: {e}")
			raise e	


class R2(Evaluation):
	"""
	Evaluation strategy with R2
	"""
	def calculate_scores(self,  y_true: np.ndarray, y_pred: np.ndarray):
		try:
			logger.info("Calculating MSE...")
			r2 = r2_score(y_true, y_pred)
			logger.info(f"R2: {r2}")
			return r2
		except Exception as e:
			logger.error(f"Error in calculating R2: {e}")
			raise e	


class RMSE(Evaluation):
	"""
	Evaluation strategy with MSE
	"""
	def calculate_scores(self,  y_true: np.ndarray, y_pred: np.ndarray):
		try:
			logging.info("Calculating MSE...")
			rmse = np.sqrt(mean_squared_error(y_true, y_pred))
			logger.info(f"RMSE: {rmse}")
			return rmse
		except Exception as e:
			logger.error(f"Error in calculating RMSE: {e}")
			raise e	
