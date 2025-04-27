from steps.ingest_data import ingest_df
from steps.cleaning_data import clean_df
from steps.model_train import RegressionModel #train_model
from steps.evaluation import evaluate_model
import argparse
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
from utils.methods import read_config, get_feats_target



def train_pipeline(config_path: str):
	config = read_config(config_path)
	train_data_path = config["processed_data_config"]["train_data_csv"]
	test_data_path = config["processed_data_config"]["test_data_csv"]
	target = config["raw_data_config"]["target"]
	model_name = config['model_config']['model_name']
	model_params = config['model_config']['params']
	preprocessor_config = config['preprocess_config']
	train = ingest_df(train_data_path)
	test = ingest_df(test_data_path)
	X_train, y_train = get_feats_target(train, target)
	X_test, y_test = get_feats_target(test, target)
	signature = infer_signature(X_train, y_train)
	model = RegressionModel(preprocessor_config, model_name, **model_params)
	model.fit(X_train, y_train)
	r2_score, rmse = evaluate_model(model, X_test, y_test)
	mlflow_config = config["mlflow_config"]
	remote_server_uri = mlflow_config["remote_server_uri"]
	mlflow.set_tracking_uri(remote_server_uri)
	mlflow.set_experiment(mlflow_config["experiment_name"])
	
	with mlflow.start_run(run_name = mlflow_config["run_name"]) as mlops_run:
		mlflow.log_params(model_params)
		mlflow.log_metrics({'r2_score': r2_score, 'rmse': rmse})
		
		tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
		print(f'tracking_url_type_store: {tracking_url_type_store}')
		if tracking_url_type_store != "file":
			mlflow.sklearn.log_model(
                model, 
                "model", 
                signature = signature,
				input_example = X_train,
                registered_model_name=mlflow_config["registered_model_name"])
		else:
			mlflow.sklearn.load_model(model, "regression_model")
 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_pipeline(config_path=parsed_args.config)
