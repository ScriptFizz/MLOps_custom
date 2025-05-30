import joblib
import mlflow
import argparse
from pprint import pprint
from utils.methods import read_config
from mlflow.tracking import MlflowClient

def log_production_model(config_path: str):
	config = read_config(config_path)
	mlflow_config = config["mlflow_config"]
	model_name = mlflow_config["registered_model_name"]
	model_dir = config["model_dir"]
	remote_server_uri = mlflow_config["remote_server_uri"]
	
	mlflow.set_tracking_uri(remote_server_uri)
	runs = mlflow.search_runs(experiment_ids = "1")
	min_rmse = min(runs["metrics.rmse"])
	min_rmse_run_id = list(runs[runs["metrics.rmse"] == min_rmse]["run_id"])[0]
	client = MlflowClient()
	for mv in client.search_model_versions(f"name='{model_name}'"):
		mv = dict(mv)
		if mv["run_id"] == min_rmse_run_id:
			current_version = mv["version"]
			logged_model = mv["source"]
			pprint(mv, indent = 4)
			client.transition_model_version_stage(
				name = model_name,
				version = current_version,
				stage = "Production"
				)
			print(f'logged_model: {logged_model}')
		else:
			current_version = mv["version"]
			client.transition_model_version_stage(
			name = model_name,
			version = current_version,
			stage = "Staging"
			)
	loaded_model = mlflow.pyfunc.load_model(logged_model)
	joblib.dump(loaded_model, model_dir)
		
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)
