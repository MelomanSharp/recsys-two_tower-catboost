import mlflow
import os
import logging
from src.config import Config

def load_latest_models_from_registry():
    """Downloads current CatBoost and Two-Tower weights from MLflow."""
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    
    catboost_path = os.path.join(Config.MODEL_DIR, "catboost_ranker.cbm")
    
    try:
        # In production, this would be mlflow.pyfunc.load_model("models:/CatBoostRanker/Production")
        # But for CatBoost and PyTorch we just pull the artifacts of the last successful run
        experiment = mlflow.get_experiment_by_name(Config.MLFLOW_EXPERIMENT_NAME)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attribute.start_time DESC"])
        
        if runs.empty:
            logging.warning("No MLflow runs found. Using local artifacts.")
            return

        latest_run_id = runs.iloc[0]['run_id']
        
        # Downloading artifacts
        local_dir = mlflow.artifacts.download_artifacts(run_id=latest_run_id, artifact_path="models", dst_path=Config.MODEL_DIR)
        logging.info(f"Downloaded latest models from MLflow run {latest_run_id}")
        
    except Exception as e:
        logging.error(f"MLflow registry fetch failed: {e}. Falling back to local.")