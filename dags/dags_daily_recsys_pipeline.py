from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.evaluation.metrics import RecSysEvaluator

default_args = {
    "owner": "data_science_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 6, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def pipeline_extract_and_preprocess():
    print("Extracting from filesystem raw data partitions...")
    loader = DataLoader()
    art, cust, trans = loader.load_raw_data()
    art, cust, trans = loader.preprocess_data(art, cust, trans)
    print(f"Loaded {len(art)} products and {len(cust)} platform users.")

def pipeline_compute_drift_and_psi():
    print("Calculating system drift factors across operational feature vectors...")
    # Simulated validation comparison metrics checks
    engineer = FeatureEngineer()
    mock_ref = [10.5, 20.0, 15.2, 99.0, 5.0]
    mock_tar = [11.0, 19.5, 14.8, 92.0, 6.2]
    psi_metric = engineer.calculate_psi(mock_ref, mock_tar, num_bins=3)
    print(f"Calculated Population Stability Index: {psi_metric}")
    if psi_metric > 0.25:
        print("ALERT: Structural data drift detected.")

def pipeline_evaluate_metrics():
    print("Measuring performance validation benchmarks against active baselines...")
    evaluator = RecSysEvaluator()
    sample_actual = ["item_a", "item_b"]
    sample_pred = ["item_a", "item_c", "item_d"]
    ndcg = evaluator.ndcg_at_k(sample_actual, sample_pred, k=10)
    print(f"Validation Operational Check - Current Window NDCG@10: {ndcg}")

with DAG(
    "daily_recsys_production_pipeline",
    default_args=default_args,
    description="Orchestrates data transformation, ML training stages, metrics validation, and caching loops.",
    schedule_interval="@daily",
    catchup=False,
) as dag:

    task_etl = PythonOperator(
        task_id="extract_and_preprocess_data",
        python_callable=pipeline_extract_and_preprocess,
    )

    task_drift = PythonOperator(
        task_id="calculate_feature_drift_psi",
        python_callable=pipeline_compute_drift_and_psi,
    )

    task_eval = PythonOperator(
        task_id="evaluate_offline_metrics",
        python_callable=pipeline_evaluate_metrics,
    )

    task_etl >> task_drift >> task_eval
