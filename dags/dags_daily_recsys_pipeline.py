from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import (
    PythonOperator,
    BranchPythonOperator,
)
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.exceptions import AirflowSkipException
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.evaluation.metrics import RecSysEvaluator


REFERENCE_WINDOW_DAYS = 30
TARGET_WINDOW_DAYS = 7




# ---------------- TASK FUNCTIONS ----------------

def extract_and_preprocess(**context):
    """ETL: load latest partition, preprocess, persist to processed/."""
    print("Extracting raw data partitions...")
    loader = DataLoader()
    art, cust, trans = loader.load_raw_data()
    art, cust, trans = loader.preprocess_data(art, cust, trans)

    # Save snapshot for downstream tasks
    out_dir = os.path.join(os.path.dirname(__file__), "../data/processed/daily")
    os.makedirs(out_dir, exist_ok=True)
    trans.to_parquet(os.path.join(out_dir, "transactions.parquet"))
    art.to_parquet(os.path.join(out_dir, "articles.parquet"))
    cust.to_parquet(os.path.join(out_dir, "customers.parquet"))

    print(f"Persisted snapshot: {len(trans)} transactions, {len(art)} articles, {len(cust)} users")
    return {"transactions": len(trans), "articles": len(art)}


def compute_drift_and_psi(**context):
    """Computes PSI based on EDA findings: price (threshold 0.15) and trendiness."""
    base_dir = os.path.join(os.path.dirname(__file__), "../data/processed/daily")
    trans = pd.read_parquet(os.path.join(base_dir, "transactions.parquet"))
    trans["t_dat"] = pd.to_datetime(trans["t_dat"])
    
    max_date = trans["t_dat"].max()
    target_start = max_date - timedelta(days=TARGET_WINDOW_DAYS)
    ref_start = target_start - timedelta(days=REFERENCE_WINDOW_DAYS)
    
    ref = trans[(trans["t_dat"] >= ref_start) & (trans["t_dat"] < target_start)]
    tar = trans[(trans["t_dat"] >= target_start) & (trans["t_dat"] <= max_date)]
    
    if ref.empty or tar.empty:
        raise AirflowSkipException

    engineer = FeatureEngineer()
    
    # 1. Price PSI (Порог 0.15, так как 0.25 недостижим и бесполезен)
    psi_price = engineer.calculate_psi(ref["price"].values, tar["price"].values, num_bins=10)
    print(f"PSI[price] = {psi_price:.4f} (Threshold: 0.15)")
    context["ti"].xcom_push("psi_price", float(psi_price))
    
    # 2. Popularity / Trendiness PSI (То, что РЕАЛЬНО дрейфует)
    ref_pop = ref.groupby("article_id").size()
    tar_pop = tar.groupby("article_id").size()
    common = ref_pop.index.intersection(tar_pop.index)
    
    psi_trend = 0.0
    if len(common) > 10:
        psi_trend = engineer.calculate_psi(ref_pop.loc[common].values, tar_pop.loc[common].values, num_bins=10)
    print(f"PSI[item_trendiness] = {psi_trend:.4f} (Threshold: 0.20)")
    context["ti"].xcom_push("psi_trendiness", float(psi_trend))

    # 3. Решение о ретрейнинге на основе выводов из EDA
    # Ловим сезонность (цена > 0.15) или сдвиг ассортимента (тренды > 0.20)
    if psi_price > 0.15 or psi_trend > 0.20:
        return "trigger_retrain"
    
    return "no_action"

def evaluate_offline_metrics(**context):
    """Computes NDCG@K on a real recent holdout window."""
    base_dir = os.path.join(os.path.dirname(__file__), "../data/processed/daily")
    trans = pd.read_parquet(os.path.join(base_dir, "transactions.parquet"))
    trans["t_dat"] = pd.to_datetime(trans["t_dat"])

    holdout_start = trans["t_dat"].max() - timedelta(days=TARGET_WINDOW_DAYS)
    holdout = trans[trans["t_dat"] >= holdout_start]

    evaluator = RecSysEvaluator()
    # For each user: top-N by popularity vs their actual next purchases
    popularity = trans[trans["t_dat"] < holdout_start].groupby("article_id").size().sort_values(ascending=False)
    top_pop_items = popularity.head(50).index.tolist()

    sampled_users = holdout["customer_id"].unique()[:500]
    ndcgs, recalls = [], []
    for uid in sampled_users:
        actual = holdout[holdout["customer_id"] == uid]["article_id"].tolist()
        if not actual:
            continue
        ndcgs.append(evaluator.ndcg_at_k(actual, top_pop_items, k=12))
        recalls.append(evaluator.recall_at_k(actual, top_pop_items, k=12))

    metrics = {
        "ndcg@12": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "recall@12": float(np.mean(recalls)) if recalls else 0.0,
        "evaluated_users": len(sampled_users),
    }
    print(f"Baseline popularity metrics: {metrics}")
    context["ti"].xcom_push("offline_metrics", metrics)
    return metrics

def build_features(**context):
    """Generates and persists engineered user and item features."""
    print("Building engineered features...")
    base_dir = os.path.join(os.path.dirname(__file__), "../data/processed/daily")
    trans = pd.read_parquet(os.path.join(base_dir, "transactions.parquet"))
    art = pd.read_parquet(os.path.join(base_dir, "articles.parquet"))
    cust = pd.read_parquet(os.path.join(base_dir, "customers.parquet"))
    
    engineer = FeatureEngineer()
    user_feat, item_feat = engineer.build_static_features(art, cust, trans)
    
    user_feat.to_parquet(os.path.join(base_dir, "user_features.parquet"))
    item_feat.to_parquet(os.path.join(base_dir, "item_features.parquet"))
    print(f"Persisted features: {len(user_feat)} users, {len(item_feat)} items")



def trigger_retrain_callable(**context):
    print("🚨 Significant drift detected — flagging model for retraining.")
    flag_path = os.path.join(os.path.dirname(__file__), "../data/retrain_flag.txt")
    with open(flag_path, "w") as f:
        f.write(datetime.utcnow().isoformat())


def no_action_callable(**context):
    print("✅ No drift — model remains in production.")


# ---------------- DAG ----------------

default_args = {
    "owner": "data_science_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 6, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "daily_recsys_production_pipeline",
    default_args=default_args,
    description="Production ETL + drift monitoring + offline evaluation",
    schedule_interval="@daily",
    catchup=False,
) as dag:

    task_etl = PythonOperator(
        task_id="extract_and_preprocess_data",
        python_callable=extract_and_preprocess,
    )

    task_fe = PythonOperator(task_id="build_features", python_callable=build_features)
    

    task_drift = BranchPythonOperator(
        task_id="calculate_feature_drift_psi",
        python_callable=compute_drift_and_psi,
    )

    task_retrain = PythonOperator(
        task_id="trigger_retrain",
        python_callable=trigger_retrain_callable,
    )

    task_no_action = PythonOperator(
        task_id="no_action",
        python_callable=no_action_callable,
    )

    task_eval = PythonOperator(
        task_id="evaluate_offline_metrics",
        python_callable=evaluate_offline_metrics,
        trigger_rule="none_failed_min_one_success",
    )

    task_etl >> task_fe >> task_drift >> [task_retrain, task_no_action] >> task_eval