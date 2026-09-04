import os, sys, time
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines import PopularityBaseline, ItemItemCFBaseline
from src.pipeline.recommendation_pipeline import RecSysPipeline
from src.data.loader import DataLoader
from src.evaluation.metrics import RecSysEvaluator

def main():
    print("🔄 Loading data for baseline evaluation...")
    loader = DataLoader()
    _, _, trans = loader.load_raw_data()
    trans['t_dat'] = pd.to_datetime(trans['t_dat'])
    
    split_date = "2019-09-01"
    train = trans[trans['t_dat'] < split_date]
    val = trans[trans['t_dat'] >= split_date]
    
    print("🛠️ Fitting baselines...")
    pop_model = PopularityBaseline()
    pop_model.fit(train)
    
    cf_model = ItemItemCFBaseline()
    cf_model.fit(train)
    
    print("🚀 Loading Production Pipeline for comparison...")
    pipeline = RecSysPipeline(use_mlflow=False)
    pipeline.load_artifacts()
    
    active_users = val.groupby('customer_id').size().nlargest(500).index.tolist()
    results = []
    
    for name, model in [("Popularity", pop_model), ("Item-Item CF", cf_model), ("Two-Tower + CatBoost", pipeline)]:
        ndcgs, recalls = [], []
        start_time = time.time()
        
        for uid in active_users:
            actual = val[val['customer_id'] == uid]['article_id'].tolist()
            if not actual: continue
            
            if name == "Two-Tower + CatBoost":
                preds = model.generate(uid, top_k=12)
            else:
                preds = model.recommend(uid, top_k=12)
                
            ndcgs.append(RecSysEvaluator.ndcg_at_k(actual, preds, k=12))
            recalls.append(RecSysEvaluator.recall_at_k(actual, preds, k=12))
            
        latency = (time.time() - start_time) / len(active_users) * 1000
        results.append({
            "model": name, "NDCG@12": np.mean(ndcgs), 
            "Recall@12": np.mean(recalls), "latency_ms_per_user": latency
        })
        
    df_res = pd.DataFrame(results)
    print("\n=== BASELINE COMPARISON ===")
    print(df_res)
    os.makedirs("reports", exist_ok=True)
    df_res.to_csv("reports/baseline_metrics.csv", index=False)

if __name__ == "__main__":
    main()