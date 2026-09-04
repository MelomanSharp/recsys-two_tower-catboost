import pandas as pd
import numpy as np
import os, json
from src.evaluation.metrics import RecSysEvaluator

class PerformanceTracker:
    def __init__(self, history_file="models/performance_history.json"):
        self.history_file = history_file
        self.history = self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []

    def evaluate_and_track(self, model, holdout_tx: pd.DataFrame, k=12):
        active_users = holdout_tx.groupby('customer_id').size().nlargest(500).index.tolist()
        ndcgs, recalls = [], []
        
        for uid in active_users:
            actual = holdout_tx[holdout_tx['customer_id'] == uid]['article_id'].tolist()
            if not actual: continue
            try:
                preds = model.generate(uid, top_k=k)
                ndcgs.append(RecSysEvaluator.ndcg_at_k(actual, preds, k=k))
                recalls.append(RecSysEvaluator.recall_at_k(actual, preds, k=k))
            except Exception: continue
            
        metrics = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "ndcg@12": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "recall@12": float(np.mean(recalls)) if recalls else 0.0
        }
        self.history.append(metrics)
        self._save_history()
        return metrics

    def check_degradation(self, threshold=0.05):
        if len(self.history) < 2: return False
        latest = self.history[-1]["ndcg@12"]
        previous = np.mean([m["ndcg@12"] for m in self.history[:-1]])
        return (previous - latest) > threshold

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f)