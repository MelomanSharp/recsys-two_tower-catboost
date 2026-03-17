import numpy as np
import pandas as pd

class RecSysEvaluator:
    @staticmethod
    def recall_at_k(actual, predicted, k=100):
        """Calculates retrieval level coverage capability via Recall@K."""
        if not actual:
            return 0.0
        predicted_k = set(predicted[:k])
        actual_set = set(actual)
        return len(predicted_k.intersection(actual_set)) / min(len(actual_set), k)

    @staticmethod
    def ndcg_at_k(actual, predicted, k=10):
        """Measures structural relevance ranking accuracy using NDCG@K."""
        if not actual:
            return 0.0
        predicted_k = predicted[:k]
        dcg = 0.0
        for i, p in enumerate(predicted_k):
            if p in actual:
                dcg += 1.0 / np.log2(i + 2)
                
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual), k))])
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def run_offline_replay_evaluation(historical_logs, policy_recommendations):
        """Simulates production A/B-tests via offline replay strategy evaluation logs."""
        matched_conversions = 0
        total_eval_sessions = len(historical_logs)
        
        for session_id, actual_buys in historical_logs.items():
            recommended = policy_recommendations.get(session_id, [])
            # Intersect with top structural placements
            if any(item in actual_buys for item in recommended[:10]):
                matched_conversions += 1
                
        estimated_ctr_uplift = (matched_conversions / total_eval_sessions) if total_eval_sessions > 0 else 0.0
        return estimated_ctr_uplift
