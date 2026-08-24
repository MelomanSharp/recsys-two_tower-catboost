from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np

class CatBoostRanker:
    def __init__(self, iterations=1000, lr=0.05):
        self.model = None
        self.iterations = iterations
        self.lr = lr
        
    def train(self, X, y, group_ids, categorical_features):
        """Trains a CatBoost model optimized via the business-standard YetiRank framework."""
        # Format dataset pool optimized explicitly for ranking structures
        train_pool = Pool(
            data=X,
            label=y,
            group_id=group_ids,
            cat_features=categorical_features
        )
        
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.lr,
            loss_function="YetiRank",
            custom_metric=["NDCG:top=10", "Recall:top=100"],
            random_seed=42,
            verbose=100
        )
        
        self.model.fit(train_pool)
        return self.model
        
    def predict_scores(self, X, cat_features):
        """Extracts raw likelihood prediction signals for multi-candidate re-ranking."""
        pool = Pool(data=X, cat_features=cat_features)
        return self.model.predict_proba(pool)[:, 1]
