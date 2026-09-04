
from catboost import CatBoostRanker as CBCatBoostRanker, Pool
import pandas as pd
import numpy as np

class CatBoostRanker:
    def __init__(self, iterations=1000, lr=0.05):
        self.model = None
        self.iterations = iterations
        self.lr = lr
        
    def train(self, X, y, group_ids, categorical_features):
        """Trains a CatBoost model optimized via the business-standard YetiRank framework."""
        
        # Cast categorical features to the category dtype before creating the Pool.
        for col in categorical_features:
            if col in X.columns:
                X[col] = X[col].astype("category")
                
        train_pool = Pool(
            data=X,
            label=y,
            group_id=group_ids,
            cat_features=categorical_features
        )
        
        # Use CatBoostRanker instead of a classifier. PFound:top=10 is supported
        # here, while Recall:top=K is not a valid CatBoost ranking metric.
        self.model = CBCatBoostRanker(
            iterations=self.iterations,
            learning_rate=self.lr,
            loss_function="YetiRank",
            custom_metric=["NDCG:top=10", "PFound:top=10"],
            random_seed=42,
            verbose=100,
            depth=6,
            border_count=128,
        )
        
        self.model.fit(train_pool)
        return self.model
        
    def predict_scores(self, X, cat_features):
        """Extracts raw ranking prediction signals for multi-candidate re-ranking."""
        # Keep feature dtypes consistent during inference.
        for col in cat_features:
            if col in X.columns:
                X[col] = X[col].astype("category")
                
        pool = Pool(data=X, cat_features=cat_features)
        
        # CatBoostRanker returns raw relevance scores through predict,
        # rather than class probabilities through predict_proba.
        return self.model.predict(pool)