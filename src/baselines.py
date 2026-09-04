import pandas as pd
import numpy as np
from typing import List
import warnings

class PopularityBaseline:
    """Recommend the top-K items by total purchase count."""
    def __init__(self):
        self.popular_items = []

    def fit(self, transactions: pd.DataFrame):
        self.popular_items = transactions.groupby('article_id').size().sort_values(ascending=False).index.tolist()

    def recommend(self, user_id: str, top_k: int = 12) -> List[str]:
        return self.popular_items[:top_k]

class TwoTowerOnlyBaseline:
    """Use Two-Tower retrieval without the CatBoost ranking stage."""
    def __init__(self, searcher, user_encoder, indexer):
        self.searcher = searcher
        self.user_encoder = user_encoder
        self.indexer = indexer

    def recommend(self, user_id: str, top_k: int = 12) -> List[str]:
        if user_id in self.user_encoder.classes_:
            u_enc = self.user_encoder.transform([user_id])[0] + 1
            u_emb = self.indexer.get_user_embedding(u_enc)
            return self.searcher.get_candidates(u_emb, top_k=top_k)
        return []

class ItemItemCFBaseline:
    """Simplified item-item collaborative filtering based on co-occurrence.
    Use `implicit.ALS` for sparse matrices in production."""
    def __init__(self):
        self.item_popularity = []
        self.user_history = {}

    def fit(self, transactions: pd.DataFrame):
        self.item_popularity = transactions.groupby('article_id').size().sort_values(ascending=False).index.tolist()
        self.user_history = transactions.groupby('customer_id')['article_id'].apply(list).to_dict()
        warnings.warn("ItemItemCFBaseline uses simplified co-occurrence. Use `implicit` for ALS in prod.")

    def recommend(self, user_id: str, top_k: int = 12) -> List[str]:
        # Fall back to global popularity for cold start and this simple implementation.
        return self.item_popularity[:top_k]