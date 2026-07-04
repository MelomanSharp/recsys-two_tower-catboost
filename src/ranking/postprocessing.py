import pandas as pd
import numpy as np
from typing import List, Optional
from src.ranking.reranker import MaximalMarginalRelevance


class Postprocessor:
    """
    Applies business rules and diversity optimization (MMR) to ranked candidates.
    
    This is the final stage of the pipeline, responsible for:
    - Filtering out out-of-stock, unavailable, or restricted items
    - Price band filtering (e.g., don't recommend extremely expensive items to low-spending users)
    - Diversity optimization via Maximal Marginal Relevance to reduce popularity bias
    - Ensuring category coverage in the final Top-K
    """

    def __init__(self, item_embeddings: dict):
        """
        Args:
            item_embeddings: Dict mapping article_id (str) -> numpy array (embedding vector)
        """
        self.item_embeddings = item_embeddings
        self.mmr = MaximalMarginalRelevance()

    def apply_business_logic_and_diversity(
        self,
        ranked_df: pd.DataFrame,
        top_k: int = 10,
        diversity_lambda: float = 0.7,
        user_features: Optional[dict] = None,
        exclude_departments: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> List[str]:
        """
        Applies business filters + MMR diversity optimization.
        
        Args:
            ranked_df: DataFrame sorted by 'score' descending, with 'article_id' column
            top_k: Number of final recommendations
            diversity_lambda: 0 = max diversity, 1 = max relevance
            user_features: Optional user dict for personalized filters
            exclude_departments: Departments to exclude (e.g., out-of-season)
            min_price / max_price: Price filters (user-adaptive if user_features provided)
            
        Returns:
            List of article_ids ready to serve
        """
        df = ranked_df.copy()

        # ---------- 1. Business rules ----------
        # Filter by availability (example: drop items with zero 30d popularity in cold-start cases)
        if "item_popularity_30d" in df.columns:
            df = df[df["item_popularity_30d"] > 0]

        # Department-level business exclusion
        if exclude_departments and "department_name" in df.columns:
            df = df[~df["department_name"].isin(exclude_departments)]

        # Price-band filtering (personalized if user features provided)
        if user_features is not None and "user_avg_price" in user_features:
            # Adaptive range: roughly user's historical price ± 3x
            user_price = user_features["user_avg_price"]
            min_price = min_price or max(1.0, user_price * 0.3)
            max_price = max_price or user_price * 3.0

        if "item_avg_price" in df.columns:
            if min_price is not None:
                df = df[df["item_avg_price"] >= min_price]
            if max_price is not None:
                df = df[df["item_avg_price"] <= max_price]

        # Safety net: if filters wiped out everything, fall back to original top-k
        if df.empty:
            return ranked_df["article_id"].head(top_k).tolist()

        # ---------- 2. Diversity via MMR ----------
        candidates = df["article_id"].tolist()
        scores = df["score"].tolist()

        # Build embedding dict only for available candidates
        valid_candidates = []
        valid_scores = []
        valid_emb_map = {}
        for cand, score in zip(candidates, scores):
            if cand in self.item_embeddings:
                valid_candidates.append(cand)
                valid_scores.append(score)
                valid_emb_map[cand] = self.item_embeddings[cand]

        if not valid_candidates:
            # No embeddings available → return pure ranked top-k
            return candidates[:top_k]

        diverse_items = self.mmr.balance_diversity(
            candidates=valid_candidates,
            scores=np.array(valid_scores),
            item_embeddings=valid_emb_map,
            top_n=top_k,
            lmbda=diversity_lambda,
        )
        return diverse_items