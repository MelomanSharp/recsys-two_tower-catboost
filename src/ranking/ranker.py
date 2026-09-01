import pandas as pd
import numpy as np
from typing import Optional, List
from src.models.catboost_ranker import CatBoostRanker


class MLRanker:
    """Two-stage ranker using CatBoost with YetiRank on top of retrieval candidates."""

    def __init__(self, user_features: pd.DataFrame, item_features: pd.DataFrame):
        self.user_features = user_features
        self.item_features = item_features
        self.ranker = CatBoostRanker(iterations=500, lr=0.05)
        self.is_trained = False

        self.cat_features = [
            "user_fav_department", "user_fav_color",
            "department_name", "colour_group_name", "section_name",
        ]
        self.cat_features = [
            c for c in self.cat_features
            if c in self.user_features.columns or c in self.item_features.columns
        ]

    # ------------------------------------------------------------------ #
    #                      TRAINING                                       #
    # ------------------------------------------------------------------ #
    def train(self, transactions: pd.DataFrame, val_split_date: str = "2019-09-01") -> dict:
        """
        Builds a pairwise training set (positive = purchased, negative = sampled)
        and fits CatBoost with YetiRank.
        
        Returns a dict of training metrics for MLflow logging.
        """
        print("🧠 Building ranking training set...")
        transactions = transactions.copy()
        transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])

        # Temporal split — train on past, validate on future
        train_tx = transactions[transactions["t_dat"] < val_split_date]
        val_tx = transactions[transactions["t_dat"] >= val_split_date]

        # Positive pairs
        pos_train = self._build_feature_pairs(train_tx)
        pos_train["target"] = 1

        # Negative sampling: 4 negatives per positive from popular items
        negatives = self._sample_negatives(pos_train, n_neg=4)
        training_df = pd.concat([pos_train, negatives], ignore_index=True)

        # Train-val split (70/30 by users, to avoid leakage)
        unique_users = training_df["customer_id"].unique()
        np.random.seed(42)
        val_users = np.random.choice(unique_users, size=int(0.3 * len(unique_users)), replace=False)

        train_df = training_df[~training_df["customer_id"].isin(val_users)]
        val_df = training_df[training_df["customer_id"].isin(val_users)]

        print(f"   Train pairs: {len(train_df)} | Val pairs: {len(val_df)}")

        # Fit
        cols_to_drop = ["customer_id", "article_id"]
        X_train = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
        y_train = train_df["target"].values
        groups_train = train_df["customer_id"].astype(str).values

        X_val = val_df.drop(columns=[c for c in cols_to_drop if c in val_df.columns])
        y_val = val_df["target"].values
        groups_val = val_df["customer_id"].astype(str).values

        self.ranker.train(
            X=X_train, y=y_train, group_ids=groups_train,
            categorical_features=[c for c in self.cat_features if c in X_train.columns],
        )
        self.is_trained = True

        # Validation ranking metrics on held-out period
        val_metrics = self._evaluate_on_holdout(val_tx, k=12)
        return {
            "train_pairs": len(train_df),
            "val_pairs": len(val_df),
            "val_ndcg@12": val_metrics["ndcg@12"],
            "val_recall@12": val_metrics["recall@12"],
        }

    def _build_feature_pairs(self, tx_df: pd.DataFrame) -> pd.DataFrame:
        """Joins transactions with user & item feature tables."""
        positives = tx_df[["customer_id", "article_id"]].drop_duplicates()
        return self._attach_features(positives)

    def _sample_negatives(self, positive_df: pd.DataFrame, n_neg: int = 4) -> pd.DataFrame:
        """Vectorized negative sampling for speed (replaces slow iterrows)."""
        if positive_df.empty:
            return pd.DataFrame(columns=["customer_id", "article_id", "target"])
            
        all_items = self.item_features["article_id"].values
        
        user_pos_counts = positive_df.groupby("customer_id").size()
        
        neg_dfs = []
        for user_id, count in user_pos_counts.items():
            num_samples = n_neg * count
            sampled_items = np.random.choice(all_items, size=num_samples, replace=True)
            
            neg_dfs.append(pd.DataFrame({
                "customer_id": np.full(num_samples, user_id),
                "article_id": sampled_items,
                "target": 0
            }))
            
        negatives = pd.concat(neg_dfs, ignore_index=True)
        
        positives_flag = positive_df[["customer_id", "article_id"]].drop_duplicates()
        positives_flag["is_positive"] = True
        
        merged = negatives.merge(positives_flag, on=["customer_id", "article_id"], how="left")
        negatives = merged[merged["is_positive"].isna()].drop(columns=["is_positive"])
        
        return self._attach_features(negatives[["customer_id", "article_id"]])
    

    def _attach_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        u = self.user_features
        i = self.item_features
        df = pairs.merge(u, on="customer_id", how="left").merge(i, on="article_id", how="left")

        # Cross features
        if {"user_fav_department", "department_name"}.issubset(df.columns):
            df["is_same_department"] = (df["user_fav_department"] == df["department_name"]).astype(int)
        if {"user_fav_color", "colour_group_name"}.issubset(df.columns):
            df["is_same_color"] = (df["user_fav_color"] == df["colour_group_name"]).astype(int)
        if {"item_avg_price", "user_avg_price"}.issubset(df.columns):
            df["price_ratio"] = df["item_avg_price"] / (df["user_avg_price"] + 1e-6)
            df["price_diff"] = np.abs(df["item_avg_price"] - df["user_avg_price"])
        return df

    def _evaluate_on_holdout(self, holdout_tx: pd.DataFrame, k: int = 12) -> dict:
        """Offline evaluation on the holdout time window."""
        from src.evaluation.metrics import RecSysEvaluator
        # Sample 1000 active users for speed
        active_users = holdout_tx.groupby("customer_id").size().nlargest(1000).index.tolist()

        ndcgs, recalls = [], []
        for uid in active_users:
            candidates = holdout_tx[holdout_tx["customer_id"] == uid]["article_id"].tolist()
            if not candidates:
                continue
            # Expand candidates with 50 random items (simulate retrieval output)
            random_extra = np.random.choice(self.item_features["article_id"].values, size=50, replace=False).tolist()
            pool = list(set(candidates + random_extra))

            df_cand = self.prepare_ranking_dataset(uid, pool)
            ranked = self.rank_candidates(df_cand)

            actual = candidates
            predicted = ranked["article_id"].tolist()
            ndcgs.append(RecSysEvaluator.ndcg_at_k(actual, predicted, k=k))
            recalls.append(RecSysEvaluator.recall_at_k(actual, predicted, k=k))

        return {
            "ndcg@12": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "recall@12": float(np.mean(recalls)) if recalls else 0.0,
        }

    # ------------------------------------------------------------------ #
    #                      INFERENCE                                      #
    # ------------------------------------------------------------------ #
    def prepare_ranking_dataset(self, user_id: str, candidate_items: list) -> pd.DataFrame:
        u_feat = self.user_features[self.user_features["customer_id"] == user_id].copy()
        if u_feat.empty:
            u_feat = self.user_features.iloc[[0]].copy()
            u_feat["customer_id"] = user_id

        i_feat = self.item_features[self.item_features["article_id"].isin(candidate_items)].copy()

        missing_items = set(candidate_items) - set(i_feat["article_id"])
        if missing_items:
            mock_df = pd.DataFrame({"article_id": list(missing_items)})
            for col in self.item_features.columns:
                if col != "article_id":
                    mock_df[col] = None
            i_feat = pd.concat([i_feat, mock_df], ignore_index=True)

        u_feat["join_key"] = 1
        i_feat["join_key"] = 1
        df = u_feat.merge(i_feat, on="join_key").drop("join_key", axis=1)

        if {"user_fav_department", "department_name"}.issubset(df.columns):
            df["is_same_department"] = (df["user_fav_department"] == df["department_name"]).astype(int)
        if {"user_fav_color", "colour_group_name"}.issubset(df.columns):
            df["is_same_color"] = (df["user_fav_color"] == df["colour_group_name"]).astype(int)
        if {"item_avg_price", "user_avg_price"}.issubset(df.columns):
            df["price_ratio"] = df["item_avg_price"] / (df["user_avg_price"] + 1e-6)
            df["price_diff"] = np.abs(df["item_avg_price"] - df["user_avg_price"])
        return df

    def rank_candidates(self, df_candidates: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = ["customer_id", "article_id"]
        X = df_candidates.drop(columns=[c for c in cols_to_drop if c in df_candidates.columns])

        if not self.is_trained:
            # Mock predict: heuristic fallback (same department + popularity)
            df_candidates["score"] = (
                df_candidates.get("is_same_department", 0) * 0.5
                + df_candidates.get("item_popularity_30d", 0) / 1000
            )
        else:
            cat_feats = [c for c in self.cat_features if c in X.columns]
            scores = self.ranker.predict_scores(X, cat_feats)
            df_candidates["score"] = scores

        return df_candidates.sort_values("score", ascending=False)