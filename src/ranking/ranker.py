import pandas as pd
import numpy as np
import gc
from typing import Optional, List
from src.models.catboost_ranker import CatBoostRanker


class MLRanker:
    """Two-stage ranker using CatBoost with YetiRank on top of retrieval candidates."""

    def __init__(self, user_features: pd.DataFrame, item_features: pd.DataFrame):
        self.user_features = user_features
        self.item_features = item_features
        self.ranker = CatBoostRanker(iterations=500, lr=0.05)
        self.is_trained = False

        self.known_cat_features = [
            "user_fav_department", "user_fav_color",
            "department_name", "colour_group_name", "section_name",
        ]

    # ------------------------------------------------------------------ #
    #                      TRAINING                                       #
    # ------------------------------------------------------------------ #
    def train(self, transactions: pd.DataFrame, val_split_date: str = "2019-09-01", sample_frac: float = 0.2, n_neg: int = 2) -> dict:
        """
        Builds a pairwise training set (positive = purchased, negative = sampled)
        and fits CatBoost with YetiRank. Optimized for low memory footprint.
        """
        print("🧠 Building ranking training set...")
        transactions = transactions.copy()
        transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])

        train_tx = transactions[transactions["t_dat"] < val_split_date]
        val_tx = transactions[transactions["t_dat"] >= val_split_date]

        # === FIX 1: Sample users. ===
        if sample_frac < 1.0:
            unique_users = train_tx["customer_id"].unique()
            sampled_users = np.random.choice(unique_users, size=int(len(unique_users) * sample_frac), replace=False)
            train_tx = train_tx[train_tx["customer_id"].isin(sampled_users)].copy()
            print(f"   Sampled {len(sampled_users)} users ({sample_frac*100:.0f}%) to prevent OOM.")

        # === FIX 2: Generate ID pairs only. ===
        print("   Generating positive pairs...")
        pos_pairs = train_tx[["customer_id", "article_id"]].drop_duplicates()
        pos_pairs["target"] = 1

        print(f"   Generating {n_neg} negative samples per positive...")
        neg_pairs = self._sample_negatives_fast(pos_pairs, n_neg=n_neg)

        print("   Concatenating pairs...")
        all_pairs = pd.concat([pos_pairs, neg_pairs], ignore_index=True)
        
        del pos_pairs, neg_pairs, train_tx, transactions
        gc.collect()

        # === FIX 3: Attach features once. ===
        print("   Attaching features to all pairs...")
        training_df = self._attach_features(all_pairs)
        del all_pairs
        gc.collect()

        print("   Splitting train/val...")
        unique_users = training_df["customer_id"].unique()
        np.random.seed(42)
        val_users = np.random.choice(unique_users, size=int(0.3 * len(unique_users)), replace=False)

        train_df = training_df[~training_df["customer_id"].isin(val_users)]
        val_df = training_df[training_df["customer_id"].isin(val_users)]

        # === FIX 5: Sort by customer_id so groups are contiguous for CatBoost. ===
        print("   Sorting by customer_id for CatBoost grouping requirement...")
        train_df = train_df.sort_values("customer_id")
        val_df = val_df.sort_values("customer_id")

        print(f"   Train pairs: {len(train_df)} | Val pairs: {len(val_df)}")

        cols_to_drop = ["customer_id", "article_id", "target"]
        X_train = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
        y_train = train_df["target"].values
        groups_train = train_df["customer_id"].astype(str).values

        X_val = val_df.drop(columns=[c for c in cols_to_drop if c in val_df.columns])
        y_val = val_df["target"].values
        groups_val = val_df["customer_id"].astype(str).values

        del train_df, val_df, training_df
        gc.collect()

        # === FIX 4: Detect categorical features dynamically. ===
        cat_features = [col for col in X_train.columns if X_train[col].dtype.name in ('object', 'category')]
        print(f"   Detected {len(cat_features)} categorical features for CatBoost: {cat_features}")

        self.ranker.train(
            X=X_train, y=y_train, group_ids=groups_train,
            categorical_features=cat_features,
        )
        self.is_trained = True

        self.cat_features = cat_features

        val_metrics = self._evaluate_on_holdout(val_tx, k=12)
        return {
            "train_pairs": len(X_train),
            "val_pairs": len(X_val),
            "val_ndcg@12": val_metrics["ndcg@12"],
            "val_recall@12": val_metrics["recall@12"],
        }

    def _sample_negatives_fast(self, positive_pairs: pd.DataFrame, n_neg: int = 2) -> pd.DataFrame:
        """Perform fast vectorized negative sampling."""
        if positive_pairs.empty:
            return pd.DataFrame(columns=["customer_id", "article_id", "target"])
            
        all_items = self.item_features["article_id"].values
        
        neg_pairs = positive_pairs.loc[positive_pairs.index.repeat(n_neg)].copy()
        neg_pairs["article_id"] = np.random.choice(all_items, size=len(neg_pairs), replace=True)
        neg_pairs["target"] = 0
        
        pos_flags = positive_pairs[["customer_id", "article_id"]].drop_duplicates()
        pos_flags["is_pos"] = True
        
        neg_pairs = neg_pairs.merge(pos_flags, on=["customer_id", "article_id"], how="left")
        neg_pairs = neg_pairs[neg_pairs["is_pos"].isna()].drop(columns=["is_pos"])
        
        return neg_pairs

    def _attach_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        u = self.user_features
        i = self.item_features
        
        df = pairs.merge(u, on="customer_id", how="left").merge(i, on="article_id", how="left")

        # === Compare categorical values safely as strings. ===
        if "user_fav_department" in df.columns and "department_name" in df.columns:
            df["is_same_department"] = (
                df["user_fav_department"].astype(str) == df["department_name"].astype(str)
            ).astype(np.int8)
            
        if "user_fav_color" in df.columns and "colour_group_name" in df.columns:
            df["is_same_color"] = (
                df["user_fav_color"].astype(str) == df["colour_group_name"].astype(str)
            ).astype(np.int8)
            
        if "item_avg_price" in df.columns and "user_avg_price" in df.columns:
            df["price_ratio"] = (df["item_avg_price"] / (df["user_avg_price"] + 1e-6)).astype(np.float32)
            df["price_diff"] = np.abs(df["item_avg_price"] - df["user_avg_price"]).astype(np.float32)

        # === Downcast numeric types. ===
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["int64"]).columns:
            if col not in ["is_same_department", "is_same_color"]:
                df[col] = pd.to_numeric(df[col], downcast="integer")
                
        return df

    def _evaluate_on_holdout(self, holdout_tx: pd.DataFrame, k: int = 12) -> dict:
        """Offline evaluation on the holdout time window."""
        from src.evaluation.metrics import RecSysEvaluator
        active_users = holdout_tx.groupby("customer_id").size().nlargest(1000).index.tolist()

        ndcgs, recalls = [], []
        for uid in active_users:
            candidates = holdout_tx[holdout_tx["customer_id"] == uid]["article_id"].tolist()
            if not candidates:
                continue
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

        # Compare categorical values safely as strings.
        if "user_fav_department" in df.columns and "department_name" in df.columns:
            df["is_same_department"] = (df["user_fav_department"].astype(str) == df["department_name"].astype(str)).astype(np.int8)
        if "user_fav_color" in df.columns and "colour_group_name" in df.columns:
            df["is_same_color"] = (df["user_fav_color"].astype(str) == df["colour_group_name"].astype(str)).astype(np.int8)
        if "item_avg_price" in df.columns and "user_avg_price" in df.columns:
            df["price_ratio"] = (df["item_avg_price"] / (df["user_avg_price"] + 1e-6)).astype(np.float32)
            df["price_diff"] = np.abs(df["item_avg_price"] - df["user_avg_price"]).astype(np.float32)
            
        return df

    def rank_candidates(self, df_candidates: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = ["customer_id", "article_id"]
        X = df_candidates.drop(columns=[c for c in cols_to_drop if c in df_candidates.columns])

        if not self.is_trained:
            df_candidates["score"] = (
                df_candidates.get("is_same_department", 0) * 0.5
                + df_candidates.get("item_popularity_30d", 0) / 1000
            )
        else:
            cat_feats = [col for col in X.columns if X[col].dtype.name in ('object', 'category')]
            scores = self.ranker.predict_scores(X, cat_feats)
            df_candidates["score"] = scores

        return df_candidates.sort_values("score", ascending=False)