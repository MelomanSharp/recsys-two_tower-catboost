import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

class FeatureEngineer:
    def __init__(self):
        self.user_encoders = {}
        self.item_encoders = {}

    def build_static_features(self, articles, customers, transactions):
        """Generates historical user behavioral profiles, item statistics, and cross-features."""
        print("   [1/6] Calculating user spend...")
        # --- USER FEATURES ---
        user_spend = transactions.groupby("customer_id").agg(
            user_avg_price=("price", "mean"),
            user_total_spent=("price", "sum"),
            user_transaction_count=("price", "count")
        ).reset_index()

        print("   [2/6] Merging articles for user favorites...")
        articles_subset = articles[["article_id", "department_name", "colour_group_name"]].copy()
        user_favs = transactions[["customer_id", "article_id"]].merge(articles_subset, on="article_id", how="left")

        print("   [3/6] Calculating favorite department (vectorized)...")
        dept_counts = user_favs.groupby(["customer_id", "department_name"]).size().reset_index(name="count")
        dept_counts = dept_counts.sort_values(["customer_id", "count"], ascending=[True, False])
        fav_dept = dept_counts.drop_duplicates(subset=["customer_id"])[["customer_id", "department_name"]].rename(columns={"department_name": "user_fav_department"})

        print("   [4/6] Calculating favorite color (vectorized)...")
        color_counts = user_favs.groupby(["customer_id", "colour_group_name"]).size().reset_index(name="count")
        color_counts = color_counts.sort_values(["customer_id", "count"], ascending=[True, False])
        fav_color = color_counts.drop_duplicates(subset=["customer_id"])[["customer_id", "colour_group_name"]].rename(columns={"colour_group_name": "user_fav_color"})

        print("   [5/6] Merging user features...")
        user_features = customers.merge(user_spend, on="customer_id", how="left")
        user_features = user_features.merge(fav_dept, on="customer_id", how="left")
        user_features = user_features.merge(fav_color, on="customer_id", how="left")

        # Cold Start mitigation
        user_features["user_avg_price"] = user_features["user_avg_price"].fillna(transactions["price"].mean())
        user_features["user_total_spent"] = user_features["user_total_spent"].fillna(0.0)
        user_features["user_transaction_count"] = user_features["user_transaction_count"].fillna(0)
        user_features["user_fav_department"] = user_features["user_fav_department"].fillna("UNKNOWN")
        user_features["user_fav_color"] = user_features["user_fav_color"].fillna("UNKNOWN")

        #  Recency (Days since last purchase)
        max_date = transactions['t_dat'].max()
        recency = transactions.groupby('customer_id')['t_dat'].max().reset_index()
        recency['days_since_last_purchase'] = (max_date - recency['t_dat']).dt.days
        user_features = user_features.merge(recency[['customer_id', 'days_since_last_purchase']], on='customer_id', how='left')
        user_features['days_since_last_purchase'] = user_features['days_since_last_purchase'].fillna(recency['days_since_last_purchase'].max() if not recency.empty else 365)

        print("   [6/6] Calculating item features...")
        # --- ITEM FEATURES ---
        item_stats = transactions.groupby("article_id").agg(
            item_popularity_30d=("price", "count"),
            item_avg_price=("price", "mean")
        ).reset_index()
        item_features = articles.merge(item_stats, on="article_id", how="left")
        item_features["item_popularity_30d"] = item_features["item_popularity_30d"].fillna(0)
        item_features["item_avg_price"] = item_features["item_avg_price"].fillna(item_features["item_avg_price"].median())

        #  Trendiness (Pop 7d / Pop 30d)
        t7 = max_date - pd.Timedelta(days=7)
        t30 = max_date - pd.Timedelta(days=30)
        pop_7d = transactions[transactions['t_dat'] >= t7].groupby('article_id').size()
        pop_30d = transactions[(transactions['t_dat'] >= t30) & (transactions['t_dat'] < t7)].groupby('article_id').size()
        trend_df = pd.DataFrame({'pop_7d': pop_7d, 'pop_30d': pop_30d}).fillna(0)
        trend_df['item_trendiness'] = trend_df['pop_7d'] / (trend_df['pop_30d'] + 1)
        item_features = item_features.merge(trend_df[['item_trendiness']], left_on='article_id', right_index=True, how='left')
        item_features['item_trendiness'] = item_features['item_trendiness'].fillna(0)

        # Item Lifetime
        lifetime = transactions.groupby('article_id')['t_dat'].min().reset_index(name='min_date')
        lifetime['item_lifetime'] = (max_date - lifetime['min_date']).dt.days
        lifetime = lifetime[['article_id', 'item_lifetime']]
        item_features = item_features.merge(lifetime, on='article_id', how='left')
        item_features['item_lifetime'] = item_features['item_lifetime'].fillna(0)

        #  Price Tier (Budget, Standard, Premium)
        try:
            item_features['price_tier'] = pd.qcut(item_features['item_avg_price'].rank(method='first'), q=3, labels=['Budget', 'Standard', 'Premium'])
        except ValueError:
            item_features['price_tier'] = 'Standard'

        return user_features, item_features

    @staticmethod
    def generate_cross_features(user_df, item_df):
        """Computes explicit runtime interaction signals between users and candidate items."""
        df = user_df.merge(item_df, on="join_key") 
        df["is_same_department"] = (df["user_fav_department"] == df["department_name"]).astype(int)
        df["is_same_color"] = (df["user_fav_color"] == df["colour_group_name"]).astype(int)
        df["price_ratio"] = df["item_avg_price"] / (df["user_avg_price"] + 1e-6)
        df["price_diff"] = np.abs(df["item_avg_price"] - df["user_avg_price"])
        return df

    @staticmethod
    def calculate_psi(expected, actual, bins=10):
        """Calculates the Population Stability Index to quantify data drift metrics."""
        # Bins are now built based on the EXPECTED (reference) distribution
        bins_edges = np.histogram_bin_edges(expected, bins=bins)
        
        expected_pct = np.histogram(expected, bins=bins_edges)[0] / len(expected)
        actual_pct = np.histogram(actual, bins=bins_edges)[0] / len(actual)
        
        # Avoid zero division
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)