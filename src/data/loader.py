import pandas as pd
import numpy as np
import os
from src.config import Config

class DataLoader:
    def __init__(self):
        self.raw_dir = Config.RAW_DATA_DIR
        
    def load_raw_data(self):
        """Loads primary data tables from raw CSV files."""
        articles = pd.read_csv(os.path.join(self.raw_dir, "articles.csv"), dtype={"article_id": str})
        customers = pd.read_csv(os.path.join(self.raw_dir, "customers.csv"), dtype={"customer_id": str})
        transactions = pd.read_csv(os.path.join(self.raw_dir, "transactions_train.csv"), dtype={"customer_id": str, "article_id": str})
        
        # Parse date
        transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
        return articles, customers, transactions

    def preprocess_data(self, articles, customers, transactions):
        """Handles basic clean-ups, missing value imputations, and label formatting."""
        # Fill missing customer engagement attributes
        customers["FN"] = customers["FN"].fillna(0.0)
        customers["Active"] = customers["Active"].fillna(0.0)
        customers["club_member_status"] = customers["club_member_status"].fillna("UNKNOWN")
        customers["fashion_news_frequency"] = customers["fashion_news_frequency"].fillna("NONE")
        customers["age"] = customers["age"].fillna(customers["age"].median())
        
        # Clean article text profiles
        articles["detail_desc"] = articles["detail_desc"].fillna("")
        articles["prod_name"] = articles["prod_name"].fillna("Unknown")
        articles["text_profile"] = articles["prod_name"] + " " + articles["detail_desc"]
        
        return articles, customers, transactions

    def load_processed_data(self):
        """Loads engineered features from processed directory."""
        proc_dir = os.path.join(Config.PROCESSED_DATA_DIR, "daily")
        if not os.path.exists(proc_dir):
            raise FileNotFoundError(f"Processed data not found in {proc_dir}. Run ETL/Feature Engineering first.")
        
        user_features = pd.read_parquet(os.path.join(proc_dir, "user_features.parquet"))
        item_features = pd.read_parquet(os.path.join(proc_dir, "item_features.parquet"))
        transactions = pd.read_parquet(os.path.join(proc_dir, "transactions.parquet"))
        
        return user_features, item_features, transactions
