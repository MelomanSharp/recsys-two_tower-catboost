import os
import pandas as pd
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.config import Config

def main():
    print("🔄 Loading raw data...")
    loader = DataLoader()
    art, cust, trans = loader.load_raw_data()
    art, cust, trans = loader.preprocess_data(art, cust, trans)
    
    print("🛠️ Building features...")
    engineer = FeatureEngineer()
    user_feat, item_feat = engineer.build_static_features(art, cust, trans)
    
    # Save to processed/daily
    out_dir = os.path.join(Config.PROCESSED_DATA_DIR, "daily")
    os.makedirs(out_dir, exist_ok=True)
    
    trans.to_parquet(os.path.join(out_dir, "transactions.parquet"), index=False)
    art.to_parquet(os.path.join(out_dir, "articles.parquet"), index=False)
    cust.to_parquet(os.path.join(out_dir, "customers.parquet"), index=False)
    user_feat.to_parquet(os.path.join(out_dir, "user_features.parquet"), index=False)
    item_feat.to_parquet(os.path.join(out_dir, "item_features.parquet"), index=False)
    
    print(f"✅ ETL Complete. Files saved to {out_dir}")

if __name__ == "__main__":
    main()