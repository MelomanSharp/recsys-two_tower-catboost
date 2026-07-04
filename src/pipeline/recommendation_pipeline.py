import numpy as np
import pandas as pd
import os
from src.retrieval.indexer import TwoTowerIndexer
from src.retrieval.searcher import CandidateSearcher
from src.ranking.ranker import MLRanker
from src.ranking.postprocessing import Postprocessor
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.config import Config
from sklearn.preprocessing import LabelEncoder

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class RecSysPipeline:
    def __init__(self, use_mlflow: bool = True):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.indexer = None
        self.searcher = None
        self.ranker = None
        self.postprocessor = None
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self._item_features = None
        self._user_features = None

    def train(self):
        if self.use_mlflow:
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
            mlflow.start_run(run_name="full_pipeline_training")
            
        print("🔄 Loading data...")
        loader = DataLoader()
        
        # Try to load processed engineered features first
        base_dir = os.path.join(Config.PROCESSED_DATA_DIR, "daily")
        user_feat_path = os.path.join(base_dir, "user_features.parquet")
        
        if os.path.exists(user_feat_path):
            print("✅ Loading engineered features from processed/...")
            user_feat = pd.read_parquet(user_feat_path)
            item_feat = pd.read_parquet(os.path.join(base_dir, "item_features.parquet"))
            trans = pd.read_parquet(os.path.join(base_dir, "transactions.parquet"))
            
            # We still need raw articles for text profiles/embeddings if needed
            art, cust, _ = loader.load_raw_data()
            art, cust, _ = loader.preprocess_data(art, cust, trans)
        else:
            print("⚠️ Processed features not found. Running inline Feature Engineering...")
            art, cust, trans = loader.load_raw_data()
            art, cust, trans = loader.preprocess_data(art, cust, trans)
            engineer = FeatureEngineer()
            user_feat, item_feat = engineer.build_static_features(art, cust, trans)
            
        self._user_features = user_feat
        self._item_features = item_feat
        
        print("🔢 Encoding IDs for Two-Tower...")
        trans["customer_id_enc"] = self.user_encoder.fit_transform(trans["customer_id"]) + 1
        trans["article_id_enc"] = self.item_encoder.fit_transform(trans["article_id"]) + 1

        num_users = len(self.user_encoder.classes_) + 1
        num_items = len(self.item_encoder.classes_) + 1

        # --- Retrieval training ---
        print("🚀 Training Two-Tower Retrieval Model...")
        self.indexer = TwoTowerIndexer(num_users, num_items, self.user_encoder, self.item_encoder)
        self.indexer.train(trans)
        item_embeddings = self.indexer.build_item_index()

        if self.use_mlflow:
            mlflow.log_param("num_users", num_users)
            mlflow.log_param("num_items", num_items)
            mlflow.log_param("embedding_dim", Config.EMBEDDING_DIM)

        print("🔍 Initializing Searcher...")
        self.searcher = CandidateSearcher(self.indexer.faiss_index, self.user_encoder, self.item_encoder)

        print("⚖️ Training Ranking Model (CatBoost YetiRank)...")
        self.ranker = MLRanker(user_feat, item_feat)
        ranker_metrics = self.ranker.train(transactions=trans)

        if self.use_mlflow:
            mlflow.log_metrics(ranker_metrics)

        print("🎨 Initializing Postprocessor...")
        emb_dict = dict(zip(self.item_encoder.classes_, item_embeddings))
        self.postprocessor = Postprocessor(item_embeddings=emb_dict)

        # Persist artifacts locally
        self._save_artifacts()

        if self.use_mlflow:
            mlflow.end_run()

        print("✅ Pipeline training complete.")
        return ranker_metrics

    def _save_artifacts(self):
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        if self.ranker and self.ranker.is_trained:
            self.ranker.ranker.model.save_model(
                os.path.join(Config.MODEL_DIR, "catboost_ranker.cbm")
            )
        print(f"💾 Model artifacts saved to {Config.MODEL_DIR}/")

    def generate(self, customer_id: str, top_k: int = 10):
        # 1. Retrieval
        if customer_id in self.user_encoder.classes_:
            u_enc = self.user_encoder.transform([customer_id])[0] + 1
            u_emb = self.indexer.get_user_embedding(u_enc)
            candidates = self.searcher.get_candidates(u_emb, top_k=100)
        else:
            candidates = np.random.choice(
                self.item_encoder.classes_, size=100, replace=False
            ).tolist()

        # 2. Ranking
        df_cand = self.ranker.prepare_ranking_dataset(customer_id, candidates)
        ranked_df = self.ranker.rank_candidates(df_cand)

        # 3. Postprocessing + MMR diversity
        user_feats = None
        if self._user_features is not None:
            row = self._user_features[self._user_features["customer_id"] == customer_id]
            if not row.empty:
                user_feats = row.iloc[0].to_dict()

        final_items = self.postprocessor.apply_business_logic_and_diversity(
            ranked_df, top_k=top_k, user_features=user_feats
        )
        return final_items