import os

class Config:
    # Path settings
    RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
    PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
    MODEL_DIR = os.getenv("MODEL_DIR", "models")
    
    # Database configurations
    POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://admin:admin@localhost:5432/recsys_db")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    
    # Model Hyperparameters
    EMBEDDING_DIM = 64
    TEXT_EMB_DIM = 384  # MiniLM-L12-v2 embedding dimension
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    EPOCHS = 5
    TOP_K_CANDIDATES = 100
    
    # MLflow configurations
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = "Hybrid_RecSys_TwoTower_CatBoost"
