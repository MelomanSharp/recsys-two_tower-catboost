from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from src.serving.api import router
from src.pipeline.recommendation_pipeline import RecSysPipeline
from src.monitoring.inference_logger import InferenceLogger
from src.monitoring.mlflow_registry import load_latest_models_from_registry

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("?? Loading models from MLflow Registry...")
    load_latest_models_from_registry()
    
    print("?? Initializing RecSys Pipeline...")
    app.state.pipeline = RecSysPipeline(use_mlflow=False)
    app.state.logger = InferenceLogger()
    
    # If there are no models in MLflow, train right away (demo only)
    try:
        app.state.pipeline.generate("test_user", 1)
    except Exception:
        print("?? Models not found locally. Training inline (Demo mode)...")
        app.state.pipeline.train()
        
    yield
    # Shutdown
    print("?? Shutting down...")

app = FastAPI(title="Production Hybrid Recommendation Engine API", version="2026.1.0", lifespan=lifespan)
app.include_router(router)