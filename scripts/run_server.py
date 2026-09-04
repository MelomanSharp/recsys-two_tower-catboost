import uvicorn
from src.serving.app import init_app
from src.pipeline.recommendation_pipeline import RecSysPipeline

def main():
    print("⚙️ Initializing Recommendation Pipeline...")
    pipeline = RecSysPipeline()
    
    # To start the API quickly, leave training disabled.
    # In production, load models from S3 or MLflow instead of training at pod startup.
    # pipeline.train() 
    
    print("🚀 Starting FastAPI server...")
    app = init_app(pipeline)
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()