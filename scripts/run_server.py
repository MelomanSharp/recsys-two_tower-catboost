import uvicorn
from src.serving.app import init_app
from src.pipeline.recommendation_pipeline import RecSysPipeline

def main():
    print("⚙️ Initializing Recommendation Pipeline...")
    pipeline = RecSysPipeline()
    
    # Для быстрого старта API без долгого обучения можно закомментировать .train()
    # В production-среде модели должны загружаться из S3/MLflow, а не обучаться при старте поды
    # pipeline.train() 
    
    print("🚀 Starting FastAPI server...")
    app = init_app(pipeline)
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()