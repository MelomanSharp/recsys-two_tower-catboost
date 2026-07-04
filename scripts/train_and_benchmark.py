from src.pipeline.recommendation_pipeline import RecSysPipeline
import subprocess

def main():
    print("Starting Full Pipeline Training...")
    # Disable MLflow to prevent handling local server 
    pipeline = RecSysPipeline(use_mlflow=False) 
    metrics = pipeline.train()
    
    print("\n=== OFFLINE METRICS ===")
    print(f"Val NDCG@12: {metrics['val_ndcg@12']:.4f}")
    print(f"Val Recall@12: {metrics['val_recall@12']:.4f}")
    
    print("\nRunning Latency Benchmark...")
    # Launch existing benchmark-scritp
    subprocess.run(["python", "scripts/benchmark_latency.py"])

if __name__ == "__main__":
    main()