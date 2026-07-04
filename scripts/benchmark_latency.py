# scripts/benchmark_latency.py
import time
import numpy as np
import pandas as pd
from src.pipeline.recommendation_pipeline import RecSysPipeline

def run_benchmark(num_requests=100):
    print("?? Initializing pipeline for benchmark...")
    pipeline = RecSysPipeline(use_mlflow=False)
    # pipeline.train() # Uncomment if the model stil doesen't exist
    
    # We take random users
    users = pipeline.user_encoder.classes_
    sample_users = np.random.choice(users, size=min(num_requests, len(users)), replace=False)
    
    latencies = []
    print(f"?? Running {len(sample_users)} inference requests...")
    
    for uid in sample_users:
        start = time.perf_counter()
        pipeline.generate(uid, top_k=10)
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # ms
        
    latencies = np.array(latencies)
    print("\n=== LATENCY BENCHMARK RESULTS ===")
    print(f"p50: {np.percentile(latencies, 50):.2f} ms")
    print(f"p95: {np.percentile(latencies, 95):.2f} ms")
    print(f"p99: {np.percentile(latencies, 99):.2f} ms")
    print(f"Mean: {np.mean(latencies):.2f} ms")

if __name__ == "__main__":
    run_benchmark(500)