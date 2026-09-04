import time, psutil, os, json
import numpy as np
from src.pipeline.recommendation_pipeline import RecSysPipeline
from src.baselines import PopularityBaseline
from src.data.loader import DataLoader

def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) # MB

def run_benchmarks():
    loader = DataLoader()
    _, _, trans = loader.load_raw_data()
    users = trans['customer_id'].unique()[:500]
    
    pipeline = RecSysPipeline(use_mlflow=False)
    pipeline.load_artifacts()
    
    pop = PopularityBaseline()
    pop.fit(trans)
    
    models = {
        "Popularity": pop,
        "Two-Tower + CatBoost": pipeline
    }
    
    results = []
    for name, model in models.items():
        latencies = []
        mem_before = get_memory_usage()
        
        for uid in users:
            start = time.perf_counter()
            if name == "Two-Tower + CatBoost":
                model.generate(uid, top_k=12)
            else:
                model.recommend(uid, top_k=12)
            latencies.append((time.perf_counter() - start) * 1000)
            
        mem_after = get_memory_usage()
        
        # Model size
        size_mb = sum(os.path.getsize(os.path.join("models", f)) 
                      for f in os.listdir("models") if not f.startswith('.')) / (1024*1024) if name != "Popularity" else 0.05
        
        results.append({
            "model": name,
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "ram_delta_mb": mem_after - mem_before,
            "model_size_mb": size_mb
        })
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_benchmarks()