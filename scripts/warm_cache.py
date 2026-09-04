import redis, json, os
from src.pipeline.recommendation_pipeline import RecSysPipeline
from src.data.loader import DataLoader

def warm_up_cache():
    print("🔥 Warming up Redis cache for top 10% active users...")
    pipeline = RecSysPipeline(use_mlflow=False)
    pipeline.load_artifacts()
    
    loader = DataLoader()
    _, _, trans = loader.load_raw_data()
    
    # Select the top 10% most active users.
    user_counts = trans.groupby('customer_id').size()
    top_users = user_counts.nlargest(int(len(user_counts) * 0.1)).index.tolist()
    
    r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0, decode_responses=True)
    
    success = 0
    for uid in top_users:
        try:
            recs = pipeline.generate(uid, top_k=12)
            r.set(f"recsys:{uid}", json.dumps(recs), ex=86400) # TTL: one day.
            success += 1
        except Exception:
            continue
            
    print(f"✅ Successfully cached recommendations for {success}/{len(top_users)} users.")

if __name__ == "__main__":
    warm_up_cache()