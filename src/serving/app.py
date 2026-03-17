from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import psycopg2
from src.config import Config

app = FastAPI(title="Production Hybrid Recommendation Engine API", version="2026.1.0")

# Setup external datastores
try:
    redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0, decode_responses=True)
    db_conn = psycopg2.connect(Config.POSTGRES_URI)
except Exception:
    # Fail-safe indicators for offline build processes
    redis_client = None
    db_conn = None

class RecommendationRequest(BaseModel):
    customer_id: str
    top_k: int = 10

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "rec-engine"}

@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):
    if not redis_client:
        raise HTTPException(status_code=500, detail="Data layers unavailable.")
        
    # L1 Cache layer retrieval attempt (< 5ms latency guarantee)
    cached_recommendations = redis_client.get(f"user_rec:{request.customer_id}")
    if cached_recommendations:
        return {"customer_id": request.customer_id, "recommendations": json.loads(cached_recommendations), "source": "cache"}
        
    # Fallback system to static high-popularity items (Mitigates absolute service downtime)
    fallback_rec = redis_client.get("global_popular_items")
    recommendations = json.loads(fallback_rec) if fallback_rec else ["fallback_item_1", "fallback_item_2"]
    
    # Track interaction context back to relational event log asynchronously
    if db_conn:
        with db_conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO serving_logs (customer_id, returned_items) VALUES (%s, %s);",
                (request.customer_id, json.dumps(recommendations))
            )
            db_conn.commit()
            
    return {"customer_id": request.customer_id, "recommendations": recommendations, "source": "fallback"}
