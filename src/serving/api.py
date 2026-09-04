from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.serving.cache import RecommendationCache
from src.serving.system_health import HealthChecker

router = APIRouter()
cache = RecommendationCache()
health_checker = HealthChecker()

class RecommendationRequest(BaseModel):
    customer_id: str
    top_k: int = 10

class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: list
    source: str

# Global pipeline reference injected during application startup.
rec_pipeline = None

def set_pipeline(pipeline):
    global rec_pipeline
    rec_pipeline = pipeline

@router.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    # 1. L1 Cache Check
    cached_recs = cache.get(request.customer_id)
    if cached_recs:
        return RecommendationResponse(
            customer_id=request.customer_id, recommendations=cached_recs[:request.top_k], source="cache"
        )
        
    # 2. ML Pipeline Fallback
    if rec_pipeline is None:
        raise HTTPException(status_code=503, detail="ML Pipeline not initialized")
        
    try:
        recommendations = rec_pipeline.generate(request.customer_id, request.top_k)
        cache.set(request.customer_id, recommendations)
        return RecommendationResponse(
            customer_id=request.customer_id, recommendations=recommendations, source="ml_pipeline"
        )
    except Exception as e:
        # 3. Hard fallback to popular items.
        fallback = cache.get("global_popular_items") or ["fallback_item_1", "fallback_item_2"]
        return RecommendationResponse(
            customer_id=request.customer_id, recommendations=fallback[:request.top_k], source="fallback"
        )

@router.get("/health")
def health_check():
    return health_checker.check_all()