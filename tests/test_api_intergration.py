import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.serving.app import app
from src.serving.api import set_pipeline

@pytest.fixture
def client():
    # Mock the pipeline so tests do not load FAISS and CatBoost.
    mock_pipeline = MagicMock()
    mock_pipeline.generate.return_value = ["item_1", "item_2", "item_3"]
    set_pipeline(mock_pipeline)
    return TestClient(app)

def test_recommend_endpoint(client):
    response = client.post("/recommend", json={"customer_id": "test_user_123", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "test_user_123"
    assert len(data["recommendations"]) == 3
    assert data["source"] in ["ml_pipeline", "cache"]

def test_fallback_on_pipeline_error(client):
    # Force the pipeline to fail.
    from src.serving.api import rec_pipeline
    rec_pipeline.generate.side_effect = Exception("OOM Error")
    
    # Pre-populate the fallback cache to emulate the production path.
    from src.serving.cache import RecommendationCache
    cache = RecommendationCache()
    if cache.is_available:
        cache.set("global_popular_items", ["fallback_1", "fallback_2"])
        
    response = client.post("/recommend", json={"customer_id": "error_user", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["source"] == "fallback"