import redis
import json
from src.config import Config

class RecommendationCache:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0, decode_responses=True
            )
            self.redis_client.ping()
            self.is_available = True
        except Exception:
            self.is_available = False
            self.redis_client = None

    def get(self, key):
        if not self.is_available: return None
        try:
            cached = self.redis_client.get(f"recsys:{key}")
            return json.loads(cached) if cached else None
        except Exception: return None

    def set(self, key, value, ttl=3600):
        if not self.is_available: return
        try:
            self.redis_client.set(f"recsys:{key}", json.dumps(value), ex=ttl)
        except Exception: pass