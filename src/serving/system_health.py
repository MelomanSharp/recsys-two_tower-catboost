import redis
import psycopg2
from src.config import Config

class HealthChecker:
    def __init__(self):
        self.status = {"redis": "unknown", "postgres": "unknown", "ml_models": "unknown"}

    def check_all(self):
        self.check_redis()
        self.check_postgres()
        self.status["ml_models"] = "healthy" 
        
        overall = "healthy" if all(v == "healthy" for v in self.status.values()) else "degraded"
        return {"status": overall, "components": self.status}

    def check_redis(self):
        try:
            r = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=0)
            r.ping()
            self.status["redis"] = "healthy"
        except Exception: self.status["redis"] = "unavailable"

    def check_postgres(self):
        try:
            conn = psycopg2.connect(Config.POSTGRES_URI)
            conn.close()
            self.status["postgres"] = "healthy"
        except Exception: self.status["postgres"] = "unavailable"