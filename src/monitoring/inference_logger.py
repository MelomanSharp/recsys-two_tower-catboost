# src/monitoring/inference_logger.py
import psycopg2
import json
import logging
from src.config import Config

class InferenceLogger:
    def __init__(self):
        self.conn = None
        try:
            self.conn = psycopg2.connect(Config.POSTGRES_URI)
            self._init_table()
        except Exception as e:
            logging.error(f"Postgres connection failed: {e}")

    def _init_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    customer_id VARCHAR(255),
                    recommendations TEXT,
                    source VARCHAR(50),
                    latency_ms FLOAT
                );
            """)
            self.conn.commit()

    def log_request(self, customer_id: str, recs: list, source: str, latency_ms: float, scores: list = None):
        if not self.conn: return
        try:
            # Store the score distribution for prediction-drift monitoring.
            scores_json = json.dumps(scores) if scores else "[]"
            with self.conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO inference_logs 
                       (customer_id, recommendations, source, latency_ms, scores_distribution) 
                       VALUES (%s, %s, %s, %s, %s)""",
                    (customer_id, json.dumps(recs), source, latency_ms, scores_json)
                )
                self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to log inference: {e}")