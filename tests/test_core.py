import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from src.data.features import FeatureEngineer
from src.ranking.reranker import MaximalMarginalRelevance
from src.serving.app import app

# 1. test of logics (PSI)
def test_psi_identical_distributions():
    fe = FeatureEngineer()
    data = np.random.normal(0, 1, 1000)
    psi = fe.calculate_psi(data, data, bins=10)
    assert psi == 0.0

def test_psi_seasonal_shift():
    """Tests that PSI catches the seasonal shift (as in September 2019, ~0.17)."""
    fe = FeatureEngineer()
    # Emulating a stable ref and a shifted tar (seasonal peak)
    ref = np.random.beta(2, 5, 1000) 
    tar = np.random.beta(5, 2, 1000) # Сдвиг
    psi = fe.calculate_psi(ref, tar, bins=10)
    
    # We know from EDA that the actual drift here is around 0.15 - 0.20
    assert psi > 0.10  #recording a moderate drift
    assert psi < 0.50  # But this is not a catastrophic shift.

# 2. MMR (Diversification) Test
def test_mmr_diversity():
    mmr = MaximalMarginalRelevance()
    candidates = ["A", "B", "C"]
    scores = np.array([0.9, 0.8, 0.7])
    # A and B are identical, C is different
    embeddings = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([0.99, 0.0]),
        "C": np.array([0.0, 1.0])
    }
    # With a high diversity requirement (lmbda=0.3) C should rise
    res = mmr.balance_diversity(candidates, scores, embeddings, top_n=2, lmbda=0.3)
    assert "C" in res

# 3. API integration test
def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()