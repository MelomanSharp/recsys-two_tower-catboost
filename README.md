# Production Recommendation System: H&M Personalized Fashion Recommendations

An end-to-end recommendation system for the Kaggle **H&M Personalized Fashion Recommendations** dataset. The current production path combines Two-Tower retrieval with CatBoost YetiRank re-ranking and exposes recommendations through a FastAPI service.

> **Status:** Two-Tower + CatBoost pipeline runs end to end. Initial ablation results are recorded; comparisons with independent baselines are still pending.

## Overview

The system follows a two-stage architecture:

```text
User history and catalog
          |
          v
Feature engineering
          |
          v
Two-Tower retrieval + FAISS       -> candidate items
          |
          v
CatBoost YetiRank                 -> ranked recommendations
          |
          v
FastAPI + Redis cache
```

This structure keeps the expensive ranking step focused on a limited candidate set while preserving a path to low-latency online serving.

## Current results

The latest recorded ablation results are in [reports/ablation_results.csv](reports/ablation_results.csv). They cover candidate-set size and embedding dimension. The columns are:

* `candidates`: number of retrieved candidates;
* `emb_dim`: Two-Tower embedding dimension;
* `ndcg`: recorded validation NDCG@12;
* `latency`: recorded inference latency in milliseconds;
* `cost`: relative cost index used by the experiment.

Selected recorded configurations:

| Candidates | Embedding dimension | NDCG@12 | Latency (ms) | Cost index |
|---:|---:|---:|---:|---:|
| 32 | 32 | 0.7787 | 710.82 | 0.5248 |
| 100 | 64 | 0.8478 | 1491.12 | 0.5550 |
| 200 | 128 | 0.9692 | 2842.48 | 0.6100 |
| 500 | 128 | 0.9900 | 4673.20 | 0.6550 |

The highest recorded NDCG in this grid is **0.9900** at 500 candidates and a 128-dimensional embedding. This is an internal ablation result, not evidence that the full system outperforms another recommender: popularity, collaborative-filtering, retrieval-only, and other baselines have not yet been evaluated on the same split and protocol.

## Repository layout

```text
.
├── data/
│   ├── raw/                    # Kaggle CSV files
│   └── processed/daily/        # ETL parquet outputs
├── dags/                       # Airflow drift and evaluation DAG
├── docker/                     # Application and Airflow images
├── reports/                    # Experiment outputs
├── scripts/
│   ├── download_data.py
│   ├── run_local_etl.py
│   ├── train_and_benchmark.py
│   ├── benchmark_latency.py
│   ├── ablation_study.py
│   └── retraining_simulation.py
├── src/
│   ├── data/                   # Loading and feature engineering
│   ├── models/                 # Two-Tower and CatBoost components
│   ├── retrieval/              # FAISS indexing and search
│   ├── ranking/                # Candidate ranking and postprocessing
│   ├── evaluation/             # Recommendation metrics
│   ├── monitoring/             # Drift and inference monitoring
│   ├── pipeline/               # End-to-end orchestration
│   └── serving/                # FastAPI, cache, and health checks
└── tests/
```

## Requirements

* Python 3.10+ is recommended.
* The packages pinned in [requirments.txt](requirments.txt). The filename is kept for compatibility with the existing project.
* Kaggle credentials configured for `kagglehub` when downloading the dataset.
* Redis and PostgreSQL for the complete serving setup. The local training and evaluation scripts do not require the Docker services.

## Quick start

Create an environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirments.txt
```

Download the Kaggle data and build processed parquet files:

```bash
python scripts/download_data.py
python scripts/run_local_etl.py
```

Train the Two-Tower + CatBoost pipeline and run the latency benchmark:

```bash
python scripts/train_and_benchmark.py
```

Run the latency benchmark against saved artifacts, or train automatically when artifacts are missing:

```bash
python -m scripts.benchmark_latency
```

Generate the quality-latency-cost ablation plot:

```bash
python scripts/ablation_study.py
```

## Serving locally

The API entry point is `scripts/run_server.py`. For the full local stack, start Redis, PostgreSQL, and the FastAPI container with:

```bash
docker compose up --build
```

The service listens on `http://localhost:8000`. The main endpoint is `POST /recommend`:

```json
{
  "customer_id": "customer-id",
  "top_k": 10
}
```

Health checks are available at `GET /health`.

## Evaluation and monitoring

The repository includes:

* NDCG@K and Recall@K evaluation for holdout interactions;
* popularity, retrieval-only, and item-item collaborative-filtering baseline implementations;
* latency benchmarking with p50, p95, and p99 measurements;
* PSI-based drift detection for price and item trendiness;
* an Airflow decision path for production-model evaluation and retraining triggers;
* Redis caching and PostgreSQL inference logging.

The next evaluation step is to run every baseline and the Two-Tower + CatBoost pipeline against the same temporal holdout, then report quality, latency, coverage, and resource usage in one comparison table. Until that experiment is complete, the ablation values above should be treated as configuration results rather than a model leaderboard.

## Tests

Run the test suite with:

```bash
pytest
```

The tests cover API fallback behavior, PSI calculations, diversification, and core retrieval behavior. Full pipeline tests require the project dependencies and the corresponding data or model artifacts.

## Technology stack

* Python, pandas, NumPy, scikit-learn
* PyTorch for Two-Tower training
* FAISS for vector retrieval
* CatBoost for ranking
* FastAPI and Uvicorn for serving
* Redis for caching
* PostgreSQL for inference logs
* Airflow and MLflow for orchestration and experiment tracking
* Docker Compose for local services

## Project direction

The project is intended to measure the complete operational trade-off, not only offline ranking quality:

```text
quality <-> latency <-> infrastructure cost <-> robustness <-> maintenance
```

Planned comparison work includes a shared temporal evaluation protocol, baseline benchmarking, candidate coverage and diversity metrics, resource measurements, and a retraining policy based on both drift and model performance.
