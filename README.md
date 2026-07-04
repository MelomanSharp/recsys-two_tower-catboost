# Production Recommendation System: H&M Personalized Fashion Recommendations

Production-oriented рекомендательная система для e-commerce, построенная на данных соревнования Kaggle **H&M Personalized Fashion Recommendations**.

Проект исследует не только качество рекомендательных моделей, но и вопрос их практической эксплуатации: **насколько усложнение ML-архитектуры оправдано с точки зрения качества, вычислительных затрат, latency и устойчивости модели к изменению данных в production.**

> **Статус:** 🚧 In Progress
> Архитектура и эксперименты находятся в разработке. Данный README описывает цели, гипотезы и план проекта.

---

## 🎯 Цель проекта

В реальном e-commerce рекомендательная система должна решать не только задачу построения точных рекомендаций.

Усложнение модели обычно приводит к росту вычислительных затрат, latency, сложности инфраструктуры и стоимости поддержки. При этом поведение пользователей, ассортимент и популярность товаров постоянно меняются, поэтому даже хорошо обученная модель может постепенно терять эффективность.

Основная цель проекта:

> **Исследовать, какой подход к персонализации обеспечивает оптимальный баланс между качеством рекомендаций, computational cost, latency и сложностью эксплуатации, а также определить, когда деградация модели становится достаточной причиной для retraining или замены модели.**

Таким образом, проект рассматривает рекомендательную систему как **production ML system**, а не как отдельный обученный алгоритм.

---

## 🏪 Бизнес-контекст

Представим e-commerce платформу с большим каталогом товаров и историей взаимодействий пользователей.

Для каждого пользователя система должна сформировать персональную подборку товаров, которая потенциально увеличивает:

* вероятность взаимодействия с рекомендованным товаром;
* add-to-cart rate;
* conversion rate;
* revenue per session/user.

При этом система должна работать с приемлемой latency и стоимостью инфраструктуры.

Кроме того, рекомендации должны оставаться актуальными при изменении:

* пользовательских предпочтений;
* ассортимента;
* популярности товаров;
* сезонности;
* распределения трафика.

---

## 🔬 Основные исследовательские вопросы

### 1. Даёт ли более сложная архитектура действительно лучшее качество?

Будут сравнены несколько подходов:

```text
Popularity baseline
        ↓
Collaborative / interaction-based baseline
        ↓
Candidate Retrieval
        ↓
Retrieval + Ranking
```

Для моделей будут исследоваться:

* Recall@K
* Precision@K
* NDCG@K
* MAP@K
* catalog coverage
* diversity

Главный вопрос:

> **Какой прирост качества даёт усложнение модели?**

---

### 2. Какова цена этого прироста?

Для различных архитектур будут измеряться:

* inference latency;
* throughput;
* CPU / RAM / GPU utilization;
* размер модели;
* стоимость inference;
* стоимость обучения;
* стоимость обновления индекса;
* время retraining.

Таким образом, сравнение моделей будет проводиться не только по offline ML metrics.

Условно:

```text
             Model quality
                  ↑
                  │
                  │        ● Complex model
                  │
                  │   ● Two-stage
                  │
                  │ ● Baseline
                  │
                  └────────────────────→
                     Computational cost
```

Цель состоит не в том, чтобы автоматически выбрать самую сложную модель, а в том, чтобы определить **Pareto-optimal trade-off между качеством и стоимостью**.

---

## ⚡ Two-stage recommendation architecture

Основная production-архитектура проекта предполагает разделение рекомендаций на два этапа.

```text
                    User
                      │
                      ↓
             ┌─────────────────┐
             │    Retrieval    │
             │                 │
             │  User embedding │
             │       +         │
             │ Item embeddings │
             └────────┬────────┘
                      │
                 ~100-500 items
                      │
                      ↓
             ┌─────────────────┐
             │     Ranking     │
             │                 │
             │    CatBoost     │
             │   / LightGBM    │
             └────────┬────────┘
                      │
                      ↓
                 Top-K items
```

Retrieval ограничивает множество кандидатов, которые необходимо рассматривать ranking-модели.

Это позволяет исследовать компромисс:

> **качество × latency × computational cost**

и проверить, насколько двухэтапная архитектура оправдана по сравнению с более простыми подходами.

---

## 🧠 Production ML / MLOps

Отдельная часть проекта посвящена жизненному циклу модели после deployment.

Модель может продолжать работать технически корректно, одновременно становясь менее полезной.

Например:

```text
User behavior changes
        ↓
Feature distribution changes
        ↓
Candidate distribution changes
        ↓
Model performance decreases
        ↓
Business metrics decrease
```

Поэтому мониторинг будет разделён как минимум на несколько уровней.

### Data drift

Контроль изменения распределений входных данных.

Возможные методы:

* PSI;
* KL divergence;
* Jensen-Shannon divergence;
* Wasserstein distance;
* statistical hypothesis tests.

### Prediction drift

Контроль изменения распределения model scores и результатов ranking.

### Model performance drift

Контроль изменения offline performance на новых данных:

* NDCG@K;
* Recall@K;
* MAP@K;
* другие релевантные metrics.

### Business-level drift

По возможности будет исследоваться связь ML degradation с proxy business metrics:

* CTR;
* add-to-cart rate;
* conversion-related metrics.

---

## 🔄 Когда необходимо переобучать модель?

Одна из ключевых задач проекта:

> **Не считать любой обнаруженный data drift автоматическим основанием для retraining.**

Например:

```text
Data drift detected
        ↓
Does model performance degrade?
        │
   ┌────┴────┐
   │         │
  No        Yes
   │         │
   ↓         ↓
Keep      Evaluate
model     severity
             │
             ↓
       Retraining decision
```

Таким образом, система должна различать:

**Statistical change**

> Данные изменились, но модель по-прежнему работает хорошо.

и

**Meaningful model degradation**

> Изменение данных сопровождается устойчивым ухудшением качества рекомендаций.

Планируется исследовать различные стратегии принятия решения о retraining:

```text
Fixed schedule
       vs
Drift-triggered retraining
       vs
Performance-triggered retraining
       vs
Hybrid strategy
```

Это позволит оценить не только качество моделей, но и **стоимость их поддержания**.

---

## 🏗️ Предварительная архитектура

```text
                    Raw Data
                       │
                       ↓
                ┌─────────────┐
                │     ETL     │
                └──────┬──────┘
                       │
                       ↓
              Feature Generation
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
        User features       Item features
             │                   │
             └─────────┬─────────┘
                       ↓
                  Retrieval
                       │
                       ↓
                Candidate Set
                       │
                       ↓
                   Ranking
                       │
                       ↓
                Recommendations
                       │
                       ↓
                     API
                       │
                       ↓
                   Clients
                       │
                       ↓
                User Events
                       │
                       └─────────────┐
                                     ↓
                              Monitoring
                                     │
                       ┌─────────────┴─────────────┐
                       ↓                           ↓
                  Data Drift                Performance
                       │                           │
                       └─────────────┬─────────────┘
                                     ↓
                              Retraining Logic
                                     │
                                     ↓
                               Model Registry
                                     │
                                     ↓
                              New Model Version
```

---

## 🧪 Offline evaluation

Для предотвращения temporal leakage данные будут разделяться с учётом временной структуры interactions.

Основные эксперименты будут включать:

1. Baseline evaluation.
2. Retrieval evaluation.
3. Ranking evaluation.
4. End-to-end evaluation.
5. Ablation studies.
6. Latency / resource benchmarks.
7. Drift simulation.
8. Retraining strategy evaluation.

Особое внимание будет уделено temporal validation, поскольку случайный train/test split плохо отражает production-сценарий для recommendation system.

---

## 📊 Что будет считаться успешным результатом?

Проект не ставит целью получить максимальный возможный score любой ценой.

Успешным будет считаться решение, которое позволяет обоснованно ответить на следующие вопросы:

### Quality

Насколько улучшается качество рекомендаций относительно baseline?

### Cost

Сколько дополнительных вычислительных ресурсов требуется для этого улучшения?

### Latency

Как изменяется latency при увеличении сложности модели и количества кандидатов?

### Scalability

Как система ведёт себя при увеличении числа пользователей, товаров и запросов?

### Robustness

Как быстро деградирует качество при изменении распределения данных?

### Maintenance

Как часто необходимо переобучение и сколько ресурсов оно требует?

### Decision policy

Можно ли построить достаточно надёжное правило:

> **"Текущую модель следует оставить" / "модель необходимо переобучить" / "текущую архитектуру следует заменить".**

---

## 🛠️ Technology Stack

### Data & ML

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost / LightGBM
* PyTorch
* FAISS

### Data processing

* ETL pipelines
* Feature engineering
* Temporal data splitting

### MLOps

* Experiment tracking
* Model versioning
* Data / model monitoring
* Drift detection
* Automated evaluation
* Retraining pipeline

### Serving

* REST API
* Caching
* Batch / online inference

### Infrastructure

Будет определено в ходе разработки в зависимости от требований к production-like deployment.

---

## 📁 Planned project structure

```text
.
├── data/
├── notebooks/
├── src/
│   ├── etl/
│   ├── features/
│   ├── retrieval/
│   ├── ranking/
│   ├── evaluation/
│   ├── monitoring/
│   └── serving/
│
├── pipelines/
│   ├── training/
│   ├── evaluation/
│   └── retraining/
│
├── tests/
├── configs/
├── docker/
├── models/
├── monitoring/
├── README.md
└── requirements.txt
```

---

## 🚧 Development Roadmap

* [ ] Data ingestion and ETL
* [ ] Temporal train/validation/test split
* [ ] Popularity baseline
* [ ] Initial recommendation baseline
* [ ] Retrieval model
* [ ] Candidate generation benchmark
* [ ] Ranking model
* [ ] End-to-end evaluation
* [ ] Latency and resource benchmarks
* [ ] Caching layer
* [ ] REST API
* [ ] Data quality monitoring
* [ ] Drift detection
* [ ] Model performance monitoring
* [ ] Model registry / versioning
* [ ] Retraining pipeline
* [ ] Retraining decision policy
* [ ] Architecture and cost comparison
* [ ] Production-like deployment

---

## 💡 Project thesis

> **A recommendation model is not successful merely because it achieves a high offline metric.**

A production recommendation system must provide a reasonable balance between:

```text
                 ┌──────────────┐
                 │    Quality   │
                 └──────┬───────┘
                        │
      ┌─────────────────┼─────────────────┐
      ↓                 ↓                 ↓
   Latency             Cost            Robustness
      │                 │                 │
      └─────────────────┼─────────────────┘
                        ↓
                 Maintainability
                        ↓
                 Business Value
```

The project therefore focuses on the **full lifecycle of a recommendation system**, from candidate generation and ranking to monitoring, evaluation and retraining decisions.
