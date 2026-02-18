# Documentation Index

Complete documentation for the **Customer Churn Prediction & Behavioral Segmentation System**.

---

## Table of Contents

### System Overview
| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System design, data flow diagrams, component interactions, and design decisions |
| [Project Structure](project-structure.md) | Every folder and file explained with purpose and contents |

### Data Pipeline
| Document | Description |
|----------|-------------|
| [Data Pipeline](data-pipeline.md) | Full ETL pipeline: ingestion, synthetic data generation, preprocessing, feature engineering, SQL storage |

### Machine Learning
| Document | Description |
|----------|-------------|
| [ML Models](ml-models.md) | Training methodology, hyperparameter tuning, evaluation metrics, model selection, and feature importance |
| [Clustering & Segmentation](clustering-segmentation.md) | K-Means behavioral segmentation, elbow method, silhouette analysis, and business segment interpretation |

### Deployment & Operations
| Document | Description |
|----------|-------------|
| [API Reference](api-reference.md) | FastAPI endpoints, request/response schemas, usage examples, and error handling |
| [Deployment Guide](deployment.md) | Docker setup, environment configuration, local development, cloud deployment strategies |
| [Configuration Guide](configuration.md) | Environment variables, settings, logging, database, and AWS simulation setup |

### Business Intelligence
| Document | Description |
|----------|-------------|
| [BI & Tableau Guide](bi-outputs.md) | Tableau-ready datasets, executive KPIs, visualizations, and dashboard building instructions |

---

## Quick Links

- **Run the pipeline:** `python main.py`
- **Start the API:** `uvicorn src.api.app:app --reload`
- **Run tests:** `pytest tests/ -v`
- **Docker:** `docker-compose up churn-api`

---

## Document Conventions

- Code references use `monospace` formatting
- File paths are relative to the project root
- Configuration values show defaults in parentheses
- Function signatures include parameter types and return types
