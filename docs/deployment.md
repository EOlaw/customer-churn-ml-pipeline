# Deployment Guide

This document covers how to run the system locally, containerize with Docker, and deploy to cloud environments.

---

## Local Development Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd customer-churn-ml-pipeline

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your values (optional for local dev)

# 5. Run the full pipeline (generates data, trains models, creates outputs)
python main.py

# 6. Start the API server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# 7. Run tests
pip install pytest
pytest tests/ -v
```

### What `python main.py` Produces

Running the main pipeline creates:
- `data/raw/customer_churn_raw.csv` — 100K raw records
- `data/processed/churn_data.db` — SQLite database
- `data/processed/customer_churn_processed.csv` — Model-ready data
- `models/saved/*.joblib` — Trained models and scaler
- `models/versioned/*.joblib` — Timestamped model copies
- `outputs/visualizations/*.png` — Charts and plots
- `outputs/reports/*.csv` and `*.json` — Evaluation reports
- `outputs/tableau/*.csv` — Tableau-ready datasets
- `logs/pipeline.log` — Full execution log

---

## Docker Deployment

### Dockerfile Explained

```dockerfile
FROM python:3.11-slim              # Lightweight Python base image
WORKDIR /app                       # Set working directory
COPY requirements.txt .            # Copy deps first (layer caching)
RUN pip install --no-cache-dir -r requirements.txt  # Install deps
COPY . .                           # Copy project files
EXPOSE 8000                        # Document the port
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why `python:3.11-slim`?**
- `slim` variant is ~150MB vs ~900MB for the full image
- Contains only essential system packages
- Sufficient for our Python ML dependencies

### docker-compose.yml Explained

The compose file defines two services:

#### `churn-api` — Production API Server

```yaml
churn-api:
  build: .                    # Build from Dockerfile
  ports:
    - "8000:8000"             # Map host:container ports
  volumes:
    - ./data:/app/data        # Persist data directory
    - ./models:/app/models    # Persist trained models
    - ./outputs:/app/outputs  # Persist outputs
    - ./logs:/app/logs        # Persist logs
  env_file:
    - .env                    # Load environment variables
  restart: unless-stopped     # Auto-restart on crash
```

**Volumes:** Data, models, outputs, and logs are mounted from the host, so they persist across container restarts and are accessible outside the container.

#### `train` — Training Pipeline Runner

```yaml
train:
  build: .
  command: python -m src.pipelines.training_pipeline
  volumes: [same as above]
  env_file: [same]
  profiles:
    - training               # Only runs when explicitly requested
```

**The `profiles: [training]` key** means this service won't start with `docker-compose up`. You must explicitly request it.

### Docker Commands

```bash
# Build and start the API
docker-compose up churn-api

# Build and start in background
docker-compose up -d churn-api

# Run the training pipeline
docker-compose --profile training run train

# View logs
docker-compose logs -f churn-api

# Stop everything
docker-compose down

# Rebuild after code changes
docker-compose up --build churn-api
```

### Build the Image Manually

```bash
# Build
docker build -t churn-prediction-api .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  --env-file .env \
  churn-prediction-api
```

---

## Cloud Deployment Strategies

### AWS Deployment

#### Option 1: ECS (Elastic Container Service)

Best for: Production workloads with auto-scaling.

```
                    ┌─────────────────┐
                    │   ALB (Load     │
                    │   Balancer)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ ECS Task 1 │  │ ECS Task 2 │  │ ECS Task 3 │
     │ (API)      │  │ (API)      │  │ (API)      │
     └────────────┘  └────────────┘  └────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │   S3 Bucket     │
                    │   (Models)      │
                    └─────────────────┘
```

Steps:
1. Push Docker image to **Amazon ECR** (Elastic Container Registry)
2. Create an ECS cluster with Fargate (serverless containers)
3. Define a task definition using the Docker image
4. Create an ECS service with an Application Load Balancer
5. Store models in S3, load at container startup
6. Set environment variables in the task definition

```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag churn-prediction-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/churn-prediction-api:latest
```

#### Option 2: Lambda + API Gateway

Best for: Low-traffic, cost-optimized deployment.

- Package the model and prediction code as a Lambda function
- Use API Gateway to expose the `/predict-churn` endpoint
- Cold starts may add 5-10 seconds for model loading
- Maximum execution time: 15 minutes

#### Option 3: SageMaker Endpoint

Best for: ML-specific deployment with built-in monitoring.

- Deploy the XGBoost model as a SageMaker endpoint
- Built-in model monitoring and A/B testing
- Auto-scaling based on inference traffic

### Production Changes Required

When moving from local/simulation to production, update these:

| Component | Local (Current) | Production |
|-----------|----------------|------------|
| Database | SQLite (`churn_data.db`) | Amazon Redshift or PostgreSQL on RDS |
| Storage | Local directories | Amazon S3 |
| Model loading | `joblib.load(local_path)` | Download from S3, then load |
| Logging | File-based (`logs/pipeline.log`) | CloudWatch Logs |
| Secrets | `.env` file | AWS Secrets Manager or SSM Parameter Store |
| Training | Local `python main.py` | SageMaker Training Job or Step Functions |

### Environment Variables for Production

Update `.env` with real credentials:

```bash
# Real AWS credentials
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-production-bucket

# Real Redshift
REDSHIFT_HOST=your-cluster.xxx.us-east-1.redshift.amazonaws.com
REDSHIFT_PORT=5439
REDSHIFT_DB=churn_analytics

# Production settings
LOG_LEVEL=WARNING
MODEL_VERSION=1.0.0
```

---

## CI/CD Pipeline (Recommended)

```
Git Push → GitHub Actions → Run Tests → Build Docker → Push to ECR → Deploy to ECS
```

Example GitHub Actions workflow:

```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push to ECR
        run: |
          docker build -t churn-api .
          # Push to ECR...
      - name: Deploy to ECS
        run: |
          # Update ECS service...
```

---

## Monitoring and Observability

### Logging

All pipeline and API activities are logged to `logs/pipeline.log`:

```
2026-02-17 22:07:42 | src.data.ingestion | INFO | Generating 100000 synthetic customer records...
2026-02-17 22:08:43 | src.models.train | INFO | xgboost | Best CV roc_auc: 0.9322
```

### Health Checks

The `/health` endpoint returns model status:
- Use with Docker `HEALTHCHECK` directive
- Use with load balancer health checks
- Use with Kubernetes liveness/readiness probes

### Model Monitoring (Recommended Additions)

For production, consider adding:
- **Data drift detection:** Compare incoming feature distributions to training data
- **Model performance monitoring:** Track prediction accuracy over time
- **Prediction logging:** Store all predictions for audit and retraining
- **Alerting:** Trigger alerts when model accuracy drops below threshold
