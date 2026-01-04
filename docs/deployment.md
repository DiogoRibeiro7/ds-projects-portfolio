# 🚀 Deployment Guide

Complete guide for deploying the Data Science Portfolio in production environments.

## Deployment Options

### 1. Docker Deployment

#### Dockerfile for ML API

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 5000 8050

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models
ENV DATA_PATH=/app/data

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/mldb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./models:/app/models
      - ./data:/app/data

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8050:8050"
    environment:
      - API_URL=http://ml-api:5000
      - REDIS_URL=redis://redis:6379
    depends_on:
      - ml-api
      - redis

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=mldb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ml-api
      - dashboard

volumes:
  postgres_data:
```

### 2. Kubernetes Deployment

#### Deployment Configuration

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  labels:
    app: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: your-registry/ml-api:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ml-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: ml-config
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 3. Cloud Platform Deployment

#### AWS Deployment

```bash
# Deploy to AWS ECS using Terraform
terraform init
terraform plan -var-file="production.tfvars"
terraform apply -auto-approve

# Using AWS CLI
aws ecs create-cluster --cluster-name ml-cluster
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster ml-cluster --service-name ml-api --task-definition ml-api:1
```

#### Azure Deployment

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group ml-rg \
  --name ml-api \
  --image your-registry.azurecr.io/ml-api:latest \
  --dns-name-label ml-api \
  --ports 5000 \
  --environment-variables \
    DATABASE_URL=$DATABASE_URL \
    REDIS_URL=$REDIS_URL
```

#### Google Cloud Platform

```bash
# Deploy to Cloud Run
gcloud run deploy ml-api \
  --image gcr.io/project-id/ml-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL=$DATABASE_URL,REDIS_URL=$REDIS_URL
```

## Production Configuration

### Environment Variables

```bash
# .env.production
# Application
APP_ENV=production
DEBUG=false
LOG_LEVEL=info

# Security
SECRET_KEY=your-production-secret-key-here
JWT_SECRET=your-jwt-secret-here
API_KEY=your-api-key-here

# Database
DATABASE_URL=postgresql://user:password@db-host:5432/production_db
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://redis-host:6379/0
REDIS_MAX_CONNECTIONS=50

# Model Configuration
MODEL_PATH=/models
MODEL_VERSION=v1.2.0
MODEL_CACHE_TTL=3600

# Monitoring
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project
DATADOG_API_KEY=your-datadog-key
PROMETHEUS_PORT=9090

# Performance
WORKERS=4
THREADS=2
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50
TIMEOUT=30
```

### Nginx Configuration

```nginx
# nginx.conf
upstream ml_api {
    least_conn;
    server ml-api-1:5000 weight=1;
    server ml-api-2:5000 weight=1;
    server ml-api-3:5000 weight=1;
}

upstream dashboard {
    server dashboard-1:8050;
    server dashboard-2:8050;
}

server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Content-Security-Policy "default-src 'self'" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location /api/ {
        proxy_pass http://ml_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }

    location / {
        proxy_pass http://dashboard/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Database Setup

### PostgreSQL Production Setup

```sql
-- Create production database
CREATE DATABASE ml_production;

-- Create user with limited privileges
CREATE USER ml_api_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE ml_production TO ml_api_user;
GRANT USAGE ON SCHEMA public TO ml_api_user;
GRANT CREATE ON SCHEMA public TO ml_api_user;

-- Create tables
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_data BYTEA,
    metadata JSONB,
    accuracy FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES models(id),
    input_data JSONB,
    prediction JSONB,
    probability FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_models_name_version ON models(name, version);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
CREATE INDEX idx_predictions_model_id ON predictions(model_id);

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

### Redis Configuration

```conf
# redis.conf
# Basic configuration
port 6379
bind 0.0.0.0
protected-mode yes
requirepass your_redis_password

# Persistence
save 900 1
save 300 10
save 60 10000
dbfilename dump.rdb
dir /data

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Performance
tcp-keepalive 300
timeout 0
tcp-backlog 511

# Security
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG ""
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml-api'
    static_configs:
      - targets: ['ml-api:9090']
    metrics_path: '/metrics'

  - job_name: 'dashboard'
    static_configs:
      - targets: ['dashboard:9090']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### Logging Configuration

```python
# logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging(app_name='ml-api'):
    """Configure structured logging for production."""

    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.INFO)

    # JSON formatter
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(levelname)s %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        f'/var/log/{app_name}.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Sentry handler for errors
    if os.getenv('SENTRY_DSN'):
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_logging = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR
        )

        sentry_sdk.init(
            dsn=os.getenv('SENTRY_DSN'),
            integrations=[sentry_logging],
            traces_sample_rate=0.1,
            environment=os.getenv('APP_ENV', 'production')
        )

    return logger
```

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          pytest tests/ --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: |
          docker build -t ${{ secrets.DOCKER_REGISTRY }}/ml-api:${{ github.ref_name }} .
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push ${{ secrets.DOCKER_REGISTRY }}/ml-api:${{ github.ref_name }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ml-api ml-api=${{ secrets.DOCKER_REGISTRY }}/ml-api:${{ github.ref_name }}
          kubectl rollout status deployment/ml-api
```

## Health Checks

```python
# health.py
from flask import Blueprint, jsonify
import psutil
import redis
from sqlalchemy import text

health_bp = Blueprint('health', __name__)

@health_bp.route('/health')
def health():
    """Basic health check."""
    return jsonify({'status': 'healthy'}), 200

@health_bp.route('/ready')
def ready():
    """Readiness probe with dependency checks."""
    checks = {
        'database': check_database(),
        'redis': check_redis(),
        'model': check_model_loaded(),
        'disk': check_disk_space(),
        'memory': check_memory()
    }

    if all(checks.values()):
        return jsonify({'status': 'ready', 'checks': checks}), 200
    else:
        return jsonify({'status': 'not ready', 'checks': checks}), 503

def check_database():
    try:
        db.session.execute(text('SELECT 1'))
        return True
    except:
        return False

def check_redis():
    try:
        r = redis.Redis.from_url(os.getenv('REDIS_URL'))
        r.ping()
        return True
    except:
        return False

def check_model_loaded():
    return hasattr(app, 'model') and app.model is not None

def check_disk_space():
    usage = psutil.disk_usage('/')
    return usage.percent < 90

def check_memory():
    memory = psutil.virtual_memory()
    return memory.percent < 90
```

## Rollback Strategy

```bash
#!/bin/bash
# rollback.sh

DEPLOYMENT="ml-api"
NAMESPACE="production"

# Get current revision
CURRENT_REVISION=$(kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE | tail -2 | head -1 | awk '{print $1}')

echo "Current revision: $CURRENT_REVISION"

# Rollback to previous revision
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE

# Wait for rollout to complete
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

# Verify health
./health_check.sh

if [ $? -eq 0 ]; then
    echo "Rollback successful"
else
    echo "Rollback failed, manual intervention required"
    exit 1
fi
```

## Security Checklist

- [ ] Use HTTPS everywhere
- [ ] Implement rate limiting
- [ ] Enable CORS properly
- [ ] Use environment variables for secrets
- [ ] Implement proper authentication
- [ ] Regular security updates
- [ ] Database connection pooling
- [ ] Input validation
- [ ] SQL injection prevention
- [ ] XSS protection
- [ ] CSRF protection
- [ ] Regular backups
- [ ] Monitoring and alerting
- [ ] Incident response plan

## Performance Optimization

1. **Enable caching** (Redis/Memcached)
2. **Use CDN** for static assets
3. **Database indexing** and query optimization
4. **Connection pooling** for databases
5. **Async processing** for heavy tasks
6. **Load balancing** across multiple instances
7. **Auto-scaling** based on metrics
8. **Code profiling** and optimization

## Useful Commands

```bash
# Docker commands
docker build -t ml-api:latest .
docker run -d -p 5000:5000 ml-api:latest
docker-compose up -d
docker-compose logs -f ml-api

# Kubernetes commands
kubectl apply -f k8s-deployment.yaml
kubectl get pods -n production
kubectl logs -f deployment/ml-api -n production
kubectl scale deployment ml-api --replicas=5

# Monitoring
curl http://localhost:5000/health
curl http://localhost:5000/metrics

# Database backup
pg_dump -h localhost -U postgres ml_production > backup.sql

# Redis backup
redis-cli --rdb /backup/dump.rdb
```

## Support and Troubleshooting

For deployment issues:
1. Check application logs
2. Verify environment variables
3. Test database connectivity
4. Check resource limits
5. Review security groups/firewall rules

For more help, see [Troubleshooting Guide](troubleshooting.md).