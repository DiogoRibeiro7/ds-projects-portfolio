# MLflow Server Docker Image
FROM python:3.9-slim

WORKDIR /mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.0 \
    azure-storage-blob==12.19.0 \
    google-cloud-storage==2.13.0 \
    pymysql==1.1.0 \
    protobuf==4.25.1 \
    sqlalchemy==2.0.23

# Create MLflow user
RUN useradd -m -u 1000 mlflow && \
    mkdir -p /mlflow/artifacts && \
    chown -R mlflow:mlflow /mlflow

USER mlflow

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_BACKEND_STORE_URI=${BACKEND_STORE_URI} \
    MLFLOW_DEFAULT_ARTIFACT_ROOT=${DEFAULT_ARTIFACT_ROOT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose MLflow port
EXPOSE 5000

# Start MLflow server
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
    --serve-artifacts