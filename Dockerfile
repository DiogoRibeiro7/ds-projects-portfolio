# Multi-stage Dockerfile for Data Science Portfolio

# Stage 1: Base image with system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

WORKDIR /tmp

# Copy requirements files
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install gunicorn uvicorn[standard]

# Stage 3: Application
FROM base as application

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy installed dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Stage 4: ML API Service
FROM application as ml-api

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run ML API
CMD ["uvicorn", "src.api.ml_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# Stage 5: Dashboard Service
FROM application as dashboard

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run Dashboard
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "dashboard_enhanced.app:app"]

# Stage 6: Jupyter Notebook Environment (for development)
FROM application as notebook

# Install Jupyter
RUN pip install jupyter jupyterlab ipywidgets

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 7: Production optimized image
FROM python:3.10-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy only necessary files from application stage
COPY --from=application --chown=appuser:appuser /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=application --chown=appuser:appuser /usr/local/bin /usr/local/bin
COPY --from=application --chown=appuser:appuser /app/src /app/src
COPY --from=application --chown=appuser:appuser /app/dashboard_enhanced /app/dashboard_enhanced
COPY --from=application --chown=appuser:appuser /app/statistical_methods /app/statistical_methods
COPY --from=application --chown=appuser:appuser /app/modern_bank_churn /app/modern_bank_churn

# Copy configuration files
COPY --chown=appuser:appuser pyproject.toml setup.py setup.cfg ./

# Switch to non-root user
USER appuser

# Default command (can be overridden)
CMD ["uvicorn", "src.api.ml_api:app", "--host", "0.0.0.0", "--port", "8000"]