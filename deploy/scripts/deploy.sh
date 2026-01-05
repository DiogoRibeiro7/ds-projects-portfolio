#!/bin/bash

# DS Portfolio Deployment Script
# Automates deployment to Kubernetes cluster

set -e

# Configuration
NAMESPACE="ml-portfolio"
REGISTRY="${REGISTRY:-your-registry}"
VERSION="${VERSION:-latest}"
CLUSTER="${CLUSTER:-production}"
ENVIRONMENT="${ENVIRONMENT:-prod}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi

    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi

    # Check if helm is installed (optional)
    if ! command -v helm &> /dev/null; then
        log_warning "helm is not installed (optional)"
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."

    # Build ML API image
    log_info "Building ML API image..."
    docker build -t ${REGISTRY}/ds-portfolio-ml-api:${VERSION} \
        -f docker/Dockerfile.ml-api .

    # Build Dashboard image
    log_info "Building Dashboard image..."
    docker build -t ${REGISTRY}/ds-portfolio-dashboard:${VERSION} \
        -f docker/Dockerfile.dashboard .

    # Build AutoML image
    log_info "Building AutoML image..."
    docker build -t ${REGISTRY}/ds-portfolio-automl:${VERSION} \
        -f docker/Dockerfile.automl .

    log_success "Docker images built successfully"
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."

    docker push ${REGISTRY}/ds-portfolio-ml-api:${VERSION}
    docker push ${REGISTRY}/ds-portfolio-dashboard:${VERSION}
    docker push ${REGISTRY}/ds-portfolio-automl:${VERSION}

    log_success "Images pushed to registry"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace ${NAMESPACE}..."

    kubectl apply -f kubernetes/namespace.yaml

    # Wait for namespace to be active
    kubectl wait --for=condition=Active namespace/${NAMESPACE} --timeout=30s

    log_success "Namespace created"
}

# Deploy secrets and configmaps
deploy_configs() {
    log_info "Deploying configurations..."

    # Check if secrets exist, create if not
    if ! kubectl get secret ml-portfolio-secrets -n ${NAMESPACE} &> /dev/null; then
        log_info "Creating secrets..."
        kubectl apply -f kubernetes/configmap.yaml
    else
        log_warning "Secrets already exist, skipping..."
    fi

    log_success "Configurations deployed"
}

# Deploy database and storage
deploy_storage() {
    log_info "Deploying database and storage..."

    kubectl apply -f kubernetes/database.yaml

    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n ${NAMESPACE} --timeout=120s

    log_success "Database and storage deployed"
}

# Deploy ML API
deploy_ml_api() {
    log_info "Deploying ML API..."

    # Update image tag
    sed -i "s|your-registry/ds-portfolio-ml-api:latest|${REGISTRY}/ds-portfolio-ml-api:${VERSION}|g" \
        kubernetes/ml-api-deployment.yaml

    kubectl apply -f kubernetes/ml-api-deployment.yaml

    # Wait for deployment to be ready
    kubectl rollout status deployment/ml-api -n ${NAMESPACE} --timeout=300s

    log_success "ML API deployed"
}

# Deploy AutoML
deploy_automl() {
    log_info "Deploying AutoML with Ray..."

    # Update image tag
    sed -i "s|your-registry/ds-portfolio-automl:latest|${REGISTRY}/ds-portfolio-automl:${VERSION}|g" \
        kubernetes/automl-deployment.yaml

    kubectl apply -f kubernetes/automl-deployment.yaml

    # Wait for Ray head to be ready
    kubectl wait --for=condition=ready pod -l app=automl-ray-head -n ${NAMESPACE} --timeout=300s

    log_success "AutoML deployed"
}

# Deploy Ingress
deploy_ingress() {
    log_info "Deploying Ingress..."

    kubectl apply -f kubernetes/ingress.yaml

    # Get ingress IP
    INGRESS_IP=$(kubectl get ingress ml-portfolio-ingress -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

    if [ -n "$INGRESS_IP" ]; then
        log_success "Ingress deployed at IP: ${INGRESS_IP}"
    else
        log_warning "Ingress deployed but external IP not yet assigned"
    fi
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."

    # Test ML API health
    API_POD=$(kubectl get pod -l app=ml-api -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -it ${API_POD} -n ${NAMESPACE} -- curl -f http://localhost:8000/health || {
        log_error "ML API health check failed"
        exit 1
    }

    # Test database connectivity
    kubectl exec -it ${API_POD} -n ${NAMESPACE} -- python -c "
import psycopg2
conn = psycopg2.connect(host='postgres-service', port=5432, database='mldb',
                        user='mluser', password='changeme-prod-password')
print('Database connection successful')
conn.close()
" || {
        log_error "Database connectivity test failed"
        exit 1
    }

    log_success "Smoke tests passed"
}

# Rollback deployment
rollback() {
    log_warning "Rolling back deployment..."

    kubectl rollout undo deployment/ml-api -n ${NAMESPACE}
    kubectl rollout undo statefulset/automl-ray-head -n ${NAMESPACE}

    log_success "Rollback completed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment to ${CLUSTER} cluster..."

    check_prerequisites

    # Build and push images if not using pre-built images
    if [ "${BUILD_IMAGES}" = "true" ]; then
        build_images
        push_images
    fi

    # Deploy components
    create_namespace
    deploy_configs
    deploy_storage
    deploy_ml_api
    deploy_automl
    deploy_ingress

    # Run tests
    if [ "${SKIP_TESTS}" != "true" ]; then
        run_smoke_tests
    fi

    log_success "Deployment completed successfully!"

    # Print access information
    echo ""
    log_info "Access Information:"
    echo "  ML API: http://api.ml-portfolio.example.com"
    echo "  Dashboard: http://dashboard.ml-portfolio.example.com"
    echo "  Ray Dashboard: http://ray.ml-portfolio.example.com"
    echo "  Grafana: http://grafana.ml-portfolio.example.com"
    echo ""
    echo "To check deployment status:"
    echo "  kubectl get all -n ${NAMESPACE}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --build)
            BUILD_IMAGES="true"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --rollback)
            rollback
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --registry <registry>  Docker registry (default: your-registry)"
            echo "  --version <version>    Image version tag (default: latest)"
            echo "  --cluster <cluster>    Target cluster (default: production)"
            echo "  --build                Build images before deployment"
            echo "  --skip-tests           Skip smoke tests"
            echo "  --rollback             Rollback to previous version"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run deployment
deploy