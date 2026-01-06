# 📚 API Reference

Complete API documentation for all modules in the Data Science Portfolio.

## 📑 Table of Contents

1. [ML Pipeline APIs](#ml-pipeline-apis)
2. [Statistical Methods APIs](#statistical-methods-apis)
3. [Dashboard APIs](#dashboard-apis)
4. [REST API Endpoints](#rest-api-endpoints)
5. [GraphQL Schema](#graphql-schema)
6. [WebSocket Events](#websocket-events)

---

## 🤖 ML Pipeline APIs

### MLPipelineOrchestrator

```python
from modern_bank_churn.ml_pipeline_orchestrator import MLPipelineOrchestrator

orchestrator = MLPipelineOrchestrator(config)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `run_pipeline(data)` | Execute complete ML pipeline | `data`: pd.DataFrame | `PipelineResults` |
| `train_models(X, y)` | Train all configured models | `X`: features, `y`: labels | `ModelResults` |
| `evaluate_models(models, X_test, y_test)` | Evaluate model performance | `models`: list, `X_test`, `y_test` | `EvaluationResults` |
| `select_best_model(results)` | Select optimal model | `results`: EvaluationResults | `Model` |
| `save_pipeline(path)` | Save pipeline configuration | `path`: str | `None` |
| `load_pipeline(path)` | Load pipeline configuration | `path`: str | `MLPipelineOrchestrator` |

#### Example

```python
# Initialize with configuration
config = PipelineConfig(
    feature_selection_method='boruta',
    model_type='ensemble',
    hyperparameter_tuning=True,
    cross_validation_folds=5
)

orchestrator = MLPipelineOrchestrator(config)

# Run complete pipeline
results = orchestrator.run_pipeline(train_data)
print(f"Best model: {results.best_model}")
print(f"AUC-ROC: {results.metrics['auc_roc']:.4f}")

# Save pipeline
orchestrator.save_pipeline('models/pipeline_v1.pkl')
```

### FeatureEngineer

```python
from modern_bank_churn.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `create_features(df)` | Generate engineered features | `df`: pd.DataFrame | pd.DataFrame |
| `create_interaction_features(df, cols)` | Create feature interactions | `df`, `cols`: list | pd.DataFrame |
| `create_polynomial_features(df, degree)` | Generate polynomial features | `df`, `degree`: int | pd.DataFrame |
| `create_time_features(df, date_col)` | Extract time-based features | `df`, `date_col`: str | pd.DataFrame |
| `encode_categorical(df, method)` | Encode categorical variables | `df`, `method`: str | pd.DataFrame |
| `select_features(X, y, method, k)` | Feature selection | `X`, `y`, `method`, `k` | list |

### ModelEvaluator

```python
from modern_bank_churn.evaluation_enhancements import ModelEvaluator

evaluator = ModelEvaluator()
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `evaluate(model, X_test, y_test)` | Comprehensive evaluation | `model`, `X_test`, `y_test` | dict |
| `cross_validate(model, X, y, cv)` | Cross-validation | `model`, `X`, `y`, `cv`: int | dict |
| `calculate_metrics(y_true, y_pred)` | Calculate all metrics | `y_true`, `y_pred` | dict |
| `plot_roc_curve(y_true, y_score)` | ROC curve visualization | `y_true`, `y_score` | Figure |
| `plot_confusion_matrix(y_true, y_pred)` | Confusion matrix | `y_true`, `y_pred` | Figure |
| `feature_importance(model, X)` | Feature importance analysis | `model`, `X` | pd.DataFrame |

---

## 📊 Statistical Methods APIs

### StatisticalAnalyzer

```python
from statistical_methods.statistical_analyzer import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(data=df, confidence_level=0.95)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `generate_summary()` | Comprehensive statistics | `include_plots`: bool | dict |
| `distribution_analysis(column)` | Test distributions | `column`: str, `test_distributions`: list | dict |
| `correlation_analysis(method)` | Correlation with significance | `method`: str, `threshold`: float | pd.DataFrame |
| `outlier_detection(method)` | Detect outliers | `method`: str, `columns`: list | pd.DataFrame |
| `normality_test(column)` | Test for normality | `column`: str | dict |

### HypothesisTester

```python
from statistical_methods.hypothesis_tester import HypothesisTester

tester = HypothesisTester(alpha=0.05, correction_method='bonferroni')
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `t_test(group1, group2)` | Two-sample t-test | `group1`, `group2`, `alternative`, `paired` | dict |
| `anova(data, groups, values)` | One-way ANOVA | `data`, `groups`, `values`, `post_hoc` | dict |
| `chi_square_test(observed)` | Chi-square test | `observed`, `expected` | dict |
| `mann_whitney_u(group1, group2)` | Non-parametric test | `group1`, `group2` | dict |
| `kruskal_wallis(*groups)` | Non-parametric ANOVA | `*groups` | dict |
| `apply_correction(results)` | Multiple testing correction | `results`: dict | dict |

### CausalInference

```python
from statistical_methods.causal_inference import CausalInference

ci = CausalInference(method='propensity')
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `estimate_ate(treatment, outcome, confounders)` | Average Treatment Effect | Various | dict |
| `propensity_score_matching(data, treatment_col)` | PS matching | Various | pd.DataFrame |
| `instrumental_variable(data, instrument)` | IV estimation | Various | dict |
| `regression_discontinuity(data, cutoff)` | RDD analysis | Various | dict |
| `sensitivity_analysis(effect, range)` | Sensitivity to confounding | Various | dict |

---

## 🎨 Dashboard APIs

### EnhancedDashboard

```python
from dashboard_framework import EnhancedDashboard, DashboardConfig

dashboard = EnhancedDashboard(config)
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `register_data_source(name, func)` | Add data source | `name`: str, `func`: callable | None |
| `add_chart(chart_id, chart_type)` | Add visualization | `chart_id`, `chart_type`, `**kwargs` | None |
| `add_filter(filter_id, type)` | Add filter component | `filter_id`, `type`, `**kwargs` | None |
| `add_page(page_id, layout_func)` | Add dashboard page | `page_id`, `layout_func` | None |
| `enable_realtime(channel)` | Enable real-time updates | `channel`, `interval` | None |
| `export_to_pdf(figures, filename)` | Export to PDF | `figures`: list, `filename` | str |
| `run(host, port, debug)` | Start dashboard server | Various | None |

### InteractiveVisualizations

```python
from visualization_components import InteractiveVisualizations

viz = InteractiveVisualizations()
```

#### Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `create_animated_time_series(df, x_col, y_cols)` | Animated line chart | Various | Figure |
| `create_interactive_3d_scatter(df, x, y, z)` | 3D scatter plot | Various | Figure |
| `create_sankey_diagram(df, source, target)` | Flow visualization | Various | Figure |
| `create_sunburst_chart(df, path_cols)` | Hierarchical chart | Various | Figure |
| `create_parallel_coordinates(df, dimensions)` | Multi-dimensional | Various | Figure |

---

## 🌐 REST API Endpoints

### Authentication

#### POST `/api/auth/register`
Register new user.

**Request Body:**
```json
{
    "username": "string",
    "password": "string",
    "email": "string"
}
```

**Response:**
```json
{
    "user_id": "string",
    "username": "string",
    "created_at": "datetime"
}
```

#### POST `/api/auth/login`
User login.

**Request:**
```http
Authorization: Basic base64(username:password)
```

**Response:**
```json
{
    "token": "jwt_token",
    "expires_in": 3600,
    "user": {
        "id": "string",
        "username": "string"
    }
}
```

### Data Operations

#### GET `/api/v1/data`
Retrieve data with filtering.

**Query Parameters:**
- `start_date`: ISO date string
- `end_date`: ISO date string
- `category`: string (multiple allowed)
- `limit`: integer
- `offset`: integer

**Headers:**
```http
Authorization: Bearer jwt_token
```

**Response:**
```json
{
    "data": [...],
    "total": 1000,
    "page": 1,
    "per_page": 100
}
```

#### POST `/api/v1/data`
Create new data entry.

**Request Body:**
```json
{
    "metric": "string",
    "value": "number",
    "timestamp": "datetime",
    "metadata": {}
}
```

**Response:**
```json
{
    "id": "string",
    "created_at": "datetime",
    "status": "success"
}
```

#### PUT `/api/v1/data/{id}`
Update data entry.

**Request Body:**
```json
{
    "value": "number",
    "metadata": {}
}
```

**Response:**
```json
{
    "id": "string",
    "updated_at": "datetime",
    "status": "success"
}
```

#### DELETE `/api/v1/data/{id}`
Delete data entry.

**Response:**
```json
{
    "deleted": true,
    "deleted_at": "datetime"
}
```

### Analytics

#### GET `/api/v1/analytics/summary`
Get analytical summary.

**Query Parameters:**
- `metrics`: comma-separated list
- `group_by`: string
- `aggregation`: string (mean, sum, count)

**Response:**
```json
{
    "summary": {
        "total_records": 10000,
        "date_range": {
            "start": "2024-01-01",
            "end": "2024-12-31"
        },
        "metrics": {...}
    }
}
```

#### POST `/api/v1/analytics/forecast`
Generate forecast.

**Request Body:**
```json
{
    "metric": "string",
    "periods": 30,
    "method": "prophet",
    "confidence_level": 0.95
}
```

**Response:**
```json
{
    "forecast": [...],
    "confidence_intervals": {...},
    "model_info": {...}
}
```

---

## 🔮 GraphQL Schema

### Queries

```graphql
type Query {
    # Get all metrics
    metrics(
        startDate: DateTime
        endDate: DateTime
        category: String
    ): [Metric!]!

    # Get specific metric
    metric(id: ID!): Metric

    # Get dashboard data
    dashboard(
        filters: DashboardFilters
    ): DashboardData!

    # Search data
    search(
        query: String!
        limit: Int = 10
    ): SearchResults!
}
```

### Mutations

```graphql
type Mutation {
    # Create metric
    createMetric(input: MetricInput!): Metric!

    # Update metric
    updateMetric(id: ID!, input: MetricInput!): Metric!

    # Delete metric
    deleteMetric(id: ID!): DeleteResponse!

    # Run analysis
    runAnalysis(
        type: AnalysisType!
        parameters: JSON
    ): AnalysisResult!
}
```

### Types

```graphql
type Metric {
    id: ID!
    name: String!
    value: Float!
    timestamp: DateTime!
    category: String
    metadata: JSON
}

input MetricInput {
    name: String!
    value: Float!
    category: String
    metadata: JSON
}

type DashboardData {
    charts: [ChartData!]!
    filters: [Filter!]!
    summary: Summary!
}

type AnalysisResult {
    id: ID!
    type: AnalysisType!
    results: JSON!
    created_at: DateTime!
}

enum AnalysisType {
    REGRESSION
    CLASSIFICATION
    CLUSTERING
    TIME_SERIES
}
```

### Example Query

```graphql
query GetDashboardData {
    dashboard(filters: {
        startDate: "2024-01-01"
        endDate: "2024-12-31"
        categories: ["sales", "marketing"]
    }) {
        charts {
            id
            type
            data
        }
        summary {
            totalRevenue
            totalOrders
            conversionRate
        }
    }
}
```

---

## 🔌 WebSocket Events

### Client to Server Events

#### `subscribe`
Subscribe to data channel.

```javascript
socket.emit('subscribe', {
    channel: 'metrics',
    filters: {
        category: 'sales'
    }
});
```

#### `unsubscribe`
Unsubscribe from channel.

```javascript
socket.emit('unsubscribe', {
    channel: 'metrics'
});
```

#### `request_data`
Request specific data.

```javascript
socket.emit('request_data', {
    type: 'latest_metrics',
    limit: 10
});
```

### Server to Client Events

#### `realtime_data`
Real-time data update.

```javascript
socket.on('realtime_data', (data) => {
    console.log('New data:', data);
    // data = {
    //     channel: 'metrics',
    //     timestamp: '2024-01-01T00:00:00Z',
    //     payload: {...}
    // }
});
```

#### `notification`
System notification.

```javascript
socket.on('notification', (notification) => {
    console.log('Notification:', notification);
    // notification = {
    //     type: 'info',
    //     message: 'Data updated',
    //     timestamp: '...'
    // }
});
```

#### `error`
Error event.

```javascript
socket.on('error', (error) => {
    console.error('Error:', error);
    // error = {
    //     code: 'SUBSCRIPTION_FAILED',
    //     message: 'Invalid channel',
    //     details: {...}
    // }
});
```

### Connection Management

```javascript
// Connect with authentication
const socket = io('http://localhost:5000', {
    auth: {
        token: 'jwt_token'
    }
});

// Handle connection events
socket.on('connect', () => {
    console.log('Connected:', socket.id);
});

socket.on('disconnect', (reason) => {
    console.log('Disconnected:', reason);
});

socket.on('reconnect', (attemptNumber) => {
    console.log('Reconnected after', attemptNumber, 'attempts');
});
```

---

## 🔐 Authentication & Security

### JWT Token Structure

```json
{
    "header": {
        "alg": "HS256",
        "typ": "JWT"
    },
    "payload": {
        "user_id": "string",
        "username": "string",
        "roles": ["user", "admin"],
        "exp": 1234567890,
        "iat": 1234567890
    },
    "signature": "..."
}
```

### Rate Limiting

Default limits per endpoint:

| Endpoint | Rate Limit |
|----------|------------|
| `/api/auth/*` | 10 per minute |
| `/api/v1/data` (GET) | 1000 per hour |
| `/api/v1/data` (POST) | 100 per hour |
| `/api/v1/analytics/*` | 100 per hour |
| `/graphql` | 500 per hour |

### Error Responses

Standard error format:

```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable message",
        "details": {
            "field": "Additional context"
        },
        "timestamp": "2024-01-01T00:00:00Z",
        "request_id": "uuid"
    }
}
```

Common error codes:

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## 📦 SDK Examples

### Python SDK

```python
from portfolio_sdk import PortfolioClient

# Initialize client
client = PortfolioClient(
    base_url='http://localhost:5000',
    api_key='your_api_key'
)

# Get data
data = client.data.list(
    start_date='2024-01-01',
    end_date='2024-12-31',
    limit=100
)

# Run analysis
result = client.analytics.forecast(
    metric='revenue',
    periods=30,
    method='prophet'
)

# WebSocket streaming
@client.on('realtime_data')
def handle_data(data):
    print(f"Received: {data}")

client.subscribe('metrics')
client.start()
```

### JavaScript SDK

```javascript
import { PortfolioClient } from 'portfolio-sdk';

// Initialize client
const client = new PortfolioClient({
    baseURL: 'http://localhost:5000',
    apiKey: 'your_api_key'
});

// Get data
const data = await client.data.list({
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    limit: 100
});

// GraphQL query
const result = await client.graphql.query(`
    query {
        metrics(category: "sales") {
            id
            name
            value
        }
    }
`);

// WebSocket streaming
client.on('realtime_data', (data) => {
    console.log('Received:', data);
});

client.subscribe('metrics');
```

---

## 📄 API Versioning

The API uses URL versioning:

- Current version: `v1`
- Base URL: `/api/v1/`
- Deprecated versions are supported for 6 months
- Version header: `X-API-Version: 1`

### Migration Guide

When migrating from older versions:

```python
# Old (deprecated)
response = requests.get('/api/data')

# New (v1)
response = requests.get('/api/v1/data')
```

---

## 🔧 Testing APIs

### Using curl

```bash
# Authentication
TOKEN=$(curl -X POST http://localhost:5000/api/auth/login \
    -u username:password \
    | jq -r '.token')

# GET request
curl -H "Authorization: Bearer $TOKEN" \
    http://localhost:5000/api/v1/data

# POST request
curl -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"metric": "revenue", "value": 1000}' \
    http://localhost:5000/api/v1/data
```

### Using Postman

Import the Postman collection:
```json
{
    "info": {
        "name": "Portfolio API",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    },
    "auth": {
        "type": "bearer",
        "bearer": [{
            "key": "token",
            "value": "{{jwt_token}}",
            "type": "string"
        }]
    },
    "item": [...]
}
```

---

For more details, see:
- [ML Pipeline Module](modules/ml_pipeline.md)
- [Statistical Methods Module](modules/statistics.md)
- [Dashboard Module](modules/dashboard.md)
- [Main Documentation](../README_ENHANCED.md)