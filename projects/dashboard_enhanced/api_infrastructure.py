"""API Infrastructure for Dashboard

REST API, GraphQL endpoint, WebSocket support, authentication, and documentation
for the enhanced dashboard system.

Author: Portfolio Team
Date: 2024
"""

import json
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import graphene
import jwt
import numpy as np
import pandas as pd
import redis
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth, HTTPTokenAuth
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restful import Api, Resource
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_swagger_ui import get_swaggerui_blueprint
from graphene import Field, Float, Int, ObjectType, Schema, String
from graphene import List as GraphList
from werkzeug.security import check_password_hash, generate_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
@dataclass
class APIConfig:
    """API Configuration"""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = True

    # Security
    secret_key: str = "your-secret-key-here"
    jwt_secret: str = "your-jwt-secret-here"
    jwt_expiration_hours: int = 24

    # Rate limiting
    rate_limit_default: str = "100 per hour"
    rate_limit_burst: str = "10 per minute"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # CORS
    cors_origins: list[str] = None

    # API Documentation
    api_title: str = "Dashboard API"
    api_version: str = "v1"
    swagger_url: str = "/api/docs"
    api_url: str = "/static/swagger.json"


class DashboardAPI:
    """Complete API infrastructure for dashboard with REST, GraphQL,
    WebSocket, authentication, and documentation.
    """

    def __init__(self, config: APIConfig | None = None):
        """Initialize API infrastructure.

        Args:
            config: API configuration
        """
        self.config = config or APIConfig()

        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = self.config.secret_key

        # Initialize extensions
        CORS(self.app, origins=self.config.cors_origins or ["*"])
        self.api = Api(self.app)
        self.socketio = SocketIO(
            self.app, cors_allowed_origins="*", async_mode="threading"
        )

        # Initialize rate limiter
        self.limiter = Limiter(
            self.app,
            key_func=get_remote_address,
            default_limits=[self.config.rate_limit_default],
            storage_uri=f"redis://{self.config.redis_host}:{self.config.redis_port}",
        )

        # Initialize authentication
        self.basic_auth = HTTPBasicAuth()
        self.token_auth = HTTPTokenAuth(scheme="Bearer")

        # User store (in production, use database)
        self.users = {}
        self.tokens = {}

        # Initialize Redis for real-time data
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True,
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except:
            logger.warning("Redis not available")
            self.redis_client = None

        # Setup routes
        self._setup_rest_api()
        self._setup_graphql()
        self._setup_websocket()
        self._setup_authentication()
        self._setup_swagger()

        logger.info("API Infrastructure initialized")

    def _setup_authentication(self):
        """Setup authentication handlers."""

        @self.basic_auth.verify_password
        def verify_password(username, password):
            if username in self.users:
                return check_password_hash(self.users[username]["password"], password)
            return False

        @self.token_auth.verify_token
        def verify_token(token):
            try:
                payload = jwt.decode(
                    token, self.config.jwt_secret, algorithms=["HS256"]
                )
                username = payload.get("username")
                if username and username in self.users:
                    return username
            except jwt.ExpiredSignatureError:
                logger.warning("Expired token")
            except jwt.InvalidTokenError:
                logger.warning("Invalid token")
            return None

        # User registration endpoint
        @self.app.route("/api/auth/register", methods=["POST"])
        @self.limiter.limit("5 per hour")
        def register():
            data = request.get_json()
            username = data.get("username")
            password = data.get("password")
            email = data.get("email")

            if not username or not password:
                return jsonify({"error": "Username and password required"}), 400

            if username in self.users:
                return jsonify({"error": "User already exists"}), 409

            # Create user
            self.users[username] = {
                "password": generate_password_hash(password),
                "email": email,
                "created_at": datetime.utcnow().isoformat(),
                "api_key": secrets.token_urlsafe(32),
            }

            return jsonify(
                {
                    "message": "User created successfully",
                    "api_key": self.users[username]["api_key"],
                }
            ), 201

        # Login endpoint
        @self.app.route("/api/auth/login", methods=["POST"])
        @self.limiter.limit("10 per hour")
        def login():
            auth = request.authorization

            if not auth or not auth.username or not auth.password:
                return jsonify({"error": "Authentication required"}), 401

            if not verify_password(auth.username, auth.password):
                return jsonify({"error": "Invalid credentials"}), 401

            # Generate JWT token
            payload = {
                "username": auth.username,
                "exp": datetime.utcnow()
                + timedelta(hours=self.config.jwt_expiration_hours),
            }
            token = jwt.encode(payload, self.config.jwt_secret, algorithm="HS256")

            self.tokens[token] = {
                "username": auth.username,
                "created_at": datetime.utcnow().isoformat(),
            }

            return jsonify(
                {"token": token, "expires_in": self.config.jwt_expiration_hours * 3600}
            ), 200

        # Logout endpoint
        @self.app.route("/api/auth/logout", methods=["POST"])
        @self.token_auth.login_required
        def logout():
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            if token in self.tokens:
                del self.tokens[token]
            return jsonify({"message": "Logged out successfully"}), 200

    def _setup_rest_api(self):
        """Setup REST API endpoints."""

        class DataResource(Resource):
            """Resource for data operations."""

            decorators = [self.token_auth.login_required]

            def get(self, dataset_id=None):
                """Get data from specified dataset."""
                if dataset_id:
                    # Get specific dataset
                    data = self._get_dataset(dataset_id)
                    if data is None:
                        return {"error": "Dataset not found"}, 404
                    return {"data": data}, 200
                else:
                    # List all datasets
                    datasets = self._list_datasets()
                    return {"datasets": datasets}, 200

            def post(self):
                """Upload new dataset."""
                data = request.get_json()
                dataset_id = data.get("id")
                dataset = data.get("data")

                if not dataset_id or not dataset:
                    return {"error": "Dataset ID and data required"}, 400

                # Store dataset
                self._store_dataset(dataset_id, dataset)

                return {"message": "Dataset uploaded", "id": dataset_id}, 201

            def put(self, dataset_id):
                """Update existing dataset."""
                data = request.get_json()
                dataset = data.get("data")

                if not dataset:
                    return {"error": "Dataset data required"}, 400

                # Update dataset
                self._update_dataset(dataset_id, dataset)

                return {"message": "Dataset updated", "id": dataset_id}, 200

            def delete(self, dataset_id):
                """Delete dataset."""
                if self._delete_dataset(dataset_id):
                    return {"message": "Dataset deleted"}, 200
                return {"error": "Dataset not found"}, 404

            def _get_dataset(self, dataset_id):
                """Get dataset from storage."""
                if self.redis_client:
                    data = self.redis_client.get(f"dataset:{dataset_id}")
                    if data:
                        return json.loads(data)
                return None

            def _list_datasets(self):
                """List all available datasets."""
                if self.redis_client:
                    keys = self.redis_client.keys("dataset:*")
                    return [key.split(":")[1] for key in keys]
                return []

            def _store_dataset(self, dataset_id, dataset):
                """Store dataset."""
                if self.redis_client:
                    self.redis_client.set(
                        f"dataset:{dataset_id}",
                        json.dumps(dataset),
                        ex=3600,  # Expire after 1 hour
                    )

            def _update_dataset(self, dataset_id, dataset):
                """Update dataset."""
                self._store_dataset(dataset_id, dataset)

            def _delete_dataset(self, dataset_id):
                """Delete dataset."""
                if self.redis_client:
                    return self.redis_client.delete(f"dataset:{dataset_id}") > 0
                return False

        class MetricsResource(Resource):
            """Resource for metrics operations."""

            decorators = [self.token_auth.login_required]

            def get(self, metric_name=None):
                """Get metrics data."""
                # Generate sample metrics
                metrics = {
                    "cpu_usage": np.random.uniform(20, 80),
                    "memory_usage": np.random.uniform(30, 70),
                    "request_count": np.random.randint(100, 1000),
                    "response_time": np.random.uniform(10, 100),
                    "error_rate": np.random.uniform(0, 5),
                }

                if metric_name:
                    if metric_name in metrics:
                        return {
                            "metric": metric_name,
                            "value": metrics[metric_name],
                        }, 200
                    return {"error": "Metric not found"}, 404

                return {"metrics": metrics}, 200

        class AnalyticsResource(Resource):
            """Resource for analytics operations."""

            decorators = [self.token_auth.login_required]

            def post(self):
                """Run analytics query."""
                data = request.get_json()
                query_type = data.get("type")
                parameters = data.get("parameters", {})

                # Process analytics query
                if query_type == "aggregate":
                    result = self._run_aggregate_query(parameters)
                elif query_type == "timeseries":
                    result = self._run_timeseries_query(parameters)
                elif query_type == "correlation":
                    result = self._run_correlation_query(parameters)
                else:
                    return {"error": "Invalid query type"}, 400

                return {"result": result}, 200

            def _run_aggregate_query(self, params):
                """Run aggregate analytics query."""
                # Sample implementation
                return {
                    "sum": np.random.uniform(1000, 10000),
                    "avg": np.random.uniform(10, 100),
                    "max": np.random.uniform(100, 1000),
                    "min": np.random.uniform(1, 10),
                    "count": np.random.randint(100, 1000),
                }

            def _run_timeseries_query(self, params):
                """Run time series analytics query."""
                # Sample implementation
                dates = pd.date_range("2024-01-01", periods=30)
                return {
                    "dates": [d.isoformat() for d in dates],
                    "values": np.random.uniform(10, 100, 30).tolist(),
                }

            def _run_correlation_query(self, params):
                """Run correlation analytics query."""
                # Sample implementation
                return {
                    "correlation_matrix": np.random.uniform(-1, 1, (5, 5)).tolist(),
                    "features": ["A", "B", "C", "D", "E"],
                }

        # Register resources
        self.api.add_resource(
            DataResource, "/api/v1/data", "/api/v1/data/<string:dataset_id>"
        )
        self.api.add_resource(
            MetricsResource, "/api/v1/metrics", "/api/v1/metrics/<string:metric_name>"
        )
        self.api.add_resource(AnalyticsResource, "/api/v1/analytics")

        # Health check endpoint
        @self.app.route("/api/health")
        def health_check():
            """Health check endpoint."""
            status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": self.config.api_version,
                "services": {
                    "api": "up",
                    "redis": "up"
                    if self.redis_client and self.redis_client.ping()
                    else "down",
                    "websocket": "up",
                },
            }
            return jsonify(status), 200

    def _setup_graphql(self):
        """Setup GraphQL endpoint."""

        # Define GraphQL types
        class MetricType(ObjectType):
            name = String()
            value = Float()
            timestamp = String()

        class DatasetType(ObjectType):
            id = String()
            name = String()
            rows = Int()
            columns = Int()
            created_at = String()

        class AnalyticsResultType(ObjectType):
            query_type = String()
            result = String()  # JSON string
            execution_time = Float()

        # Define queries
        class Query(ObjectType):
            metric = Field(MetricType, name=String())
            all_metrics = GraphList(MetricType)
            dataset = Field(DatasetType, id=String())
            all_datasets = GraphList(DatasetType)
            analytics = Field(
                AnalyticsResultType, query_type=String(), parameters=String()
            )

            def resolve_metric(self, info, name):
                """Resolve single metric."""
                value = np.random.uniform(0, 100)
                return MetricType(
                    name=name, value=value, timestamp=datetime.utcnow().isoformat()
                )

            def resolve_all_metrics(self, info):
                """Resolve all metrics."""
                metrics = ["cpu", "memory", "disk", "network", "requests"]
                return [
                    MetricType(
                        name=m,
                        value=np.random.uniform(0, 100),
                        timestamp=datetime.utcnow().isoformat(),
                    )
                    for m in metrics
                ]

            def resolve_dataset(self, info, id):
                """Resolve single dataset."""
                return DatasetType(
                    id=id,
                    name=f"Dataset {id}",
                    rows=np.random.randint(100, 10000),
                    columns=np.random.randint(5, 50),
                    created_at=datetime.utcnow().isoformat(),
                )

            def resolve_all_datasets(self, info):
                """Resolve all datasets."""
                return [
                    DatasetType(
                        id=f"ds_{i}",
                        name=f"Dataset {i}",
                        rows=np.random.randint(100, 10000),
                        columns=np.random.randint(5, 50),
                        created_at=datetime.utcnow().isoformat(),
                    )
                    for i in range(5)
                ]

            def resolve_analytics(self, info, query_type, parameters=None):
                """Resolve analytics query."""
                start_time = time.time()

                # Process query
                if query_type == "aggregate":
                    result = {"sum": 1000, "avg": 50, "count": 20}
                elif query_type == "trend":
                    result = {"trend": "increasing", "slope": 0.5}
                else:
                    result = {"error": "Unknown query type"}

                execution_time = time.time() - start_time

                return AnalyticsResultType(
                    query_type=query_type,
                    result=json.dumps(result),
                    execution_time=execution_time,
                )

        # Define mutations
        class CreateDataset(graphene.Mutation):
            class Arguments:
                id = String(required=True)
                name = String(required=True)
                data = String(required=True)  # JSON string

            dataset = Field(DatasetType)
            success = graphene.Boolean()
            message = String()

            def mutate(self, info, id, name, data):
                """Create new dataset."""
                try:
                    # Store dataset
                    # In production, store in database
                    dataset = DatasetType(
                        id=id,
                        name=name,
                        rows=100,
                        columns=10,
                        created_at=datetime.utcnow().isoformat(),
                    )

                    return CreateDataset(
                        dataset=dataset,
                        success=True,
                        message="Dataset created successfully",
                    )
                except Exception as e:
                    return CreateDataset(dataset=None, success=False, message=str(e))

        class UpdateMetric(graphene.Mutation):
            class Arguments:
                name = String(required=True)
                value = Float(required=True)

            metric = Field(MetricType)
            success = graphene.Boolean()

            def mutate(self, info, name, value):
                """Update metric value."""
                metric = MetricType(
                    name=name, value=value, timestamp=datetime.utcnow().isoformat()
                )

                # Store in Redis if available
                if self.redis_client:
                    self.redis_client.set(
                        f"metric:{name}",
                        json.dumps({"value": value, "timestamp": metric.timestamp}),
                        ex=3600,
                    )

                return UpdateMetric(metric=metric, success=True)

        class Mutation(ObjectType):
            create_dataset = CreateDataset.Field()
            update_metric = UpdateMetric.Field()

        # Create schema
        schema = Schema(query=Query, mutation=Mutation)

        # Add GraphQL endpoint without flask-graphql dependency
        def _graphql_view():
            if request.method == "GET":
                return """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>GraphiQL</title>
                    <style>body { margin: 0; }</style>
                </head>
                <body>
                    <div id="root"></div>
                    <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
                    <script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
                    <script src="https://unpkg.com/graphiql/graphiql.min.js"></script>
                    <link rel="stylesheet" href="https://unpkg.com/graphiql/graphiql.min.css" />
                    <script>
                        ReactDOM.render(
                            React.createElement(GraphiQL, {
                                fetcher: GraphiQL.createFetcher({url: '/graphql'}),
                            }),
                            document.getElementById('root')
                        );
                    </script>
                </body>
                </html>
                """

            payload = request.get_json() or {}
            query = payload.get("query")
            variables = payload.get("variables")
            result = schema.execute(query, variable_values=variables)
            response = {"data": result.data}
            if result.errors:
                response["errors"] = [str(e) for e in result.errors]
            return jsonify(response)

        self.app.add_url_rule(
            "/graphql",
            view_func=_graphql_view,
            methods=["GET", "POST"],
        )

    def _setup_websocket(self):
        """Setup WebSocket handlers."""

        @self.socketio.on("connect")
        def handle_connect():
            """Handle client connection."""
            client_id = request.sid
            logger.info(f"Client connected: {client_id}")
            emit("connected", {"client_id": client_id})

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Handle client disconnection."""
            client_id = request.sid
            logger.info(f"Client disconnected: {client_id}")

        @self.socketio.on("subscribe")
        def handle_subscribe(data):
            """Handle subscription to data channel."""
            channel = data.get("channel")
            client_id = request.sid

            if channel:
                join_room(channel)
                logger.info(f"Client {client_id} subscribed to {channel}")
                emit("subscribed", {"channel": channel})

                # Start sending real-time data
                self.socketio.start_background_task(
                    self._send_realtime_data, channel, client_id
                )

        @self.socketio.on("unsubscribe")
        def handle_unsubscribe(data):
            """Handle unsubscription from data channel."""
            channel = data.get("channel")
            client_id = request.sid

            if channel:
                leave_room(channel)
                logger.info(f"Client {client_id} unsubscribed from {channel}")
                emit("unsubscribed", {"channel": channel})

        @self.socketio.on("message")
        def handle_message(data):
            """Handle client message."""
            message_type = data.get("type")
            payload = data.get("payload")

            # Process message based on type
            if message_type == "query":
                result = self._process_query(payload)
                emit("query_result", result)
            elif message_type == "update":
                self._process_update(payload)
                # Broadcast update to all clients in room
                room = data.get("room", "global")
                emit("data_update", payload, room=room, include_self=False)

    def _send_realtime_data(self, channel: str, client_id: str):
        """Send real-time data to subscribed client."""
        while True:
            try:
                # Generate sample real-time data
                data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "channel": channel,
                    "value": np.random.uniform(0, 100),
                    "metrics": {
                        "cpu": np.random.uniform(20, 80),
                        "memory": np.random.uniform(30, 70),
                        "requests": np.random.randint(10, 100),
                    },
                }

                # Send to specific room
                self.socketio.emit("realtime_data", data, room=channel)

                # Sleep before next update
                self.socketio.sleep(5)

            except Exception as e:
                logger.error(f"Error sending real-time data: {e}")
                break

    def _process_query(self, payload: dict) -> dict:
        """Process WebSocket query."""
        query_type = payload.get("type")

        if query_type == "status":
            return {"status": "operational", "timestamp": datetime.utcnow().isoformat()}
        elif query_type == "metrics":
            return {
                "metrics": {
                    "active_connections": len(self.socketio.server.manager.rooms),
                    "messages_processed": np.random.randint(1000, 10000),
                }
            }
        else:
            return {"error": "Unknown query type"}

    def _process_update(self, payload: dict):
        """Process WebSocket update."""
        # Store update in Redis if available
        if self.redis_client:
            update_id = payload.get("id", "update")
            self.redis_client.set(f"update:{update_id}", json.dumps(payload), ex=3600)

    def _setup_swagger(self):
        """Setup Swagger documentation."""
        # Swagger configuration
        SWAGGER_URL = self.config.swagger_url
        API_URL = self.config.api_url

        swaggerui_blueprint = get_swaggerui_blueprint(
            SWAGGER_URL, API_URL, config={"app_name": self.config.api_title}
        )

        self.app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

        # Serve Swagger JSON
        @self.app.route("/static/swagger.json")
        def swagger_json():
            """Serve Swagger JSON specification."""
            swagger_spec = {
                "openapi": "3.0.0",
                "info": {
                    "title": self.config.api_title,
                    "version": self.config.api_version,
                    "description": "Enhanced Dashboard API with REST, GraphQL, and WebSocket support",
                },
                "servers": [
                    {
                        "url": f"http://{self.config.host}:{self.config.port}",
                        "description": "Development server",
                    }
                ],
                "components": {
                    "securitySchemes": {
                        "bearerAuth": {
                            "type": "http",
                            "scheme": "bearer",
                            "bearerFormat": "JWT",
                        },
                        "basicAuth": {"type": "http", "scheme": "basic"},
                    },
                    "schemas": {
                        "Dataset": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "data": {"type": "object"},
                                "created_at": {"type": "string", "format": "date-time"},
                            },
                        },
                        "Metric": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "number"},
                                "timestamp": {"type": "string", "format": "date-time"},
                            },
                        },
                        "Error": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "message": {"type": "string"},
                                "code": {"type": "integer"},
                            },
                        },
                    },
                },
                "paths": {
                    "/api/health": {
                        "get": {
                            "summary": "Health check",
                            "tags": ["System"],
                            "responses": {
                                "200": {"description": "System health status"}
                            },
                        }
                    },
                    "/api/auth/register": {
                        "post": {
                            "summary": "Register new user",
                            "tags": ["Authentication"],
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "username": {"type": "string"},
                                                "password": {"type": "string"},
                                                "email": {
                                                    "type": "string",
                                                    "format": "email",
                                                },
                                            },
                                            "required": ["username", "password"],
                                        }
                                    }
                                },
                            },
                            "responses": {
                                "201": {"description": "User created"},
                                "400": {"description": "Invalid request"},
                                "409": {"description": "User already exists"},
                            },
                        }
                    },
                    "/api/auth/login": {
                        "post": {
                            "summary": "Login user",
                            "tags": ["Authentication"],
                            "security": [{"basicAuth": []}],
                            "responses": {
                                "200": {
                                    "description": "Login successful",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "token": {"type": "string"},
                                                    "expires_in": {"type": "integer"},
                                                },
                                            }
                                        }
                                    },
                                },
                                "401": {"description": "Invalid credentials"},
                            },
                        }
                    },
                    "/api/v1/data": {
                        "get": {
                            "summary": "List all datasets",
                            "tags": ["Data"],
                            "security": [{"bearerAuth": []}],
                            "responses": {
                                "200": {
                                    "description": "List of datasets",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "datasets": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    }
                                                },
                                            }
                                        }
                                    },
                                }
                            },
                        },
                        "post": {
                            "summary": "Upload new dataset",
                            "tags": ["Data"],
                            "security": [{"bearerAuth": []}],
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/Dataset"
                                        }
                                    }
                                },
                            },
                            "responses": {
                                "201": {"description": "Dataset created"},
                                "400": {"description": "Invalid request"},
                            },
                        },
                    },
                    "/api/v1/data/{dataset_id}": {
                        "get": {
                            "summary": "Get specific dataset",
                            "tags": ["Data"],
                            "security": [{"bearerAuth": []}],
                            "parameters": [
                                {
                                    "name": "dataset_id",
                                    "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"},
                                }
                            ],
                            "responses": {
                                "200": {
                                    "description": "Dataset data",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "$ref": "#/components/schemas/Dataset"
                                            }
                                        }
                                    },
                                },
                                "404": {"description": "Dataset not found"},
                            },
                        },
                        "put": {
                            "summary": "Update dataset",
                            "tags": ["Data"],
                            "security": [{"bearerAuth": []}],
                            "parameters": [
                                {
                                    "name": "dataset_id",
                                    "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"},
                                }
                            ],
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "$ref": "#/components/schemas/Dataset"
                                        }
                                    }
                                },
                            },
                            "responses": {
                                "200": {"description": "Dataset updated"},
                                "404": {"description": "Dataset not found"},
                            },
                        },
                        "delete": {
                            "summary": "Delete dataset",
                            "tags": ["Data"],
                            "security": [{"bearerAuth": []}],
                            "parameters": [
                                {
                                    "name": "dataset_id",
                                    "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"},
                                }
                            ],
                            "responses": {
                                "200": {"description": "Dataset deleted"},
                                "404": {"description": "Dataset not found"},
                            },
                        },
                    },
                    "/api/v1/metrics": {
                        "get": {
                            "summary": "Get all metrics",
                            "tags": ["Metrics"],
                            "security": [{"bearerAuth": []}],
                            "responses": {
                                "200": {
                                    "description": "Metrics data",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "metrics": {"type": "object"}
                                                },
                                            }
                                        }
                                    },
                                }
                            },
                        }
                    },
                    "/api/v1/analytics": {
                        "post": {
                            "summary": "Run analytics query",
                            "tags": ["Analytics"],
                            "security": [{"bearerAuth": []}],
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "string"},
                                                "parameters": {"type": "object"},
                                            },
                                        }
                                    }
                                },
                            },
                            "responses": {
                                "200": {"description": "Analytics result"},
                                "400": {"description": "Invalid query"},
                            },
                        }
                    },
                    "/graphql": {
                        "post": {
                            "summary": "GraphQL endpoint",
                            "tags": ["GraphQL"],
                            "security": [{"bearerAuth": []}],
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "query": {"type": "string"},
                                                "variables": {"type": "object"},
                                            },
                                        }
                                    }
                                },
                            },
                            "responses": {"200": {"description": "GraphQL response"}},
                        }
                    },
                },
                "tags": [
                    {"name": "System", "description": "System operations"},
                    {
                        "name": "Authentication",
                        "description": "Authentication endpoints",
                    },
                    {"name": "Data", "description": "Data management endpoints"},
                    {"name": "Metrics", "description": "Metrics endpoints"},
                    {"name": "Analytics", "description": "Analytics endpoints"},
                    {"name": "GraphQL", "description": "GraphQL endpoint"},
                ],
            }

            return jsonify(swagger_spec)

    def run(self):
        """Run the API server."""
        logger.info(
            f"Starting API server on http://{self.config.host}:{self.config.port}"
        )
        logger.info(
            f"Swagger documentation: http://{self.config.host}:{self.config.port}{self.config.swagger_url}"
        )
        logger.info(
            f"GraphiQL interface: http://{self.config.host}:{self.config.port}/graphql"
        )

        self.socketio.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
        )


if __name__ == "__main__":
    # Create and run API
    config = APIConfig(
        host="0.0.0.0",
        port=5000,
        debug=True,
        secret_key="development-secret-key",
        jwt_secret="development-jwt-secret",
    )

    api = DashboardAPI(config)
    api.run()
