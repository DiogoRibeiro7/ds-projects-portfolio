"""
Streaming Data Processing Module
Real-time data processing with Kafka, Kinesis, and structured streaming
"""

import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import threading
from enum import Enum

import numpy as np
import pandas as pd

# Kafka imports
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import KafkaError
import confluent_kafka
from confluent_kafka import Producer, Consumer, KafkaException
from confluent_kafka.avro import AvroProducer, AvroConsumer
from confluent_kafka.schema_registry import SchemaRegistryClient

# Stream processing
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, window, from_json, to_json,
    udf, pandas_udf, avg, sum, count,
    stddev, min, max, collect_list
)
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, IntegerType, TimestampType
)
from pyspark.sql.streaming import StreamingQuery
import pyspark.sql.functions as F

# AWS Kinesis
try:
    import boto3
    from botocore.exceptions import ClientError
    KINESIS_AVAILABLE = True
except ImportError:
    KINESIS_AVAILABLE = False

# Redis Streams
import redis

# Apache Pulsar (alternative to Kafka)
try:
    import pulsar
    PULSAR_AVAILABLE = True
except ImportError:
    PULSAR_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of streaming systems"""
    KAFKA = "kafka"
    KINESIS = "kinesis"
    REDIS = "redis"
    PULSAR = "pulsar"
    SPARK = "spark"


@dataclass
class StreamConfig:
    """Configuration for streaming"""
    stream_type: StreamType = StreamType.KAFKA
    bootstrap_servers: str = "localhost:9092"
    topic: str = "ml-predictions"
    consumer_group: str = "ml-consumer-group"
    batch_size: int = 100
    batch_timeout_ms: int = 1000
    max_retries: int = 3
    enable_auto_commit: bool = True
    compression_type: str = "gzip"
    schema_registry_url: Optional[str] = None
    aws_region: Optional[str] = "us-east-1"
    redis_host: str = "localhost"
    redis_port: int = 6379


@dataclass
class StreamMessage:
    """Standard message format for streaming"""
    message_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    message_type: str = "prediction"
    version: str = "1.0"

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            'message_id': self.message_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'metadata': self.metadata,
            'message_type': self.message_type,
            'version': self.version
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'StreamMessage':
        """Create from JSON string"""
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class KafkaStreamProcessor:
    """Kafka-based stream processing"""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.producer = None
        self.consumer = None
        self.admin = None
        self._initialize_kafka()

    def _initialize_kafka(self):
        """Initialize Kafka components"""
        # Create admin client
        self.admin = KafkaAdminClient(
            bootstrap_servers=self.config.bootstrap_servers,
            client_id='ml-admin'
        )

        # Create producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type=self.config.compression_type,
            retries=self.config.max_retries
        )

        # Create consumer
        self.consumer = KafkaConsumer(
            self.config.topic,
            bootstrap_servers=self.config.bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=self.config.enable_auto_commit,
            group_id=self.config.consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        logger.info("Kafka stream processor initialized")

    def create_topic(self, partitions: int = 3, replication_factor: int = 1):
        """Create Kafka topic if it doesn't exist"""
        topic = NewTopic(
            name=self.config.topic,
            num_partitions=partitions,
            replication_factor=replication_factor
        )
        try:
            self.admin.create_topics([topic])
            logger.info(f"Created topic: {self.config.topic}")
        except Exception as e:
            logger.warning(f"Topic might already exist: {e}")

    def send_message(self, message: StreamMessage) -> bool:
        """Send message to Kafka"""
        try:
            future = self.producer.send(
                self.config.topic,
                value=asdict(message)
            )
            record_metadata = future.get(timeout=10)
            logger.debug(f"Message sent to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def consume_messages(self, process_func: Callable[[StreamMessage], Any],
                        max_messages: Optional[int] = None):
        """Consume messages from Kafka"""
        message_count = 0
        try:
            for message in self.consumer:
                # Parse message
                stream_message = StreamMessage.from_json(json.dumps(message.value))

                # Process message
                process_func(stream_message)

                message_count += 1
                if max_messages and message_count >= max_messages:
                    break

        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
        finally:
            self.consumer.close()

    def consume_batch(self, batch_process_func: Callable[[List[StreamMessage]], Any]):
        """Consume messages in batches"""
        batch = []
        last_batch_time = time.time()

        try:
            for message in self.consumer:
                # Parse message
                stream_message = StreamMessage.from_json(json.dumps(message.value))
                batch.append(stream_message)

                # Process batch if size or timeout reached
                current_time = time.time()
                if (len(batch) >= self.config.batch_size or
                    (current_time - last_batch_time) * 1000 >= self.config.batch_timeout_ms):

                    batch_process_func(batch)
                    batch = []
                    last_batch_time = current_time

        except Exception as e:
            logger.error(f"Error consuming batch: {e}")
        finally:
            # Process remaining messages
            if batch:
                batch_process_func(batch)
            self.consumer.close()


class SparkStreamProcessor:
    """Spark Structured Streaming processor"""

    def __init__(self, app_name: str = "MLStreamProcessor"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.streaming.backpressure.enabled", "true") \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("WARN")
        self.streams = {}

    def create_kafka_stream(self, config: StreamConfig) -> 'DataFrame':
        """Create Kafka streaming DataFrame"""
        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", config.bootstrap_servers) \
            .option("subscribe", config.topic) \
            .option("startingOffsets", "latest") \
            .load()

        # Parse JSON values
        schema = StructType([
            StructField("message_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("data", StringType(), True),
            StructField("metadata", StringType(), True),
            StructField("message_type", StringType(), True),
            StructField("version", StringType(), True)
        ])

        parsed_df = df.select(
            from_json(col("value").cast("string"), schema).alias("message")
        ).select("message.*")

        return parsed_df

    def add_ml_predictions(self, df: 'DataFrame', model_path: str) -> 'DataFrame':
        """Add ML predictions to streaming data"""
        # Load model (implement based on your model format)
        # This is a placeholder for the actual model loading

        @pandas_udf(returnType=DoubleType())
        def predict_udf(features_series: pd.Series) -> pd.Series:
            # Load and apply model
            import joblib
            model = joblib.load(model_path)

            # Parse features and predict
            features = features_series.apply(json.loads)
            features_df = pd.DataFrame(features.tolist())
            predictions = model.predict_proba(features_df)[:, 1]

            return pd.Series(predictions)

        # Add predictions
        df_with_predictions = df.withColumn(
            "prediction_score",
            predict_udf(col("data"))
        )

        return df_with_predictions

    def aggregate_stream(self, df: 'DataFrame', window_duration: str = "1 minute",
                        slide_duration: str = "30 seconds") -> 'DataFrame':
        """Aggregate streaming data over time windows"""
        aggregated = df \
            .withWatermark("timestamp", "10 seconds") \
            .groupBy(
                window(col("timestamp"), window_duration, slide_duration),
                col("message_type")
            ) \
            .agg(
                count("*").alias("count"),
                avg("prediction_score").alias("avg_score"),
                stddev("prediction_score").alias("stddev_score"),
                min("prediction_score").alias("min_score"),
                max("prediction_score").alias("max_score")
            )

        return aggregated

    def detect_anomalies(self, df: 'DataFrame', threshold: float = 3.0) -> 'DataFrame':
        """Detect anomalies in streaming data"""
        # Calculate rolling statistics
        stats_df = df \
            .withWatermark("timestamp", "10 seconds") \
            .groupBy(
                window(col("timestamp"), "5 minutes", "1 minute")
            ) \
            .agg(
                avg("prediction_score").alias("mean"),
                stddev("prediction_score").alias("std")
            )

        # Join with original data and detect anomalies
        anomaly_df = df.join(
            stats_df,
            df.timestamp >= stats_df.window.start & df.timestamp < stats_df.window.end
        ).withColumn(
            "z_score",
            (col("prediction_score") - col("mean")) / col("std")
        ).withColumn(
            "is_anomaly",
            F.abs(col("z_score")) > threshold
        )

        return anomaly_df

    def write_stream(self, df: 'DataFrame', output_mode: str = "append",
                    trigger_interval: str = "10 seconds") -> StreamingQuery:
        """Write streaming results"""
        query = df.writeStream \
            .outputMode(output_mode) \
            .trigger(processingTime=trigger_interval) \
            .format("console") \
            .option("truncate", False) \
            .start()

        return query

    def write_to_kafka(self, df: 'DataFrame', topic: str,
                      bootstrap_servers: str) -> StreamingQuery:
        """Write streaming data back to Kafka"""
        # Convert to JSON
        kafka_df = df.select(
            to_json(F.struct("*")).alias("value")
        )

        query = kafka_df.writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", bootstrap_servers) \
            .option("topic", topic) \
            .option("checkpointLocation", "/tmp/checkpoint") \
            .start()

        return query


class KinesisStreamProcessor:
    """AWS Kinesis stream processor"""

    def __init__(self, config: StreamConfig):
        if not KINESIS_AVAILABLE:
            raise ImportError("boto3 not available for Kinesis")

        self.config = config
        self.kinesis_client = boto3.client(
            'kinesis',
            region_name=config.aws_region
        )
        self.stream_name = config.topic

    def create_stream(self, shard_count: int = 2):
        """Create Kinesis stream"""
        try:
            response = self.kinesis_client.create_stream(
                StreamName=self.stream_name,
                ShardCount=shard_count
            )
            logger.info(f"Created Kinesis stream: {self.stream_name}")

            # Wait for stream to be active
            waiter = self.kinesis_client.get_waiter('stream_exists')
            waiter.wait(StreamName=self.stream_name)

        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceInUseException':
                logger.warning("Stream already exists")
            else:
                raise

    def put_record(self, message: StreamMessage) -> bool:
        """Put single record to Kinesis"""
        try:
            response = self.kinesis_client.put_record(
                StreamName=self.stream_name,
                Data=message.to_json(),
                PartitionKey=message.message_id
            )
            logger.debug(f"Record sent: {response['ShardId']}:{response['SequenceNumber']}")
            return True
        except ClientError as e:
            logger.error(f"Failed to put record: {e}")
            return False

    def put_records_batch(self, messages: List[StreamMessage]) -> int:
        """Put multiple records to Kinesis"""
        records = [
            {
                'Data': message.to_json(),
                'PartitionKey': message.message_id
            }
            for message in messages
        ]

        try:
            response = self.kinesis_client.put_records(
                Records=records,
                StreamName=self.stream_name
            )
            failed_count = response.get('FailedRecordCount', 0)
            if failed_count > 0:
                logger.warning(f"{failed_count} records failed to send")
            return len(messages) - failed_count
        except ClientError as e:
            logger.error(f"Failed to put records: {e}")
            return 0

    def consume_stream(self, process_func: Callable[[StreamMessage], Any],
                       shard_iterator_type: str = 'TRIM_HORIZON'):
        """Consume from Kinesis stream"""
        # Get shards
        response = self.kinesis_client.describe_stream(StreamName=self.stream_name)
        shards = response['StreamDescription']['Shards']

        for shard in shards:
            # Get shard iterator
            iterator_response = self.kinesis_client.get_shard_iterator(
                StreamName=self.stream_name,
                ShardId=shard['ShardId'],
                ShardIteratorType=shard_iterator_type
            )
            shard_iterator = iterator_response['ShardIterator']

            # Consume records
            while shard_iterator:
                try:
                    response = self.kinesis_client.get_records(
                        ShardIterator=shard_iterator,
                        Limit=100
                    )

                    records = response['Records']
                    for record in records:
                        message = StreamMessage.from_json(record['Data'].decode('utf-8'))
                        process_func(message)

                    shard_iterator = response.get('NextShardIterator')

                    # Sleep if no records
                    if not records:
                        time.sleep(1)

                except ClientError as e:
                    logger.error(f"Error consuming stream: {e}")
                    break


class RedisStreamProcessor:
    """Redis Streams processor"""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
        self.stream_key = f"stream:{config.topic}"

    def add_to_stream(self, message: StreamMessage) -> str:
        """Add message to Redis stream"""
        message_id = self.redis_client.xadd(
            self.stream_key,
            {
                'data': message.to_json(),
                'type': message.message_type,
                'timestamp': message.timestamp.isoformat()
            }
        )
        return message_id

    def read_stream(self, process_func: Callable[[StreamMessage], Any],
                   block_ms: int = 1000, count: int = 10):
        """Read from Redis stream"""
        last_id = '0'

        while True:
            try:
                # Read messages
                messages = self.redis_client.xread(
                    {self.stream_key: last_id},
                    block=block_ms,
                    count=count
                )

                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            # Parse and process message
                            message = StreamMessage.from_json(data['data'])
                            process_func(message)
                            last_id = message_id

            except Exception as e:
                logger.error(f"Error reading stream: {e}")
                time.sleep(1)

    def create_consumer_group(self, group_name: str):
        """Create consumer group for Redis stream"""
        try:
            self.redis_client.xgroup_create(
                self.stream_key,
                group_name,
                id='0'
            )
            logger.info(f"Created consumer group: {group_name}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.warning("Consumer group already exists")
            else:
                raise


class StreamingMLPipeline:
    """End-to-end streaming ML pipeline"""

    def __init__(self, stream_config: StreamConfig):
        self.config = stream_config
        self.processor = None
        self.model = None
        self.feature_buffer = deque(maxlen=1000)
        self.prediction_cache = {}

        self._initialize_processor()

    def _initialize_processor(self):
        """Initialize appropriate stream processor"""
        if self.config.stream_type == StreamType.KAFKA:
            self.processor = KafkaStreamProcessor(self.config)
        elif self.config.stream_type == StreamType.KINESIS:
            self.processor = KinesisStreamProcessor(self.config)
        elif self.config.stream_type == StreamType.REDIS:
            self.processor = RedisStreamProcessor(self.config)
        else:
            raise ValueError(f"Unsupported stream type: {self.config.stream_type}")

    def load_model(self, model_path: str):
        """Load ML model for streaming predictions"""
        import joblib
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

    def extract_features(self, message: StreamMessage) -> np.ndarray:
        """Extract features from stream message"""
        # Implement feature extraction logic
        features = message.data.get('features', {})

        # Convert to numpy array
        feature_array = np.array(list(features.values()))

        # Add to buffer for drift detection
        self.feature_buffer.append(feature_array)

        return feature_array

    def make_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction on features"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # Make prediction
        prediction = self.model.predict([features])[0]

        # Get probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba([features])[0].tolist()

        return {
            'prediction': prediction,
            'probability': probability,
            'timestamp': datetime.now().isoformat()
        }

    def detect_drift(self) -> float:
        """Detect data drift in streaming features"""
        if len(self.feature_buffer) < 100:
            return 0.0

        # Simple drift detection using mean shift
        recent_features = list(self.feature_buffer)[-50:]
        older_features = list(self.feature_buffer)[-100:-50]

        recent_mean = np.mean(recent_features, axis=0)
        older_mean = np.mean(older_features, axis=0)

        drift_score = np.linalg.norm(recent_mean - older_mean)
        return drift_score

    def process_message(self, message: StreamMessage):
        """Process single streaming message"""
        try:
            # Extract features
            features = self.extract_features(message)

            # Make prediction
            prediction_result = self.make_prediction(features)

            # Check for drift
            drift_score = self.detect_drift()
            if drift_score > 0.5:
                logger.warning(f"High drift detected: {drift_score}")

            # Create output message
            output_message = StreamMessage(
                message_id=f"pred_{message.message_id}",
                timestamp=datetime.now(),
                data={
                    'original_id': message.message_id,
                    'prediction': prediction_result['prediction'],
                    'probability': prediction_result['probability'],
                    'drift_score': drift_score
                },
                metadata={
                    'model_version': '1.0',
                    'processing_time_ms': time.time() * 1000
                },
                message_type='prediction_result'
            )

            # Send to output stream
            if hasattr(self.processor, 'send_message'):
                self.processor.send_message(output_message)

            logger.info(f"Processed message: {message.message_id}")

        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")

    def run(self):
        """Run streaming pipeline"""
        logger.info("Starting streaming ML pipeline")

        if hasattr(self.processor, 'consume_messages'):
            self.processor.consume_messages(self.process_message)
        elif hasattr(self.processor, 'consume_stream'):
            self.processor.consume_stream(self.process_message)
        elif hasattr(self.processor, 'read_stream'):
            self.processor.read_stream(self.process_message)


# Example usage
if __name__ == "__main__":
    # Configure streaming
    config = StreamConfig(
        stream_type=StreamType.KAFKA,
        bootstrap_servers="localhost:9092",
        topic="ml-predictions",
        consumer_group="ml-processors"
    )

    # Create pipeline
    pipeline = StreamingMLPipeline(config)

    # Load model (example path)
    # pipeline.load_model("models/production_model.pkl")

    # Run pipeline
    # pipeline.run()