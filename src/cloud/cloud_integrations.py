"""Cloud Provider Integrations
Support for AWS, GCP, and Azure ML services
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# AWS imports
try:
    import boto3
    import sagemaker
    from botocore.exceptions import ClientError, NoCredentialsError
    from sagemaker.estimator import Estimator
    from sagemaker.model import Model
    from sagemaker.predictor import Predictor
    from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
    from sagemaker.sklearn import SKLearn, SKLearnProcessor
    from sagemaker.tuner import (
        ContinuousParameter,
        HyperparameterTuner,
        IntegerParameter,
    )
    from sagemaker.xgboost import XGBoost, XGBoostProcessor

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logging.warning("AWS SDK not available")

# GCP imports
try:
    import google.auth
    from google.cloud import aiplatform, bigquery, storage
    from google.cloud.aiplatform import gapic as aip
    from google.oauth2 import service_account

    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    logging.warning("GCP SDK not available")

# Azure imports
try:
    import azureml.core
    from azure.ai.ml import MLClient
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import (
        CodeConfiguration,
        Environment,
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
    )
    from azure.ai.ml.entities import (
        Model as AzureModel,
    )
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobClient, BlobServiceClient
    from azureml.core import Dataset, Experiment, Run, Workspace
    from azureml.core.compute import AmlCompute, ComputeTarget
    from azureml.core.model import Model as AzureMLModel
    from azureml.train.automl import AutoMLConfig

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logging.warning("Azure SDK not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    """Configuration for cloud provider"""

    provider: str  # "aws", "gcp", "azure"
    region: str
    project_id: str | None = None
    bucket_name: str | None = None
    credentials_path: str | None = None
    resource_group: str | None = None
    subscription_id: str | None = None
    workspace_name: str | None = None


class CloudStorageInterface:
    """Base interface for cloud storage"""

    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload a local artifact and return the provider-specific URI."""
        raise NotImplementedError

    def download_file(self, remote_path: str, local_path: str) -> str:
        """Download a remote artifact into ``local_path`` and return the path."""
        raise NotImplementedError

    def list_files(self, prefix: str = "") -> list[str]:
        """List available objects matching ``prefix``."""
        raise NotImplementedError

    def delete_file(self, remote_path: str) -> bool:
        """Delete the remote artifact and return ``True`` when successful."""
        raise NotImplementedError


class AWSIntegration(CloudStorageInterface):
    """AWS cloud integration"""

    def __init__(self, config: CloudConfig):
        if not AWS_AVAILABLE:
            raise ImportError(
                "AWS SDK not available. Install: pip install boto3 sagemaker"
            )

        self.config = config
        self.s3_client = boto3.client("s3", region_name=config.region)
        self.sagemaker_client = boto3.client("sagemaker", region_name=config.region)
        self.sagemaker_session = sagemaker.Session()
        self.role = self._get_execution_role()

    def _get_execution_role(self) -> str:
        """Get SageMaker execution role"""
        try:
            return sagemaker.get_execution_role()
        except Exception:
            # Fallback to creating a role ARN
            sts_client = boto3.client("sts")
            account_id = sts_client.get_caller_identity()["Account"]
            return f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"

    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(local_path, self.config.bucket_name, remote_path)
            s3_uri = f"s3://{self.config.bucket_name}/{remote_path}"
            logger.info(f"Uploaded to {s3_uri}")
            return s3_uri
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            raise

    def download_file(self, remote_path: str, local_path: str) -> str:
        """Download file from S3"""
        try:
            self.s3_client.download_file(
                self.config.bucket_name, remote_path, local_path
            )
            logger.info(f"Downloaded to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download file: {e}")
            raise

    def list_files(self, prefix: str = "") -> list[str]:
        """List files in S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.bucket_name, Prefix=prefix
            )
            files = [obj["Key"] for obj in response.get("Contents", [])]
            return files
        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            raise

    def delete_file(self, remote_path: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(
                Bucket=self.config.bucket_name, Key=remote_path
            )
            logger.info(f"Deleted {remote_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file: {e}")
            return False

    def train_sagemaker_model(
        self,
        train_data_s3: str,
        val_data_s3: str,
        algorithm: str = "xgboost",
        hyperparameters: dict[str, Any] = None,
    ) -> str:
        """Train model using SageMaker"""
        if algorithm == "xgboost":
            estimator = XGBoost(
                role=self.role,
                instance_type="ml.m5.xlarge",
                instance_count=1,
                framework_version="1.5-1",
                sagemaker_session=self.sagemaker_session,
                hyperparameters=hyperparameters
                or {
                    "objective": "binary:logistic",
                    "num_round": 100,
                    "max_depth": 5,
                    "eta": 0.3,
                },
            )
        else:
            # Use generic Estimator
            estimator = Estimator(
                role=self.role,
                instance_type="ml.m5.xlarge",
                instance_count=1,
                image_uri=sagemaker.image_uris.retrieve(algorithm, self.config.region),
                sagemaker_session=self.sagemaker_session,
                hyperparameters=hyperparameters or {},
            )

        # Fit the model
        estimator.fit({"train": train_data_s3, "validation": val_data_s3})

        return estimator.model_data

    def hyperparameter_tuning(
        self, train_data_s3: str, val_data_s3: str, param_ranges: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform hyperparameter tuning with SageMaker"""
        # Create base estimator
        xgb_estimator = XGBoost(
            role=self.role,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            framework_version="1.5-1",
            sagemaker_session=self.sagemaker_session,
        )

        # Define hyperparameter ranges
        hyperparameter_ranges = {}
        for param, config in param_ranges.items():
            if config["type"] == "integer":
                hyperparameter_ranges[param] = IntegerParameter(
                    config["min"], config["max"]
                )
            elif config["type"] == "continuous":
                hyperparameter_ranges[param] = ContinuousParameter(
                    config["min"], config["max"]
                )

        # Create tuner
        tuner = HyperparameterTuner(
            xgb_estimator,
            "validation:auc",
            hyperparameter_ranges,
            max_jobs=20,
            max_parallel_jobs=5,
            strategy="Bayesian",
        )

        # Start tuning
        tuner.fit({"train": train_data_s3, "validation": val_data_s3})

        # Get best model
        best_training_job = tuner.best_training_job()

        return {
            "best_job": best_training_job,
            "best_params": tuner.best_estimator().hyperparameters(),
            "model_data": tuner.best_estimator().model_data,
        }

    def deploy_endpoint(
        self, model_data: str, endpoint_name: str, instance_type: str = "ml.t2.medium"
    ) -> str:
        """Deploy model to SageMaker endpoint"""
        # Create model
        model = Model(
            model_data=model_data,
            role=self.role,
            sagemaker_session=self.sagemaker_session,
            predictor_cls=Predictor,
        )

        # Deploy endpoint
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

        logger.info(f"Deployed endpoint: {endpoint_name}")
        return predictor.endpoint_name

    def batch_transform(
        self,
        model_data: str,
        input_data_s3: str,
        output_data_s3: str,
        instance_type: str = "ml.m5.xlarge",
    ):
        """Perform batch transform job"""
        # Create transformer
        transformer = sagemaker.transformer.Transformer(
            model_name=model_data,
            instance_type=instance_type,
            instance_count=1,
            output_path=output_data_s3,
            sagemaker_session=self.sagemaker_session,
        )

        # Start transform job
        transformer.transform(input_data_s3, content_type="text/csv", split_type="Line")

        transformer.wait()
        logger.info(f"Batch transform completed: {output_data_s3}")


class GCPIntegration(CloudStorageInterface):
    """Google Cloud Platform integration"""

    def __init__(self, config: CloudConfig):
        if not GCP_AVAILABLE:
            raise ImportError(
                "GCP SDK not available. Install: pip install google-cloud-storage google-cloud-aiplatform"
            )

        self.config = config

        # Set up credentials
        if config.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.credentials_path

        # Initialize clients
        self.storage_client = storage.Client(project=config.project_id)
        self.bucket = self.storage_client.bucket(config.bucket_name)

        # Initialize AI Platform
        aiplatform.init(project=config.project_id, location=config.region)
        self.bigquery_client = bigquery.Client(project=config.project_id)

    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload file to GCS"""
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        gcs_uri = f"gs://{self.config.bucket_name}/{remote_path}"
        logger.info(f"Uploaded to {gcs_uri}")
        return gcs_uri

    def download_file(self, remote_path: str, local_path: str) -> str:
        """Download file from GCS"""
        blob = self.bucket.blob(remote_path)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded to {local_path}")
        return local_path

    def list_files(self, prefix: str = "") -> list[str]:
        """List files in GCS bucket"""
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]

    def delete_file(self, remote_path: str) -> bool:
        """Delete file from GCS"""
        try:
            blob = self.bucket.blob(remote_path)
            blob.delete()
            logger.info(f"Deleted {remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False

    def train_vertex_model(
        self,
        training_script: str,
        train_data_gcs: str,
        model_display_name: str,
        machine_type: str = "n1-standard-4",
        accelerator_type: str | None = None,
    ) -> str:
        """Train model using Vertex AI"""
        # Create custom training job
        job = aiplatform.CustomTrainingJob(
            display_name=f"training_{model_display_name}",
            script_path=training_script,
            container_uri="gcr.io/cloud-aiplatform/training/tf-cpu.2-8:latest",
            model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-8:latest",
        )

        # Run training
        model = job.run(
            dataset=train_data_gcs,
            model_display_name=model_display_name,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=1 if accelerator_type else 0,
        )

        logger.info(f"Trained model: {model.resource_name}")
        return model.resource_name

    def deploy_vertex_endpoint(
        self,
        model_name: str,
        endpoint_display_name: str,
        machine_type: str = "n1-standard-2",
    ) -> str:
        """Deploy model to Vertex AI endpoint"""
        # Get or create endpoint
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_display_name}"'
        )

        if endpoints:
            endpoint = endpoints[0]
        else:
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)

        # Deploy model
        model = aiplatform.Model(model_name)
        deployed_model = model.deploy(
            endpoint=endpoint,
            machine_type=machine_type,
            min_replica_count=1,
            max_replica_count=3,
            accelerator_type=None,
            accelerator_count=0,
        )

        logger.info(f"Deployed to endpoint: {endpoint.resource_name}")
        return endpoint.resource_name

    def automl_training(
        self,
        dataset_display_name: str,
        target_column: str,
        model_display_name: str,
        budget_hours: int = 1,
    ) -> str:
        """Train AutoML model on Vertex AI"""
        # Create dataset
        dataset = aiplatform.TabularDataset.create(
            display_name=dataset_display_name,
            gcs_source=f"gs://{self.config.bucket_name}/data.csv",
        )

        # Create AutoML training job
        job = aiplatform.AutoMLTabularTrainingJob(
            display_name=f"automl_{model_display_name}",
            optimization_prediction_type="classification",
            optimization_objective="minimize-log-loss",
        )

        # Run training
        model = job.run(
            dataset=dataset,
            target_column=target_column,
            model_display_name=model_display_name,
            budget_milli_node_hours=budget_hours * 1000,
        )

        logger.info(f"AutoML model trained: {model.resource_name}")
        return model.resource_name

    def bigquery_ml(self, query: str, model_name: str) -> str:
        """Create and train BigQuery ML model"""
        # Create model using SQL
        create_model_query = f"""
        CREATE OR REPLACE MODEL `{self.config.project_id}.{model_name}`
        OPTIONS(model_type='logistic_reg') AS
        {query}
        """

        # Execute query
        job = self.bigquery_client.query(create_model_query)
        job.result()

        logger.info(f"BigQuery ML model created: {model_name}")
        return f"{self.config.project_id}.{model_name}"


class AzureIntegration(CloudStorageInterface):
    """Microsoft Azure integration"""

    def __init__(self, config: CloudConfig):
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure SDK not available. Install: pip install azure-ai-ml azure-storage-blob azureml-core"
            )

        self.config = config

        # Initialize Azure credentials
        self.credential = DefaultAzureCredential()

        # Initialize storage client
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{config.bucket_name}.blob.core.windows.net",
            credential=self.credential,
        )

        # Initialize ML client
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=config.subscription_id,
            resource_group_name=config.resource_group,
            workspace_name=config.workspace_name,
        )

        # Initialize AzureML workspace
        self.workspace = Workspace(
            subscription_id=config.subscription_id,
            resource_group=config.resource_group,
            workspace_name=config.workspace_name,
        )

    def upload_file(self, local_path: str, remote_path: str) -> str:
        """Upload file to Azure Blob Storage"""
        container_name, blob_name = remote_path.split("/", 1)
        blob_client = self.blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        blob_uri = (
            f"https://{self.config.bucket_name}.blob.core.windows.net/{remote_path}"
        )
        logger.info(f"Uploaded to {blob_uri}")
        return blob_uri

    def download_file(self, remote_path: str, local_path: str) -> str:
        """Download file from Azure Blob Storage"""
        container_name, blob_name = remote_path.split("/", 1)
        blob_client = self.blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        with open(local_path, "wb") as data:
            data.write(blob_client.download_blob().readall())

        logger.info(f"Downloaded to {local_path}")
        return local_path

    def list_files(self, prefix: str = "") -> list[str]:
        """List files in Azure Blob Storage"""
        container_name = prefix.split("/")[0] if "/" in prefix else "default"
        container_client = self.blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blobs]

    def delete_file(self, remote_path: str) -> bool:
        """Delete file from Azure Blob Storage"""
        try:
            container_name, blob_name = remote_path.split("/", 1)
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            blob_client.delete_blob()
            logger.info(f"Deleted {remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False

    def train_azure_ml(
        self,
        training_script: str,
        compute_target: str,
        environment_name: str,
        experiment_name: str,
    ) -> str:
        """Train model using Azure ML"""
        # Get or create compute target
        try:
            compute = ComputeTarget(workspace=self.workspace, name=compute_target)
        except:
            # Create compute if not exists
            compute_config = AmlCompute.provisioning_configuration(
                vm_size="Standard_D2_v2", max_nodes=4
            )
            compute = ComputeTarget.create(
                self.workspace, compute_target, compute_config
            )
            compute.wait_for_completion(show_output=True)

        # Create experiment
        experiment = Experiment(self.workspace, experiment_name)

        # Configure run
        from azureml.core import ScriptRunConfig

        config = ScriptRunConfig(
            source_directory=".",
            script=training_script,
            compute_target=compute,
            environment=Environment.get(self.workspace, environment_name),
        )

        # Submit run
        run = experiment.submit(config)
        run.wait_for_completion(show_output=True)

        # Register model
        model = run.register_model(
            model_name=f"{experiment_name}_model", model_path="outputs/model.pkl"
        )

        logger.info(f"Model registered: {model.name}:{model.version}")
        return f"{model.name}:{model.version}"

    def automl_azure(
        self,
        dataset_name: str,
        target_column: str,
        compute_target: str,
        experiment_name: str,
    ) -> str:
        """Train AutoML model on Azure"""
        # Get dataset
        dataset = Dataset.get_by_name(self.workspace, dataset_name)

        # Configure AutoML
        automl_config = AutoMLConfig(
            experiment_timeout_minutes=30,
            task="classification",
            primary_metric="AUC_weighted",
            training_data=dataset,
            label_column_name=target_column,
            compute_target=compute_target,
            enable_early_stopping=True,
            featurization="auto",
            max_concurrent_iterations=4,
            max_cores_per_iteration=-1,
            verbosity=logging.INFO,
        )

        # Create experiment
        experiment = Experiment(self.workspace, experiment_name)

        # Submit AutoML run
        automl_run = experiment.submit(automl_config)
        automl_run.wait_for_completion(show_output=True)

        # Get best model
        best_run, fitted_model = automl_run.get_output()

        # Register model
        model = best_run.register_model(
            model_name=f"automl_{experiment_name}", model_path="outputs/model.pkl"
        )

        logger.info(f"AutoML model registered: {model.name}:{model.version}")
        return f"{model.name}:{model.version}"

    def deploy_azure_endpoint(
        self,
        model_name: str,
        endpoint_name: str,
        instance_type: str = "Standard_F2s_v2",
    ) -> str:
        """Deploy model to Azure ML endpoint"""
        # Create endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name, description="ML model endpoint", auth_mode="key"
        )
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        # Get model
        model = self.ml_client.models.get(name=model_name, version="1")

        # Create deployment
        deployment = ManagedOnlineDeployment(
            name="default",
            endpoint_name=endpoint_name,
            model=model,
            instance_type=instance_type,
            instance_count=1,
        )

        self.ml_client.online_deployments.begin_create_or_update(deployment).result()

        # Update traffic
        endpoint.traffic = {"default": 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        logger.info(f"Deployed to endpoint: {endpoint_name}")
        return endpoint_name


class CloudMLPlatform:
    """Unified interface for cloud ML platforms"""

    def __init__(self, config: CloudConfig):
        self.config = config
        self.provider = None

        # Initialize appropriate provider
        if config.provider.lower() == "aws":
            self.provider = AWSIntegration(config)
        elif config.provider.lower() == "gcp":
            self.provider = GCPIntegration(config)
        elif config.provider.lower() == "azure":
            self.provider = AzureIntegration(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    def upload_model(self, model_path: str, model_name: str) -> str:
        """Upload model to cloud storage"""
        remote_path = (
            f"models/{model_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}/model.pkl"
        )
        return self.provider.upload_file(model_path, remote_path)

    def download_model(self, model_uri: str, local_path: str) -> str:
        """Download model from cloud storage"""
        return self.provider.download_file(model_uri, local_path)

    def list_models(self, prefix: str = "models/") -> list[str]:
        """List available models"""
        return self.provider.list_files(prefix)

    def train_model(self, **kwargs) -> str:
        """Train model on cloud platform"""
        if hasattr(self.provider, "train_sagemaker_model"):
            return self.provider.train_sagemaker_model(**kwargs)
        elif hasattr(self.provider, "train_vertex_model"):
            return self.provider.train_vertex_model(**kwargs)
        elif hasattr(self.provider, "train_azure_ml"):
            return self.provider.train_azure_ml(**kwargs)
        else:
            raise NotImplementedError(
                f"Training not implemented for {self.config.provider}"
            )

    def deploy_model(self, model_uri: str, endpoint_name: str, **kwargs) -> str:
        """Deploy model to cloud endpoint"""
        if hasattr(self.provider, "deploy_endpoint"):
            return self.provider.deploy_endpoint(model_uri, endpoint_name, **kwargs)
        elif hasattr(self.provider, "deploy_vertex_endpoint"):
            return self.provider.deploy_vertex_endpoint(
                model_uri, endpoint_name, **kwargs
            )
        elif hasattr(self.provider, "deploy_azure_endpoint"):
            return self.provider.deploy_azure_endpoint(
                model_uri, endpoint_name, **kwargs
            )
        else:
            raise NotImplementedError(
                f"Deployment not implemented for {self.config.provider}"
            )


# Example usage
if __name__ == "__main__":
    # AWS Example
    if AWS_AVAILABLE:
        aws_config = CloudConfig(
            provider="aws", region="us-east-1", bucket_name="ml-portfolio-bucket"
        )
        aws_platform = CloudMLPlatform(aws_config)
        # aws_platform.upload_model("local_model.pkl", "xgboost_v1")

    # GCP Example
    if GCP_AVAILABLE:
        gcp_config = CloudConfig(
            provider="gcp",
            region="us-central1",
            project_id="ml-portfolio-project",
            bucket_name="ml-portfolio-bucket",
        )
        gcp_platform = CloudMLPlatform(gcp_config)
        # gcp_platform.upload_model("local_model.pkl", "xgboost_v1")

    # Azure Example
    if AZURE_AVAILABLE:
        azure_config = CloudConfig(
            provider="azure",
            region="eastus",
            subscription_id="your-subscription-id",
            resource_group="ml-portfolio-rg",
            workspace_name="ml-portfolio-ws",
            bucket_name="mlportfoliostorage",
        )
        azure_platform = CloudMLPlatform(azure_config)
        # azure_platform.upload_model("local_model.pkl", "xgboost_v1")
