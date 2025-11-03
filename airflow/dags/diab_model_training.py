"""
Model Training DAG
Trains diabetes readmission prediction models with prerequisite checks
"""

import os
import json
import boto3
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import ShortCircuitOperator, PythonOperator
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator
from airflow.models.baseoperator import chain


AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
ECS_CLUSTER = os.environ.get("ECS_CLUSTER", "")

# Use dedicated model training task definition (higher resources: 2vCPU/4GB)
ECS_MODEL_TRAINING_TASK_DEF_RAW = os.environ.get("ECS_MODEL_TRAINING_TASK_DEF", "")
ECS_MODEL_TRAINING_TASK_DEF = ECS_MODEL_TRAINING_TASK_DEF_RAW.split(":")[0] if ":" in ECS_MODEL_TRAINING_TASK_DEF_RAW else ECS_MODEL_TRAINING_TASK_DEF_RAW

ECS_SUBNETS = os.environ.get("ECS_SUBNETS", "").split(",") if os.environ.get("ECS_SUBNETS") else []
ECS_SECURITY_GROUPS = os.environ.get("ECS_SECURITY_GROUPS", "").split(",") if os.environ.get("ECS_SECURITY_GROUPS") else []
DATAMART_BASE_URI = os.environ.get("DATAMART_BASE_URI", "s3a://diab-readmit-123456-datamart/")
MODEL_CONFIG_S3_PATH = os.environ.get("MODEL_CONFIG_S3_PATH", "s3://diab-readmit-123456-datamart/config/model_config.json")


def check_gold_data_exists(**context):
    """
    Check if gold layer data exists before training
    Returns True if data exists, False otherwise
    """
    print("=" * 80)
    print("Checking Gold Layer Data Availability")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Parse S3 URI
    datamart_uri = DATAMART_BASE_URI.replace("s3a://", "s3://")
    bucket = datamart_uri.split("/")[2]
    
    # Check feature store
    feature_prefix = "gold/feature_store/"
    print(f"Checking s3://{bucket}/{feature_prefix}")
    
    try:
        feature_response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=feature_prefix,
            MaxKeys=1
        )
        has_features = feature_response.get('KeyCount', 0) > 0
        print(f"✓ Feature store exists: {has_features}")
    except Exception as e:
        print(f"✗ Error checking feature store: {e}")
        has_features = False
    
    # Check label store
    label_prefix = "gold/label_store/"
    print(f"Checking s3://{bucket}/{label_prefix}")
    
    try:
        label_response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=label_prefix,
            MaxKeys=1
        )
        has_labels = label_response.get('KeyCount', 0) > 0
        print(f"✓ Label store exists: {has_labels}")
    except Exception as e:
        print(f"✗ Error checking label store: {e}")
        has_labels = False
    
    # Both must exist
    data_exists = has_features and has_labels
    
    if data_exists:
        print("\n✓ Gold layer data is available. Proceeding with training.")
    else:
        print("\n✗ Gold layer data is NOT available. Skipping training.")
        print("   Run the data processing DAG first (diab_medallion_ecs)")
    
    return data_exists


def check_sufficient_training_data(**context):
    """
    Check if there's sufficient data in the training window
    Returns True if sufficient data exists, False otherwise
    """
    print("=" * 80)
    print("Checking Training Data Sufficiency")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Load model config from S3
    try:
        config_s3_path = MODEL_CONFIG_S3_PATH.replace("s3://", "")
        config_bucket = config_s3_path.split("/")[0]
        config_key = "/".join(config_s3_path.split("/")[1:])
        
        print(f"Loading config from s3://{config_bucket}/{config_key}")
        response = s3_client.get_object(Bucket=config_bucket, Key=config_key)
        config = json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"✗ Error loading config from S3: {e}")
        print("   Make sure to upload conf/model_config.json to S3")
        return False
    
    temporal_splits = config['temporal_splits']
    train_start = temporal_splits['train']['start_date']
    train_end = temporal_splits['train']['end_date']
    
    print(f"Training window: {train_start} to {train_end}")
    
    # Check if we have parquet files in gold layer
    # This is a simplified check - actual check would count records
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    datamart_uri = DATAMART_BASE_URI.replace("s3a://", "s3://")
    bucket = datamart_uri.split("/")[2]
    
    feature_prefix = "gold/feature_store/"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=feature_prefix
        )
        
        # Count parquet files
        parquet_count = sum(1 for obj in response.get('Contents', []) 
                           if obj['Key'].endswith('.parquet'))
        
        print(f"Found {parquet_count} feature parquet partitions")
        
        # Require at least 10 partitions (rough heuristic)
        sufficient = parquet_count >= 10
        
        if sufficient:
            print(f"✓ Sufficient training data available ({parquet_count} partitions)")
        else:
            print(f"✗ Insufficient training data ({parquet_count} partitions, need >= 10)")
        
        return sufficient
        
    except Exception as e:
        print(f"✗ Error checking training data: {e}")
        return False


def validate_model_config(**context):
    """
    Validate model configuration file
    Returns True if config is valid, False otherwise
    """
    print("=" * 80)
    print("Validating Model Configuration")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    try:
        # Load config from S3
        config_s3_path = MODEL_CONFIG_S3_PATH.replace("s3://", "")
        config_bucket = config_s3_path.split("/")[0]
        config_key = "/".join(config_s3_path.split("/")[1:])
        
        print(f"Loading config from s3://{config_bucket}/{config_key}")
        response = s3_client.get_object(Bucket=config_bucket, Key=config_key)
        config = json.loads(response['Body'].read().decode('utf-8'))
        
        # Check required fields
        required_fields = ['temporal_splits', 'model_config', 'training_config']
        for field in required_fields:
            if field not in config:
                print(f"✗ Missing required field: {field}")
                return False
        
        # Check temporal splits
        required_splits = ['train', 'test', 'oot']
        for split in required_splits:
            if split not in config['temporal_splits']:
                print(f"✗ Missing temporal split: {split}")
                return False
            
            split_config = config['temporal_splits'][split]
            if 'start_date' not in split_config or 'end_date' not in split_config:
                print(f"✗ Missing start_date/end_date in {split} split")
                return False
        
        print("✓ Model configuration is valid")
        print(f"  Training algorithms: {config['training_config'].get('algorithms', [])}")
        print(f"  Train window: {config['temporal_splits']['train']['start_date']} to {config['temporal_splits']['train']['end_date']}")
        print(f"  Test window: {config['temporal_splits']['test']['start_date']} to {config['temporal_splits']['test']['end_date']}")
        print(f"  OOT window: {config['temporal_splits']['oot']['start_date']} to {config['temporal_splits']['oot']['end_date']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error validating config: {e}")
        print(f"   Config path: {MODEL_CONFIG_S3_PATH}")
        print("   Make sure to upload conf/model_config.json to S3:")
        print(f"   aws s3 cp conf/model_config.json {MODEL_CONFIG_S3_PATH}")
        return False


default_args = {
    "owner": "ml-eng",
    "retries": 0,  # Disable auto-retry for model training (takes hours, manual intervention better)
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="diab_model_training",
    start_date=datetime(2025, 10, 1),
    schedule_interval=None,  # Trigger manually after data processing
    catchup=False,
    max_active_runs=1,  # Only one DAG run at a time to prevent duplicate executions
    default_args=default_args,
    description="Train diabetes readmission prediction models with prerequisite checks",
    tags=["ml", "training", "diabetes"],
) as dag:
    
    # Prerequisite Check 1: Validate model configuration
    check_config = ShortCircuitOperator(
        task_id="check_model_config",
        python_callable=validate_model_config,
        doc_md="""
        ### Validate Model Configuration
        Checks that model_config.json is valid and contains all required fields:
        - temporal_splits (train/test/oot with start_date/end_date)
        - model_config
        - training_config
        
        **Skip downstream tasks if validation fails.**
        """
    )
    
    # Prerequisite Check 2: Check if gold data exists
    check_data = ShortCircuitOperator(
        task_id="check_gold_data_exists",
        python_callable=check_gold_data_exists,
        doc_md="""
        ### Check Gold Layer Data
        Verifies that both feature_store and label_store exist in S3 gold layer.
        
        **Skip downstream tasks if data doesn't exist.**
        Run `diab_medallion_ecs` DAG first to create gold layer data.
        """
    )
    
    # Prerequisite Check 3: Check sufficient training data
    check_sufficient_data = ShortCircuitOperator(
        task_id="check_sufficient_training_data",
        python_callable=check_sufficient_training_data,
        doc_md="""
        ### Check Training Data Sufficiency
        Ensures there are enough data partitions in the training window.
        Requires at least 10 parquet partitions.
        
        **Skip downstream tasks if insufficient data.**
        """
    )
    
    # Data Preprocessing Task - Runs ONCE before all model training
    # Creates preprocessed data in gold/preprocessed/ folder
    preprocess_data = EcsRunTaskOperator(
        task_id="preprocess_training_data",
        aws_conn_id="aws_default",
        cluster=ECS_CLUSTER,
        task_definition=ECS_MODEL_TRAINING_TASK_DEF,
        launch_type="FARGATE",
        region_name=AWS_REGION,
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ECS_SUBNETS,
                "securityGroups": ECS_SECURITY_GROUPS,
                "assignPublicIp": "ENABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": os.environ.get("ECS_CONTAINER_NAME", "app"),
                    "command": ["python", "preprocess_train_data.py"],
                    "environment": [
                        {"name": "AWS_REGION", "value": AWS_REGION},
                        {"name": "DATAMART_BASE_URI", "value": DATAMART_BASE_URI},
                        {"name": "MODEL_CONFIG_S3_URI", "value": MODEL_CONFIG_S3_PATH},
                    ],
                }
            ]
        },
        propagate_tags="TASK_DEFINITION",
        reattach=True,
        execution_timeout=timedelta(minutes=30),
        doc_md="""
        ### Preprocess Training Data
        Loads gold layer data, applies temporal splits, and preprocesses features.
        Saves preprocessed data to S3 gold/preprocessed/ folder.
        
        This runs ONCE before model training to avoid redundant preprocessing.
        
        **Steps:**
        - Load feature_store and label_store from gold layer
        - Split into train/test/oot temporal windows
        - Apply StandardScaler on numeric features
        - Save preprocessed data to S3 as Parquet
        - Create 'latest' pointer for downstream tasks
        
        **Timeout: 30 minutes**
        """
    )
    
    # Model Training Tasks (ECS Fargate) - One per algorithm for parallel execution
    # Uses dedicated model training task definition with 2vCPU/4GB (vs 1vCPU/2GB for data processing)
    train_logistic_regression = EcsRunTaskOperator(
        task_id="train_logistic_regression",
        aws_conn_id="aws_default",
        cluster=ECS_CLUSTER,
        task_definition=ECS_MODEL_TRAINING_TASK_DEF,
        launch_type="FARGATE",
        region_name=AWS_REGION,
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ECS_SUBNETS,
                "securityGroups": ECS_SECURITY_GROUPS,
                "assignPublicIp": "ENABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": os.environ.get("ECS_CONTAINER_NAME", "app"),
                    "command": ["python", "model_train.py"],
                    "environment": [
                        {"name": "AWS_REGION", "value": AWS_REGION},
                        {"name": "DATAMART_BASE_URI", "value": DATAMART_BASE_URI},
                        {"name": "MODEL_CONFIG_S3_URI", "value": MODEL_CONFIG_S3_PATH},
                        {"name": "ALGORITHM", "value": "logistic_regression"},
                    ],
                }
            ]
        },
        propagate_tags="TASK_DEFINITION",
        reattach=True,  # Reattach to existing task if Airflow worker restarts
        execution_timeout=timedelta(hours=2),
        doc_md="""
        ### Train Logistic Regression
        Trains Logistic Regression model with hyperparameter tuning:
        - L1/L2 regularization
        - Class weight balancing
        - 5-fold cross-validation
        
        **Timeout: 2 hours**
        """
    )
    
    train_random_forest = EcsRunTaskOperator(
        task_id="train_random_forest",
        aws_conn_id="aws_default",
        cluster=ECS_CLUSTER,
        task_definition=ECS_MODEL_TRAINING_TASK_DEF,
        launch_type="FARGATE",
        region_name=AWS_REGION,
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ECS_SUBNETS,
                "securityGroups": ECS_SECURITY_GROUPS,
                "assignPublicIp": "ENABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": os.environ.get("ECS_CONTAINER_NAME", "app"),
                    "command": ["python", "model_train.py"],
                    "environment": [
                        {"name": "AWS_REGION", "value": AWS_REGION},
                        {"name": "DATAMART_BASE_URI", "value": DATAMART_BASE_URI},
                        {"name": "MODEL_CONFIG_S3_URI", "value": MODEL_CONFIG_S3_PATH},
                        {"name": "ALGORITHM", "value": "random_forest"},
                    ],
                }
            ]
        },
        propagate_tags="TASK_DEFINITION",
        reattach=True,  # Reattach to existing task if Airflow worker restarts
        execution_timeout=timedelta(hours=3),
        doc_md="""
        ### Train Random Forest
        Trains Random Forest model with hyperparameter tuning:
        - Tree depth and count optimization
        - Min samples per leaf/split
        - Class weight balancing
        - 5-fold cross-validation
        
        **Timeout: 3 hours** (more intensive than LogReg)
        """
    )
    
    train_xgboost = EcsRunTaskOperator(
        task_id="train_xgboost",
        aws_conn_id="aws_default",
        cluster=ECS_CLUSTER,
        task_definition=ECS_MODEL_TRAINING_TASK_DEF,
        launch_type="FARGATE",
        region_name=AWS_REGION,
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ECS_SUBNETS,
                "securityGroups": ECS_SECURITY_GROUPS,
                "assignPublicIp": "ENABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": os.environ.get("ECS_CONTAINER_NAME", "app"),
                    "command": ["python", "model_train.py"],
                    "environment": [
                        {"name": "AWS_REGION", "value": AWS_REGION},
                        {"name": "DATAMART_BASE_URI", "value": DATAMART_BASE_URI},
                        {"name": "MODEL_CONFIG_S3_URI", "value": MODEL_CONFIG_S3_PATH},
                        {"name": "ALGORITHM", "value": "xgboost"},
                    ],
                }
            ]
        },
        propagate_tags="TASK_DEFINITION",
        reattach=True,  # Reattach to existing task if Airflow worker restarts
        execution_timeout=timedelta(hours=3),
        doc_md="""
        ### Train XGBoost
        Trains XGBoost model with hyperparameter tuning:
        - Learning rate optimization
        - Tree depth and count
        - Subsampling and column sampling
        - Scale_pos_weight for imbalance
        - 5-fold cross-validation
        
        **Timeout: 3 hours**
        """
    )
    
    # Generate model comparison after all training completes
    generate_comparison = EcsRunTaskOperator(
        task_id="generate_model_comparison",
        aws_conn_id="aws_default",
        cluster=ECS_CLUSTER,
        task_definition=ECS_MODEL_TRAINING_TASK_DEF,
        launch_type="FARGATE",
        region_name=AWS_REGION,
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ECS_SUBNETS,
                "securityGroups": ECS_SECURITY_GROUPS,
                "assignPublicIp": "ENABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": os.environ.get("ECS_CONTAINER_NAME", "app"),
                    "command": ["python", "generate_model_comparison.py"],
                    "environment": [
                        {"name": "AWS_REGION", "value": AWS_REGION},
                        {"name": "DATAMART_BASE_URI", "value": DATAMART_BASE_URI},
                        {"name": "MODEL_CONFIG_S3_URI", "value": MODEL_CONFIG_S3_PATH},
                    ],
                }
            ]
        },
        propagate_tags="TASK_DEFINITION",
        reattach=True,
        execution_timeout=timedelta(minutes=10),
        doc_md="""
        ### Generate Model Comparison
        Creates comprehensive comparison of all trained models:
        - Loads performance metrics from S3
        - Generates comparison table with all metrics
        - Identifies best model by OOT GINI
        - Saves comparison as JSON and CSV to S3
        
        **Output:**
        - `latest_model_comparison.json` - Full comparison data
        - `latest_model_comparison.csv` - Excel-friendly format
        
        **Timeout: 10 minutes**
        """
    )
    
    # Define task dependencies
    # Preprocessing runs once after checks, then all models train in parallel,
    # finally comparison is generated
    chain(
        check_config,
        check_data,
        check_sufficient_data,
        preprocess_data,
    )
    
    preprocess_data >> [train_logistic_regression, train_random_forest, train_xgboost] >> generate_comparison
