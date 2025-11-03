"""
Model Inference DAG
Generates predictions using the best trained model
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
ECS_CONTAINER_NAME = os.environ.get("ECS_CONTAINER_NAME", "app")

# Use model training task definition (2vCPU/4GB) for inference
ECS_MODEL_TRAINING_TASK_DEF_RAW = os.environ.get("ECS_MODEL_TRAINING_TASK_DEF", "")
ECS_MODEL_TRAINING_TASK_DEF = ECS_MODEL_TRAINING_TASK_DEF_RAW.split(":")[0] if ":" in ECS_MODEL_TRAINING_TASK_DEF_RAW else ECS_MODEL_TRAINING_TASK_DEF_RAW

ECS_SUBNETS = os.environ.get("ECS_SUBNETS", "").split(",") if os.environ.get("ECS_SUBNETS") else []
ECS_SECURITY_GROUPS = os.environ.get("ECS_SECURITY_GROUPS", "").split(",") if os.environ.get("ECS_SECURITY_GROUPS") else []
DATAMART_BASE_URI = os.environ.get("DATAMART_BASE_URI", "s3a://diab-readmit-123456-datamart/")
MODEL_CONFIG_S3_PATH = os.environ.get("MODEL_CONFIG_S3_PATH", "s3://diab-readmit-123456-datamart/config/model_config.json")


def check_trained_model_exists(**context):
    """
    Check if at least one trained model exists
    Returns True if model exists, False otherwise
    """
    print("=" * 80)
    print("Checking for Trained Models")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Load model config to get registry location
    try:
        config_uri = MODEL_CONFIG_S3_PATH
        bucket = config_uri.split("/")[2]
        key = "/".join(config_uri.split("/")[3:])
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        config = json.loads(response['Body'].read().decode('utf-8'))
        
        registry_bucket = config["model_config"]["model_registry_bucket"]
        registry_prefix = config["model_config"]["model_registry_prefix"]
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False
    
    # Check for trained models
    algorithms = ["logistic_regression", "random_forest", "xgboost"]
    models_found = []
    
    for algorithm in algorithms:
        model_key = f"{registry_prefix}{algorithm}/latest/model.pkl"
        
        try:
            s3_client.head_object(Bucket=registry_bucket, Key=model_key)
            models_found.append(algorithm)
            print(f"✓ Found model: {algorithm}")
        except:
            print(f"  No model found for: {algorithm}")
    
    if models_found:
        print(f"\n✓ Found {len(models_found)} trained model(s): {models_found}")
        print("  Proceeding with inference.")
        return True
    else:
        print("\n✗ No trained models found.")
        print("  Run the model training DAG first (diab_model_training)")
        return False


def check_model_comparison_exists(**context):
    """
    Check if model comparison exists to identify best model
    If not, will fall back to default model
    Returns True (always proceed)
    """
    print("=" * 80)
    print("Checking for Model Comparison")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    try:
        # Load model config
        config_uri = MODEL_CONFIG_S3_PATH
        bucket = config_uri.split("/")[2]
        key = "/".join(config_uri.split("/")[3:])
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        config = json.loads(response['Body'].read().decode('utf-8'))
        
        registry_bucket = config["model_config"]["model_registry_bucket"]
        registry_prefix = config["model_config"]["model_registry_prefix"]
        
        # Check for comparison file
        comparison_key = f"{registry_prefix}latest_model_comparison.json"
        
        s3_client.head_object(Bucket=registry_bucket, Key=comparison_key)
        print(f"✓ Model comparison found: s3://{registry_bucket}/{comparison_key}")
        print("  Will use recommended best model")
        
    except Exception as e:
        print(f"⚠ Model comparison not found")
        print("  Will use default model (xgboost)")
    
    # Always return True - comparison is optional, not required
    return True


def check_inference_data_exists(**context):
    """
    Check if feature data exists for inference
    Returns True if data exists, False otherwise
    """
    print("=" * 80)
    print("Checking Inference Data Availability")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Parse S3 URI
    datamart_uri = DATAMART_BASE_URI.replace("s3a://", "s3://")
    bucket = datamart_uri.split("/")[2]
    
    # Check feature store
    feature_prefix = "gold/feature_store/"
    print(f"Checking s3://{bucket}/{feature_prefix}")
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=feature_prefix,
            MaxKeys=1
        )
        has_data = response.get('KeyCount', 0) > 0
        
        if has_data:
            print(f"✓ Feature data is available")
            print("  Proceeding with inference.")
        else:
            print(f"✗ No feature data found")
            print("  Run the data processing DAG first (diab_medallion_ecs)")
        
        return has_data
        
    except Exception as e:
        print(f"✗ Error checking feature data: {e}")
        return False


def check_preprocessing_artifacts_exist(**context):
    """
    Check if preprocessing artifacts (scaler) exist
    Returns True if artifacts exist, False otherwise
    """
    print("=" * 80)
    print("Checking Preprocessing Artifacts")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Parse S3 URI
    datamart_uri = DATAMART_BASE_URI.replace("s3a://", "s3://")
    bucket = datamart_uri.split("/")[2]
    
    # Check for preprocessing artifacts
    preprocessed_prefix = "gold/preprocessed/"
    print(f"Checking s3://{bucket}/{preprocessed_prefix}")
    
    try:
        # Check for latest.txt
        latest_key = f"{preprocessed_prefix}latest.txt"
        
        try:
            s3_client.head_object(Bucket=bucket, Key=latest_key)
            print(f"✓ Found latest preprocessing pointer")
            has_artifacts = True
        except:
            # Check for any preprocessing folder
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=f"{preprocessed_prefix}train_data_",
                Delimiter='/',
                MaxKeys=1
            )
            has_artifacts = len(response.get('CommonPrefixes', [])) > 0
            
            if has_artifacts:
                print(f"✓ Found preprocessing artifacts")
            else:
                print(f"✗ No preprocessing artifacts found")
        
        if has_artifacts:
            print("  Proceeding with inference.")
        else:
            print("  Run the model training DAG first (diab_model_training)")
            print("  Training DAG creates preprocessing artifacts")
        
        return has_artifacts
        
    except Exception as e:
        print(f"✗ Error checking preprocessing artifacts: {e}")
        return False


# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'diab_model_inference',
    default_args=default_args,
    description='Generate predictions using best trained model',
    schedule_interval=None,  # Manual trigger or scheduled as needed
    catchup=False,
    tags=['diabetes', 'inference', 'prediction', 'ecs'],
)

# Task 1: Check if trained model exists
check_model = ShortCircuitOperator(
    task_id='check_trained_model_exists',
    python_callable=check_trained_model_exists,
    provide_context=True,
    dag=dag,
)

# Task 2: Check if model comparison exists (optional - always proceeds)
check_comparison = ShortCircuitOperator(
    task_id='check_model_comparison_exists',
    python_callable=check_model_comparison_exists,
    provide_context=True,
    dag=dag,
)

# Task 3: Check if inference data exists
check_data = ShortCircuitOperator(
    task_id='check_inference_data_exists',
    python_callable=check_inference_data_exists,
    provide_context=True,
    dag=dag,
)

# Task 4: Check if preprocessing artifacts exist
check_preprocessing = ShortCircuitOperator(
    task_id='check_preprocessing_artifacts_exist',
    python_callable=check_preprocessing_artifacts_exist,
    provide_context=True,
    dag=dag,
)

# Task 5: Run inference
run_inference = EcsRunTaskOperator(
    task_id='run_model_inference',
    dag=dag,
    aws_conn_id='aws_default',
    cluster=ECS_CLUSTER,
    task_definition=ECS_MODEL_TRAINING_TASK_DEF,
    launch_type='FARGATE',
    region_name=AWS_REGION,
    network_configuration={
        'awsvpcConfiguration': {
            'subnets': ECS_SUBNETS,
            'securityGroups': ECS_SECURITY_GROUPS,
            'assignPublicIp': 'ENABLED',
        },
    },
    overrides={
        'containerOverrides': [
            {
                'name': ECS_CONTAINER_NAME,
                'command': ['python', 'model_inference.py'],
                'environment': [
                    {'name': 'AWS_REGION', 'value': AWS_REGION},
                    {'name': 'DATAMART_BASE_URI', 'value': DATAMART_BASE_URI},
                    {'name': 'MODEL_CONFIG_S3_URI', 'value': MODEL_CONFIG_S3_PATH},
                    # Optional: Set specific inference date
                    # {'name': 'INFERENCE_DATE', 'value': '2008-12-01'},
                ],
            },
        ],
    },
    propagate_tags='TASK_DEFINITION',
    reattach=True,
    execution_timeout=timedelta(minutes=30),  # Inference should be faster than training
)

# Define task dependencies
# All checks must pass before inference runs
chain(
    check_model,
    check_comparison,
    check_data,
    check_preprocessing,
    run_inference
)
