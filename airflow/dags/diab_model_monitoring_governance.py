"""
Model Monitoring and Governance DAG
Monitors model performance over time and makes governance decisions

This DAG should run after predictions are generated to:
1. Compare predictions against actual labels
2. Calculate performance metrics (AUC, GINI)
3. Calculate drift metrics (PSI)
4. Make governance decisions (retrain, schedule retrain, or no action)
"""

import os
import json
import boto3
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import ShortCircuitOperator
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator
from airflow.models.baseoperator import chain


AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
ECS_CLUSTER = os.environ.get("ECS_CLUSTER", "")
ECS_CONTAINER_NAME = os.environ.get("ECS_CONTAINER_NAME", "app")

# Use model training task definition
ECS_MODEL_TRAINING_TASK_DEF_RAW = os.environ.get("ECS_MODEL_TRAINING_TASK_DEF", "")
ECS_MODEL_TRAINING_TASK_DEF = ECS_MODEL_TRAINING_TASK_DEF_RAW.split(":")[0] if ":" in ECS_MODEL_TRAINING_TASK_DEF_RAW else ECS_MODEL_TRAINING_TASK_DEF_RAW

ECS_SUBNETS = os.environ.get("ECS_SUBNETS", "").split(",") if os.environ.get("ECS_SUBNETS") else []
ECS_SECURITY_GROUPS = os.environ.get("ECS_SECURITY_GROUPS", "").split(",") if os.environ.get("ECS_SECURITY_GROUPS") else []
DATAMART_BASE_URI = os.environ.get("DATAMART_BASE_URI", "s3a://diab-readmit-123456-datamart/")
MODEL_CONFIG_S3_PATH = os.environ.get("MODEL_CONFIG_S3_PATH", "s3://diab-readmit-123456-datamart/config/model_config.json")

# Model name for monitoring/governance
DEFAULT_MODEL_ALGORITHM = os.environ.get("DEFAULT_MODEL_ALGORITHM", "xgboost")


def check_predictions_exist(**context):
    """
    Check if model predictions exist
    Returns True if predictions exist, False otherwise
    """
    print("=" * 80)
    print("Checking Model Predictions")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Parse S3 URI
    datamart_uri = DATAMART_BASE_URI.replace("s3a://", "s3://")
    bucket = datamart_uri.split("/")[2]
    
    # Get model algorithm from context or use default
    model_algorithm = context.get('dag_run').conf.get('model_algorithm', DEFAULT_MODEL_ALGORITHM)
    
    # Check for predictions
    predictions_prefix = f"gold/model_predictions/{model_algorithm}/"
    print(f"Checking s3://{bucket}/{predictions_prefix}")
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=predictions_prefix,
            MaxKeys=1
        )
        has_predictions = response.get('KeyCount', 0) > 0
        
        if has_predictions:
            print(f"✓ Predictions found for {model_algorithm}")
            print("  Proceeding with monitoring.")
        else:
            print(f"✗ No predictions found for {model_algorithm}")
            print("  Run the inference DAG first (diab_model_inference)")
        
        return has_predictions
        
    except Exception as e:
        print(f"✗ Error checking predictions: {e}")
        return False


def check_labels_exist(**context):
    """
    Check if label store exists (ground truth for monitoring)
    Returns True if labels exist, False otherwise
    """
    print("=" * 80)
    print("Checking Label Store")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Parse S3 URI
    datamart_uri = DATAMART_BASE_URI.replace("s3a://", "s3://")
    bucket = datamart_uri.split("/")[2]
    
    # Check for labels
    label_prefix = "gold/label_store/"
    print(f"Checking s3://{bucket}/{label_prefix}")
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=label_prefix,
            MaxKeys=1
        )
        has_labels = response.get('KeyCount', 0) > 0
        
        if has_labels:
            print(f"✓ Label store is available")
            print("  Proceeding with monitoring.")
        else:
            print(f"✗ No labels found")
            print("  Run the data processing DAG first (diab_medallion_ecs)")
        
        return has_labels
        
    except Exception as e:
        print(f"✗ Error checking labels: {e}")
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
    'diab_model_monitoring_governance',
    default_args=default_args,
    description='Monitor model performance and make governance decisions',
    schedule_interval=None,  # Manual trigger or scheduled (e.g., weekly)
    catchup=False,
    tags=['diabetes', 'monitoring', 'governance', 'mlops'],
)

# Task 1: Check if predictions exist
check_predictions = ShortCircuitOperator(
    task_id='check_predictions_exist',
    python_callable=check_predictions_exist,
    provide_context=True,
    dag=dag,
)

# Task 2: Check if labels exist
check_labels = ShortCircuitOperator(
    task_id='check_labels_exist',
    python_callable=check_labels_exist,
    provide_context=True,
    dag=dag,
)

# Task 3: Run model monitoring
# Compares predictions vs actual labels, calculates AUC, GINI, PSI
run_monitoring = EcsRunTaskOperator(
    task_id='run_model_monitoring',
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
                'command': ['python', 'model_monitoring.py'],
                'environment': [
                    {'name': 'AWS_REGION', 'value': AWS_REGION},
                    {'name': 'DATAMART_BASE_URI', 'value': DATAMART_BASE_URI},
                    # Model algorithm - can be overridden via DAG conf
                    #{'name': 'MODEL_ALGORITHM', 'value': "{{ dag_run.conf.get('model_algorithm', '%s') }}" % DEFAULT_MODEL_ALGORITHM},
                ],
            },
        ],
    },
    propagate_tags='TASK_DEFINITION',
    reattach=True,
    execution_timeout=timedelta(minutes=20),
)

# Task 4: Run governance decision
# Reviews monitoring results and decides: retrain, schedule retrain, or no action
run_governance = EcsRunTaskOperator(
    task_id='run_model_governance',
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
                'command': ['python', 'model_governance.py'],
                'environment': [
                    {'name': 'AWS_REGION', 'value': AWS_REGION},
                    {'name': 'DATAMART_BASE_URI', 'value': DATAMART_BASE_URI},
                    # Model algorithm - can be overridden via DAG conf
                    #{'name': 'MODEL_ALGORITHM', 'value': "{{ dag_run.conf.get('model_algorithm', '%s') }}" % DEFAULT_MODEL_ALGORITHM},
                    # Auto-retrain flag - can be set via DAG conf
                    {'name': 'AUTO_RETRAIN', 'value': "{{ dag_run.conf.get('auto_retrain', 'false') }}"},
                ],
            },
        ],
    },
    propagate_tags='TASK_DEFINITION',
    reattach=True,
    execution_timeout=timedelta(minutes=15),
)

# Define task dependencies
# Sequential flow: check predictions → check labels → monitor → governance
chain(
    check_predictions,
    check_labels,
    run_monitoring,
    run_governance
)
