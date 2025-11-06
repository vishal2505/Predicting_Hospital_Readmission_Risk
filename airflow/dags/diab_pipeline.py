import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.baseoperator import chain
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator


AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

# These should be set via Airflow Variables or env on the EC2 instance
# You can also hardcode for a quick demo and update later.
ECS_CLUSTER = os.environ.get("ECS_CLUSTER", "")
ECS_TASK_DEF_RAW = os.environ.get("ECS_TASK_DEF", "")
# Strip revision number to always use latest active revision
# e.g., "family:8" becomes "family" which ECS resolves to latest
ECS_TASK_DEF = ECS_TASK_DEF_RAW.split(":")[0] if ":" in ECS_TASK_DEF_RAW else ECS_TASK_DEF_RAW
ECS_SUBNETS = os.environ.get("ECS_SUBNETS", "").split(",") if os.environ.get("ECS_SUBNETS") else []
ECS_SECURITY_GROUPS = os.environ.get("ECS_SECURITY_GROUPS", "").split(",") if os.environ.get("ECS_SECURITY_GROUPS") else []

# Base env shared by all stages
BASE_ENV = {
    "AWS_REGION": AWS_REGION,
    # Switch between local and S3. Example S3: s3a://diab-readmit-123456-datamart/
    "DATAMART_BASE_URI": os.environ.get("DATAMART_BASE_URI", "datamart/"),
    # Backfill window (override in Airflow UI if needed)
    "START_DATE": os.environ.get("START_DATE", "1999-01-01"),
    "END_DATE": os.environ.get("END_DATE", "2008-12-31"),
}

default_args = {
    "owner": "data-eng",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="diab_medallion_ecs",
    start_date=datetime(2025, 10, 1),
    schedule_interval=None,  # trigger manually or add cron like "0 */6 * * *"
    catchup=False,
    default_args=default_args,
) as dag:

    # Bronze task
    bronze = EcsRunTaskOperator(
        task_id="bronze",
        aws_conn_id="aws_default",
        cluster=ECS_CLUSTER,
        task_definition=ECS_TASK_DEF,
        launch_type="FARGATE",
        region_name=AWS_REGION,
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": ECS_SUBNETS,
                "securityGroups": ECS_SECURITY_GROUPS,
                "assignPublicIp": "ENABLED",  # avoid NAT cost
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": os.environ.get("ECS_CONTAINER_NAME", "app"),
                    "environment": [
                        {"name": k, "value": v} for k, v in {
                            **BASE_ENV,
                            "RUN_BRONZE": "true",
                            "RUN_SILVER": "false",
                            "RUN_GOLD": "false",
                        }.items()
                    ],
                    # Optional: override command
                    # "command": ["python", "main.py"],
                }
            ]
        },
        propagate_tags="TASK_DEFINITION",
        execution_timeout=timedelta(hours=2),
    )

    # Silver task
    silver = EcsRunTaskOperator(
        task_id="silver",
        aws_conn_id="aws_default",
        cluster=ECS_CLUSTER,
        task_definition=ECS_TASK_DEF,
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
                    "environment": [
                        {"name": k, "value": v} for k, v in {
                            **BASE_ENV,
                            "RUN_BRONZE": "false",
                            "RUN_SILVER": "true",
                            "RUN_GOLD": "false",
                        }.items()
                    ],
                }
            ]
        },
        propagate_tags="TASK_DEFINITION",
        execution_timeout=timedelta(hours=2),
    )

    # Gold task
    gold = EcsRunTaskOperator(
        task_id="gold",
        aws_conn_id="aws_default",
        cluster=ECS_CLUSTER,
        task_definition=ECS_TASK_DEF,
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
                    "environment": [
                        {"name": k, "value": v} for k, v in {
                            **BASE_ENV,
                            "RUN_BRONZE": "false",
                            "RUN_SILVER": "false",
                            "RUN_GOLD": "true",
                        }.items()
                    ],
                }
            ]
        },
        propagate_tags="TASK_DEFINITION",
        execution_timeout=timedelta(hours=2),
    )

    chain(bronze, silver, gold)
