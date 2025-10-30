import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# Initialize SparkSession with S3 support (uses env credentials if present)
aws_region = os.environ.get("AWS_REGION", "us-east-1")

builder = (
    pyspark.sql.SparkSession.builder
    .appName("dev")
    .master("local[*]")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        ",".join([
            "com.amazonaws.auth.EnvironmentVariableCredentialsProvider",
            "com.amazonaws.auth.profile.ProfileCredentialsProvider",
            "com.amazonaws.auth.InstanceProfileCredentialsProvider",
            "com.amazonaws.auth.WebIdentityTokenCredentialsProvider",
        ]),
    )
    .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_region}.amazonaws.com")
)

# Prefer baked-in JARs; fall back to remote packages if not present (e.g., running outside Docker)
hadoop_aws_jar = "/opt/spark/jars-extra/hadoop-aws-3.3.4.jar"
aws_bundle_jar = "/opt/spark/jars-extra/aws-java-sdk-bundle-1.12.639.jar"
if os.path.exists(hadoop_aws_jar) and os.path.exists(aws_bundle_jar):
    builder = builder.config("spark.jars", f"{hadoop_aws_jar},{aws_bundle_jar}")
else:
    builder = builder.config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.639")

spark = builder.getOrCreate()

# Disable profiling to avoid "profile file cannot be null" error
spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.python.profile", "false")

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config (overridable via env for orchestration)
snapshot_date_str = os.environ.get("SNAPSHOT_DATE", "1999-01-01")
start_date_str = os.environ.get("START_DATE", "1999-01-01")
end_date_str = os.environ.get("END_DATE", "2008-12-31")

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# Base datamart location (local folder or S3 URI). Example for S3: s3a://diab-readmit-123456-datamart/
DATAMART_BASE_URI = os.environ.get("DATAMART_BASE_URI", "datamart/")
is_s3a = DATAMART_BASE_URI.startswith("s3a://")

# create bronze datalake
bronze_diabetes_directory = os.path.join(DATAMART_BASE_URI, "bronze/diabetes/")

if not is_s3a and not os.path.exists(bronze_diabetes_directory):
    os.makedirs(bronze_diabetes_directory)

# which stages to run (toggle via env)
RUN_BRONZE = os.environ.get("RUN_BRONZE", "false").lower() == "true"
RUN_SILVER = os.environ.get("RUN_SILVER", "false").lower() == "true"
RUN_GOLD = os.environ.get("RUN_GOLD", "true").lower() == "true"

# run bronze backfill
if RUN_BRONZE:
    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table_diabetes(date_str, bronze_diabetes_directory, spark)

# create silver datalake
silver_diabetes_monthly_directory = os.path.join(DATAMART_BASE_URI, "silver/diabetes/")

if not is_s3a and not os.path.exists(silver_diabetes_monthly_directory):
    os.makedirs(silver_diabetes_monthly_directory)

bronze_diabetes_directory = os.path.join(DATAMART_BASE_URI, "bronze/diabetes/")

# run silver backfill
if RUN_SILVER:
    for date_str in dates_str_lst:
        utils.data_processing_silver_table.process_silver_table(date_str, bronze_diabetes_directory, silver_diabetes_monthly_directory, spark)

# create gold datalake
gold_label_store_directory = os.path.join(DATAMART_BASE_URI, "gold/label_store/")
gold_feature_store_directory = os.path.join(DATAMART_BASE_URI, "gold/feature_store/")

if not is_s3a and (not os.path.exists(gold_label_store_directory) or not os.path.exists(gold_feature_store_directory)):
    os.makedirs(gold_label_store_directory, exist_ok=True)
    os.makedirs(gold_feature_store_directory, exist_ok=True)

if RUN_GOLD:
    for date_str in dates_str_lst:
        utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_diabetes_monthly_directory, gold_label_store_directory, spark)
        utils.data_processing_gold_table.process_features_gold_table(date_str, silver_diabetes_monthly_directory, gold_feature_store_directory, spark)

if RUN_GOLD:
    folder_path = gold_label_store_directory
    # Spark can read all partitions with a wildcard; works for local or S3
    df = spark.read.option("header", "true").parquet(folder_path + "*")
    print("row_count:", df.count())
    df.show()

    folder_path = gold_feature_store_directory
    df = spark.read.option("header", "true").parquet(folder_path + "*")
    print("row_count:", df.count())
    df.show()