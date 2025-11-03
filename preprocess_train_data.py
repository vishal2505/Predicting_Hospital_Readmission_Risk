"""
Data Preprocessing for Model Training
Loads gold layer data, applies temporal splits and scaling, saves to S3
This runs once before model training to avoid redundant preprocessing
"""

import os
import json
import boto3
import pickle
from datetime import datetime
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


def load_config(config_path="conf/model_config.json"):
    """Load model configuration from S3 or local file"""
    s3_uri = os.environ.get("MODEL_CONFIG_S3_URI")
    if s3_uri:
        if s3_uri.startswith("s3://"):
            try:
                _, _, rest = s3_uri.partition("s3://")
                bucket, _, key = rest.partition("/")
                local_tmp = "/tmp/model_config.json"
                s3 = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
                print(f"Downloading model config from s3://{bucket}/{key}")
                s3.download_file(bucket, key, local_tmp)
                with open(local_tmp, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"✗ Error downloading config from {s3_uri}: {e}")
                raise

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    raise FileNotFoundError(f"Model config not found at {config_path} and no valid S3 config provided.")


def init_spark(app_name="DataPreprocessing"):
    """Initialize Spark session with S3 support"""
    aws_region = os.environ.get("AWS_REGION", "ap-southeast-1")

    provider_chain = ",".join([
        "com.amazonaws.auth.ContainerCredentialsProvider",
        "com.amazonaws.auth.EnvironmentVariableCredentialsProvider",
        "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider",
    ])

    builder = (
        pyspark.sql.SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", provider_chain)
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{aws_region}.amazonaws.com")
        .config("spark.python.profile", "false")
    )

    jar_dir = "/opt/spark/jars-extra"
    if os.path.exists(jar_dir):
        hadoop_jar = os.path.join(jar_dir, "hadoop-aws-3.3.4.jar")
        aws_jar = os.path.join(jar_dir, "aws-java-sdk-bundle-1.12.639.jar")
        if os.path.exists(hadoop_jar) and os.path.exists(aws_jar):
            builder = builder.config("spark.jars", f"{hadoop_jar},{aws_jar}")
    else:
        builder = builder.config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.639"
        )

    return builder.getOrCreate()


def load_gold_data(spark, base_uri, config):
    """Load feature and label stores from gold layer with .parquet folder structures"""
    print("\n" + "=" * 80)
    print("Loading Gold Layer Data")
    print("=" * 80)

    # --------------------------
    # LABEL STORE
    # --------------------------
    label_store_path = f"{base_uri.rstrip('/')}/gold/label_store/"
    print(f"Loading labels from: {label_store_path}")

    # Patterns to match e.g.
    # s3://.../gold/label_store/gold_label_store_1999_01.parquet/part-00000.parquet
    label_patterns = [
        f"{label_store_path}*.parquet/*.parquet",     # gold_label_store_YYYY_MM.parquet/part-*.parquet
        f"{label_store_path}*/*.parquet",             # fallback for nested folders
        f"{label_store_path}*.parquet",               # direct parquet files
        label_store_path,                             # auto-detect
    ]

    labels_sdf = None
    for pattern in label_patterns:
        try:
            print(f"  Trying pattern: {pattern}")
            labels_sdf = spark.read.option("mergeSchema", "true").parquet(pattern)
            count = labels_sdf.count()
            print(f"  ✓ Success with pattern: {pattern} → {count} records")
            break
        except Exception as e:
            print(f"  ✗ Failed ({pattern}): {str(e)[:150]}")
            continue

    if labels_sdf is None:
        raise FileNotFoundError(f"Could not load label_store from {label_store_path}")

    # --------------------------
    # FEATURE STORE
    # --------------------------
    feature_store_path = f"{base_uri.rstrip('/')}/gold/feature_store/"
    print(f"\nLoading features from: {feature_store_path}")

    feature_patterns = [
        f"{feature_store_path}*.parquet/*.parquet",   # gold_feature_store_YYYY_MM.parquet/part-*.parquet
        f"{feature_store_path}*/*.parquet",
        f"{feature_store_path}*.parquet",
        feature_store_path,
    ]

    features_sdf = None
    for pattern in feature_patterns:
        try:
            print(f"  Trying pattern: {pattern}")
            features_sdf = spark.read.option("mergeSchema", "true").parquet(pattern)
            count = features_sdf.count()
            print(f"  ✓ Success with pattern: {pattern} → {count} records")
            break
        except Exception as e:
            print(f"  ✗ Failed ({pattern}): {str(e)[:150]}")
            continue

    if features_sdf is None:
        raise FileNotFoundError(f"Could not load feature_store from {feature_store_path}")

    # --------------------------
    # JOIN FEATURES + LABELS
    # --------------------------
    print("\nJoining features and labels...")
    data_sdf = labels_sdf.join(
        features_sdf,
        on=["encounter_id", "snapshot_date"],
        how="inner"
    )

    print(f"✓ Joined data: {data_sdf.count()} records\n")
    return data_sdf


def split_temporal_windows(data_sdf, temporal_splits):
    """Split data into train/test/oot windows"""
    print("\n" + "=" * 80)
    print("Splitting Temporal Windows")
    print("=" * 80)
    
    train_start = temporal_splits["train"]["start_date"]
    train_end = temporal_splits["train"]["end_date"]
    test_start = temporal_splits["test"]["start_date"]
    test_end = temporal_splits["test"]["end_date"]
    oot_start = temporal_splits["oot"]["start_date"]
    oot_end = temporal_splits["oot"]["end_date"]
    
    train_sdf = data_sdf.filter(
        (col("snapshot_date") >= train_start) & (col("snapshot_date") <= train_end)
    )
    test_sdf = data_sdf.filter(
        (col("snapshot_date") >= test_start) & (col("snapshot_date") <= test_end)
    )
    oot_sdf = data_sdf.filter(
        (col("snapshot_date") >= oot_start) & (col("snapshot_date") <= oot_end)
    )
    
    print(f"Train: {train_start} to {train_end} → {train_sdf.count():,} records")
    print(f"Test:  {test_start} to {test_end} → {test_sdf.count():,} records")
    print(f"OOT:   {oot_start} to {oot_end} → {oot_sdf.count():,} records")
    
    return train_sdf, test_sdf, oot_sdf


def prepare_datasets(train_sdf, test_sdf, oot_sdf):
    """
    Prepare train/test/oot datasets with StandardScaler preprocessing
    Matches the working notebook preprocessing logic
    """
    print("\n" + "=" * 80)
    print("Preparing Datasets with Preprocessing")
    print("=" * 80)
    
    # Convert to pandas
    train_pdf = train_sdf.toPandas()
    test_pdf = test_sdf.toPandas()
    oot_pdf = oot_sdf.toPandas()
    
    # Exclude non-feature columns and diagnosis codes
    exclude_cols = ['encounter_id', 'snapshot_date', 'label', 'medical_specialty', 
                    'diag_1', 'diag_2', 'diag_3']
    
    feature_cols = [c for c in train_pdf.columns if c not in exclude_cols]
    
    # Separate features and labels
    X_train = train_pdf[feature_cols]
    y_train = train_pdf['label']
    X_test = test_pdf[feature_cols]
    y_test = test_pdf['label']
    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf['label']
    
    print(f"✓ X_train: {X_train.shape}, Readmission rate: {y_train.mean():.3f}")
    print(f"✓ X_test:  {X_test.shape}, Readmission rate: {y_test.mean():.3f}")
    print(f"✓ X_oot:   {X_oot.shape}, Readmission rate: {y_oot.mean():.3f}")
    
    # Define columns that need log1p transformation (from working notebook)
    # Only these 3 columns have skewed distributions that benefit from log transform
    log_transform_cols = [
        'age_midpoint',
        'severity_x_visits',
        'medication_density'
    ]
    
    # Define all numeric columns for scaling
    all_numeric_cols = [
        'age_midpoint',
        'admission_severity_score',
        'admission_source_risk_score',
        'metformin_ord',
        'insulin_ord',
        'severity_x_visits',
        'medication_density'
    ]
    
    # Filter to only include columns that exist in the dataset
    log_transform_cols = [c for c in log_transform_cols if c in feature_cols]
    all_numeric_cols = [c for c in all_numeric_cols if c in feature_cols]
    
    # Columns that get scaled but NOT log-transformed
    scale_only_cols = [c for c in all_numeric_cols if c not in log_transform_cols]
    
    print(f"\nApplying preprocessing transformations...")
    print(f"Log1p transform (3 columns): {log_transform_cols}")
    print(f"Scale only (4 columns): {scale_only_cols}")
    
    # Define log1p transformation function
    def log1p_transform(x):
        """Apply log1p transformation to handle skewed distributions"""
        return np.log1p(x)
    
    # Create pipeline with log transformation followed by scaling
    log_then_scale_pipeline = Pipeline(steps=[
        ('log', FunctionTransformer(log1p_transform, validate=False)),
        ('scaler', StandardScaler())
    ])
    
    # Create pipeline with just scaling (no log transform)
    scale_only_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Create ColumnTransformer with both pipelines
    scaler = ColumnTransformer(
        transformers=[
            ('log_scale', log_then_scale_pipeline, log_transform_cols),
            ('scale_only', scale_only_pipeline, scale_only_cols)
        ],
        remainder='passthrough'  # Keep other features as-is
    )
    
    # Fit on training data and transform all sets
    scaler.fit(X_train)
    X_train_processed = scaler.transform(X_train)
    X_test_processed = scaler.transform(X_test)
    X_oot_processed = scaler.transform(X_oot)
    
    print(f"✓ X_train_processed: {X_train_processed.shape}")
    print(f"✓ X_test_processed:  {X_test_processed.shape}")
    print(f"✓ X_oot_processed:   {X_oot_processed.shape}")
    
    return X_train_processed, y_train, X_test_processed, y_test, X_oot_processed, y_oot, feature_cols, scaler


def save_preprocessed_data_to_s3(X_train, y_train, X_test, y_test, X_oot, y_oot, 
                                  feature_cols, scaler, config):
    """
    Save preprocessed data and scaler to S3 in gold/preprocessed/ folder
    """
    print("\n" + "=" * 80)
    print("Saving Preprocessed Data to S3")
    print("=" * 80)
    
    # Get bucket from config
    datamart_uri = os.environ.get("DATAMART_BASE_URI", "s3a://diab-readmit-123456-datamart/")
    bucket = datamart_uri.replace("s3a://", "").replace("s3://", "").rstrip("/").split("/")[0]
    
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # S3 prefix for preprocessed data
    prefix = f"gold/preprocessed/train_data_{timestamp}/"
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    
    # Save as parquet for efficiency
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Convert to pandas DataFrames for easier handling
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df['label'] = y_train.values
    
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df['label'] = y_test.values
    
    oot_df = pd.DataFrame(X_oot, columns=feature_cols)
    oot_df['label'] = y_oot.values
    
    # Save train data
    train_path = f"/tmp/train_processed.parquet"
    train_df.to_parquet(train_path, index=False)
    s3_client.upload_file(train_path, bucket, f"{prefix}train_processed.parquet")
    print(f"✓ Uploaded: s3://{bucket}/{prefix}train_processed.parquet")
    
    # Save test data
    test_path = f"/tmp/test_processed.parquet"
    test_df.to_parquet(test_path, index=False)
    s3_client.upload_file(test_path, bucket, f"{prefix}test_processed.parquet")
    print(f"✓ Uploaded: s3://{bucket}/{prefix}test_processed.parquet")
    
    # Save OOT data
    oot_path = f"/tmp/oot_processed.parquet"
    oot_df.to_parquet(oot_path, index=False)
    s3_client.upload_file(oot_path, bucket, f"{prefix}oot_processed.parquet")
    print(f"✓ Uploaded: s3://{bucket}/{prefix}oot_processed.parquet")
    
    # Save scaler
    scaler_path = "/tmp/scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    s3_client.upload_file(scaler_path, bucket, f"{prefix}scaler.pkl")
    print(f"✓ Uploaded: s3://{bucket}/{prefix}scaler.pkl")
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "oot_shape": oot_df.shape,
        "feature_cols": feature_cols,
        "train_readmission_rate": float(y_train.mean()),
        "test_readmission_rate": float(y_test.mean()),
        "oot_readmission_rate": float(y_oot.mean()),
        "s3_prefix": f"s3://{bucket}/{prefix}",
        "temporal_splits": config["temporal_splits"]
    }
    
    metadata_path = "/tmp/preprocessing_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    s3_client.upload_file(metadata_path, bucket, f"{prefix}metadata.json")
    print(f"✓ Uploaded: s3://{bucket}/{prefix}metadata.json")
    
    # Create a "latest" pointer
    latest_path = "gold/preprocessed/latest.txt"
    latest_content = f"{prefix}\n"
    s3_client.put_object(Bucket=bucket, Key=latest_path, Body=latest_content.encode())
    print(f"✓ Updated latest pointer: s3://{bucket}/{latest_path}")
    
    return f"s3://{bucket}/{prefix}"


def main():
    """Main preprocessing pipeline"""
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING FOR MODEL TRAINING")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    datamart_uri = os.environ.get("DATAMART_BASE_URI", "s3a://diab-readmit-123456-datamart/")
    
    # Initialize Spark
    spark = init_spark()
    
    # Load data from gold layer
    data_sdf = load_gold_data(spark, datamart_uri, config)
    
    # Split into temporal windows
    train_sdf, test_sdf, oot_sdf = split_temporal_windows(
        data_sdf, 
        config["temporal_splits"]
    )
    
    # Prepare datasets with preprocessing
    X_train, y_train, X_test, y_test, X_oot, y_oot, feature_cols, scaler = prepare_datasets(
        train_sdf, test_sdf, oot_sdf
    )
    
    # Save preprocessed data to S3
    s3_prefix = save_preprocessed_data_to_s3(
        X_train, y_train, X_test, y_test, X_oot, y_oot,
        feature_cols, scaler, config
    )
    
    print("\n" + "=" * 80)
    print("✓ Preprocessing Completed Successfully!")
    print("=" * 80)
    print(f"Preprocessed data saved to: {s3_prefix}")
    print("\nNext steps:")
    print("1. Model training tasks will load preprocessed data from S3")
    print("2. No redundant preprocessing needed")
    print("3. Faster training execution")
    
    spark.stop()


if __name__ == "__main__":
    main()
