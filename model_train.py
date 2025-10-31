"""
Model Training Pipeline
Trains diabetes readmission prediction models using temporal window split
"""

import os
import json
import pickle
import boto3
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, make_scorer, classification_report
)
from sklearn.preprocessing import StandardScaler


def load_config(config_path="conf/model_config.json"):
    """Load model configuration"""
    # Priority order:
    # 1. Environment variable MODEL_CONFIG_S3_URI -> download from S3
    # 2. Local file at config_path
    # 3. Environment variables MODEL_CONFIG_BUCKET and MODEL_CONFIG_KEY

    s3_uri = os.environ.get("MODEL_CONFIG_S3_URI")
    if s3_uri:
        # Expect s3://bucket/key/path.json
        if s3_uri.startswith("s3://"):
            try:
                _, _, rest = s3_uri.partition("s3://")
                bucket, _, key = rest.partition("/")
                local_tmp = "/tmp/model_config.json"
                s3 = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
                print(f"Downloading model config from s3://{bucket}/{key} to {local_tmp}")
                s3.download_file(bucket, key, local_tmp)
                with open(local_tmp, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"✗ Error downloading config from {s3_uri}: {e}")
        else:
            print(f"MODEL_CONFIG_S3_URI provided but does not start with s3://: {s3_uri}")

    # Fallback: local file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"✗ Error reading local config {config_path}: {e}")

    # Last resort: build from MODEL_CONFIG_BUCKET and MODEL_CONFIG_KEY
    bucket = os.environ.get('MODEL_CONFIG_BUCKET')
    key = os.environ.get('MODEL_CONFIG_KEY')
    if bucket and key:
        try:
            local_tmp = '/tmp/model_config.json'
            s3 = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
            print(f"Downloading model config from s3://{bucket}/{key} to {local_tmp}")
            s3.download_file(bucket, key, local_tmp)
            with open(local_tmp, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"✗ Error downloading config from s3://{bucket}/{key}: {e}")

    raise FileNotFoundError(f"Model config not found locally at {config_path} and no valid S3 config provided.")


def init_spark(app_name="ModelTraining"):
    """Initialize Spark session with S3 support"""
    # Initialize SparkSession with S3 support (uses env credentials if present)
    aws_region = os.environ.get("AWS_REGION", "ap-southeast-1")


    provider_chain = ",".join([
        "com.amazonaws.auth.ContainerCredentialsProvider",          # ECS / EKS / Fargate
        "com.amazonaws.auth.EnvironmentVariableCredentialsProvider",# if AWS_* are set
        "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider",  # EC2/ECS host role
    ])

    builder = (
        pyspark.sql.SparkSession.builder
        .appName("dev")
        .master("local[*]")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", provider_chain)
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
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_gold_data(spark, datamart_uri, config):
    """Load features and labels from gold layer"""
    print("=" * 80)
    print("Loading Gold Layer Data")
    print("=" * 80)
    
    # Load label store - use wildcard to read all partitions
    label_store_path = f"{datamart_uri}gold/label_store/"
    print(f"Loading labels from: {label_store_path}")
    
    # Try to read with mergeSchema option to handle partitioned data
    try:
        labels_sdf = spark.read.option("mergeSchema", "true").parquet(f"{label_store_path}*/*.parquet")
    except Exception as e:
        print(f"⚠ Failed to read with partition pattern, trying direct path: {e}")
        # Fallback: try reading directly (non-partitioned)
        labels_sdf = spark.read.parquet(label_store_path)
    
    print(f"✓ Label store loaded: {labels_sdf.count()} records")
    
    # Load feature store - use wildcard to read all partitions
    feature_store_path = f"{datamart_uri}gold/feature_store/"
    print(f"Loading features from: {feature_store_path}")
    
    try:
        features_sdf = spark.read.option("mergeSchema", "true").parquet(f"{feature_store_path}*/*.parquet")
    except Exception as e:
        print(f"⚠ Failed to read with partition pattern, trying direct path: {e}")
        # Fallback: try reading directly (non-partitioned)
        features_sdf = spark.read.parquet(feature_store_path)
    
    print(f"✓ Feature store loaded: {features_sdf.count()} records")
    
    # Join features and labels
    data_sdf = labels_sdf.join(
        features_sdf,
        on=["encounter_id", "snapshot_date"],
        how="inner"
    )
    print(f"✓ Joined data: {data_sdf.count()} records")
    
    return data_sdf


def split_temporal_windows(data_sdf, temporal_splits):
    """Split data into train/test/oot using temporal windows"""
    print("\n" + "=" * 80)
    print("Temporal Window Split")
    print("=" * 80)
    
    train_config = temporal_splits["train"]
    test_config = temporal_splits["test"]
    oot_config = temporal_splits["oot"]
    
    print(f"Train: {train_config['start_date']} to {train_config['end_date']}")
    print(f"Test:  {test_config['start_date']} to {test_config['end_date']}")
    print(f"OOT:   {oot_config['start_date']} to {oot_config['end_date']}")
    
    # Split data
    train_sdf = data_sdf.filter(
        (col("snapshot_date") >= train_config["start_date"]) &
        (col("snapshot_date") <= train_config["end_date"])
    )
    
    test_sdf = data_sdf.filter(
        (col("snapshot_date") >= test_config["start_date"]) &
        (col("snapshot_date") <= test_config["end_date"])
    )
    
    oot_sdf = data_sdf.filter(
        (col("snapshot_date") >= oot_config["start_date"]) &
        (col("snapshot_date") <= oot_config["end_date"])
    )
    
    print(f"\n✓ Train: {train_sdf.count()} records")
    print(f"✓ Test:  {test_sdf.count()} records")
    print(f"✓ OOT:   {oot_sdf.count()} records")
    
    # Check label distribution
    print("\nLabel Distribution:")
    for name, sdf in [("Train", train_sdf), ("Test", test_sdf), ("OOT", oot_sdf)]:
        dist = sdf.groupBy("label").count().collect()
        print(f"  {name}: {dist}")
    
    return train_sdf, test_sdf, oot_sdf


def prepare_datasets(train_sdf, test_sdf, oot_sdf):
    """Convert Spark DataFrames to pandas and prepare X, y with scaling"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    
    print("\n" + "=" * 80)
    print("Preparing Datasets")
    print("=" * 80)
    
    # Get feature columns (exclude meta columns and diagnosis codes)
    all_cols = train_sdf.columns
    feature_cols = [c for c in all_cols 
                   if c not in ["encounter_id", "snapshot_date", "label", "medical_specialty", 
                               "diag_1", "diag_2", "diag_3"]]
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Features: {feature_cols[:10]}... (showing first 10)")
    
    # Convert to pandas
    print("\nConverting to pandas...")
    train_pdf = train_sdf.toPandas()
    test_pdf = test_sdf.toPandas()
    oot_pdf = oot_sdf.toPandas()
    
    # Prepare X and y (raw features)
    X_train = train_pdf[feature_cols]
    y_train = train_pdf["label"]
    X_test = test_pdf[feature_cols]
    y_test = test_pdf["label"]
    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    
    print(f"\n✓ X_train: {X_train.shape}, Readmission rate: {y_train.mean():.3f}")
    print(f"✓ X_test:  {X_test.shape}, Readmission rate: {y_test.mean():.3f}")
    print(f"✓ X_oot:   {X_oot.shape}, Readmission rate: {y_oot.mean():.3f}")
    
    # Apply StandardScaler on specific numeric columns
    print("\nApplying StandardScaler to numeric features...")
    numeric_cols = [
        'age_midpoint', 'admission_severity_score', 'admission_source_risk_score',
        'metformin_ord', 'insulin_ord', 'severity_x_visits', 'medication_density'
    ]
    
    # Filter to only include numeric_cols that exist in feature_cols
    numeric_cols = [c for c in numeric_cols if c in feature_cols]
    print(f"Scaling {len(numeric_cols)} numeric columns: {numeric_cols}")
    
    scaler = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_cols)],
        remainder='passthrough'
    )
    
    scaler.fit(X_train)
    X_train_processed = scaler.transform(X_train)
    X_test_processed = scaler.transform(X_test)
    X_oot_processed = scaler.transform(X_oot)
    
    print(f"✓ X_train_processed: {X_train_processed.shape}")
    print(f"✓ X_test_processed:  {X_test_processed.shape}")
    print(f"✓ X_oot_processed:   {X_oot_processed.shape}")
    
    return X_train_processed, y_train, X_test_processed, y_test, X_oot_processed, y_oot, feature_cols


def train_logistic_regression(X_train, y_train, config):
    """Train Logistic Regression with hyperparameter tuning"""
    print("\n" + "=" * 80)
    print("Training Logistic Regression")
    print("=" * 80)
    
    param_dist = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced']
    }
    
    log_reg = LogisticRegression(
        solver='liblinear',
        random_state=config.get("random_state", 42),
        max_iter=1000
    )
    
    random_search = RandomizedSearchCV(
        estimator=log_reg,
        param_distributions=param_dist,
        n_iter=20,
        cv=config.get("cv_folds", 5),
        scoring=make_scorer(roc_auc_score),
        random_state=config.get("random_state", 42),
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting hyperparameter search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {random_search.best_params_}")
    print(f"✓ Best CV AUC: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def train_random_forest(X_train, y_train, config):
    """Train Random Forest with hyperparameter tuning"""
    print("\n" + "=" * 80)
    print("Training Random Forest")
    print("=" * 80)
    
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    
    rf = RandomForestClassifier(
        random_state=config.get("random_state", 42),
        n_jobs=-1
    )
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=config.get("cv_folds", 5),
        scoring=make_scorer(roc_auc_score),
        random_state=config.get("random_state", 42),
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting hyperparameter search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {random_search.best_params_}")
    print(f"✓ Best CV AUC: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def train_xgboost(X_train, y_train, config):
    """Train XGBoost with hyperparameter tuning"""
    print("\n" + "=" * 80)
    print("Training XGBoost")
    print("=" * 80)
    
    param_dist = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, y_train.value_counts()[0] / y_train.value_counts()[1]]
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=config.get("random_state", 42),
        n_jobs=-1,
        eval_metric='auc'
    )
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=config.get("cv_folds", 5),
        scoring=make_scorer(roc_auc_score),
        random_state=config.get("random_state", 42),
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting hyperparameter search...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {random_search.best_params_}")
    print(f"✓ Best CV AUC: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test, X_oot, y_oot, model_name):
    """Evaluate model on test and OOT sets"""
    print("\n" + "=" * 80)
    print(f"Evaluating {model_name}")
    print("=" * 80)
    
    results = {}
    
    for dataset_name, X, y in [("Test", X_test, y_test), ("OOT", X_oot, y_oot)]:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_pred_proba)
        }
        
        results[dataset_name.lower()] = metrics
        
        print(f"\n{dataset_name} Set Performance:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    return results


def save_model_to_s3(model, model_name, model_metadata, config):
    """Save model and metadata to S3 model registry"""
    print("\n" + "=" * 80)
    print("Saving Model to S3 Model Registry")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    
    bucket = config["model_config"]["model_registry_bucket"]
    prefix = config["model_config"]["model_registry_prefix"]
    
    # Create versioned model path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"{model_name}_v{timestamp}"
    model_key = f"{prefix}{model_version}.pkl"
    metadata_key = f"{prefix}{model_version}_metadata.json"
    
    # Save model locally first
    local_model_path = f"/tmp/{model_version}.pkl"
    with open(local_model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Upload to S3
    print(f"Uploading model to s3://{bucket}/{model_key}")
    s3_client.upload_file(local_model_path, bucket, model_key)
    
    # Upload metadata
    print(f"Uploading metadata to s3://{bucket}/{metadata_key}")
    s3_client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(model_metadata, indent=2)
    )
    
    # Update latest symlink
    latest_key = f"{prefix}{model_name}_latest.pkl"
    latest_metadata_key = f"{prefix}{model_name}_latest_metadata.json"
    
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': model_key},
        Key=latest_key
    )
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': metadata_key},
        Key=latest_metadata_key
    )
    
    print(f"✓ Model saved: s3://{bucket}/{model_key}")
    print(f"✓ Latest link: s3://{bucket}/{latest_key}")
    
    # Cleanup
    os.remove(local_model_path)
    
    return f"s3://{bucket}/{model_key}"


def main():
    """Main training pipeline"""
    print("\n" + "=" * 80)
    print("DIABETES READMISSION MODEL TRAINING PIPELINE")
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
    
    # Prepare datasets
    X_train, y_train, X_test, y_test, X_oot, y_oot, feature_cols = prepare_datasets(
        train_sdf, test_sdf, oot_sdf
    )
    
    # Train models
    models = {}
    training_config = config.get("training_config", {})
    algorithms = training_config.get("algorithms", ["logistic_regression"])
    
    if "logistic_regression" in algorithms:
        models["logistic_regression"] = train_logistic_regression(X_train, y_train, config["model_config"])
    
    if "random_forest" in algorithms:
        models["random_forest"] = train_random_forest(X_train, y_train, config["model_config"])
    
    if "xgboost" in algorithms:
        models["xgboost"] = train_xgboost(X_train, y_train, config["model_config"])
    
    # Evaluate and save models
    for model_name, model in models.items():
        results = evaluate_model(model, X_test, y_test, X_oot, y_oot, model_name)
        
        # Prepare metadata
        metadata = {
            "model_name": model_name,
            "training_date": datetime.now().isoformat(),
            "temporal_splits": config["temporal_splits"],
            "feature_count": len(feature_cols),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "oot_samples": len(X_oot),
            "performance": results,
            "config": config
        }
        
        # Save to S3
        save_model_to_s3(model, model_name, metadata, config)
    
    print("\n" + "=" * 80)
    print("✓ Training Pipeline Completed Successfully!")
    print("=" * 80)
    
    spark.stop()


if __name__ == "__main__":
    main()
