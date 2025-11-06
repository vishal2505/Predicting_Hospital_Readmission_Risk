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
    recall_score, f1_score, make_scorer, classification_report,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
import numpy as np


def log1p_transform(x):
    """
    Apply log1p transformation to handle skewed distributions
    Note: Must be defined at module level for pickle serialization
    """
    return np.log1p(x)


def load_preprocessed_data_from_s3():
    """
    Load preprocessed data from S3 gold/preprocessed/ folder
    This avoids redundant preprocessing across parallel training tasks
    """
    print("\n" + "=" * 80)
    print("Loading Preprocessed Data from S3")
    print("=" * 80)
    
    # Get bucket from environment
    datamart_uri = os.environ.get("DATAMART_BASE_URI", "s3a://diab-readmit-123456-datamart/")
    bucket = datamart_uri.replace("s3a://", "").replace("s3://", "").rstrip("/").split("/")[0]
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    
    # Get the latest preprocessing run
    latest_key = "gold/preprocessed/latest.txt"
    try:
        response = s3_client.get_object(Bucket=bucket, Key=latest_key)
        prefix = response['Body'].read().decode('utf-8').strip()
        print(f"✓ Found latest preprocessing: {prefix}")
    except Exception as e:
        raise FileNotFoundError(f"No preprocessed data found. Run preprocess_train_data.py first. Error: {e}")
    
    # Download metadata
    metadata_key = f"{prefix}metadata.json"
    metadata_path = "/tmp/preprocessing_metadata.json"
    s3_client.download_file(bucket, metadata_key, metadata_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Loaded metadata: {metadata['timestamp']}")
    print(f"  Train shape: {metadata['train_shape']}")
    print(f"  Test shape: {metadata['test_shape']}")
    print(f"  OOT shape: {metadata['oot_shape']}")
    
    # Download train data
    train_key = f"{prefix}train_processed.parquet"
    train_path = "/tmp/train_processed.parquet"
    s3_client.download_file(bucket, train_key, train_path)
    train_df = pd.read_parquet(train_path)
    print(f"✓ Loaded train data: {train_df.shape}")
    
    # Download test data
    test_key = f"{prefix}test_processed.parquet"
    test_path = "/tmp/test_processed.parquet"
    s3_client.download_file(bucket, test_key, test_path)
    test_df = pd.read_parquet(test_path)
    print(f"✓ Loaded test data: {test_df.shape}")
    
    # Download OOT data
    oot_key = f"{prefix}oot_processed.parquet"
    oot_path = "/tmp/oot_processed.parquet"
    s3_client.download_file(bucket, oot_key, oot_path)
    oot_df = pd.read_parquet(oot_path)
    print(f"✓ Loaded OOT data: {oot_df.shape}")
    
    # Download scaler
    scaler_key = f"{prefix}scaler.pkl"
    scaler_path = "/tmp/scaler.pkl"
    s3_client.download_file(bucket, scaler_key, scaler_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Loaded scaler")
    
    # Separate features and labels
    feature_cols = [c for c in train_df.columns if c != 'label']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label']
    X_test = test_df[feature_cols].values
    y_test = test_df['label']
    X_oot = oot_df[feature_cols].values
    y_oot = oot_df['label']
    
    print(f"\n✓ Ready for training:")
    print(f"  X_train: {X_train.shape}, Readmission rate: {y_train.mean():.3f}")
    print(f"  X_test:  {X_test.shape}, Readmission rate: {y_test.mean():.3f}")
    print(f"  X_oot:   {X_oot.shape}, Readmission rate: {y_oot.mean():.3f}")
    
    return X_train, y_train, X_test, y_test, X_oot, y_oot, feature_cols


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
    """Evaluate model on test and OOT sets with comprehensive metrics"""
    print("\n" + "=" * 80)
    print(f"Evaluating {model_name}")
    print("=" * 80)
    
    results = {}
    
    for dataset_name, X, y in [("Test", X_test, y_test), ("OOT", X_oot, y_oot)]:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate AUC-ROC
        auc_roc = roc_auc_score(y, y_pred_proba)
        
        # Calculate GINI coefficient (2*AUC - 1)
        gini = 2 * auc_roc - 1
        
        # Calculate PR-AUC (better for imbalanced datasets)
        pr_auc = average_precision_score(y, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc_roc': auc_roc,
            'gini': gini,
            'pr_auc': pr_auc
        }
        
        results[dataset_name.lower()] = metrics
        
        print(f"\n{dataset_name} Set Performance:")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"  GINI:      {metrics['gini']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    return results


def extract_feature_importance(model, feature_cols, model_name, top_n=20):
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained sklearn model
        feature_cols: List of feature column names
        model_name: Name of the algorithm (for appropriate extraction method)
        top_n: Number of top features to return
    
    Returns:
        dict: Feature importance data with 'importance' list and 'top_features' list
    """
    print(f"\nExtracting feature importance for {model_name}...")
    
    importance_data = {
        'method': None,
        'importance': [],
        'top_features': []
    }
    
    try:
        if model_name == 'logistic_regression':
            # For logistic regression, use coefficients
            if hasattr(model, 'coef_'):
                coefficients = model.coef_[0]
                importance_list = [
                    {'feature': feat, 'importance': float(coef), 'abs_importance': abs(float(coef))}
                    for feat, coef in zip(feature_cols, coefficients)
                ]
                # Sort by absolute value
                importance_list.sort(key=lambda x: x['abs_importance'], reverse=True)
                importance_data['method'] = 'coefficients'
                importance_data['importance'] = importance_list
                importance_data['top_features'] = importance_list[:top_n]
                
                print(f"  ✓ Extracted coefficients for {len(feature_cols)} features")
                print(f"\n  Top {min(top_n, len(importance_list))} Features:")
                for i, item in enumerate(importance_list[:top_n], 1):
                    print(f"    {i}. {item['feature']}: {item['importance']:.4f}")
        
        elif model_name in ['random_forest', 'xgboost']:
            # For tree-based models, use feature_importances_
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_list = [
                    {'feature': feat, 'importance': float(imp)}
                    for feat, imp in zip(feature_cols, importances)
                ]
                # Sort by importance
                importance_list.sort(key=lambda x: x['importance'], reverse=True)
                importance_data['method'] = 'feature_importances'
                importance_data['importance'] = importance_list
                importance_data['top_features'] = importance_list[:top_n]
                
                print(f"  ✓ Extracted feature importances for {len(feature_cols)} features")
                print(f"\n  Top {min(top_n, len(importance_list))} Features:")
                for i, item in enumerate(importance_list[:top_n], 1):
                    print(f"    {i}. {item['feature']}: {item['importance']:.4f}")
        
        else:
            print(f"  ⚠ Feature importance extraction not supported for {model_name}")
    
    except Exception as e:
        print(f"  ✗ Error extracting feature importance: {e}")
    
    return importance_data


def save_model_to_s3(model, model_name, model_metadata, config):
    """
    Save model and metadata to S3 model registry with organized folder structure
    Structure: model_registry/{algorithm}/{version}/
    """
    print("\n" + "=" * 80)
    print("Saving Model to S3 Model Registry")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    
    bucket = config["model_config"]["model_registry_bucket"]
    base_prefix = config["model_config"]["model_registry_prefix"]
    
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Organize by algorithm: model_registry/{algorithm}/v{timestamp}/
    algorithm_folder = f"{base_prefix}{model_name}/"
    version_folder = f"{algorithm_folder}v{timestamp}/"
    
    model_key = f"{version_folder}model.pkl"
    metadata_key = f"{version_folder}metadata.json"
    
    # Save model locally first
    local_model_path = f"/tmp/{model_name}_{timestamp}.pkl"
    with open(local_model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Upload model to S3
    print(f"Uploading model to s3://{bucket}/{model_key}")
    s3_client.upload_file(local_model_path, bucket, model_key)
    
    # Upload metadata
    print(f"Uploading metadata to s3://{bucket}/{metadata_key}")
    s3_client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(model_metadata, indent=2).encode('utf-8')
    )
    
    # Save dedicated performance metrics file
    performance_key = f"{version_folder}performance.json"
    performance_data = {
        "algorithm": model_name,
        "version": timestamp,
        "training_date": model_metadata.get("training_date"),
        "metrics": {
            "test": {
                "auc_roc": model_metadata.get("performance", {}).get("test", {}).get("auc_roc", 0.0),
                "gini": model_metadata.get("performance", {}).get("test", {}).get("gini", 0.0),
                "pr_auc": model_metadata.get("performance", {}).get("test", {}).get("pr_auc", 0.0),
                "accuracy": model_metadata.get("performance", {}).get("test", {}).get("accuracy", 0.0),
                "precision": model_metadata.get("performance", {}).get("test", {}).get("precision", 0.0),
                "recall": model_metadata.get("performance", {}).get("test", {}).get("recall", 0.0),
                "f1": model_metadata.get("performance", {}).get("test", {}).get("f1", 0.0)
            },
            "oot": {
                "auc_roc": model_metadata.get("performance", {}).get("oot", {}).get("auc_roc", 0.0),
                "gini": model_metadata.get("performance", {}).get("oot", {}).get("gini", 0.0),
                "pr_auc": model_metadata.get("performance", {}).get("oot", {}).get("pr_auc", 0.0),
                "accuracy": model_metadata.get("performance", {}).get("oot", {}).get("accuracy", 0.0),
                "precision": model_metadata.get("performance", {}).get("oot", {}).get("precision", 0.0),
                "recall": model_metadata.get("performance", {}).get("oot", {}).get("recall", 0.0),
                "f1": model_metadata.get("performance", {}).get("oot", {}).get("f1", 0.0)
            }
        },
        "data_info": {
            "training_samples": model_metadata.get("training_samples"),
            "test_samples": model_metadata.get("test_samples"),
            "oot_samples": model_metadata.get("oot_samples"),
            "feature_count": model_metadata.get("feature_count")
        },
        "temporal_splits": model_metadata.get("temporal_splits", {}),
        "feature_importance": model_metadata.get("feature_importance", {})
    }
    
    print(f"Uploading performance metrics to s3://{bucket}/{performance_key}")
    s3_client.put_object(
        Bucket=bucket,
        Key=performance_key,
        Body=json.dumps(performance_data, indent=2).encode('utf-8')
    )
    
    # Create/update latest symlinks in algorithm folder
    latest_model_key = f"{algorithm_folder}latest/model.pkl"
    latest_metadata_key = f"{algorithm_folder}latest/metadata.json"
    latest_performance_key = f"{algorithm_folder}latest/performance.json"
    
    print(f"Updating latest links...")
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': model_key},
        Key=latest_model_key
    )
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': metadata_key},
        Key=latest_metadata_key
    )
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': performance_key},
        Key=latest_performance_key
    )
    
    # Create version index file
    version_info = {
        "version": timestamp,
        "model_path": f"s3://{bucket}/{model_key}",
        "metadata_path": f"s3://{bucket}/{metadata_key}",
        "performance_path": f"s3://{bucket}/{performance_key}",
        "created_at": timestamp,
        "algorithm": model_name,
        "test_auc": model_metadata.get("performance", {}).get("test", {}).get("auc_roc", 0.0),
        "test_gini": model_metadata.get("performance", {}).get("test", {}).get("gini", 0.0),
        "test_pr_auc": model_metadata.get("performance", {}).get("test", {}).get("pr_auc", 0.0),
        "oot_auc": model_metadata.get("performance", {}).get("oot", {}).get("auc_roc", 0.0),
        "oot_gini": model_metadata.get("performance", {}).get("oot", {}).get("gini", 0.0),
        "oot_pr_auc": model_metadata.get("performance", {}).get("oot", {}).get("pr_auc", 0.0),
        "test_accuracy": model_metadata.get("performance", {}).get("test", {}).get("accuracy", 0.0),
        "oot_accuracy": model_metadata.get("performance", {}).get("oot", {}).get("accuracy", 0.0)
    }
    
    version_index_key = f"{algorithm_folder}versions.json"
    
    # Try to load existing versions index
    versions_list = []
    try:
        response = s3_client.get_object(Bucket=bucket, Key=version_index_key)
        versions_list = json.loads(response['Body'].read().decode('utf-8'))
    except s3_client.exceptions.NoSuchKey:
        print("Creating new versions index")
    
    # Append new version
    versions_list.append(version_info)
    
    # Sort by timestamp descending (newest first)
    versions_list.sort(key=lambda x: x['version'], reverse=True)
    
    # Upload updated versions index
    s3_client.put_object(
        Bucket=bucket,
        Key=version_index_key,
        Body=json.dumps(versions_list, indent=2).encode('utf-8')
    )
    
    print(f"\n✓ Model saved to: s3://{bucket}/{model_key}")
    print(f"✓ Metadata saved to: s3://{bucket}/{metadata_key}")
    print(f"✓ Performance metrics saved to: s3://{bucket}/{performance_key}")
    print(f"✓ Latest links: s3://{bucket}/{algorithm_folder}latest/")
    print(f"✓ Version index: s3://{bucket}/{version_index_key}")
    print(f"✓ Total versions for {model_name}: {len(versions_list)}")
    
    # Cleanup local file
    os.remove(local_model_path)
    
    return f"s3://{bucket}/{model_key}"


def main():
    """
    Main training pipeline
    Loads preprocessed data from S3 and trains specified algorithms
    """
    print("\n" + "=" * 80)
    print("DIABETES READMISSION MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Load preprocessed data from S3 (no redundant preprocessing!)
    X_train, y_train, X_test, y_test, X_oot, y_oot, feature_cols = load_preprocessed_data_from_s3()
    
    # Determine which algorithms to train
    training_config = config.get("training_config", {})
    
    # Check if running in single-algorithm mode (from DAG)
    target_algorithm = os.environ.get("ALGORITHM")
    
    if target_algorithm:
        # Single algorithm mode (triggered by DAG task)
        algorithms_to_train = [target_algorithm]
        print(f"\n Single Algorithm Mode: Training only '{target_algorithm}'")
    else:
        # Multi-algorithm mode (local/manual execution)
        enabled = training_config.get("enabled_algorithms", {})
        algorithms_to_train = [
            alg for alg, is_enabled in enabled.items() 
            if is_enabled
        ]
        # Fallback to old-style config if enabled_algorithms not present
        if not algorithms_to_train:
            algorithms_to_train = training_config.get("algorithms", ["logistic_regression"])
        print(f"\n Multi-Algorithm Mode: Training {len(algorithms_to_train)} algorithms")
    
    print(f"Algorithms: {algorithms_to_train}\n")
    
    # Train models
    models = {}
    
    if "logistic_regression" in algorithms_to_train:
        models["logistic_regression"] = train_logistic_regression(X_train, y_train, config["model_config"])
    
    if "random_forest" in algorithms_to_train:
        models["random_forest"] = train_random_forest(X_train, y_train, config["model_config"])
    
    if "xgboost" in algorithms_to_train:
        models["xgboost"] = train_xgboost(X_train, y_train, config["model_config"])
    
    # Evaluate and save models
    for model_name, model in models.items():
        results = evaluate_model(model, X_test, y_test, X_oot, y_oot, model_name)
        
        # Extract feature importance
        feature_importance = extract_feature_importance(model, feature_cols, model_name, top_n=20)
        
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
            "feature_importance": feature_importance,
            "config": config,
            "execution_mode": "single_algorithm" if target_algorithm else "multi_algorithm"
        }
        
        # Save to S3
        save_model_to_s3(model, model_name, metadata, config)
    
    print("\n" + "=" * 80)
    print("✓ Training Pipeline Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
