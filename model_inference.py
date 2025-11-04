"""
Model Inference Pipeline
Loads trained model and generates predictions for a specific snapshot date
Based on model_inference.ipynb notebook
"""

import os
import json
import boto3
import pickle
import numpy as np
import pandas as pd
from datetime import datetime


def log1p_transform(x):
    """
    Apply log1p transformation to handle skewed distributions
    Note: Must be defined at module level for pickle serialization
    """
    return np.log1p(x)


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


def get_best_model_info(config):
    """
    Load model comparison to identify the best model
    Returns: (algorithm_name, model_s3_path)
    """
    print("\n" + "=" * 80)
    print("Identifying Best Model from Comparison")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    
    bucket = config["model_config"]["model_registry_bucket"]
    base_prefix = config["model_config"]["model_registry_prefix"]
    
    # Try to load latest model comparison
    comparison_key = f"{base_prefix}latest_model_comparison.json"
    
    try:
        print(f"Loading comparison from s3://{bucket}/{comparison_key}")
        response = s3_client.get_object(Bucket=bucket, Key=comparison_key)
        comparison = json.loads(response['Body'].read().decode('utf-8'))
        
        recommended_model = comparison.get('recommended_model')
        recommendation_reason = comparison.get('recommendation_reason')
        
        print(f"\n🏆 Best Model: {recommended_model}")
        print(f"   Reason: {recommendation_reason}")
        
        return recommended_model
        
    except s3_client.exceptions.NoSuchKey:
        print("\n⚠ No model comparison found. Using default: xgboost")
        return "xgboost"
    except Exception as e:
        print(f"\n✗ Error loading comparison: {e}")
        print("   Falling back to default: xgboost")
        return "xgboost"


def load_model_from_s3(algorithm, config):
    """
    Load trained model from S3
    Note: Scaler is loaded separately from preprocessing artifacts
    """
    print("\n" + "=" * 80)
    print(f"Loading Model: {algorithm}")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    
    bucket = config["model_config"]["model_registry_bucket"]
    base_prefix = config["model_config"]["model_registry_prefix"]
    
    # Load model from latest folder
    model_key = f"{base_prefix}{algorithm}/latest/model.pkl"
    metadata_key = f"{base_prefix}{algorithm}/latest/metadata.json"
    
    print(f"Model path: s3://{bucket}/{model_key}")
    
    # Download model
    local_model_path = "/tmp/model.pkl"
    s3_client.download_file(bucket, model_key, local_model_path)
    
    with open(local_model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded successfully")
    
    # Download metadata
    local_metadata_path = "/tmp/metadata.json"
    s3_client.download_file(bucket, metadata_key, local_metadata_path)
    
    with open(local_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Metadata loaded")
    print(f"   Training date: {metadata.get('training_date', 'N/A')}")
    print(f"   Feature count: {metadata.get('feature_count', 'N/A')}")
    
    return model, metadata


def load_scaler_from_s3():
    """
    Load the fitted scaler from preprocessing artifacts
    """
    print("\n" + "=" * 80)
    print("Loading Preprocessing Scaler")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    datamart_base = os.environ.get("DATAMART_BASE_URI", "s3://bucket/")
    
    # Parse S3 URI
    if datamart_base.startswith("s3a://"):
        datamart_base = datamart_base.replace("s3a://", "s3://")
    
    bucket = datamart_base.split("/")[2]
    
    # Get latest preprocessing run
    latest_key = "gold/preprocessed/latest.txt"
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=latest_key)
        latest_folder = response['Body'].read().decode('utf-8').strip()
        print(f"Latest preprocessing: {latest_folder}")
    except:
        # Fallback to searching for latest
        print("⚠ latest.txt not found, searching for latest preprocessing folder...")
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix="gold/preprocessed/train_data_",
            Delimiter='/'
        )
        folders = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
        if not folders:
            raise FileNotFoundError("No preprocessing folders found")
        latest_folder = sorted(folders)[-1]
        print(f"Found latest: {latest_folder}")
    
    # Download scaler
    scaler_key = f"{latest_folder}scaler.pkl"
    local_scaler_path = "/tmp/scaler.pkl"
    
    print(f"Downloading s3://{bucket}/{scaler_key}")
    s3_client.download_file(bucket, scaler_key, local_scaler_path)
    
    with open(local_scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✓ Scaler loaded successfully")
    
    return scaler


def load_feature_store_for_snapshot(snapshot_date):
    """
    Load feature store for a specific snapshot date from S3
    
    Args:
        snapshot_date: Date string in format YYYY-MM-DD
    
    Returns:
        DataFrame with features for the snapshot date
    """
    print("\n" + "=" * 80)
    print(f"Loading Feature Store for Snapshot Date: {snapshot_date}")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    datamart_base = os.environ.get("DATAMART_BASE_URI", "s3://bucket/")
    
    # Parse S3 URI
    if datamart_base.startswith("s3a://"):
        datamart_base = datamart_base.replace("s3a://", "s3://")
    
    bucket = datamart_base.split("/")[2]
    feature_prefix = "gold/feature_store/"
    
    print(f"Loading from s3://{bucket}/{feature_prefix}")
    
    # Convert snapshot_date to year and month for filtering
    snapshot_dt = datetime.strptime(snapshot_date, "%Y-%m-%d")
    target_year = snapshot_dt.year
    target_month = snapshot_dt.month
    
    print(f"Filtering for Year: {target_year}, Month: {target_month}")
    
    # List all partition folders
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=feature_prefix,
        Delimiter='/'
    )
    
    partitions = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
    
    if not partitions:
        raise FileNotFoundError(f"No data found in s3://{bucket}/{feature_prefix}")
    
    # Download and filter parquet files
    all_data = []
    
    for partition in partitions:
        # List parquet files in partition
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=partition
        )
        
        parquet_files = [obj['Key'] for obj in response.get('Contents', []) 
                        if obj['Key'].endswith('.parquet')]
        
        for file_key in parquet_files:
            local_path = f"/tmp/{os.path.basename(file_key)}"
            s3_client.download_file(bucket, file_key, local_path)
            
            # Read parquet file
            df = pd.read_parquet(local_path)
            all_data.append(df)
    
    # Combine all data
    feature_df = pd.concat(all_data, ignore_index=True)
    
    # Convert snapshot_date to datetime if it's string
    if 'snapshot_date' in feature_df.columns:
        feature_df['snapshot_date'] = pd.to_datetime(feature_df['snapshot_date'])
    
    # Filter for the specific snapshot date (year and month)
    filtered_df = feature_df[
        (feature_df['snapshot_date'].dt.year == target_year) &
        (feature_df['snapshot_date'].dt.month == target_month)
    ].copy()
    
    if len(filtered_df) == 0:
        raise ValueError(f"No data found for snapshot date {snapshot_date} (year={target_year}, month={target_month})")
    
    print(f"✓ Loaded {len(filtered_df)} records for {snapshot_date}")
    print(f"  Columns: {list(filtered_df.columns)}")
    
    return filtered_df


def preprocess_and_predict(features_df, model, scaler, algorithm, snapshot_date):
    """
    Preprocess features and generate predictions
    
    Args:
        features_df: DataFrame with features
        model: Trained model
        scaler: Fitted StandardScaler from preprocessing
        algorithm: Algorithm name
        snapshot_date: Snapshot date string
    
    Returns:
        DataFrame with predictions
    """
    print("\n" + "=" * 80)
    print("Preprocessing and Generating Predictions")
    print("=" * 80)
    
    # Extract feature columns (exclude ID and metadata columns)
    feature_cols = [c for c in features_df.columns 
                   if c not in ["encounter_id", "snapshot_date", "label", "partition_date"]]
    
    print(f"Using {len(feature_cols)} features")
    
    # Prepare X_inference
    X_inference = features_df[feature_cols].copy()
    
    # Apply scaler transformation
    X_inference_scaled = scaler.transform(X_inference)
    
    print(f"✓ Applied standard scaler transformation")
    print(f"  Shape: {X_inference_scaled.shape}")
    
    # Generate predictions
    y_pred_proba = model.predict_proba(X_inference_scaled)[:, 1]
    
    print(f"✓ Generated {len(y_pred_proba)} predictions")
    print(f"  Average probability: {y_pred_proba.mean():.4f}")
    
    # Prepare output DataFrame
    predictions_df = features_df[["encounter_id", "snapshot_date"]].copy()
    predictions_df["model_algorithm"] = algorithm
    predictions_df["model_predictions"] = y_pred_proba
    predictions_df["prediction_timestamp"] = datetime.now().isoformat()
    
    return predictions_df


def save_predictions_to_s3(predictions_df, algorithm, snapshot_date):
    """
    Save model predictions to S3 datamart using pandas
    """
    print("\n" + "=" * 80)
    print("Saving Predictions to S3")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    datamart_base = os.environ.get("DATAMART_BASE_URI", "s3://bucket/")
    
    # Parse S3 URI
    if datamart_base.startswith("s3a://"):
        datamart_base = datamart_base.replace("s3a://", "s3://")
    
    bucket = datamart_base.split("/")[2]
    
    # Verify snapshot_date column exists
    print(f"  Columns to save: {list(predictions_df.columns)}")
    if 'snapshot_date' not in predictions_df.columns:
        print(f"  ⚠ Warning: snapshot_date column missing, adding it")
        predictions_df['snapshot_date'] = snapshot_date
    
    # Ensure snapshot_date is string format for consistency
    predictions_df['snapshot_date'] = pd.to_datetime(predictions_df['snapshot_date']).dt.strftime('%Y-%m-%d')
    
    print(f"  ✓ Verified snapshot_date column exists: {predictions_df['snapshot_date'].unique()}")
    
    # Create predictions folder structure
    snapshot_date_formatted = snapshot_date.replace('-', '')
    
    # Save timestamped version
    timestamped_key = f"gold/model_predictions/{algorithm}/{algorithm}_predictions_{snapshot_date_formatted}.parquet"
    local_timestamped = "/tmp/predictions_timestamped.parquet"
    predictions_df.to_parquet(local_timestamped, index=False)
    
    print(f"Uploading timestamped predictions: s3://{bucket}/{timestamped_key}")
    s3_client.upload_file(local_timestamped, bucket, timestamped_key)
    
    # Save as latest (overwrites previous)
    latest_key = f"gold/model_predictions/{algorithm}/latest_predictions.parquet"
    local_latest = "/tmp/predictions_latest.parquet"
    predictions_df.to_parquet(local_latest, index=False)
    
    print(f"Uploading latest predictions: s3://{bucket}/{latest_key}")
    s3_client.upload_file(local_latest, bucket, latest_key)
    
    print(f"✓ Predictions saved successfully")
    print(f"  Timestamped: s3://{bucket}/{timestamped_key}")
    print(f"  Latest: s3://{bucket}/{latest_key}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'snapshot_date': snapshot_date,
        'algorithm': algorithm,
        'num_predictions': len(predictions_df),
        'avg_probability': float(predictions_df['model_predictions'].mean()),
        'predictions_path': f"s3://{bucket}/{timestamped_key}",
        'latest_path': f"s3://{bucket}/{latest_key}"
    }
    
    metadata_key = f"gold/model_predictions/{algorithm}/metadata_{snapshot_date_formatted}.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode('utf-8')
    )
    
    print(f"✓ Metadata saved: s3://{bucket}/{metadata_key}")
    
    return f"s3://{bucket}/{latest_key}"


def main():
    """
    Main inference pipeline
    1. Get snapshot date
    2. Load model and scaler
    3. Load feature store for that snapshot date
    4. Preprocess and predict
    5. Save predictions
    """
    print("\n" + "=" * 80)
    print("DIABETES READMISSION MODEL INFERENCE PIPELINE")
    print("=" * 80)
    
    # Get snapshot date from environment variable
    snapshot_date = os.environ.get("SNAPSHOT_DATE")
    
    if not snapshot_date:
        # Default to a recent date if not provided
        print("⚠ SNAPSHOT_DATE not provided, using default: 2008-03-01")
        snapshot_date = "2008-03-01"
    
    print(f"\n📅 Snapshot Date: {snapshot_date}")
    
    # Load configuration
    config = load_config()
    
    # Identify best model
    best_algorithm = get_best_model_info(config)
    
    # Load trained model
    model, metadata = load_model_from_s3(best_algorithm, config)
    
    # Load preprocessing scaler
    scaler = load_scaler_from_s3()
    
    # Load feature store for the snapshot date
    features_df = load_feature_store_for_snapshot(snapshot_date)
    
    # Preprocess and generate predictions
    predictions_df = preprocess_and_predict(
        features_df, 
        model,
        scaler,
        best_algorithm,
        snapshot_date
    )
    
    # Save predictions to S3
    predictions_path = save_predictions_to_s3(predictions_df, best_algorithm, snapshot_date)
    
    print("\n" + "=" * 80)
    print("✓ Inference Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Snapshot Date: {snapshot_date}")
    print(f"   Model: {best_algorithm}")
    print(f"   Records processed: {len(predictions_df)}")
    print(f"   Average prediction: {predictions_df['model_predictions'].mean():.4f}")
    print(f"   Predictions saved: {predictions_path}")


if __name__ == "__main__":
    main()
