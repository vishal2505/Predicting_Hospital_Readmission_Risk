"""
Model Inference Pipeline
Loads best trained model and generates predictions for new data
"""

import os
import json
import boto3
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline


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
    Load trained model and metadata from S3
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


def load_preprocessed_scaler_from_s3():
    """
    Load the fitted scaler from preprocessing step
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


def load_inference_data_from_s3(inference_date=None):
    """
    Load feature data for inference from S3
    
    Args:
        inference_date: Date string in format YYYY-MM-DD (optional)
                       If None, loads most recent data
    """
    print("\n" + "=" * 80)
    print("Loading Inference Data from S3")
    print("=" * 80)
    
    s3_client = boto3.client('s3', region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
    datamart_base = os.environ.get("DATAMART_BASE_URI", "s3://bucket/")
    
    # Parse S3 URI
    if datamart_base.startswith("s3a://"):
        datamart_base = datamart_base.replace("s3a://", "s3://")
    
    bucket = datamart_base.split("/")[2]
    feature_prefix = "gold/feature_store/"
    
    print(f"Loading from s3://{bucket}/{feature_prefix}")
    
    # List all partition folders
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=feature_prefix,
        Delimiter='/'
    )
    
    partitions = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
    
    if not partitions:
        raise FileNotFoundError(f"No data found in s3://{bucket}/{feature_prefix}")
    
    # Filter by inference_date if provided
    if inference_date:
        target_partition = f"{feature_prefix}partition_date={inference_date}/"
        if target_partition not in partitions:
            raise FileNotFoundError(f"No data found for date {inference_date}")
        partitions_to_load = [target_partition]
        print(f"Loading data for {inference_date}")
    else:
        # Load most recent partition
        partitions_to_load = [sorted(partitions)[-1]]
        partition_date = partitions_to_load[0].split("=")[-1].rstrip("/")
        print(f"Loading most recent data: {partition_date}")
    
    # Download parquet files from partition(s)
    all_data = []
    
    for partition in partitions_to_load:
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
    inference_df = pd.concat(all_data, ignore_index=True)
    
    print(f"✓ Loaded {len(inference_df)} records")
    print(f"  Columns: {list(inference_df.columns)}")
    
    return inference_df


def log1p_transform(x):
    """Apply log1p transformation to handle skewed distributions"""
    return np.log1p(x)


def preprocess_inference_data(df, scaler, feature_cols):
    """
    Apply same preprocessing as training data
    
    Args:
        df: Raw feature DataFrame
        scaler: Fitted StandardScaler from training
        feature_cols: List of feature columns (from model metadata)
    
    Returns:
        X_processed: Preprocessed feature matrix
        entity_ids: encounter_id for tracking predictions
    """
    print("\n" + "=" * 80)
    print("Preprocessing Inference Data")
    print("=" * 80)
    
    # Extract entity IDs for later
    entity_ids = df['encounter_id'].values if 'encounter_id' in df.columns else None
    
    # Select feature columns
    X = df[feature_cols].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    
    # Apply log1p transformation ONLY to the 3 columns that need it (same as training)
    # These are the columns with skewed distributions
    log_transform_cols = [
        'age_midpoint',
        'severity_x_visits', 
        'medication_density'
    ]
    
    # Only transform features that exist in the data
    log_transform_cols_present = [f for f in log_transform_cols if f in X.columns]
    
    if log_transform_cols_present:
        print(f"Applying log1p to {len(log_transform_cols_present)} columns: {log_transform_cols_present}")
        X[log_transform_cols_present] = np.log1p(X[log_transform_cols_present])
    
    # Apply scaling (using fitted scaler from training)
    # The scaler was fitted with log1p already applied, so we apply it first
    X_scaled = scaler.transform(X)
    
    print("✓ Preprocessing complete")
    
    return X_scaled, entity_ids


def generate_predictions(model, X, entity_ids, algorithm):
    """
    Generate predictions using trained model
    
    Returns:
        DataFrame with encounter_id and predictions
    """
    print("\n" + "=" * 80)
    print("Generating Predictions")
    print("=" * 80)
    
    # Get probability predictions (for binary classification)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Get class predictions
    y_pred = model.predict(X)
    
    print(f"✓ Generated {len(y_pred)} predictions")
    print(f"  Positive rate: {y_pred.mean():.2%}")
    print(f"  Average probability: {y_pred_proba.mean():.4f}")
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'encounter_id': entity_ids,
        'prediction_probability': y_pred_proba,
        'prediction_class': y_pred,
        'model_algorithm': algorithm,
        'prediction_timestamp': datetime.now().isoformat()
    })
    
    return predictions_df


def save_predictions_to_s3(predictions_df, algorithm):
    """
    Save predictions to S3 datamart
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
    
    # Create predictions folder structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_key = f"gold/model_predictions/{algorithm}/predictions_{timestamp}.parquet"
    
    # Save to local temp file first
    local_path = "/tmp/predictions.parquet"
    predictions_df.to_parquet(local_path, index=False)
    
    # Upload to S3
    print(f"Uploading to s3://{bucket}/{predictions_key}")
    s3_client.upload_file(local_path, bucket, predictions_key)
    
    # Update latest pointer
    latest_key = f"gold/model_predictions/{algorithm}/latest_predictions.parquet"
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': predictions_key},
        Key=latest_key
    )
    
    print(f"✓ Predictions saved successfully")
    print(f"  Path: s3://{bucket}/{predictions_key}")
    print(f"  Latest: s3://{bucket}/{latest_key}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'algorithm': algorithm,
        'num_predictions': len(predictions_df),
        'positive_rate': float(predictions_df['prediction_class'].mean()),
        'avg_probability': float(predictions_df['prediction_probability'].mean()),
        'predictions_path': f"s3://{bucket}/{predictions_key}"
    }
    
    metadata_key = f"gold/model_predictions/{algorithm}/metadata_{timestamp}.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode('utf-8')
    )
    
    print(f"✓ Metadata saved: s3://{bucket}/{metadata_key}")
    
    return f"s3://{bucket}/{predictions_key}"


def main():
    """
    Main inference pipeline
    """
    print("\n" + "=" * 80)
    print("DIABETES READMISSION MODEL INFERENCE PIPELINE")
    print("=" * 80)
    
    # Get inference date from environment (optional)
    inference_date = os.environ.get("INFERENCE_DATE")  # Format: YYYY-MM-DD
    
    # Load configuration
    config = load_config()
    
    # Identify best model
    best_algorithm = get_best_model_info(config)
    
    # Load trained model
    model, metadata = load_model_from_s3(best_algorithm, config)
    
    # Load preprocessing scaler
    scaler = load_preprocessed_scaler_from_s3()
    
    # Load inference data
    inference_df = load_inference_data_from_s3(inference_date)
    
    # Get feature columns from model metadata
    feature_cols = metadata.get('config', {}).get('feature_columns', [])
    if not feature_cols:
        # Fallback: exclude known non-feature columns
        feature_cols = [c for c in inference_df.columns 
                       if c not in ['encounter_id', 'snapshot_date', 'label', 'partition_date']]
    
    print(f"\nUsing {len(feature_cols)} features")
    
    # Preprocess data
    X_processed, entity_ids = preprocess_inference_data(inference_df, scaler, feature_cols)
    
    # Generate predictions
    predictions_df = generate_predictions(model, X_processed, entity_ids, best_algorithm)
    
    # Save predictions to S3
    predictions_path = save_predictions_to_s3(predictions_df, best_algorithm)
    
    print("\n" + "=" * 80)
    print("✓ Inference Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Model: {best_algorithm}")
    print(f"   Records processed: {len(predictions_df)}")
    print(f"   Predictions saved: {predictions_path}")


if __name__ == "__main__":
    main()
