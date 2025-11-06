"""
Model Monitoring Script - Production Version
Monitors model performance by comparing predictions against actual labels

This script:
1. Loads model predictions from S3 (gold/model_predictions/)
2. Loads actual labels from S3 (gold/label_store/)
3. Merges on encounter_id + snapshot_date
4. Calculates metrics per month:
   - AUC (Area Under Curve)
   - GINI coefficient
   - PSI (Population Stability Index)
5. Saves monitoring results as JSON to model-registry bucket

Environment Variables:
    AWS_REGION: AWS region (default: ap-southeast-1)
    DATAMART_BASE_URI: S3 base URI (e.g., s3a://bucket/prefix/)
    MODEL_ALGORITHM: Model algorithm name (optional, auto-selects best if not provided)
    SNAPSHOT_DATE: Specific snapshot date to monitor (optional, reads from predictions if not set)
    MODEL_CONFIG_S3_PATH: S3 path to model config (for auto-selection)
"""

import os
import sys
import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
DATAMART_BASE_URI = os.environ.get("DATAMART_BASE_URI", "s3a://diab-readmit-123456-datamart/")
MODEL_CONFIG_S3_PATH = os.environ.get("MODEL_CONFIG_S3_PATH", "s3://diab-readmit-123456-datamart/config/model_config.json")
SNAPSHOT_DATE = os.environ.get("SNAPSHOT_DATE", '2008-03-01')  # Optional: specific date to monitor

# Model algorithm - can be overridden, otherwise auto-selects best
MODEL_ALGORITHM_OVERRIDE = os.environ.get("MODEL_ALGORITHM", None)

# Ensure S3A protocol
if DATAMART_BASE_URI.startswith("s3://"):
    DATAMART_BASE_URI = DATAMART_BASE_URI.replace("s3://", "s3a://")


def psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI)
    
    Measures the shift in population distribution between two datasets.
    PSI < 0.10: No significant change
    PSI 0.10 - 0.25: Moderate change (schedule retrain)
    PSI > 0.25: Significant change (retrain now)
    
    Args:
        expected: Reference distribution (array-like, typically first month)
        actual: Current distribution (array-like, month being compared)
        buckets: Number of bins for histogram (default 10)
    
    Returns:
        PSI value (float)
    """
    # Create bin edges based on percentiles (0 to 1 for probabilities)
    breakpoints = np.linspace(0, 1, buckets + 1)
    
    # Calculate histograms
    expected_perc, _ = np.histogram(expected, bins=breakpoints)
    actual_perc, _ = np.histogram(actual, bins=breakpoints)
    
    # Convert to percentages
    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)
    
    # Avoid division by zero
    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)
    
    # Calculate PSI
    psi_val = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return psi_val


def csi(expected, actual):
    """
    Characteristic Stability Index (CSI)
    
    Similar to PSI, but for categorical variables.
    Measures distribution shift in categorical features.
    
    Args:
        expected: Reference categorical distribution
        actual: Current categorical distribution
    
    Returns:
        CSI value (float)
    """
    expected_counts = pd.Series(expected).value_counts(normalize=True)
    actual_counts = pd.Series(actual).value_counts(normalize=True)
    
    # Get all categories
    categories = expected_counts.index.union(actual_counts.index)
    
    # Reindex to include all categories
    expected_perc = expected_counts.reindex(categories, fill_value=1e-6)
    actual_perc = actual_counts.reindex(categories, fill_value=1e-6)
    
    # Calculate CSI
    csi_val = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return csi_val


def get_best_model_algorithm():
    """
    Get the best model algorithm from model comparison results
    Same logic as model inference
    
    Returns:
        str: Model algorithm name (e.g., 'xgboost')
    """
    print("=" * 80)
    print("Identifying Best Model")
    print("=" * 80)
    
    # If MODEL_ALGORITHM is explicitly provided, use it
    if MODEL_ALGORITHM_OVERRIDE:
        print(f"✓ Using explicitly specified model: {MODEL_ALGORITHM_OVERRIDE}")
        return MODEL_ALGORITHM_OVERRIDE
    
    # Otherwise, load from model comparison
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Try to get model registry info from config
    try:
        # Step 1: Parse config S3 path (config is in datamart bucket)
        config_s3_path = MODEL_CONFIG_S3_PATH.replace("s3://", "").replace("s3a://", "")
        config_bucket = config_s3_path.split("/")[0]
        config_key = "/".join(config_s3_path.split("/")[1:])
        
        print(f"Step 1: Loading config from s3://{config_bucket}/{config_key}")
        response = s3_client.get_object(Bucket=config_bucket, Key=config_key)
        config = json.loads(response['Body'].read().decode('utf-8'))
        
        # Step 2: Extract model registry bucket from config (different from config bucket!)
        model_registry_bucket = config["model_config"]["model_registry_bucket"]
        model_registry_prefix = config["model_config"]["model_registry_prefix"]
        
        print(f"Step 2: Model registry is at s3://{model_registry_bucket}/{model_registry_prefix}")
        
        # Step 3: Load comparison from model registry bucket
        comparison_key = f"{model_registry_prefix}latest_model_comparison.json"
        print(f"Step 3: Loading comparison from s3://{model_registry_bucket}/{comparison_key}")
        
        response = s3_client.get_object(Bucket=model_registry_bucket, Key=comparison_key)
        comparison = json.loads(response['Body'].read().decode('utf-8'))
        
        recommended_model = comparison.get('recommended_model')
        recommendation_reason = comparison.get('recommendation_reason')
        
        print(f"✓ Best Model: {recommended_model}")
        print(f"  Reason: {recommendation_reason}")
        print()
        
        return recommended_model
        
    except Exception as e:
        print(f"⚠ Could not load model comparison: {e}")
        print("  Falling back to default: xgboost")
        print()
        return "xgboost"


def monitor_model():
    """
    Main monitoring function
    
    Workflow:
    1. Identify best model algorithm (or use override)
    2. Initialize Spark with S3 configuration
    3. Load predictions from S3
    4. Load labels from S3
    5. Merge datasets on encounter_id + snapshot_date
    6. Group by month and calculate metrics
    7. Save monitoring results as JSON to model-registry bucket
    """
    print("=" * 80)
    print("Model Monitoring")
    print("=" * 80)
    
    # Load model registry config
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    config_s3_path = MODEL_CONFIG_S3_PATH.replace("s3://", "").replace("s3a://", "")
    config_bucket = config_s3_path.split("/")[0]
    config_key = "/".join(config_s3_path.split("/")[1:])
    
    response = s3_client.get_object(Bucket=config_bucket, Key=config_key)
    config = json.loads(response['Body'].read().decode('utf-8'))
    model_registry_bucket = config["model_config"]["model_registry_bucket"]
    model_registry_prefix = config["model_config"]["model_registry_prefix"]
    
    # Get the model algorithm to monitor
    MODEL_ALGORITHM = get_best_model_algorithm()
    
    print(f"Monitoring Model: {MODEL_ALGORITHM}")
    print(f"Datamart URI: {DATAMART_BASE_URI}")
    if SNAPSHOT_DATE:
        print(f"Snapshot Date: {SNAPSHOT_DATE}")
    print()
    
    # Initialize Spark with S3 configuration
    print("Initializing Spark...")
    
    # Credentials provider chain (same as model_train.py)
    provider_chain = ",".join([
        "com.amazonaws.auth.EnvironmentVariableCredentialsProvider",
        "com.amazonaws.auth.InstanceProfileCredentialsProvider",
        "com.amazonaws.auth.ContainerCredentialsProvider",
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
    ])
    
    builder = (
        SparkSession.builder
        .appName("ModelMonitoring")
        .master("local[*]")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", provider_chain)
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{AWS_REGION}.amazonaws.com")
    )
    
    # Prefer baked-in JARs; fall back to remote packages if not present
    hadoop_aws_jar = "/opt/spark/jars-extra/hadoop-aws-3.3.4.jar"
    aws_bundle_jar = "/opt/spark/jars-extra/aws-java-sdk-bundle-1.12.639.jar"
    if os.path.exists(hadoop_aws_jar) and os.path.exists(aws_bundle_jar):
        builder = builder.config("spark.jars", f"{hadoop_aws_jar},{aws_bundle_jar}")
    else:
        builder = builder.config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.639")
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark initialized\n")
    
    # Paths
    # If SNAPSHOT_DATE is specified, read from timestamped partition
    # Otherwise read from latest_predictions.parquet
    if SNAPSHOT_DATE:
        snapshot_date_formatted = SNAPSHOT_DATE.replace('-', '')
        pred_path = f"{DATAMART_BASE_URI}gold/model_predictions/{MODEL_ALGORITHM}/{MODEL_ALGORITHM}_predictions_{snapshot_date_formatted}.parquet"
        print(f"Reading predictions for specific snapshot: {SNAPSHOT_DATE}")
    else:
        pred_path = f"{DATAMART_BASE_URI}gold/model_predictions/{MODEL_ALGORITHM}/latest_predictions.parquet"
        print(f"Reading latest predictions")
    
    label_path = f"{DATAMART_BASE_URI}gold/label_store/"
    
    print(f"Predictions path: {pred_path}")
    print(f"Labels path: {label_path}\n")
    
    # --- Load predictions ---
    print("Loading predictions...")
    try:
        # Read the latest_predictions.parquet file directly
        preds_sdf = spark.read.option("header", "true").parquet(pred_path)
        pred_count = preds_sdf.count()
        print(f"✓ Loaded {pred_count:,} prediction records")
        
        # Convert to pandas for easier processing
        preds_pdf = preds_sdf.toPandas()
        print(f"  Prediction columns: {list(preds_pdf.columns)}")
        
        # Verify snapshot_date column exists
        if 'snapshot_date' not in preds_pdf.columns:
            print(f"✗ Error: 'snapshot_date' column not found in predictions")
            print(f"  Available columns: {list(preds_pdf.columns)}")
            spark.stop()
            sys.exit(1)
        
        # Convert snapshot_date to datetime
        preds_pdf['snapshot_date'] = pd.to_datetime(preds_pdf['snapshot_date'])
        
        # Get unique snapshot dates in predictions
        pred_snapshot_dates = sorted(preds_pdf['snapshot_date'].unique())
        print(f"  Found {len(pred_snapshot_dates)} unique snapshot dates in predictions")
        print(f"  Date range: {pred_snapshot_dates[0]} to {pred_snapshot_dates[-1]}")
        
        # If SNAPSHOT_DATE is specified, filter predictions
        if SNAPSHOT_DATE:
            filter_date = pd.to_datetime(SNAPSHOT_DATE)
            preds_pdf = preds_pdf[preds_pdf['snapshot_date'] == filter_date]
            if len(preds_pdf) == 0:
                print(f"✗ Error: No predictions found for snapshot_date={SNAPSHOT_DATE}")
                spark.stop()
                sys.exit(1)
            print(f"  Filtered to snapshot_date={SNAPSHOT_DATE}: {len(preds_pdf):,} records")
        
        preds_pdf = preds_pdf.sort_values(by='encounter_id')
        print()
        
    except Exception as e:
        print(f"✗ Error loading predictions: {e}")
        print("  Make sure inference DAG has run successfully")
        import traceback
        traceback.print_exc()
        spark.stop()
        sys.exit(1)
    
    # --- Load labels ---
    print("Loading labels...")
    try:
        # If SNAPSHOT_DATE is specified, read only that specific partition
        # Otherwise read all partitions
        if SNAPSHOT_DATE:
            # Parse snapshot date to get year and month for partition filtering
            snapshot_dt = pd.to_datetime(SNAPSHOT_DATE)
            year = snapshot_dt.year
            month = str(snapshot_dt.month).zfill(2)
            
            # Read specific partition if it exists
            # Label partitions are like: gold_label_store_YYYY_MM.parquet/
            label_partition_path = f"{label_path}gold_label_store_{year}_{month}.parquet/"
            print(f"  Reading label partition: {label_partition_path}")
            
            try:
                labels_sdf = (
                    spark.read
                    .option("header", "true")
                    .option("mergeSchema", "true")
                    .parquet(label_partition_path)
                )
                label_count = labels_sdf.count()
                print(f"✓ Loaded {label_count:,} label records from partition {year}-{month}")
            except Exception as partition_error:
                print(f"  ⚠ Could not read specific partition, reading all labels and filtering...")
                # Fallback to reading all and filtering
                labels_sdf = (
                    spark.read
                    .option("header", "true")
                    .option("mergeSchema", "true")
                    .option("recursiveFileLookup", "true")
                    .parquet(label_path)
                )
                label_count = labels_sdf.count()
                print(f"✓ Loaded {label_count:,} label records (all partitions)")
        else:
            # Read all label partitions
            labels_sdf = (
                spark.read
                .option("header", "true")
                .option("mergeSchema", "true")
                .option("recursiveFileLookup", "true")
                .parquet(label_path)
            )
            label_count = labels_sdf.count()
            print(f"✓ Loaded {label_count:,} label records (all partitions)")
        
        # Convert to pandas
        labels_pdf = labels_sdf.toPandas()
        print(f"  Label columns: {list(labels_pdf.columns)}")
        
        # Verify snapshot_date column exists
        if 'snapshot_date' not in labels_pdf.columns:
            print(f"✗ Error: 'snapshot_date' column not found in labels")
            print(f"  Available columns: {list(labels_pdf.columns)}")
            spark.stop()
            sys.exit(1)
        
        # Convert snapshot_date to datetime
        labels_pdf['snapshot_date'] = pd.to_datetime(labels_pdf['snapshot_date'])
        
        # Get unique snapshot dates in labels
        label_snapshot_dates = sorted(labels_pdf['snapshot_date'].unique())
        print(f"  Found {len(label_snapshot_dates)} unique snapshot dates in labels")
        if len(label_snapshot_dates) > 0:
            print(f"  Date range: {label_snapshot_dates[0]} to {label_snapshot_dates[-1]}")
        
        # If SNAPSHOT_DATE is specified, filter labels to exact date
        if SNAPSHOT_DATE:
            filter_date = pd.to_datetime(SNAPSHOT_DATE)
            original_count = len(labels_pdf)
            labels_pdf = labels_pdf[labels_pdf['snapshot_date'] == filter_date]
            if len(labels_pdf) == 0:
                print(f"✗ Error: No labels found for exact snapshot_date={SNAPSHOT_DATE}")
                print(f"  Loaded {original_count} records but none matched the date")
                print(f"  Available dates in loaded data: {sorted(pd.to_datetime(labels_pdf['snapshot_date']).dt.date.unique()) if original_count > 0 else 'None'}")
                spark.stop()
                sys.exit(1)
            print(f"  Filtered to snapshot_date={SNAPSHOT_DATE}: {len(labels_pdf):,} records")
        
        labels_pdf = labels_pdf.sort_values(by='encounter_id')
        print()
        
    except Exception as e:
        print(f"✗ Error loading labels: {e}")
        print("  Make sure data processing DAG has created label_store")
        import traceback
        traceback.print_exc()
        spark.stop()
        sys.exit(1)
    
    # --- Merge predictions with true labels ---
    print("Merging predictions with labels...")
    print(f"Merging on: ['encounter_id', 'snapshot_date']")
    
    # Verify snapshot dates match
    pred_dates = set(preds_pdf['snapshot_date'].unique())
    label_dates = set(labels_pdf['snapshot_date'].unique())
    
    # Find common dates
    common_dates = pred_dates.intersection(label_dates)
    missing_in_labels = pred_dates - label_dates
    missing_in_preds = label_dates - pred_dates
    
    if missing_in_labels:
        print(f"⚠ Warning: Predictions exist for dates not in labels: {sorted(missing_in_labels)}")
    if missing_in_preds:
        print(f"⚠ Warning: Labels exist for dates not in predictions: {sorted(missing_in_preds)}")
    
    if not common_dates:
        print(f"✗ Error: No common snapshot dates between predictions and labels")
        print(f"  Prediction dates: {sorted(pred_dates)}")
        print(f"  Label dates: {sorted(label_dates)}")
        spark.stop()
        sys.exit(1)
    
    print(f"  Common snapshot dates: {len(common_dates)}")
    print(f"  Date range: {min(common_dates)} to {max(common_dates)}")
    
    # Select only necessary columns from labels to avoid duplicates
    merged_pdf = preds_pdf.merge(
        labels_pdf[["encounter_id", "snapshot_date", "label"]],
        on=["encounter_id", "snapshot_date"],
        how="inner"
    )
    
    print(f"✓ Merged {len(merged_pdf):,} matching records")
    
    if len(merged_pdf) == 0:
        print(f"✗ Error: No matching records found after merge")
        print(f"  Predictions: {len(preds_pdf):,} records")
        print(f"  Labels: {len(labels_pdf):,} records")
        print(f"  Check that encounter_id and snapshot_date values match")
        spark.stop()
        sys.exit(1)
    
    print()
    if merged_pdf.empty:
        print("✗ Error: Merged dataframe is empty after filtering")
        spark.stop()
        sys.exit(1)
    
    # Ensure snapshot_date is datetime
    merged_pdf["snapshot_date"] = pd.to_datetime(merged_pdf["snapshot_date"])
    
    # Create a month column in YYYY-MM format
    merged_pdf["snapshot_month"] = merged_pdf["snapshot_date"].dt.to_period("M").astype(str)
    
    months = sorted(merged_pdf["snapshot_month"].unique())
    print(f"Found {len(months)} unique months: {', '.join(months)}\n")
    
    # Use the first month as reference for stability calculations
    reference_month = months[0]
    reference_df = merged_pdf[merged_pdf["snapshot_month"] == reference_month]
    
    print(f"Using {reference_month} as reference month (n={len(reference_df):,} records)\n")
    print("=" * 80)
    print("Calculating Metrics by Month")
    print("=" * 80)
    
    monitoring_records = []
    
    for month in months:
        temp_df = merged_pdf[merged_pdf["snapshot_month"] == month]
        
        # Calculate AUC and GINI (requires at least 2 classes)
        if len(temp_df["label"].unique()) > 1:
            try:
                auc = roc_auc_score(temp_df["label"], temp_df["model_predictions"])
                gini = 2 * auc - 1
            except Exception as e:
                print(f"  Warning: Could not calculate AUC for {month}: {e}")
                auc = None
                gini = None
        else:
            auc = None
            gini = None
        
        # Calculate PSI for score distribution
        try:
            psi_val = psi(reference_df["model_predictions"], temp_df["model_predictions"])
        except Exception as e:
            print(f"  Warning: Could not calculate PSI for {month}: {e}")
            psi_val = None
        
        # Print summary
        print(f"\n{month}:")
        print(f"  Records: {len(temp_df):,}")
        if auc is not None:
            print(f"  AUC: {auc:.4f}")
            print(f"  GINI: {gini:.4f}")
        if psi_val is not None:
            print(f"  PSI: {psi_val:.4f}")
            # Interpret PSI
            if psi_val < 0.10:
                print(f"  PSI Status: ✓ No significant change")
            elif psi_val < 0.25:
                print(f"  PSI Status: ⚠️ Moderate drift")
            else:
                print(f"  PSI Status: 🚨 Significant drift")
        
        monitoring_records.append({
            "snapshot_month": month,
            "row_count": len(temp_df),
            "auc": auc,
            "gini": gini,
            "psi": psi_val,
        })
    
    # Create monitoring dataframe
    monitoring_df = pd.DataFrame(monitoring_records)
    
    print("\n" + "=" * 80)
    print("Saving Monitoring Results")
    print("=" * 80)
    
    # Prepare JSON output
    try:
        # Get latest metrics
        latest_month = monitoring_df.iloc[-1]
        
        # Create monitoring JSON structure
        monitoring_json = {
            "algorithm": MODEL_ALGORITHM,
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "latest_snapshot_month": latest_month["snapshot_month"],
            "latest_metrics": {
                "auc": float(latest_month["auc"]) if pd.notna(latest_month["auc"]) else None,
                "gini": float(latest_month["gini"]) if pd.notna(latest_month["gini"]) else None,
                "psi": float(latest_month["psi"]) if pd.notna(latest_month["psi"]) else None,
                "row_count": int(latest_month["row_count"])
            },
            "status": _get_health_status(latest_month),
            "monthly_summary": [
                {
                    "month": row["snapshot_month"],
                    "auc": float(row["auc"]) if pd.notna(row["auc"]) else None,
                    "gini": float(row["gini"]) if pd.notna(row["gini"]) else None,
                    "psi": float(row["psi"]) if pd.notna(row["psi"]) else None,
                    "row_count": int(row["row_count"])
                }
                for _, row in monitoring_df.iterrows()
            ]
        }
        
        # Save to model registry bucket
        monitoring_key = f"{model_registry_prefix}monitoring/{MODEL_ALGORITHM}_latest_monitoring.json"
        s3_client.put_object(
            Bucket=model_registry_bucket,
            Key=monitoring_key,
            Body=json.dumps(monitoring_json, indent=2),
            ContentType='application/json'
        )
        
        monitoring_s3_path = f"s3://{model_registry_bucket}/{monitoring_key}"
        print(f"✓ Monitoring results saved to:\n  {monitoring_s3_path}\n")
        
        # Print summary
        print("Summary Statistics:")
        print(monitoring_df.to_string(index=False))
        print()
        
    except Exception as e:
        print(f"✗ Error saving monitoring results: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        sys.exit(1)


def _get_health_status(metrics_row):
    """
    Determine health status based on metrics
    
    Args:
        metrics_row: Series with auc, psi values
    
    Returns:
        str: 'critical', 'warning', or 'healthy'
    """
    auc = metrics_row["auc"]
    psi = metrics_row["psi"]
    
    # Critical if AUC too low or PSI too high
    if pd.notna(auc) and auc < 0.70:
        return "critical"
    if pd.notna(psi) and psi > 0.25:
        return "critical"
    
    # Warning if PSI moderate
    if pd.notna(psi) and psi > 0.10:
        return "warning"
    
    return "healthy"
    
    print("=" * 80)
    print("Model Monitoring Complete")
    print("=" * 80)
    
    spark.stop()
    return monitoring_df


if __name__ == "__main__":
    try:
        monitor_model()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
