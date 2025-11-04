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
5. Saves monitoring results to S3 (gold/model_monitoring/)

Environment Variables:
    AWS_REGION: AWS region (default: ap-southeast-1)
    DATAMART_BASE_URI: S3 base URI (e.g., s3a://bucket/prefix/)
    MODEL_ALGORITHM: Model algorithm name (e.g., xgboost, default)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
DATAMART_BASE_URI = os.environ.get("DATAMART_BASE_URI", "s3a://diab-readmit-123456-datamart/")
MODEL_ALGORITHM = os.environ.get("MODEL_ALGORITHM", "xgboost")

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


def monitor_model():
    """
    Main monitoring function
    
    Workflow:
    1. Initialize Spark with S3 configuration
    2. Load predictions from S3
    3. Load labels from S3
    4. Merge datasets on encounter_id + snapshot_date
    5. Group by month and calculate metrics
    6. Save monitoring results to S3
    """
    print("=" * 80)
    print("Model Monitoring")
    print("=" * 80)
    print(f"Model Algorithm: {MODEL_ALGORITHM}")
    print(f"Datamart URI: {DATAMART_BASE_URI}")
    print()
    
    # Initialize Spark with S3 configuration
    print("Initializing Spark...")
    spark = (
        SparkSession.builder
        .appName("ModelMonitoring")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{AWS_REGION}.amazonaws.com")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark initialized\n")
    
    # Paths
    pred_path = f"{DATAMART_BASE_URI}gold/model_predictions/{MODEL_ALGORITHM}/"
    label_path = f"{DATAMART_BASE_URI}gold/label_store/"
    monitoring_path = f"{DATAMART_BASE_URI}gold/model_monitoring/{MODEL_ALGORITHM}_monitoring.parquet"
    
    print(f"Predictions path: {pred_path}")
    print(f"Labels path: {label_path}")
    print(f"Output path: {monitoring_path}\n")
    
    # --- Load predictions ---
    print("Loading predictions...")
    try:
        preds_sdf = spark.read.option("header", "true").parquet(pred_path)
        pred_count = preds_sdf.count()
        print(f"✓ Loaded {pred_count:,} prediction records\n")
        
        # Convert to pandas for easier processing
        preds_pdf = preds_sdf.toPandas().sort_values(by='encounter_id')
        
    except Exception as e:
        print(f"✗ Error loading predictions: {e}")
        print("  Make sure inference DAG has run successfully")
        spark.stop()
        sys.exit(1)
    
    # --- Load labels ---
    print("Loading labels...")
    try:
        labels_sdf = spark.read.option("header", "true").parquet(label_path)
        label_count = labels_sdf.count()
        print(f"✓ Loaded {label_count:,} label records\n")
        
        # Convert to pandas
        labels_pdf = labels_sdf.toPandas().sort_values(by='encounter_id')
        
    except Exception as e:
        print(f"✗ Error loading labels: {e}")
        print("  Make sure data processing DAG has created label_store")
        spark.stop()
        sys.exit(1)
    
    # --- Merge predictions with true labels ---
    print("Merging predictions with labels...")
    merged_pdf = preds_pdf.merge(
        labels_pdf[["encounter_id", "snapshot_date", "label"]],
        on=["encounter_id", "snapshot_date"],
        how="inner"
    )
    
    print(f"✓ Merged {len(merged_pdf):,} matching records\n")
    
    if merged_pdf.empty:
        print("⚠️ Warning: No matching records found!")
        print("  Check that:")
        print("  - Predictions and labels have matching encounter_ids")
        print("  - snapshot_date values align between datasets")
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
    
    # Save to S3
    try:
        spark_monitoring_df = spark.createDataFrame(monitoring_df)
        spark_monitoring_df.write.mode("overwrite").parquet(monitoring_path)
        print(f"✓ Monitoring results saved to:\n  {monitoring_path}\n")
        
        # Print summary
        print("Summary Statistics:")
        print(monitoring_df.to_string(index=False))
        print()
        
    except Exception as e:
        print(f"✗ Error saving monitoring results: {e}")
        spark.stop()
        sys.exit(1)
    
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
