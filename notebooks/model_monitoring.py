import os
import glob
import pandas as pd
from sklearn.metrics import roc_auc_score
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import numpy as np
import pandas as pd

def psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI)
    expected: reference distribution (array-like)
    actual: current distribution (array-like)
    """
    # Bin edges based on expected
    breakpoints = np.linspace(0, 1, buckets + 1)
    
    expected_perc, _ = np.histogram(expected, bins=breakpoints)
    actual_perc, _ = np.histogram(actual, bins=breakpoints)
    
    # Avoid division by zero
    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)
    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)
    
    psi_val = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return psi_val

def csi(expected, actual, buckets=10):
    """
    Characteristic Stability Index (CSI)
    Similar to PSI, but can be applied to categorical variables.
    """
    expected_counts = pd.Series(expected).value_counts(normalize=True)
    actual_counts = pd.Series(actual).value_counts(normalize=True)
    
    categories = expected_counts.index.union(actual_counts.index)
    
    expected_perc = expected_counts.reindex(categories, fill_value=1e-6)
    actual_perc = actual_counts.reindex(categories, fill_value=1e-6)
    
    csi_val = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return csi_val

def monitor_model(pred_folder, label_folder, modelname):
    # Initialize Spark
    spark = SparkSession.builder.appName("model_monitor").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    # --- Load predictions ---
    pred_files = glob.glob(os.path.join(pred_folder, "*"))
    preds_sdf = spark.read.option("header", "true").parquet(*pred_files)
    preds_pdf = preds_sdf.toPandas().sort_values(by='encounter_id')
    
    # --- Load labels ---
    label_files = glob.glob(os.path.join(label_folder, "*"))
    labels_sdf = spark.read.option("header", "true").parquet(*label_files)
    labels_pdf = labels_sdf.toPandas().sort_values(by='encounter_id')
    
    # --- Merge predictions with true labels (on customer_id + snapshot_date) ---
    merged_pdf = preds_pdf.merge(
        labels_pdf[["encounter_id", "snapshot_date", "label"]],
        on=["encounter_id", "snapshot_date"],
        how="inner"
    )
    
    print(f"Merged rows: {len(merged_pdf)}")
    if merged_pdf.empty:
        print("⚠️ Warning: No matching records found. Check date alignment.")
        spark.stop()
        return None
    
    # Ensure snapshot_date is datetime
    merged_pdf["snapshot_date"] = pd.to_datetime(merged_pdf["snapshot_date"])
    
    # Create a month column in YYYY-MM format
    merged_pdf["snapshot_month"] = merged_pdf["snapshot_date"].dt.to_period("M").astype(str)

    # Use the first month as reference for stability calculations
    reference_month = merged_pdf["snapshot_month"].min()
    reference_df = merged_pdf[merged_pdf["snapshot_month"] == reference_month]
    
    monitoring_records = []
    
    for month in sorted(merged_pdf["snapshot_month"].unique()):
        temp_df = merged_pdf[merged_pdf["snapshot_month"] == month]
        if len(temp_df["label"].unique()) > 1:  # roc_auc_score requires at least 2 classes
            auc = roc_auc_score(temp_df["label"], temp_df["model_predictions"])
            gini = 2 * auc - 1
        else:
            auc = None
            gini = None

        # PSI for score distribution
        psi_val = psi(reference_df["model_predictions"], temp_df["model_predictions"])
        
        monitoring_records.append({
            "snapshot_month": month,
            "row_count": len(temp_df),
            "auc": auc,
            "gini": gini,
            "psi": psi_val,
        })
    
    monitoring_df = pd.DataFrame(monitoring_records)
    
    # --- Save monitoring results as gold table ---
    gold_dir = f"datamart/gold/model_monitoring/"
    os.makedirs(gold_dir, exist_ok=True)
    gold_path = os.path.join(gold_dir, f"{modelname[:-4]}_monitoring.parquet")
    spark.createDataFrame(monitoring_df).write.mode("overwrite").parquet(gold_path)
    
    print(f"Monitoring results saved to {gold_path}")
    
    spark.stop()
    return monitoring_df

# Example usage
monitor_df = monitor_model(
    pred_folder="datamart/gold/model_predictions/diabetes_stackensemble_2009_01_01",
    label_folder="datamart/gold/label_store",
    modelname = "diabetes_stackensemble_2009_01_01.pkl"
)
print(monitor_df)
