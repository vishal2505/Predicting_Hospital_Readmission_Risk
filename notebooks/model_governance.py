from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc
import subprocess
import os

def main(modelname, auto_retrain=False):
    print("\n=== Starting Model Governance Check (PySpark) ===\n")

    spark = (
        SparkSession.builder
        .appName("ModelGovernance")
        .getOrCreate()
    )

    monitoring_path = f"datamart/gold/model_monitoring/{modelname[:-4]}_monitoring.parquet"
    governance_path = f"datamart/gold/model_governance/{modelname[:-4]}_governance.parquet"

    if not os.path.exists(monitoring_path):
        raise FileNotFoundError(f"Monitoring file not found: {monitoring_path}")

    # Read monitoring data
    monitoring_df = spark.read.parquet(monitoring_path)

    # Get latest snapshot by date
    latest_df = (
        monitoring_df
        .orderBy(desc("snapshot_month"))
        .limit(1)
    )

    latest_record = latest_df.collect()[0]
    auc = latest_record["auc"]
    psi = latest_record["psi"]
    snapshot_date = latest_record["snapshot_month"]

    print(f"📅 Snapshot: {snapshot_date}")
    print(f"📊 Latest AUC: {auc:.3f}")
    print(f"📈 Latest PSI: {psi:.3f}")

    # Governance thresholds
    AUC_THRESHOLD = 0.70
    PSI_WARNING = 0.10
    PSI_CRITICAL = 0.25

    # Governance logic
    if auc < AUC_THRESHOLD or psi > PSI_CRITICAL:
        decision = "Retrain"
        retrain_needed = True
        print("🚨 Model degradation detected. Retraining required.")
    elif psi > PSI_WARNING:
        decision = "Schedule Retrain"
        retrain_needed = False
        print("⚠️ Moderate drift detected. Schedule retraining.")
    else:
        decision = "No Action"
        retrain_needed = False
        print("✅ Model is healthy.")

    # Write governance decision
    result_df = spark.createDataFrame(
        [(modelname, snapshot_date, float(auc), float(psi), decision)],
        ["model_name", "latest_snapshot", "auc", "psi", "decision"]
    )

    os.makedirs(os.path.dirname(governance_path), exist_ok=True)
    result_df.write.mode("overwrite").parquet(governance_path)
    print(f"✅ Governance decision saved to: {governance_path}")

    # Trigger retraining (optional)
    if retrain_needed and auto_retrain:
        print("\n--- Initiating model retraining via train.py ---\n")
        subprocess.run(["python", "train.py"], check=True)

    print("\n=== Model Governance Check Completed ===\n")
    spark.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model Governance and Auto-Refresh (PySpark)")
    parser.add_argument("--modelname", type=str, required=True, help="Model filename (e.g., credit_model_2024_06_01.pkl)")
    parser.add_argument("--auto_retrain", action="store_true", help="Automatically trigger retraining if needed")
    args = parser.parse_args()

    main(args.modelname, args.auto_retrain)
