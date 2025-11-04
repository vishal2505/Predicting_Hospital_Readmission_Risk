"""
Model Governance Script - Production Version
Makes automated decisions about model retraining based on monitoring results

This script:
1. Reads monitoring results from S3 (gold/model_monitoring/)
2. Extracts latest snapshot metrics (AUC, PSI)
3. Applies governance thresholds:
   - AUC < 0.70 OR PSI > 0.25 → Retrain (critical)
   - PSI > 0.10 → Schedule Retrain (warning)
   - Otherwise → No Action (healthy)
4. Saves governance decision to S3 (gold/model_governance/)
5. Optionally triggers retraining (if auto_retrain=true)

Environment Variables:
    AWS_REGION: AWS region (default: ap-southeast-1)
    DATAMART_BASE_URI: S3 base URI (e.g., s3a://bucket/prefix/)
    MODEL_ALGORITHM: Model algorithm name (e.g., xgboost)
    AUTO_RETRAIN: Enable auto-retrain trigger (default: false)

Governance Thresholds:
    AUC_THRESHOLD = 0.70 (below this triggers retrain)
    PSI_WARNING = 0.10 (moderate drift)
    PSI_CRITICAL = 0.25 (significant drift, triggers retrain)
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc

# Configuration
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
DATAMART_BASE_URI = os.environ.get("DATAMART_BASE_URI", "s3a://diab-readmit-123456-datamart/")
MODEL_ALGORITHM = os.environ.get("MODEL_ALGORITHM", "xgboost")
AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "false").lower() == "true"

# Ensure S3A protocol
if DATAMART_BASE_URI.startswith("s3://"):
    DATAMART_BASE_URI = DATAMART_BASE_URI.replace("s3://", "s3a://")

# Governance thresholds
AUC_THRESHOLD = 0.70
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25


def make_governance_decision():
    """
    Main governance function
    
    Workflow:
    1. Initialize Spark with S3 configuration
    2. Read monitoring results from S3
    3. Get latest snapshot metrics
    4. Apply governance thresholds
    5. Make decision (Retrain / Schedule Retrain / No Action)
    6. Save governance decision to S3
    7. Optionally trigger retraining
    """
    print("=" * 80)
    print("Model Governance")
    print("=" * 80)
    print(f"Model Algorithm: {MODEL_ALGORITHM}")
    print(f"Datamart URI: {DATAMART_BASE_URI}")
    print(f"Auto-Retrain: {AUTO_RETRAIN}")
    print()
    
    # Initialize Spark with S3 configuration
    print("Initializing Spark...")
    spark = (
        SparkSession.builder
        .appName("ModelGovernance")
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
    monitoring_path = f"{DATAMART_BASE_URI}gold/model_monitoring/{MODEL_ALGORITHM}_monitoring.parquet"
    governance_path = f"{DATAMART_BASE_URI}gold/model_governance/{MODEL_ALGORITHM}_governance.parquet"
    
    print(f"Monitoring input: {monitoring_path}")
    print(f"Governance output: {governance_path}\n")
    
    # --- Read monitoring data ---
    print("Loading monitoring results...")
    try:
        monitoring_df = spark.read.parquet(monitoring_path)
        record_count = monitoring_df.count()
        print(f"✓ Loaded {record_count} monitoring records\n")
        
    except Exception as e:
        print(f"✗ Error loading monitoring results: {e}")
        print("  Make sure monitoring DAG task has run successfully")
        spark.stop()
        sys.exit(1)
    
    # --- Get latest snapshot by date ---
    print("Finding latest snapshot...")
    latest_df = (
        monitoring_df
        .orderBy(desc("snapshot_month"))
        .limit(1)
    )
    
    if latest_df.count() == 0:
        print("✗ No monitoring records found")
        spark.stop()
        sys.exit(1)
    
    latest_record = latest_df.collect()[0]
    
    # Extract metrics
    auc = latest_record["auc"]
    psi = latest_record["psi"]
    snapshot_month = latest_record["snapshot_month"]
    row_count = latest_record["row_count"]
    
    print(f"✓ Latest snapshot: {snapshot_month}\n")
    
    print("=" * 80)
    print("Latest Metrics")
    print("=" * 80)
    print(f"📅 Snapshot Month: {snapshot_month}")
    print(f"📊 Records: {row_count:,}")
    if auc is not None:
        print(f"📈 AUC: {auc:.4f} (threshold: {AUC_THRESHOLD:.2f})")
        gini = 2 * auc - 1
        print(f"📉 GINI: {gini:.4f}")
    else:
        print(f"📈 AUC: N/A")
    
    if psi is not None:
        print(f"🔄 PSI: {psi:.4f} (warning: {PSI_WARNING:.2f}, critical: {PSI_CRITICAL:.2f})")
    else:
        print(f"🔄 PSI: N/A")
    
    print()
    
    # --- Apply governance logic ---
    print("=" * 80)
    print("Governance Decision")
    print("=" * 80)
    
    retrain_needed = False
    decision = "No Action"
    reason = []
    
    # Check AUC threshold
    if auc is not None and auc < AUC_THRESHOLD:
        decision = "Retrain"
        retrain_needed = True
        reason.append(f"AUC ({auc:.4f}) below threshold ({AUC_THRESHOLD:.2f})")
        print(f"🚨 Critical: AUC degradation detected")
    
    # Check PSI thresholds
    if psi is not None:
        if psi > PSI_CRITICAL:
            decision = "Retrain"
            retrain_needed = True
            reason.append(f"PSI ({psi:.4f}) exceeds critical threshold ({PSI_CRITICAL:.2f})")
            print(f"🚨 Critical: Significant model drift detected")
        elif psi > PSI_WARNING and decision != "Retrain":
            decision = "Schedule Retrain"
            reason.append(f"PSI ({psi:.4f}) exceeds warning threshold ({PSI_WARNING:.2f})")
            print(f"⚠️ Warning: Moderate model drift detected")
    
    if decision == "No Action":
        print(f"✅ Model is healthy - no action required")
    
    # Print decision summary
    print()
    print(f"Decision: {decision}")
    if reason:
        print(f"Reason: {'; '.join(reason)}")
    print()
    
    # --- Save governance decision ---
    print("=" * 80)
    print("Saving Governance Decision")
    print("=" * 80)
    
    try:
        # Create governance record
        governance_record = spark.createDataFrame(
            [(
                MODEL_ALGORITHM,
                snapshot_month,
                float(auc) if auc is not None else None,
                float(psi) if psi is not None else None,
                decision,
                '; '.join(reason) if reason else "Model metrics within acceptable range"
            )],
            ["model_name", "latest_snapshot", "auc", "psi", "decision", "reason"]
        )
        
        # Save to S3
        governance_record.write.mode("overwrite").parquet(governance_path)
        print(f"✓ Governance decision saved to:\n  {governance_path}\n")
        
        # Print record
        print("Governance Record:")
        governance_record.show(truncate=False)
        
    except Exception as e:
        print(f"✗ Error saving governance decision: {e}")
        spark.stop()
        sys.exit(1)
    
    # --- Trigger retraining (optional) ---
    if retrain_needed and AUTO_RETRAIN:
        print("=" * 80)
        print("Auto-Retrain Trigger")
        print("=" * 80)
        print("🔄 Retraining required and AUTO_RETRAIN=true")
        print()
        print("Note: Automated retraining trigger would be implemented here.")
        print("Options:")
        print("  1. Trigger Airflow DAG via REST API")
        print("  2. Send SNS notification to start training pipeline")
        print("  3. Write trigger file to S3 that training DAG monitors")
        print()
        print("For production implementation, use Airflow TriggerDagRunOperator")
        print("or boto3 to invoke training DAG programmatically.")
        print()
        
        # Example: Write trigger file to S3
        trigger_path = f"{DATAMART_BASE_URI}triggers/retrain_trigger_{MODEL_ALGORITHM}.txt"
        trigger_df = spark.createDataFrame(
            [(snapshot_month, decision, '; '.join(reason))],
            ["snapshot_month", "decision", "reason"]
        )
        trigger_df.write.mode("overwrite").parquet(trigger_path)
        print(f"✓ Trigger file written to: {trigger_path}")
        print()
    
    elif retrain_needed and not AUTO_RETRAIN:
        print("=" * 80)
        print("Manual Intervention Required")
        print("=" * 80)
        print("🔄 Retraining required but AUTO_RETRAIN=false")
        print()
        print("Action needed:")
        print("  1. Review governance decision in S3")
        print("  2. Manually trigger training DAG: diab_model_training")
        print("  3. Or set AUTO_RETRAIN=true to enable automatic retraining")
        print()
    
    print("=" * 80)
    print("Model Governance Complete")
    print("=" * 80)
    
    spark.stop()
    
    # Return decision for programmatic use
    return {
        "decision": decision,
        "retrain_needed": retrain_needed,
        "auc": auc,
        "psi": psi,
        "snapshot_month": snapshot_month
    }


if __name__ == "__main__":
    try:
        result = make_governance_decision()
        
        # Exit with code based on decision
        # 0 = No Action, 1 = Schedule Retrain, 2 = Retrain
        if result["decision"] == "Retrain":
            sys.exit(2)
        elif result["decision"] == "Schedule Retrain":
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)
